#!/usr/bin/env python3
"""Resumable, paired LTE BER measurements. Partial shards are never full results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfsep.baselines.lte_estimated import approximate_wiener, load_statistics, marginal_bit_map
from rfsep.config import load_platform
from rfsep.contracts import Observation, complex_to_network, network_to_complex
from rfsep.data.manifests import build_manifest
from rfsep.evaluation.metrics import bit_errors, nmse
from rfsep.observation import build_example, load_calibration
from rfsep.priors.inference import load_prior_for_inference
from rfsep.separation.rgs import RGSConfig, separate_rgs
from rfsep.waveforms.receiver import receive_target, target_demodulate
from smoke_rgs import _direct_initial

SIRS = (-24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6)
TEST_IDS = tuple(range(2000000, 2000256))
AUDIT_IDS = tuple(range(1000000, 1000004))
TARGET = ROOT / "artifacts/prior_formal/prior-formal-20260923/target/checkpoints/last.pt"
INTERFERENCE = ROOT / "artifacts/prior_formal/prior-formal-20260923/interference/checkpoints/last.pt"
EXPECTED_SHAS = {
    "target": "bba67840419fd5951b8e2ac3a70f8be84a2efaa720d637c9e6f925a0418278be",
    "interference": "0d2270eaa46886afc9da4571e92fff042af78b348e20d185b1c531c634b13ad0",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def context(args):
    platform = load_platform(ROOT / "configs/platform/lte_v1.yaml")
    calibration = load_calibration(ROOT / "reports/formal_validation/calibration_summary.json", platform)
    ids = TEST_IDS if args.split == "test" else AUDIT_IDS[:args.audit_parents]
    manifest = build_manifest(platform, args.split, len(ids))
    # Explicit high parent IDs are not represented by build_manifest's default 0-based records.
    manifest["records"] = [dict(row, record_id=parent, example_id=f"pair:{args.split}:{parent:08d}")
                           for row, parent in zip(manifest["records"], ids)]
    for row, parent in zip(manifest["records"], ids):
        row["target_parent"] = f"target:{args.split}:{parent:08d}"
        row["interference_parent"] = f"interference:{args.split}:{parent:08d}"
    manifest.pop("manifest_digest")
    manifest["manifest_digest"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    manifest_file = run / "sample_manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("existing manifest differs")
    else:
        atomic_json(manifest_file, manifest)
    config = {
        "schema": "lte-rgs-ber-5000-v1", "platform_hash": platform.config_hash,
        "calibration_id": calibration["calibration_id"], "manifest_digest": manifest["manifest_digest"],
        "split": args.split, "parent_first": ids[0], "parent_last": ids[-1],
        "sir_re_db": SIRS if args.split == "test" else tuple(args.audit_sirs),
        "rgs_iterations": args.iterations, "rgs_max_lr": args.max_lr, "rgs_min_lr": 1e-6,
        "rgs_omega": "kappa_squared", "rgs_timesteps": [1, 49],
        "initialization": "direct_hard_qpsk_remodulation", "batch_parents": args.batch_parents,
        "target_checkpoint_sha256": EXPECTED_SHAS["target"],
        "interference_checkpoint_sha256": EXPECTED_SHAS["interference"],
        "baseline_statistics_sha256": digest(Path(args.statistics)),
    }
    config_file = run / "resolved_config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != json.loads(json.dumps(config)):
            raise ValueError("existing run config differs")
    else:
        atomic_json(config_file, config)
    return platform, calibration, ids, run, config


def expected_file(path: Path, config: dict) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if payload["config_sha256"] != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise ValueError(f"stale shard config: {path}")
    if payload["status"] != "complete":
        raise ValueError(f"noncomplete shard: {path}")
    return True


def save_shard(path: Path, config: dict, rows: list[dict], elapsed: float) -> None:
    atomic_json(path, {"status": "complete", "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                       "elapsed_seconds": elapsed, "rows": rows})


def baseline_estimates(observation, info, platform, stats):
    direct = receive_target(observation, info, platform.waveform_spec)
    wiener_wave = approximate_wiener(observation.y, info.kappa, stats)
    wiener = receive_target(Observation(observation.example_id, wiener_wave), info, platform.waveform_spec)
    bit_map = marginal_bit_map(target_demodulate(observation.y, platform.waveform_spec), info.kappa, stats)
    return (("direct", direct.target_bits, observation.y),
            ("fft_diagonal_wiener_approx", wiener.target_bits, wiener_wave),
            ("single_re_marginal_bit_map_estimated", bit_map, None))


def actual_sir_re(truth, kappa, platform):
    s = target_demodulate(truth.target_waveform, platform.waveform_spec)
    b = target_demodulate(truth.interference_waveform, platform.waveform_spec)
    return float(10 * np.log10(np.sum(np.abs(s) ** 2) /
                               (kappa * kappa * np.sum(np.abs(b) ** 2))))


def inner_modulation_counts(truth):
    return dict(Counter(mod for frame in truth.interference_modulation_map
                        for mod in frame[1:3]))


def run_baselines(args, platform, calibration, ids, run, config):
    stats = load_statistics(args.statistics, platform)
    stats_sha = digest(Path(args.statistics))
    for sir in config["sir_re_db"]:
        for parent in ids:
            path = run / "baselines" / f"sir_{sir:+03d}" / f"parent_{parent}.json"
            if expected_file(path, config):
                continue
            start = time.perf_counter()
            observation, info, truth = build_example(platform, args.split, parent, sir, calibration)
            measured_sir = actual_sir_re(truth, info.kappa, platform)
            rows = []
            for method, bits, waveform in baseline_estimates(observation, info, platform, stats):
                errors, nbits = bit_errors(bits, truth.target_bits)
                rows.append({"method": method, "sir_re_db": sir, "parent_id": parent,
                             "num_errors": errors, "num_bits": nbits,
                             "nmse": nmse(waveform, truth.target_waveform) if waveform is not None else None,
                             "statistics_sha256": stats_sha, "kappa": info.kappa,
                             "interference_inner_modulation_counts": inner_modulation_counts(truth)})
                rows[-1]["actual_sir_re_db"] = measured_sir
            for name, waveform in (("clean_audit", truth.target_waveform),
                                   ("oracle_subtraction_audit", observation.y - info.kappa * truth.interference_waveform)):
                decoded = receive_target(Observation(observation.example_id, waveform), info, platform.waveform_spec)
                errors, nbits = bit_errors(decoded.target_bits, truth.target_bits)
                if errors:
                    raise RuntimeError(f"{name} failed {parent} {sir}: {errors}/{nbits}")
                rows.append({"method": name, "sir_re_db": sir, "parent_id": parent,
                             "num_errors": errors, "num_bits": nbits})
            save_shard(path, config, rows, time.perf_counter() - start)
        print(f"baseline sir={sir} complete={len(ids)}/{len(ids)}", flush=True)


def sample_seed(sir: int, parent: int, slot: int) -> int:
    data = f"lte-rgs-5000-v1|{sir}|{parent}|{slot}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**63 - 1)


def run_rgs(args, platform, calibration, ids, run, config):
    for source, path in (("target", TARGET), ("interference", INTERFERENCE)):
        if digest(path) != EXPECTED_SHAS[source]:
            raise ValueError(f"{source} final checkpoint hash mismatch")
    device = torch.device(args.device)
    target = load_prior_for_inference(TARGET, source="target", device=device, weights="ema")
    interference = load_prior_for_inference(INTERFERENCE, source="interference", device=device, weights="ema")
    if (next(target.model.parameters()).device != device or
            next(interference.model.parameters()).device != device):
        raise RuntimeError("both complete priors must reside on this worker's GPU")
    if (target.global_step, interference.global_step) != (200000, 300000):
        raise ValueError("not final checkpoint steps")
    if target.metadata["platform_hash"] != platform.config_hash or interference.metadata["platform_hash"] != platform.config_hash:
        raise ValueError("checkpoint platform hash mismatch")
    selected = ids[args.worker::args.workers]
    for sir in config["sir_re_db"]:
        for batch_index in range(0, len(selected), args.batch_parents):
            parents = selected[batch_index:batch_index + args.batch_parents]
            path = run / "rgs" / f"sir_{sir:+03d}" / f"worker_{args.worker}_first_{parents[0]}.json"
            if expected_file(path, config):
                continue
            start = time.perf_counter()
            examples = [build_example(platform, args.split, parent, sir, calibration, dtype=np.complex64)
                        for parent in parents]
            ys, initials, seeds = [], [], []
            for parent, (observation, _, _) in zip(parents, examples):
                initial = _direct_initial(observation.y, platform)
                for slot in (0, 1):
                    sl = slice(slot * 15360, (slot + 1) * 15360)
                    ys.append(complex_to_network(observation.y[sl]))
                    initials.append(complex_to_network(initial[sl]))
                    seeds.append(sample_seed(sir, parent, slot))
            y = torch.from_numpy(np.stack(ys)).to(device)
            x0 = torch.from_numpy(np.stack(initials)).to(device)
            kappa = examples[0][1].kappa
            try:
                estimate, audit = separate_rgs(y, kappa, x0, target, interference,
                    RGSConfig(args.iterations, args.max_lr, sample_seeds=tuple(seeds), diagnostic_interval=100))
                estimates = network_to_complex(estimate.cpu().numpy())
                rows = []
                for i, (parent, (observation, info, truth)) in enumerate(zip(parents, examples)):
                    waveform = np.concatenate((estimates[2 * i], estimates[2 * i + 1]))
                    output = receive_target(Observation(observation.example_id, waveform), info, platform.waveform_spec)
                    errors, nbits = bit_errors(output.target_bits, truth.target_bits)
                    rows.append({"method": "waveform_alpha_rgs", "sir_re_db": sir,
                                 "parent_id": parent, "num_errors": errors, "num_bits": nbits,
                                 "nmse": nmse(waveform, truth.target_waveform),
                                 "kappa": kappa, "iterations": args.iterations,
                                 "actual_sir_re_db": actual_sir_re(truth, kappa, platform),
                                 "interference_inner_modulation_counts": inner_modulation_counts(truth),
                                 "slot_seeds": seeds[2 * i:2 * i + 2],
                                 "rgs_audit": audit})
                save_shard(path, config, rows, time.perf_counter() - start)
                print(f"rgs worker={args.worker} sir={sir} parents={parents[0]}..{parents[-1]} elapsed={time.perf_counter()-start:.1f}s", flush=True)
            except Exception as exc:
                atomic_json(path.with_suffix(".failure.json"), {"status": "failed", "sir": sir, "parents": parents,
                                                              "error": repr(exc), "elapsed_seconds": time.perf_counter() - start})
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baselines", "rgs"), required=True)
    parser.add_argument("--split", choices=("audit", "test"), default="test")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--statistics", default=str(ROOT / "artifacts/rgs_ber_5000/baseline_train_stats.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-parents", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max-lr", type=float, default=0.005)
    parser.add_argument("--audit-parents", type=int, default=4)
    parser.add_argument("--audit-sirs", type=lambda value: tuple(int(x) for x in value.split(",")),
                        default=(-24, -12, -6, -3, 0, 3, 6))
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        parser.error("invalid worker index")
    platform, calibration, ids, run, config = context(args)
    if args.mode == "baselines":
        run_baselines(args, platform, calibration, ids, run, config)
    else:
        run_rgs(args, platform, calibration, ids, run, config)


if __name__ == "__main__":
    main()
