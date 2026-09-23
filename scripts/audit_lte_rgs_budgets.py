#!/usr/bin/env python3
"""Paired audit-only comparison of completed and shared-path RGS budgets.

Each completed budget has its own cosine schedule but identical parent/slot
RNG streams. The 1000/2000 snapshots of the 5000-step run are separately
labelled: they share its schedule and are not completed shorter budgets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfsep.baselines.lte_estimated import load_statistics
from rfsep.config import load_platform
from rfsep.contracts import Observation, complex_to_network, network_to_complex
from rfsep.evaluation.metrics import bit_errors, nmse
from rfsep.observation import build_example, load_calibration
from rfsep.priors.inference import load_prior_for_inference
from rfsep.separation.rgs import RGSConfig, separate_rgs
from rfsep.waveforms.receiver import receive_target
from run_lte_rgs_ber_5000 import (EXPECTED_SHAS, INTERFERENCE, TARGET, actual_sir_re,
                                  atomic_json, baseline_estimates, digest, sample_seed)
from smoke_rgs import _direct_initial

SIRS = (-24, -12, -6, -3, 0, 3, 6)
PARENTS = tuple(range(1000000, 1000016))
BUDGETS = (1000, 2000, 5000)
MAX_LR = 1e-4
STATISTICS = ROOT / "artifacts/rgs_ber_5000/baseline_train_stats_final.json"


def config_payload(platform, calibration):
    config = {"schema": "lte-waveform-rgs-budget-audit-v1", "split": "audit",
              "platform_hash": platform.config_hash, "calibration_id": calibration["calibration_id"],
              "parent_ids": PARENTS, "sir_re_db": SIRS, "completed_budgets": BUDGETS,
              "shared_5000_path_snapshots": (1000, 2000), "max_lr": MAX_LR,
              "min_lr": 1e-6, "omega": "kappa_squared", "timestep_range": (1, 49),
              "initialization": "direct_hard_qpsk_remodulation",
              "target_checkpoint_sha256": EXPECTED_SHAS["target"],
              "interference_checkpoint_sha256": EXPECTED_SHAS["interference"],
              "baseline_train_statistics_sha256": digest(STATISTICS)}
    config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config


def decode_estimate(estimate, observation, info, truth, platform):
    waveform = np.concatenate(network_to_complex(estimate.detach().cpu().numpy()))
    output = receive_target(Observation(observation.example_id, waveform), info, platform.waveform_spec)
    errors, bits = bit_errors(output.target_bits, truth.target_bits)
    return {"num_errors": errors, "num_bits": bits, "nmse": nmse(waveform, truth.target_waveform)}


def run_worker(args):
    platform = load_platform(ROOT / "configs/platform/lte_v1.yaml")
    calibration = load_calibration(ROOT / "reports/formal_validation/calibration_summary.json", platform)
    config = config_payload(platform, calibration)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / "resolved_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("audit config changed; refusing resume")
    if not config_file.exists():
        atomic_json(config_file, config)
    for source, checkpoint in (("target", TARGET), ("interference", INTERFERENCE)):
        if digest(checkpoint) != EXPECTED_SHAS[source]:
            raise ValueError(f"{source} checkpoint differs from final EMA identity")
    device = torch.device(args.device)
    target = load_prior_for_inference(TARGET, source="target", device=device, weights="ema")
    interference = load_prior_for_inference(INTERFERENCE, source="interference", device=device, weights="ema")
    if (target.global_step, interference.global_step) != (200000, 300000):
        raise ValueError("not the final prior steps")
    if target.metadata["platform_hash"] != platform.config_hash or interference.metadata["platform_hash"] != platform.config_hash:
        raise ValueError("checkpoint platform mismatch")
    stats = load_statistics(STATISTICS, platform)
    for sir in SIRS:
        for parent in PARENTS[args.worker::args.workers]:
            shard = output / "shards" / f"sir_{sir:+03d}" / f"parent_{parent}.json"
            if shard.exists():
                payload = json.loads(shard.read_text())
                if payload.get("status") != "complete" or payload.get("config_sha256") != config["config_sha256"]:
                    raise ValueError(f"invalid existing audit shard: {shard}")
                continue
            start = time.perf_counter()
            observation, info, truth = build_example(platform, "audit", parent, sir, calibration,
                                                     dtype=np.complex64)
            rows = []
            for method, bits, waveform in baseline_estimates(observation, info, platform, stats):
                errors, count = bit_errors(bits, truth.target_bits)
                rows.append({"method": method, "sir_re_db": sir, "parent_id": parent,
                             "num_errors": errors, "num_bits": count,
                             "nmse": nmse(waveform, truth.target_waveform) if waveform is not None else None,
                             "actual_sir_re_db": actual_sir_re(truth, info.kappa, platform),
                             "kappa": info.kappa})
            initial = _direct_initial(observation.y, platform)
            ys, initials = [], []
            for slot in (0, 1):
                sl = slice(slot * 15360, (slot + 1) * 15360)
                ys.append(complex_to_network(observation.y[sl]))
                initials.append(complex_to_network(initial[sl]))
            y = torch.from_numpy(np.stack(ys)).to(device)
            x0 = torch.from_numpy(np.stack(initials)).to(device)
            seeds = (sample_seed(sir, parent, 0), sample_seed(sir, parent, 1))
            for budget in BUDGETS:
                tick = time.perf_counter()
                estimate, audit = separate_rgs(y, info.kappa, x0, target, interference,
                    RGSConfig(budget, MAX_LR, sample_seeds=seeds, diagnostic_interval=100,
                              snapshot_iterations=(1000, 2000) if budget == 5000 else ()))
                snapshots = audit.pop("snapshots", {})
                measured = decode_estimate(estimate, observation, info, truth, platform)
                rows.append({"method": "waveform_alpha_rgs", "schedule_total_steps": budget,
                             "recorded_step": budget, "trajectory_kind": "completed_budget",
                             "sir_re_db": sir, "parent_id": parent, "kappa": info.kappa,
                             "slot_seeds": seeds, "elapsed_seconds": time.perf_counter() - tick,
                             "rgs_audit": audit, **measured})
                if budget == 5000:
                    for step, state in sorted(snapshots.items()):
                        measured = decode_estimate(state, observation, info, truth, platform)
                        rows.append({"method": "waveform_alpha_rgs", "schedule_total_steps": 5000,
                                     "recorded_step": step, "trajectory_kind": "shared_5000_path_snapshot",
                                     "sir_re_db": sir, "parent_id": parent, "kappa": info.kappa,
                                     "slot_seeds": seeds, **measured})
            atomic_json(shard, {"status": "complete", "config_sha256": config["config_sha256"],
                                "sir_re_db": sir, "parent_id": parent,
                                "elapsed_seconds": time.perf_counter() - start, "rows": rows})
            print(f"audit worker={args.worker} sir={sir} parent={parent} "
                  f"elapsed={time.perf_counter() - start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        parser.error("invalid worker assignment")
    run_worker(args)


if __name__ == "__main__":
    main()
