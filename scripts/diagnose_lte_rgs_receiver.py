#!/usr/bin/env python3
"""Audit-only RGS receiver diagnosis; never reads the sealed test split.

Recompute selected frozen 5k audit trajectories and verify their BER against
the original atomic audit shards before recording decision and spectral facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfsep.config import load_platform
from rfsep.contracts import Observation, complex_to_network, network_to_complex
from rfsep.evaluation.metrics import bit_errors, nmse
from rfsep.observation import build_example, load_calibration
from rfsep.priors.inference import load_prior_for_inference
from rfsep.separation.rgs import RGSConfig, separate_rgs
from rfsep.waveforms.constellations import map_bits
from rfsep.waveforms.ofdm import signed_to_bins
from rfsep.waveforms.receiver import receive_target
from audit_lte_rgs_budgets import MAX_LR, STATISTICS, config_payload
from run_lte_rgs_ber_5000 import EXPECTED_SHAS, INTERFERENCE, TARGET, atomic_json, digest, sample_seed
from smoke_rgs import _direct_initial

SIRS = (-24, -6, 0, 6)
PARENTS = tuple(range(1000000, 1000004))
ORIGINAL_AUDIT = ROOT / "artifacts/rgs_ber_5000/audit_paired_budgets_20260923"


def useful_spectrum_error(waveform: np.ndarray, truth: np.ndarray, platform) -> dict:
    error = waveform - truth
    active = signed_to_bins(platform.target_subcarriers, platform.fft_size)
    selected = np.zeros(platform.fft_size, dtype=bool)
    selected[active] = True
    spectra = []
    cursor = 0
    for cp in platform.cp_lengths:
        useful = error[cursor + cp:cursor + cp + platform.fft_size]
        spectra.append(np.fft.fft(useful, norm="ortho"))
        cursor += cp + platform.fft_size
    spectrum = np.stack(spectra)
    return {"target_tones_error_energy": float(np.sum(np.abs(spectrum[:, selected]) ** 2)),
            "other_tones_error_energy": float(np.sum(np.abs(spectrum[:, ~selected]) ** 2))}


def bit_axes(grid: np.ndarray) -> np.ndarray:
    return np.stack((grid.real, grid.imag), axis=-1).reshape(-1)


def analyze(observation, info, truth, initial: np.ndarray, estimate: np.ndarray, platform) -> dict:
    direct = receive_target(observation, info, platform.waveform_spec)
    separated = receive_target(Observation(observation.example_id, estimate), info,
                               platform.waveform_spec)
    initialized = receive_target(Observation(observation.example_id, initial), info,
                                 platform.waveform_spec)
    truth_bits = truth.target_bits.reshape(-1)
    direct_bits = direct.target_bits.reshape(-1)
    rgs_bits = separated.target_bits.reshape(-1)
    if not np.array_equal(initialized.target_bits.reshape(-1), direct_bits):
        raise ValueError("hard Direct initialization does not preserve Direct decisions")
    direct_wrong = direct_bits != truth_bits
    rgs_wrong = rgs_bits != truth_bits
    flips = direct_bits != rgs_bits
    hard_grid = map_bits(direct_bits.reshape(14, 300, 2), "QPSK")
    truth_signs = 1 - 2 * truth_bits.astype(np.int8)
    signed_direct = truth_signs * bit_axes(direct.target_grid)
    signed_rgs = truth_signs * bit_axes(separated.target_grid)
    direct_errors, bits = bit_errors(direct_bits, truth_bits)
    rgs_errors, _ = bit_errors(rgs_bits, truth_bits)
    return {
        "num_bits": bits, "direct_errors": direct_errors, "rgs_errors": rgs_errors,
        "decision_flips": int(np.count_nonzero(flips)),
        "corrected_bits": int(np.count_nonzero(direct_wrong & ~rgs_wrong)),
        "damaged_bits": int(np.count_nonzero(~direct_wrong & rgs_wrong)),
        "direct_waveform_nmse": nmse(observation.y, truth.target_waveform),
        "initial_waveform_nmse": nmse(initial, truth.target_waveform),
        "rgs_waveform_nmse": nmse(estimate, truth.target_waveform),
        "direct_grid_nmse": nmse(direct.target_grid, truth.target_grid),
        "initial_grid_nmse": nmse(initialized.target_grid, truth.target_grid),
        "rgs_grid_nmse": nmse(separated.target_grid, truth.target_grid),
        "rgs_grid_distance_from_hard_initial": nmse(separated.target_grid, hard_grid),
        "direct_error_margin_mean": float(signed_direct[direct_wrong].mean()),
        "rgs_margin_on_direct_errors_mean": float(signed_rgs[direct_wrong].mean()),
        "rgs_margin_on_direct_correct_mean": float(signed_rgs[~direct_wrong].mean()),
        "mean_abs_grid_axis_shift": float(np.mean(np.abs(bit_axes(separated.target_grid) -
                                                    bit_axes(initialized.target_grid)))),
        "direct_useful_error": useful_spectrum_error(observation.y, truth.target_waveform, platform),
        "initial_useful_error": useful_spectrum_error(initial, truth.target_waveform, platform),
        "rgs_useful_error": useful_spectrum_error(estimate, truth.target_waveform, platform),
    }


def run(args) -> None:
    platform = load_platform(ROOT / "configs/platform/lte_v1.yaml")
    calibration = load_calibration(ROOT / "reports/formal_validation/calibration_summary.json", platform)
    manifest = config_payload(platform, calibration)
    original = json.loads((ORIGINAL_AUDIT / "resolved_config.json").read_text())
    if (json.loads(json.dumps(manifest)) != original or
            digest(STATISTICS) != original["baseline_train_statistics_sha256"]):
        raise ValueError("original audit provenance changed")
    for source, path in (("target", TARGET), ("interference", INTERFERENCE)):
        if digest(path) != EXPECTED_SHAS[source]:
            raise ValueError(f"{source} checkpoint identity changed")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    diag_config = {"schema": "lte-rgs-receiver-diagnostic-v1", "split": "audit",
                   "sir_re_db": SIRS, "parent_ids": PARENTS,
                   "original_audit_config_sha256": original["config_sha256"],
                   "iterations": 5000, "max_lr": MAX_LR,
                   "target_checkpoint_sha256": EXPECTED_SHAS["target"],
                   "interference_checkpoint_sha256": EXPECTED_SHAS["interference"]}
    diag_config["config_sha256"] = hashlib.sha256(json.dumps(diag_config, sort_keys=True).encode()).hexdigest()
    config_file = output / "resolved_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != json.loads(json.dumps(diag_config)):
        raise ValueError("diagnostic config changed; refusing resume")
    if not config_file.exists():
        atomic_json(config_file, diag_config)
    device = torch.device(args.device)
    target = load_prior_for_inference(TARGET, source="target", device=device, weights="ema")
    interference = load_prior_for_inference(INTERFERENCE, source="interference", device=device, weights="ema")
    if (target.global_step, interference.global_step) != (200000, 300000):
        raise ValueError("not final EMA prior steps")
    for sir in SIRS:
        for parent in PARENTS[args.worker::args.workers]:
            shard = output / "shards" / f"sir_{sir:+03d}" / f"parent_{parent}.json"
            if shard.exists():
                payload = json.loads(shard.read_text())
                if payload.get("config_sha256") != diag_config["config_sha256"]:
                    raise ValueError(f"invalid existing diagnostic shard: {shard}")
                continue
            observation, info, truth = build_example(platform, "audit", parent, sir, calibration,
                                                     dtype=np.complex64)
            initial = _direct_initial(observation.y, platform)
            ys, initials = [], []
            for slot in (0, 1):
                sl = slice(slot * 15360, (slot + 1) * 15360)
                ys.append(complex_to_network(observation.y[sl]))
                initials.append(complex_to_network(initial[sl]))
            y = torch.from_numpy(np.stack(ys)).to(device)
            x0 = torch.from_numpy(np.stack(initials)).to(device)
            seeds = (sample_seed(sir, parent, 0), sample_seed(sir, parent, 1))
            estimate, rgs_audit = separate_rgs(y, info.kappa, x0, target, interference,
                RGSConfig(5000, MAX_LR, sample_seeds=seeds, diagnostic_interval=100))
            waveform = np.concatenate(network_to_complex(estimate.detach().cpu().numpy()))
            measured = analyze(observation, info, truth, initial, waveform, platform)
            original_shard = json.loads((ORIGINAL_AUDIT / "shards" / f"sir_{sir:+03d}" /
                                         f"parent_{parent}.json").read_text())
            prior_row = next(row for row in original_shard["rows"]
                             if row["method"] == "waveform_alpha_rgs" and
                             row["trajectory_kind"] == "completed_budget" and
                             row["recorded_step"] == 5000)
            if (measured["rgs_errors"] != prior_row["num_errors"] or
                    abs(measured["rgs_waveform_nmse"] - prior_row["nmse"]) > 1e-4 or
                    rgs_audit["target_prior_calls"] != 5000 or
                    rgs_audit["interference_prior_calls"] != 5000):
                raise ValueError(f"recomputed RGS does not match original audit {sir}, {parent}")
            atomic_json(shard, {"status": "complete", "config_sha256": diag_config["config_sha256"],
                                "sir_re_db": sir, "parent_id": parent,
                                "slot_seeds": seeds, **measured})
            print(f"receiver diagnostic worker={args.worker} sir={sir} parent={parent} "
                  f"flips={measured['decision_flips']} corrected={measured['corrected_bits']} "
                  f"damaged={measured['damaged_bits']}", flush=True)


def summarize(args) -> None:
    output = Path(args.output)
    config = json.loads((output / "resolved_config.json").read_text())
    rows = []
    for sir in SIRS:
        for parent in PARENTS:
            shard = output / "shards" / f"sir_{sir:+03d}" / f"parent_{parent}.json"
            payload = json.loads(shard.read_text())
            if (payload["status"] != "complete" or payload["config_sha256"] != config["config_sha256"]
                    or payload["sir_re_db"] != sir or payload["parent_id"] != parent
                    or payload["num_bits"] != 8400
                    or payload["decision_flips"] != payload["corrected_bits"] + payload["damaged_bits"]
                    or payload["rgs_errors"] - payload["direct_errors"] !=
                    payload["damaged_bits"] - payload["corrected_bits"]):
                raise ValueError(f"invalid diagnostic shard: {shard}")
            original_shard = json.loads((ORIGINAL_AUDIT / "shards" / f"sir_{sir:+03d}" /
                                         f"parent_{parent}.json").read_text())
            direct_row = next(row for row in original_shard["rows"] if row["method"] == "direct")
            rgs_row = next(row for row in original_shard["rows"]
                           if row["method"] == "waveform_alpha_rgs" and
                           row["trajectory_kind"] == "completed_budget" and
                           row["recorded_step"] == 5000)
            if (payload["direct_errors"] != direct_row["num_errors"] or
                    payload["rgs_errors"] != rgs_row["num_errors"] or
                    abs(payload["rgs_waveform_nmse"] - rgs_row["nmse"]) > 1e-4):
                raise ValueError(f"diagnostic does not match original audit: {shard}")
            rows.append(payload)
    summaries = []
    for sir in SIRS:
        chosen = [row for row in rows if row["sir_re_db"] == sir]
        pooled = {key: sum(row[key] for row in chosen) for key in
                  ("num_bits", "direct_errors", "rgs_errors", "decision_flips", "corrected_bits",
                   "damaged_bits")}
        for method in ("direct", "initial", "rgs"):
            pooled[f"{method}_waveform_nmse_mean"] = float(np.mean(
                [row[f"{method}_waveform_nmse"] for row in chosen]))
            pooled[f"{method}_grid_nmse_mean"] = float(np.mean(
                [row[f"{method}_grid_nmse"] for row in chosen]))
            for band in ("target_tones", "other_tones"):
                pooled[f"{method}_{band}_error_energy"] = sum(
                    row[f"{method}_useful_error"][f"{band}_error_energy"] for row in chosen)
        for metric in ("direct_error_margin_mean", "rgs_margin_on_direct_errors_mean",
                       "rgs_margin_on_direct_correct_mean", "mean_abs_grid_axis_shift",
                       "rgs_grid_distance_from_hard_initial"):
            pooled[f"{metric}_mean"] = float(np.mean([row[metric] for row in chosen]))
        pooled["sir_re_db"] = sir
        pooled["num_parents"] = len(chosen)
        summaries.append(pooled)
    result = {"status": "complete", "split": "audit", "parent_ids": PARENTS,
              "sir_re_db": SIRS, "num_shards": len(rows), "rows": summaries,
              "config_sha256": config["config_sha256"]}
    atomic_json(output / "summary.json", result)
    if args.summary_output:
        atomic_json(Path(args.summary_output), result)
    print(json.dumps(summaries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "summarize"), default="run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary-output")
    args = parser.parse_args()
    if args.mode == "summarize":
        summarize(args)
    else:
        if args.workers < 1 or not 0 <= args.worker < args.workers:
            parser.error("invalid worker assignment")
        run(args)


if __name__ == "__main__":
    main()
