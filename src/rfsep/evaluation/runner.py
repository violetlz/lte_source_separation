from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import yaml

from ..baselines.direct import direct_receive
from ..config import ExperimentConfig
from ..contracts import Observation, ReceiverInfo
from ..data.manifests import build_manifest, write_manifest
from ..diagnostics.waveform_audit import run_waveform_audit
from ..observation import kappa_for_sir, load_calibration, mix_time_domain
from ..waveforms.constellations import demap_symbols
from ..waveforms.receiver import receive_target, target_demodulate
from ..waveforms.sources import generate_interference, generate_target
from .metrics import bit_errors, evm_rms
from .plotting import plot_ber, plot_cp, plot_spectrum


PER_EXAMPLE_FIELDS = [
    "run_id", "example_id", "method", "sir_re_db", "kappa",
    "nominal_sir_total_db", "actual_sir_re_db", "actual_sir_total_db", "num_errors", "num_bits",
    "target_inband_energy", "scaled_interference_inband_energy",
    "target_total_energy", "scaled_interference_total_energy", "evm",
    "status", "elapsed_seconds",
]


def _git_state(root: Path) -> dict:
    def run(*args):
        result = subprocess.run(args, cwd=root, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("git", "rev-parse", "HEAD"), "branch": run("git", "branch", "--show-current"),
            "status_porcelain": run("git", "status", "--short")}


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], float(row["sir_re_db"]))].append(row)
    output = []
    for (method, sir), items in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        errors = sum(int(x["num_errors"]) for x in items)
        bits = sum(int(x["num_bits"]) for x in items)
        target_inband = sum(float(x["target_inband_energy"]) for x in items)
        interference_inband = sum(float(x["scaled_interference_inband_energy"]) for x in items)
        target_total = sum(float(x["target_total_energy"]) for x in items)
        interference_total = sum(float(x["scaled_interference_total_energy"]) for x in items)
        output.append({
            "method": method, "sir_re_db": sir, "num_examples": len(items),
            "nominal_sir_total_db": float(items[0]["nominal_sir_total_db"]),
            "num_errors": errors, "num_bits": bits, "ber": errors / bits,
            "zero_error_95_upper_bound": 3.0 / bits if errors == 0 else "",
            "measured_sir_re_db": 10 * np.log10(target_inband / interference_inband),
            "measured_sir_total_db": 10 * np.log10(target_total / interference_total),
            "mean_evm": float(np.mean([float(x["evm"]) for x in items])),
            "failed_examples": sum(x["status"] != "ok" for x in items),
        })
    return output


def evaluate(experiment: ExperimentConfig, project_root: Path) -> Path:
    formal = experiment.raw["name"] == "platform_validation"
    calibration = load_calibration(experiment.calibration_path, experiment.platform,
                                   allow_provisional=not formal)
    dtype = np.complex128 if experiment.raw["precision"] == "complex128" else np.complex64
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{experiment.raw['name']}"
    run_dir = project_root / "artifacts" / "runs" / run_id
    figures = run_dir / "figures"
    figures.mkdir(parents=True)
    resolved = {
        "experiment": experiment.raw, "platform": experiment.platform.raw,
        "platform_hash": experiment.platform.config_hash,
        "calibration_id": calibration["calibration_id"],
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    manifest = build_manifest(experiment.platform, str(experiment.raw["split"]), int(experiment.raw["num_examples"]))
    write_manifest(manifest, project_root / "artifacts" / "manifests" / f"{run_id}.json")
    started = time.perf_counter()
    rows: list[dict] = []
    log_lines = [f"run_id={run_id}", f"started_utc={datetime.now(timezone.utc).isoformat()}"]
    example_for_plot = None
    sir_values = [float(x) for x in experiment.raw["sir_re_db"]]
    kappas = {sir: kappa_for_sir(float(calibration["q_b"]), sir) for sir in sir_values}
    for record_id in range(int(experiment.raw["num_examples"])):
        target, target_grid, target_bits = generate_target(experiment.platform, str(experiment.raw["split"]), record_id, dtype=dtype)
        interference, _, _ = generate_interference(experiment.platform, str(experiment.raw["split"]), record_id, dtype=dtype)
        ds = target_demodulate(target.waveform, experiment.platform.waveform_spec)
        db = target_demodulate(interference.waveform, experiment.platform.waveform_spec)
        target_inband_energy = float(np.sum(np.abs(ds) ** 2))
        target_total_energy = float(np.sum(np.abs(target.waveform) ** 2))
        clean_bits = demap_symbols(ds, "QPSK").reshape(-1)
        for sir in sir_values:
            kappa = kappas[sir]
            scaled_inband_energy = float(np.sum(np.abs(kappa * db) ** 2))
            scaled_total_energy = float(np.sum(np.abs(kappa * interference.waveform) ** 2))
            actual_re = float(10 * np.log10(target_inband_energy / scaled_inband_energy))
            actual_total = float(10 * np.log10(target_total_energy / scaled_total_energy))
            y = mix_time_domain(target.waveform, interference.waveform, kappa, dtype=dtype)
            info = ReceiverInfo(experiment.platform.raw["version"], experiment.platform.config_hash,
                                calibration["calibration_id"], kappa, sir)
            example_id = f"pair:{experiment.raw['split']}:{record_id:08d}"
            method_inputs = [("clean_target", target.waveform), ("direct_mixture", y),
                             ("oracle_subtraction_audit", y - kappa * interference.waveform)]
            for method, waveform in method_inputs:
                tick = time.perf_counter()
                observation = Observation(example_id, waveform)
                output = (direct_receive(observation, info, experiment.platform.waveform_spec)
                          if method == "direct_mixture" else receive_target(
                              observation, info, experiment.platform.waveform_spec))
                errors, bits = bit_errors(output.target_bits, target_bits)
                row = {
                    "run_id": run_id, "example_id": example_id, "method": method,
                    "sir_re_db": sir, "kappa": kappa,
                    "nominal_sir_total_db": float(-20 * np.log10(kappa)),
                    "actual_sir_re_db": actual_re,
                    "actual_sir_total_db": actual_total, "num_errors": errors, "num_bits": bits,
                    "target_inband_energy": target_inband_energy,
                    "scaled_interference_inband_energy": scaled_inband_energy,
                    "target_total_energy": target_total_energy,
                    "scaled_interference_total_energy": scaled_total_energy,
                    "evm": evm_rms(output.target_grid, target_grid), "status": "ok",
                    "elapsed_seconds": time.perf_counter() - tick,
                }
                rows.append(row)
            if example_for_plot is None and sir == sir_values[len(sir_values) // 2]:
                example_for_plot = (target.waveform.copy(), interference.waveform.copy(), y.copy())
        if (record_id + 1) % 32 == 0 or record_id + 1 == int(experiment.raw["num_examples"]):
            log_lines.append(f"completed_examples={record_id + 1}")
    summaries = _summary(rows)
    _write_csv(run_dir / "per_example.csv", rows, PER_EXAMPLE_FIELDS)
    _write_csv(run_dir / "summary.csv", summaries, list(summaries[0]))
    direct = [row for row in summaries if row["method"] == "direct_mixture"]
    clean = [row for row in summaries if row["method"] == "clean_target"]
    oracle = [row for row in summaries if row["method"] == "oracle_subtraction_audit"]
    expected_bits = int(experiment.raw["num_examples"]) * 8400
    checks = {
        "waveform_audit": run_waveform_audit(experiment.platform),
        "all_rows_present": {"passed": len(rows) == int(experiment.raw["num_examples"]) * len(sir_values) * 3,
                             "actual": len(rows)},
        "direct_denominators": {"passed": all(int(x["num_bits"]) == expected_bits for x in direct),
                                "expected_bits_per_sir": expected_bits},
        "clean_zero_errors": {"passed": all(int(x["num_errors"]) == 0 for x in clean),
                              "total_errors": sum(int(x["num_errors"]) for x in clean)},
        "oracle_zero_errors": {"passed": all(int(x["num_errors"]) == 0 for x in oracle),
                               "total_errors": sum(int(x["num_errors"]) for x in oracle)},
        "sir_ensemble_tolerance": {
            "passed": all(abs(float(x["measured_sir_re_db"]) - float(x["sir_re_db"])) <= 0.25 for x in direct),
            "max_abs_error_db": max(abs(float(x["measured_sir_re_db"]) - float(x["sir_re_db"])) for x in direct),
            "tolerance_db": 0.25,
        },
        "fixed_kappa_per_condition": {"passed": all(len({float(r["kappa"]) for r in rows
                                                          if r["method"] == "direct_mixture" and float(r["sir_re_db"]) == sir}) == 1
                                                         for sir in sir_values)},
        "all_status_ok_and_finite": {"passed": all(r["status"] == "ok" and np.isfinite(float(r["evm"])) for r in rows)},
    }
    checks["all_passed"] = all(value.get("passed", value.get("all_passed", False))
                               for key, value in checks.items() if key != "all_passed")
    (run_dir / "validation_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    if example_for_plot is not None:
        plot_spectrum(*example_for_plot, experiment.platform, figures / "waveform_spectrum.png")
        plot_cp(example_for_plot[0], experiment.platform, figures / "cp_comparison.png")
    plot_ber(summaries, figures / "direct_ber_vs_sir.png")
    elapsed = time.perf_counter() - started
    run_manifest = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root.resolve()), "platform_hash": experiment.platform.config_hash,
        "calibration_hash": calibration["calibration_id"], "sample_manifest_digest": manifest["manifest_digest"],
        "root_seed": experiment.platform.root_seed,
        "seed_scheme": "sha256(rfsep-v1|root|purpose|split|record_id)",
        "precision": experiment.raw["precision"], "fft_compute_precision": "numpy complex128",
        "model": "not applicable; no model or checkpoint used",
        "python": sys.version, "numpy": np.__version__, "pyyaml": yaml.__version__,
        "matplotlib": matplotlib.__version__, "host": platform.node(), "platform": platform.platform(),
        "processor": platform.processor(), "git": _git_state(project_root), "elapsed_seconds": elapsed,
        "status": "passed" if checks["all_passed"] else "failed",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    log_lines.extend([f"elapsed_seconds={elapsed:.6f}", f"status={run_manifest['status']}"])
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    return run_dir


def load_summary(run_dir: Path) -> list[dict]:
    with (run_dir / "summary.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))
