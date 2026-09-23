#!/usr/bin/env python3
"""Only creates a formal curve when every paired method and parent is present."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIRS = (-24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6)
METHODS = ("direct", "fft_diagonal_wiener_approx",
           "single_re_marginal_bit_map_estimated", "waveform_alpha_rgs")
LABELS = ("Direct OFDM", "FFT-diagonal Wiener (approx.)",
          "Single-RE marginal bit-MAP (estimated likelihood)", "Waveform α-RGS (5000 updates)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "resolved_config.json").read_text())
    if config["split"] != "test" or config["rgs_iterations"] != 5000:
        raise ValueError("formal summary requires frozen full test contract")
    expected = set(range(2000000, 2000256))
    collected = defaultdict(dict)
    run_seconds = defaultdict(float)
    baseline_pipeline_seconds = defaultdict(float)
    for parent_file in sorted((args.run_dir / "baselines").glob("sir_*/parent_*.json")):
        payload = json.loads(parent_file.read_text())
        if payload["status"] != "complete":
            raise ValueError(f"incomplete baseline {parent_file}")
        baseline_pipeline_seconds[int(payload["rows"][0]["sir_re_db"])] += float(payload["elapsed_seconds"])
        for row in payload["rows"]:
            key = (int(row["sir_re_db"]), row["method"])
            parent = int(row["parent_id"])
            if parent in collected[key]:
                raise ValueError(f"duplicate row {key} {parent}")
            collected[key][parent] = row
    for shard in sorted((args.run_dir / "rgs").glob("sir_*/worker_*.json")):
        payload = json.loads(shard.read_text())
        if payload["status"] != "complete":
            raise ValueError(f"incomplete RGS {shard}")
        for row in payload["rows"]:
            key = (int(row["sir_re_db"]), row["method"])
            parent = int(row["parent_id"])
            if parent in collected[key]:
                raise ValueError(f"duplicate row {key} {parent}")
            collected[key][parent] = row
            run_seconds[key] += float(payload["elapsed_seconds"]) / len(payload["rows"])
    progress = {f"{sir}:{method}": len(collected[(sir, method)])
                for sir in SIRS for method in METHODS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    if all(set(collected[(sir, method)]) == expected for sir in SIRS for method in METHODS[:-1]):
        baseline_rows = []
        for sir in SIRS:
            for method in METHODS[:-1]:
                values = list(collected[(sir, method)].values())
                errors = sum(int(row["num_errors"]) for row in values)
                bits = sum(int(row["num_bits"]) for row in values)
                if bits != 2150400:
                    raise ValueError(f"baseline denominator mismatch: {sir}, {method}")
                baseline_rows.append({"status": "baseline_complete_rgs_pending", "sir_re_db": sir,
                                      "method": method, "num_parents": 256,
                                      "num_errors": errors, "num_bits": bits, "ber": errors / bits})
        with (args.output_dir / "baseline_only_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(baseline_rows)
        with (args.output_dir / "baseline_only_per_parent.csv").open("w", newline="") as handle:
            fields = ("sir_re_db", "method", "parent_id", "num_errors", "num_bits",
                      "kappa", "actual_sir_re_db", "nmse")
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for sir in SIRS:
                for method in METHODS[:-1]:
                    for parent in sorted(expected):
                        row = collected[(sir, method)][parent]
                        writer.writerow({field: row.get(field, "") for field in fields})
    missing = {key: sorted(expected - set(collected[key])) for key in ((sir, method) for sir in SIRS for method in METHODS)
               if set(collected[key]) != expected}
    if missing:
        print(json.dumps({"state": "incomplete", "completed_counts": progress,
                          "missing_groups": {f"{sir}:{method}": len(value) for (sir, method), value in missing.items()}}, indent=2))
        return
    for sir in SIRS:
        for audit_method in ("clean_audit", "oracle_subtraction_audit"):
            audit_rows = collected[(sir, audit_method)]
            if set(audit_rows) != expected or any(int(row["num_errors"]) for row in audit_rows.values()):
                raise ValueError(f"clean/oracle chain audit failed: {sir}, {audit_method}")
        kappas = {float(collected[(sir, method)][parent]["kappa"])
                  for method in METHODS for parent in expected}
        if len(kappas) != 1:
            raise ValueError(f"inconsistent kappa at {sir}: {kappas}")
    rows = []
    for sir in SIRS:
        for method in METHODS:
            values = list(collected[(sir, method)].values())
            errors = sum(int(row["num_errors"]) for row in values)
            bits = sum(int(row["num_bits"]) for row in values)
            if bits != 2150400 or any(int(row["num_bits"]) != 8400 for row in values):
                raise ValueError(f"wrong denominators: {sir}, {method}, {bits}")
            if method == "waveform_alpha_rgs" and any(int(row["iterations"]) != 5000 for row in values):
                raise ValueError("RGS shard with wrong update count")
            if method == "waveform_alpha_rgs" and any(
                    int(row["rgs_audit"]["target_prior_calls"]) != 5000 or
                    int(row["rgs_audit"]["interference_prior_calls"]) != 5000
                    for row in values):
                raise ValueError("RGS shard missing prior calls")
            rows.append({"sir_re_db": sir, "method": method, "num_parents": 256,
                         "num_errors": errors, "num_bits": bits, "ber": errors / bits,
                         "zero_error_95_upper_bound": 3 / bits if errors == 0 else "",
                         "rgs_runtime_seconds": run_seconds[(sir, method)] if method == "waveform_alpha_rgs" else "",
                         "shared_baseline_pipeline_seconds": baseline_pipeline_seconds[sir] if method != "waveform_alpha_rgs" else "",
                         "mean_actual_sir_re_db": np.mean([row["actual_sir_re_db"] for row in values]),
                         "mean_nmse": np.mean([row["nmse"] for row in values if row.get("nmse") is not None])
                                      if method != "single_re_marginal_bit_map_estimated" else ""})
    rng = np.random.default_rng(20260923)
    bootstrap_rows = []
    for sir in SIRS:
        rgs = np.array([collected[(sir, "waveform_alpha_rgs")][p]["num_errors"] for p in sorted(expected)])
        indices = rng.integers(0, 256, size=(5000, 256))
        for method in METHODS[:-1]:
            baseline = np.array([collected[(sir, method)][p]["num_errors"] for p in sorted(expected)])
            diff = (rgs - baseline) / 8400
            draws = diff[indices].mean(axis=1)
            bootstrap_rows.append({"sir_re_db": sir, "baseline": method,
                                   "rgs_minus_baseline_ber": float(diff.mean()),
                                   "paired_parent_bootstrap_95_lower": float(np.quantile(draws, 0.025)),
                                   "paired_parent_bootstrap_95_upper": float(np.quantile(draws, 0.975)),
                                   "bootstrap_replicates": 5000})
    with (args.output_dir / "paired_bootstrap.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(bootstrap_rows)
    modulation_rows = []
    for sir in SIRS:
        for method in METHODS:
            for modulation in ("QPSK", "16QAM", "64QAM"):
                subset = [row for row in collected[(sir, method)].values()
                          if row.get("interference_inner_modulation_counts", {}).get(modulation, 0) >= 3]
                if subset:
                    errors = sum(int(row["num_errors"]) for row in subset)
                    bits = sum(int(row["num_bits"]) for row in subset)
                    modulation_rows.append({"sir_re_db": sir, "method": method,
                                            "diagnostic_group": f"at_least_3_of_6_inner_blocks_{modulation}",
                                            "num_parents": len(subset), "num_errors": errors,
                                            "num_bits": bits, "ber": errors / bits})
    with (args.output_dir / "modulation_diagnostic_slices.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(modulation_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(modulation_rows)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "per_parent.csv").open("w", newline="") as handle:
        fields = ("sir_re_db", "method", "parent_id", "num_errors", "num_bits", "nmse", "kappa", "iterations", "slot_seeds", "actual_sir_re_db")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for sir in SIRS:
            for method in METHODS:
                for parent in sorted(expected):
                    row = collected[(sir, method)][parent]
                    writer.writerow({field: row.get(field, "") for field in fields})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, label in zip(METHODS, LABELS):
        series = [row for row in rows if row["method"] == method]
        y = [row["ber"] if row["num_errors"] else row["zero_error_95_upper_bound"] for row in series]
        ax.semilogy(SIRS, y, marker="o", label=label)
        for x, value, row in zip(SIRS, y, series):
            if row["num_errors"] == 0:
                ax.annotate("0 errors; 95% bound", (x, value), fontsize=6, rotation=40)
    ax.set(xlabel="SIR_RE (dB)", ylabel="Uncoded target QPSK BER",
           title="LTE no-noise paired test, 256 parents per SIR")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "sir_ber_5000.png", dpi=200)
    fig.savefig(args.output_dir / "sir_ber_5000.pdf")
    plt.close(fig)
    files = ("summary.csv", "per_parent.csv", "paired_bootstrap.csv",
             "modulation_diagnostic_slices.csv", "sir_ber_5000.png", "sir_ber_5000.pdf")
    failures = list(args.run_dir.glob("rgs/sir_*/*.failure.json"))
    manifest = {
        "status": "complete", "contract": config,
        "sirs": SIRS, "methods": METHODS,
        "parents_per_sir": 256, "bits_per_method_sir": 2150400,
        "rgs_iterations_per_slot": 5000, "rgs_prior_calls_per_slot": 5000,
        "rgs_total_sample_updates": 11 * 256 * 2 * 5000,
        "rgs_runtime_seconds_sum_of_two_workers": float(sum(run_seconds[(sir, "waveform_alpha_rgs")] for sir in SIRS)),
        "baseline_runtime_seconds": float(sum(baseline_pipeline_seconds.values())),
        "failure_record_count_including_retried": len(failures),
        "failure_records": [str(path) for path in failures],
        "files_sha256": {name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                         for name in files},
        "clean_and_oracle_zero_errors": True,
        "all_parent_method_sir_rows_present": True,
        "zero_error_plot_points_are_95_percent_bounds_not_measured_ber": True,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state": "complete", "rows": len(rows), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
