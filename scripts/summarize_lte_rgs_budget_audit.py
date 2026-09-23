#!/usr/bin/env python3
"""Strict count-level summary for the paired audit budget comparison."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_lte_rgs_budgets import BUDGETS, PARENTS, SIRS

BASELINES = ("direct", "fft_diagonal_wiener_approx", "single_re_marginal_bit_map_estimated")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "resolved_config.json").read_text())
    grouped = defaultdict(dict)
    shard_count = 0
    for shard in (args.run_dir / "shards").glob("sir_*/*.json"):
        payload = json.loads(shard.read_text())
        if payload["status"] != "complete" or payload["config_sha256"] != config["config_sha256"]:
            raise ValueError(f"invalid audit shard {shard}")
        shard_count += 1
        for row in payload["rows"]:
            key = (int(row["sir_re_db"]), row["method"], row.get("trajectory_kind", "baseline"),
                   int(row.get("schedule_total_steps", 0)), int(row.get("recorded_step", 0)))
            parent = int(row["parent_id"])
            if parent in grouped[key]:
                raise ValueError(f"duplicate {key} parent {parent}")
            grouped[key][parent] = row
    expected = set(PARENTS)
    needed = []
    for sir in SIRS:
        needed.extend((sir, method, "baseline", 0, 0) for method in BASELINES)
        needed.extend((sir, "waveform_alpha_rgs", "completed_budget", step, step)
                      for step in BUDGETS)
        needed.extend((sir, "waveform_alpha_rgs", "shared_5000_path_snapshot", 5000, step)
                      for step in (1000, 2000))
    progress = {str(key): len(grouped[key]) for key in needed}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "progress.json").write_text(json.dumps({"complete_shards": shard_count,
                                                                 "expected_shards": len(SIRS) * len(PARENTS),
                                                                 "group_parent_counts": progress}, indent=2) + "\n")
    if shard_count != len(SIRS) * len(PARENTS) or any(set(grouped[key]) != expected for key in needed):
        print(json.dumps({"status": "incomplete", "complete_shards": shard_count,
                          "expected_shards": len(SIRS) * len(PARENTS)}))
        return
    summary = []
    for key in needed:
        sir, method, kind, total_steps, recorded_step = key
        values = list(grouped[key].values())
        errors = sum(int(row["num_errors"]) for row in values)
        bits = sum(int(row["num_bits"]) for row in values)
        if bits != 16 * 8400:
            raise ValueError(f"bit denominator mismatch: {key}")
        if kind == "completed_budget" and any(
                row["rgs_audit"]["target_prior_calls"] != total_steps or
                row["rgs_audit"]["interference_prior_calls"] != total_steps for row in values):
            raise ValueError(f"missing prior calls: {key}")
        summary.append({"sir_re_db": sir, "method": method, "trajectory_kind": kind,
                        "schedule_total_steps": total_steps, "recorded_step": recorded_step,
                        "num_parents": 16, "num_errors": errors, "num_bits": bits,
                        "ber": errors / bits,
                        "mean_nmse": np.mean([row["nmse"] for row in values if row.get("nmse") is not None])
                                     if method != "single_re_marginal_bit_map_estimated" else "",
                        "runtime_seconds_sum": sum(float(row.get("elapsed_seconds", 0)) for row in values)
                                               if kind == "completed_budget" else ""})
    write_csv(args.output_dir / "summary.csv", summary)
    rng = np.random.default_rng(20260923)
    sample_indices = rng.integers(0, len(PARENTS), size=(10000, len(PARENTS)))
    differences = []
    for sir in SIRS:
        for step in BUDGETS:
            rkey = (sir, "waveform_alpha_rgs", "completed_budget", step, step)
            rgs = np.array([grouped[rkey][parent]["num_errors"] for parent in PARENTS])
            for baseline in BASELINES:
                bkey = (sir, baseline, "baseline", 0, 0)
                other = np.array([grouped[bkey][parent]["num_errors"] for parent in PARENTS])
                delta = (rgs - other) / 8400
                draws = delta[sample_indices].mean(axis=1)
                differences.append({"sir_re_db": sir, "rgs_completed_budget": step,
                                    "reference": baseline, "mean_ber_difference": float(delta.mean()),
                                    "paired_bootstrap_95_lower": float(np.quantile(draws, 0.025)),
                                    "paired_bootstrap_95_upper": float(np.quantile(draws, 0.975)),
                                    "bootstrap_replicates": len(sample_indices)})
            if step != 5000:
                other = np.array([grouped[(sir, "waveform_alpha_rgs", "completed_budget", 5000, 5000)]
                                  [parent]["num_errors"] for parent in PARENTS])
                delta = (rgs - other) / 8400
                draws = delta[sample_indices].mean(axis=1)
                differences.append({"sir_re_db": sir, "rgs_completed_budget": step,
                                    "reference": "rgs_completed_5000", "mean_ber_difference": float(delta.mean()),
                                    "paired_bootstrap_95_lower": float(np.quantile(draws, 0.025)),
                                    "paired_bootstrap_95_upper": float(np.quantile(draws, 0.975)),
                                    "bootstrap_replicates": len(sample_indices)})
    write_csv(args.output_dir / "paired_differences.csv", differences)
    print(json.dumps({"status": "complete", "complete_shards": shard_count,
                      "summary": str(args.output_dir / "summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
