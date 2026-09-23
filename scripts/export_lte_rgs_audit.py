#!/usr/bin/env python3
"""Export honest audit-only integer counts from the atomic RGS shards."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for run in sorted(args.artifact_root.glob("audit*")):
        config_file = run / "resolved_config.json"
        if not config_file.exists():
            continue
        config = json.loads(config_file.read_text())
        grouped = defaultdict(list)
        for shard in (run / "rgs").glob("sir_*/*.json"):
            payload = json.loads(shard.read_text())
            if payload["status"] != "complete":
                raise ValueError(f"incomplete audit shard {shard}")
            for row in payload["rows"]:
                grouped[int(row["sir_re_db"])].append(row)
        for sir, values in sorted(grouped.items()):
            rows.append({"audit_run": run.name, "split": config["split"],
                         "sir_re_db": sir, "max_lr": config["rgs_max_lr"],
                         "iterations": config["rgs_iterations"],
                         "num_parents": len(values),
                         "num_errors": sum(int(row["num_errors"]) for row in values),
                         "num_bits": sum(int(row["num_bits"]) for row in values),
                         "mean_nmse": sum(float(row["nmse"]) for row in values) / len(values)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} audit SIR rows to {args.output}")


if __name__ == "__main__":
    main()
