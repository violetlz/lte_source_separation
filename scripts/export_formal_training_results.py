#!/usr/bin/env python3
"""Export the fixed full-validation curves without copying training artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0, 1, 4, 9, 24, 39, 49)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def export(run_root: Path, output: Path) -> list[dict]:
    columns = ["source", "optimizer_step", "validation_windows"]
    for weight in ("raw", "ema"):
        columns.append(f"{weight}_multi_noise_mean_noise_mse")
        for level in LEVELS:
            for metric in ("noise_mse", "real_mse", "nmse"):
                columns.append(f"{weight}_t{level}_{metric}")
    result = []
    for source, budget in (("target", 200000), ("interference", 300000)):
        directory = run_root / source
        status = json.loads((directory / "status.json").read_text())
        if (status.get("state") != "completed" or status.get("exit_reason") != "budget"
                or status.get("global_step") != budget
                or status.get("formal_exposures") != (budget - 2000) * 16):
            raise ValueError(f"{source} formal training is not complete or counts differ")
        checkpoint = directory / "checkpoints" / "last.pt"
        if sha256_file(checkpoint) != status["checkpoint_sha256"]:
            raise ValueError(f"{source} final checkpoint SHA-256 mismatch")
        by_step = {}
        for line in (directory / "metrics.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record.get("kind") != "validation" or record.get("validation/full") != 1:
                continue
            step = int(record["optimizer/global_step"])
            if record["validation/windows"] != 2048:
                raise ValueError(f"{source} step {step} has incomplete full validation")
            row = {"source": source, "optimizer_step": step,
                   "validation_windows": record["validation/windows"]}
            for weight in ("raw", "ema"):
                row[f"{weight}_multi_noise_mean_noise_mse"] = record[
                    f"validation/{weight}/multi_noise_mean_noise_mse"]
                for level in LEVELS:
                    for metric in ("noise_mse", "real_mse", "nmse"):
                        row[f"{weight}_t{level}_{metric}"] = record[
                            f"validation/{weight}/t{level}/{metric}"]
            if step in by_step and by_step[step] != row:
                raise ValueError(f"{source} step {step} has conflicting validations")
            by_step[step] = row
        expected = set(range(10000, budget + 1, 10000))
        if set(by_step) != expected:
            raise ValueError(f"{source} full-validation steps are incomplete")
        result.extend(by_step[step] for step in sorted(by_step))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path,
                        default=ROOT / "artifacts/prior_formal/prior-formal-20260923")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/formal_validation_full_20260923.csv")
    args = parser.parse_args()
    rows = export(args.run_root, args.output)
    print(f"exported {len(rows)} fixed full-validation rows to {args.output}")


if __name__ == "__main__":
    main()
