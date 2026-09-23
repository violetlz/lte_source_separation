#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfsep.baselines.lte_estimated import estimate_train_statistics
from rfsep.config import load_platform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=int, default=512)
    parser.add_argument("--bins", type=int, default=2048)
    parser.add_argument("--smooth-bins", type=float, default=8.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    platform = load_platform(ROOT / "configs/platform/lte_v1.yaml")
    stats = estimate_train_statistics(platform, range(500000, 500000 + args.parents),
                                      bins=args.bins, smooth_bins=args.smooth_bins)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, separators=(",", ":")) + "\n")
    print(json.dumps({key: stats[key] for key in ("platform_hash", "num_parents", "num_re_samples")}, indent=2))
    print(json.dumps(stats["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
