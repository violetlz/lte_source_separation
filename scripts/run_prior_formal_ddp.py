#!/usr/bin/env python3
"""Resume the interference formal prior on two GPUs with a global batch of 16."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfsep.training.formal import load_formal_config  # noqa: E402
from rfsep.training.formal_ddp import train_formal_ddp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/training/interference_prior_formal.yaml"))
    parser.add_argument("--run-dir", default=str(ROOT / "artifacts/prior_formal/prior-formal-20260923/interference"))
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("launch with torchrun --nproc_per_node=2")
    raw = load_formal_config(args.config)
    train_formal_ddp(raw, Path(args.run_dir).resolve(), stop_after=args.stop_after)


if __name__ == "__main__":
    main()
