#!/usr/bin/env python3
"""Measure the actual frozen-prior waveform RGS throughput before a formal run."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfsep.config import load_platform
from rfsep.contracts import complex_to_network
from rfsep.observation import build_example, load_calibration
from rfsep.priors.inference import load_prior_for_inference
from rfsep.separation.rgs import RGSConfig, separate_rgs
from smoke_rgs import _direct_initial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    platform = load_platform(ROOT / "configs/platform/lte_v1.yaml")
    calibration = load_calibration(ROOT / "reports/formal_validation/calibration_summary.json", platform)
    target = load_prior_for_inference(ROOT / "artifacts/prior_formal/prior-formal-20260923/target/checkpoints/last.pt", source="target", device=args.device)
    interference = load_prior_for_inference(ROOT / "artifacts/prior_formal/prior-formal-20260923/interference/checkpoints/last.pt", source="interference", device=args.device)
    ys, initials = [], []
    for i in range((args.batch + 1) // 2):
        obs, _, _ = build_example(platform, "audit", 1000000 + i, -6, calibration, dtype=np.complex64)
        initial = _direct_initial(obs.y, platform)
        for slot in range(2):
            sl = slice(slot * 15360, (slot + 1) * 15360)
            ys.append(complex_to_network(obs.y[sl]))
            initials.append(complex_to_network(initial[sl]))
    y = torch.from_numpy(np.stack(ys[:args.batch])).to(args.device)
    initial = torch.from_numpy(np.stack(initials[:args.batch])).to(args.device)
    _, info, _ = build_example(platform, "audit", 1000000, -6, calibration, dtype=np.complex64)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    _, audit = separate_rgs(y, info.kappa, initial, target, interference,
                            RGSConfig(args.steps, 0.005, sample_seeds=tuple(range(1000, 1000 + args.batch)), diagnostic_interval=100))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    payload = {"steps": args.steps, "batch": args.batch, "elapsed_seconds": seconds,
               "sample_steps_per_second": args.steps * args.batch / seconds,
               "full_grid_sample_steps": 11 * 256 * 2 * 5000,
               "estimated_full_grid_hours_two_gpus": 11 * 256 * 2 * 5000 / (2 * args.steps * args.batch / seconds) / 3600,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "rgs_audit": audit}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
