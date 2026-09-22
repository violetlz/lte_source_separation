#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfsep.config import load_training  # noqa: E402
from rfsep.data.single_source import SingleSourceDataset  # noqa: E402
from rfsep.models.diffwave import DiffWavePrior, parameter_count  # noqa: E402
from rfsep.priors.vp import VPSchedule  # noqa: E402
from rfsep.training.status import write_json_atomic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--timed-updates", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.timed_updates < 20:
        raise ValueError("formal engineering benchmark requires at least 20 timed updates")
    config = load_training(args.config)
    runtime = config.raw["runtime"]
    device = torch.device(args.device)
    torch.manual_seed(int(runtime["seed"]))
    model = DiffWavePrior(config.raw["model"]["residual_channels"],
                          activation_checkpointing=runtime["activation_checkpointing"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.raw["optimizer"]["learning_rate"],
                                 betas=tuple(config.raw["optimizer"]["betas"]))
    schedule = VPSchedule()
    dataset = SingleSourceDataset(config.view, config.raw["source"], "train")
    effective = int(runtime["effective_batch_size"])
    microbatch = int(runtime["microbatch_size"])
    accumulation = effective // microbatch
    generator = torch.Generator(device=device).manual_seed(int(runtime["seed"]) + 77)
    precision = runtime["precision"]
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    step_times, data_times = [], []
    cursor = 0
    torch.cuda.reset_peak_memory_stats(device)
    for update in range(args.warmup_updates + args.timed_updates):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        data_seconds = 0.0
        optimizer.zero_grad(set_to_none=True)
        for accumulation_index in range(accumulation):
            ids = range(cursor + accumulation_index * microbatch,
                        cursor + (accumulation_index + 1) * microbatch)
            tick = time.perf_counter()
            batch = np.stack([dataset.window(index).network for index in ids])
            data_seconds += time.perf_counter() - tick
            x0 = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
            timestep = torch.randint(0, 50, (microbatch,), generator=generator, device=device)
            epsilon = torch.randn(x0.shape, generator=generator, device=device)
            x_t = schedule.add_noise(x0, timestep, epsilon)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=precision in {"bf16", "fp16"}):
                loss = F.mse_loss(model.predict_noise(x_t, timestep).float(), epsilon) / accumulation
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cursor += effective
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if update >= args.warmup_updates:
            step_times.append(elapsed)
            data_times.append(data_seconds)
    payload = {
        "kind": "engineering_benchmark_not_training_conclusion",
        "source": config.raw["source"], "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda, "precision": precision,
        "parameters": parameter_count(model), "window_length": 15360,
        "microbatch_size": microbatch, "accumulation_steps": accumulation,
        "effective_batch_size": effective, "warmup_updates": args.warmup_updates,
        "timed_updates": args.timed_updates, "mean_step_seconds": float(np.mean(step_times)),
        "p95_step_seconds": float(np.percentile(step_times, 95)),
        "mean_data_seconds": float(np.mean(data_times)),
        "samples_per_second": effective / float(np.mean(step_times)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "estimated_2000_update_seconds": 2000 * float(np.mean(step_times)),
        "platform_hash": config.view.platform.config_hash,
        "training_view_hash": config.view.config_hash,
        "training_config_hash": config.config_hash,
    }
    write_json_atomic(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
