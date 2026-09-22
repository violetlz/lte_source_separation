#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfsep.models.diffwave import DiffWavePrior  # noqa: E402
from rfsep.priors.vp import VPSchedule  # noqa: E402
from rfsep.data.materialized import MaterializedPriorDataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("target", "interference"), required=True)
    parser.add_argument("--microbatches", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--trace")
    parser.add_argument("--manifest")
    parser.add_argument("--loader-only", action="store_true")
    args = parser.parse_args()
    if args.loader_only:
        if not args.manifest:
            raise ValueError("--loader-only requires --manifest")
        dataset = MaterializedPriorDataset(args.manifest)
        loader_results = []
        for workers in (2, 4, 8):
            loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=workers,
                                pin_memory=True, persistent_workers=True, prefetch_factor=2)
            iterator = iter(loader)
            for _ in range(20):
                next(iterator)
            tick = time.perf_counter()
            for _ in range(100):
                next(iterator)
            seconds = time.perf_counter() - tick
            loader_results.append({"workers": workers, "batches": 100,
                                   "samples_per_second": 1600 / seconds,
                                   "mean_batch_seconds": seconds / 100})
            del iterator, loader
        payload = {"profile": "materialized-loader-v1", "results": loader_results}
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(encoded)
        print(encoded, end="")
        return 0
    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiler process must see exactly one GPU")
    channels = 64 if args.source == "target" else 128
    outcomes = []
    for micro in args.microbatches:
        if 16 % micro:
            continue
        torch.cuda.empty_cache()
        try:
            model = DiffWavePrior(channels, 24, 12, False).cuda().train()
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, foreach=True)
            schedule = VPSchedule().cuda()
            batch = torch.randn(16, 2, 15360, device="cuda")
            times = []
            for update in range(args.warmup + args.updates):
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                for begin in range(0, 16, micro):
                    x0 = batch[begin:begin + micro]
                    timestep = torch.randint(0, 50, (micro,), device="cuda")
                    epsilon = torch.randn_like(x0)
                    x_t = schedule.add_noise(x0, timestep, epsilon, validate=False)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = F.mse_loss(model(x_t, timestep).float(), epsilon) / (16 // micro)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
                optimizer.step()
                torch.cuda.synchronize()
                if update >= args.warmup:
                    times.append(time.perf_counter() - tick)
            reserved = torch.cuda.max_memory_reserved()
            outcomes.append({"source": args.source, "microbatch": micro,
                             "updates": args.updates, "mean_step_seconds": sum(times)/len(times),
                             "samples_per_second": 16/(sum(times)/len(times)),
                             "peak_reserved_bytes": reserved, "status": "ok"})
        except torch.cuda.OutOfMemoryError:
            outcomes.append({"source": args.source, "microbatch": micro, "status": "oom"})
        finally:
            for name in ("model", "optimizer", "schedule", "batch"):
                if name in locals():
                    del locals()[name]
    payload = {"profile": "gpu-resident-compute-upper-v1", "results": outcomes}
    if args.trace:
        if not args.manifest:
            raise ValueError("--trace requires a validated --manifest")
        micro = max((item["microbatch"] for item in outcomes if item["status"] == "ok"),
                    key=lambda value: next(item["samples_per_second"] for item in outcomes
                                           if item.get("microbatch") == value))
        dataset = MaterializedPriorDataset(args.manifest)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True, prefetch_factor=2)
        iterator = iter(loader)
        model = DiffWavePrior(channels, 24, 12, False).cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, foreach=True)
        schedule = VPSchedule().cuda()

        def update():
            batch = next(iterator).cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            for begin in range(0, 16, micro):
                x0 = batch[begin:begin + micro]
                timestep = torch.randint(0, 50, (micro,), device="cuda")
                epsilon = torch.randn_like(x0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(schedule.add_noise(x0, timestep, epsilon,
                                                          validate=False), timestep)
                    loss = F.mse_loss(prediction.float(), epsilon) / (16 // micro)
                loss.backward()
            optimizer.step()

        for _ in range(20):
            update()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True,
        ) as profiler:
            for _ in range(5):
                update()
                profiler.step()
        profiler.export_chrome_trace(args.trace)
        payload["trace"] = {"path": str(Path(args.trace).resolve()),
                            "updates": 5, "microbatch": micro,
                            "top_cuda_ops": profiler.key_averages().table(
                                sort_by="self_cuda_time_total", row_limit=12)}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
