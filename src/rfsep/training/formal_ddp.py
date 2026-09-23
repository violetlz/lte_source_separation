"""Two-GPU continuation of the formal interference prior.

Each rank consumes half of the original 16-sample global batch. The durable
sampler still advances by 16 only after one synchronized optimizer update.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ..data.materialized import (MaterializedPriorDataset, RankSliceBatchSampler,
                                 RecoverableEpochBatchSampler, sha256_file)
from ..priors.vp import VPSchedule
from .checkpoint import atomic_torch_save, git_state, rng_state
from .formal import _model, _validate
from .status import append_jsonl, update_status
from .trainer import ExponentialMovingAverage


def _restore_rank_rng(state: dict, generator: torch.Generator, rank: int) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    generator.set_state(state["sampler_generator"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        torch.cuda.set_rng_state(cuda_states[min(rank, len(cuda_states) - 1)].cpu(), rank)


def train_formal_ddp(raw: dict, run_dir: Path, *, stop_after: int | None = None) -> None:
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if raw["source"] != "interference" or world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("two-GPU continuation is supported only for interference")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("each GPU must support BF16")
    dist.init_process_group("nccl", device_id=device)
    leader = rank == 0
    view = raw.pop("_view")
    config_path, config_hash = Path(raw.pop("_config_path")), raw.pop("_config_hash")
    checkpoint_dir = run_dir / "checkpoints"
    formal_last = checkpoint_dir / "last.pt"
    status_path, metrics_path = run_dir / "status.json", run_dir / "metrics.jsonl"
    stop_marker = run_dir / "stop_ddp.request"
    if not formal_last.is_file():
        raise FileNotFoundError("two-GPU training requires a completed formal checkpoint")
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    if resolved != raw:
        raise ValueError("immutable formal config changed")
    train_manifest = (config_path.parent / raw["data"]["train_manifest"]).resolve()
    validation_manifest = (config_path.parent / raw["data"]["validation_manifest"]).resolve()
    train_data = MaterializedPriorDataset(train_manifest)
    validation_data = MaterializedPriorDataset(validation_manifest) if leader else None
    if train_data.manifest["source"] != "interference" or len(train_data) != 200000:
        raise ValueError("interference training manifest mismatch")
    if leader and len(validation_data) != 2048:
        raise ValueError("validation manifest mismatch")
    payload = torch.load(formal_last, map_location="cpu", weights_only=False)
    if (payload.get("schema_version") != 2 or payload.get("source") != "interference"
            or payload.get("formal_config_hash") != config_hash
            or payload.get("dataset_id") != train_data.manifest["dataset_id"]
            or payload.get("dataset_manifest_sha256") != sha256_file(train_manifest)):
        raise ValueError("strict formal checkpoint identity mismatch")
    if int(payload["global_step"]) >= int(raw["budget"]["max_cumulative_updates"]):
        raise ValueError("checkpoint already reached formal budget")
    model = _model(raw).to(device)
    model.load_state_dict(payload["model"])
    ema = ExponentialMovingAverage(model, float(raw["optimizer"]["ema_decay"]))
    ema.model.load_state_dict(payload["ema"])
    ddp = DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                  broadcast_buffers=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(raw["optimizer"]["learning_rate"]),
                                 betas=tuple(raw["optimizer"]["betas"]),
                                 weight_decay=float(raw["optimizer"]["weight_decay"]),
                                 foreach=True)
    optimizer.load_state_dict(payload["optimizer"])
    schedule = VPSchedule(50, 0.0001, 0.05).to(device)
    noise_generator = torch.Generator(device=device)
    previous_rng = payload.get("rng_per_rank")
    if previous_rng is not None:
        if len(previous_rng) != world:
            raise ValueError("checkpoint RNG rank count mismatch")
        _restore_rank_rng(previous_rng[rank], noise_generator, rank)
    elif rank == 0:
        _restore_rank_rng(payload["rng"], noise_generator, rank)
    else:
        seed = int.from_bytes(hashlib.sha256(
            (sha256_file(formal_last) + ":rank1").encode()).digest()[:8], "big")
        noise_generator.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    sampler = RecoverableEpochBatchSampler(len(train_data), 16,
                                            int(raw["runtime"]["sampler_seed"]))
    sampler.load_state_dict(payload["dataset_sampler"])
    global_step = int(payload["global_step"])
    historical_exposures = int(payload["historical_exposures"])
    if historical_exposures + sampler.exposures != int(payload["cumulative_exposures"]):
        raise ValueError("checkpoint sample accounting mismatch")
    initial_step = global_step
    log_sequence = int(payload["log_sequence"])
    validation_count = int(payload["validation_count"])
    best_metric = float(payload["best_denoising_metric"])
    last_validation = dict(payload["last_validation"])
    stability_achieved = bool(payload.get("stability_achieved", False))
    if not stability_achieved:
        raise ValueError("two-GPU continuation requires a stable prior checkpoint")

    wandb_run = None
    if leader:
        import wandb
        tracking = raw["tracking"]
        wandb_run = wandb.init(project=tracking["project"], group=tracking["group"],
                               name=tracking["name"], id=payload["wandb"]["run_id"],
                               resume="must", mode="online",
                               tags=["interference", "stage-formal", "ddp-two-gpu"])
        wandb_run.define_metric("optimizer/global_step")
        wandb_run.define_metric("train/*", step_metric="optimizer/global_step")
        wandb_run.define_metric("validation/*", step_metric="optimizer/global_step")
        update_status(status_path, state="training", pid=os.getpid(), global_step=global_step,
                      formal_exposures=sampler.exposures, epoch=sampler.epoch,
                      epoch_offset=sampler.offset, execution_backend="ddp-nccl-2gpu",
                      gpu_uuids=[str(torch.cuda.get_device_properties(i).uuid)
                                 for i in range(world)],
                      resume_checkpoint_sha256=sha256_file(formal_last),
                      resume_step=global_step)
    dist.barrier()

    loader_kwargs = dict(num_workers=int(raw["runtime"]["num_workers"]),
                         pin_memory=True, persistent_workers=True,
                         prefetch_factor=int(raw["runtime"]["prefetch_factor"]))
    loader = DataLoader(train_data, batch_sampler=RankSliceBatchSampler(sampler, rank, world),
                        **loader_kwargs)
    iterator = iter(loader)
    # Each rank sees eight samples. DDP's gradient average yields the original
    # global-16 mean loss, with one all-reduce per optimizer update.
    local_batch = int(raw["runtime"]["effective_batch_size"]) // world
    if local_batch != int(raw["runtime"]["microbatch_size"]):
        raise ValueError("DDP continuation requires one microbatch per rank")
    benchmark_times: list[float] = []
    rank_rng: list[dict] | None = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {sig: signal.signal(sig, request_stop)
                    for sig in (signal.SIGINT, signal.SIGTERM)}

    def capture_rng() -> None:
        nonlocal rank_rng
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, rng_state(noise_generator))
        if leader:
            rank_rng = gathered

    def checkpoint_payload() -> dict:
        if rank_rng is None:
            raise RuntimeError("both rank RNG states must be captured before checkpoint")
        result = dict(payload)
        result.update({"model": model.state_dict(), "ema": ema.model.state_dict(),
                       "optimizer": optimizer.state_dict(), "global_step": global_step,
                       "formal_updates": global_step - 2000,
                       "formal_exposures": sampler.exposures,
                       "cumulative_exposures": historical_exposures + sampler.exposures,
                       "dataset_sampler": sampler.state_dict(),
                       "rng": rank_rng[0], "rng_per_rank": rank_rng,
                       "distributed_world_size": world,
                       "execution_backend": "ddp-nccl-2gpu",
                       "best_denoising_metric": best_metric,
                       "last_validation": last_validation,
                       "validation_count": validation_count,
                       "log_sequence": log_sequence,
                       "wandb": {"run_id": wandb_run.id, "run_path": wandb_run.path,
                                 "url": wandb_run.url},
                       "environment": {"python": sys.version, "torch": torch.__version__,
                                       "numpy": np.__version__, "host": platform.node(),
                                       "cuda": torch.version.cuda,
                                       "gpu_uuids": [str(torch.cuda.get_device_properties(i).uuid)
                                                     for i in range(world)]},
                       "git": git_state(Path(__file__).resolve().parents[3])})
        return result

    def save(path: Path) -> str:
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("non-finite model parameter before checkpoint")
        return atomic_torch_save(checkpoint_payload(), path)

    def validate(*, full: bool) -> None:
        nonlocal validation_count, last_validation, best_metric, log_sequence
        if leader:
            update_status(status_path, state="validation", global_step=global_step)
            last_validation = _validate(raw, (("raw", model), ("ema", ema.model)),
                                        validation_data, schedule, device, full=full)
            last_validation.update({"optimizer/global_step": global_step,
                                    "tracking/event_sequence": log_sequence})
            append_jsonl(metrics_path, {"kind": "validation", **last_validation})
            wandb_run.log(last_validation)
            log_sequence += 1
            validation_count += 1
            score = min(float(last_validation["validation/raw/multi_noise_mean_noise_mse"]),
                        float(last_validation["validation/ema/multi_noise_mean_noise_mse"]))
            if full and score < best_metric:
                best_metric = score
                save(checkpoint_dir / "best_denoising.pt")
            model.train()
        dist.barrier()

    def learning_rate(step: int) -> float:
        maximum = float(raw["optimizer"]["learning_rate"])
        minimum = float(raw["optimizer"]["min_learning_rate"])
        maximum_step = int(raw["budget"]["max_cumulative_updates"])
        decay_start = int(maximum_step * 0.8)
        if step <= decay_start:
            return maximum
        ratio = (step - decay_start) / (maximum_step - decay_start)
        return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * ratio))

    max_updates = int(raw["budget"]["max_cumulative_updates"])
    limit = min(max_updates, stop_after) if stop_after is not None else max_updates
    try:
        model.train()
        while global_step < limit:
            # The marker is the graceful stop interface for the two rank job.
            local_stop = torch.tensor(int(stop_requested or stop_marker.exists()), device=device)
            dist.all_reduce(local_stop, op=dist.ReduceOp.MAX)
            if bool(local_stop):
                break
            tick = time.perf_counter()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                loader = DataLoader(train_data,
                                    batch_sampler=RankSliceBatchSampler(sampler, rank, world),
                                    **loader_kwargs)
                iterator = iter(loader)
                cpu_batch = next(iterator)
            data_ready = time.perf_counter()
            x0 = cpu_batch.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            timestep = torch.randint(0, 50, (local_batch,), generator=noise_generator,
                                     device=device)
            epsilon = torch.randn(x0.shape, generator=noise_generator, device=device)
            x_t = schedule.add_noise(x0, timestep, epsilon, validate=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = ddp(x_t, timestep)
                loss = F.mse_loss(predicted.float(), epsilon)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss before optimizer commit")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(raw["optimizer"]["grad_clip"]),
                                                       foreach=True)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("non-finite gradient before optimizer commit")
            lr = learning_rate(global_step + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            ema.update(model)
            sampler.commit()
            global_step += 1
            step_seconds = time.perf_counter() - tick
            benchmark_times.append(step_seconds)
            benchmark_times = benchmark_times[-100:]
            if global_step % int(raw["tracking"]["log_every"]) == 0:
                mean_loss = loss.detach().clone()
                dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
                if leader:
                    metrics = {"optimizer/global_step": global_step,
                               "tracking/event_sequence": log_sequence,
                               "train/noise_mse": float(mean_loss),
                               "train/grad_norm": float(grad_norm),
                               "train/learning_rate": lr,
                               "train/formal_exposures": sampler.exposures,
                               "train/cumulative_exposures": historical_exposures + sampler.exposures,
                               "train/epoch": sampler.epoch,
                               "train/epoch_offset": sampler.offset,
                               "train/step_seconds": step_seconds,
                               "train/data_wait_seconds": data_ready - tick,
                               "train/samples_per_second": 16 / step_seconds,
                               "train/gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                               "train/distributed_world_size": world}
                    append_jsonl(metrics_path, {"kind": "train", **metrics})
                    wandb_run.log(metrics)
                    log_sequence += 1
                    update_status(status_path, state="stable_running", global_step=global_step,
                                  formal_exposures=sampler.exposures, epoch=sampler.epoch,
                                  epoch_offset=sampler.offset, last_train=metrics)
            quick = global_step % int(raw["validation"]["quick_every"]) == 0
            checkpoint = global_step % int(raw["checkpoint"]["last_every"]) == 0
            if quick or checkpoint:
                capture_rng()
            if quick:
                validate(full=global_step % int(raw["validation"]["full_every"]) == 0)
            if checkpoint:
                if leader:
                    digest = save(formal_last)
                    save(checkpoint_dir / f"step_{global_step:06d}.pt")
                    numbered = sorted(checkpoint_dir.glob("step_*.pt"))
                    for stale in numbered[:-int(raw["checkpoint"]["keep_last"])]:
                        stale.unlink()
                    update_status(status_path, state="stable_running", global_step=global_step,
                                  checkpoint=str(formal_last), checkpoint_sha256=digest)
                dist.barrier()
        capture_rng()
        if global_step >= max_updates and (not last_validation or
                int(last_validation.get("optimizer/global_step", -1)) != global_step or
                not last_validation.get("validation/full")):
            validate(full=True)
        if leader:
            digest = save(formal_last)
            state = "completed" if global_step >= max_updates else "paused"
            update_status(status_path, state=state, global_step=global_step,
                          formal_exposures=sampler.exposures, epoch=sampler.epoch,
                          epoch_offset=sampler.offset, checkpoint=str(formal_last),
                          checkpoint_sha256=digest,
                          exit_reason="budget" if state == "completed" else "signal_or_stage",
                          benchmark={"updates": len(benchmark_times),
                                     "mean_step_seconds": float(np.mean(benchmark_times)),
                                     "samples_per_second": 16 / float(np.mean(benchmark_times)),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            wandb_run.finish(exit_code=0)
            print(json.dumps({"state": state, "global_step": global_step,
                              "checkpoint_sha256": digest}, sort_keys=True), flush=True)
        dist.barrier()
    except Exception as exc:
        if leader:
            update_status(status_path, state="failed", global_step=global_step,
                          failure=f"{type(exc).__name__}: {exc}", checkpoint=str(formal_last),
                          checkpoint_sha256=sha256_file(formal_last))
            if wandb_run is not None:
                wandb_run.finish(exit_code=1)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        dist.destroy_process_group()
