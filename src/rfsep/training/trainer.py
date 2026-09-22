from __future__ import annotations

import copy
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
from torch.nn import functional as F

from ..config import TrainingConfig
from ..data.single_source import CursorSampler, SingleSourceDataset
from ..models.diffwave import DiffWavePrior, parameter_count
from ..priors.vp import VPSchedule
from .checkpoint import (SCHEMA_VERSION, atomic_torch_save, git_state, load_checkpoint,
                         restore_rng_state, rng_state)
from .status import append_jsonl, update_status, write_json_atomic
from .tracking import init_wandb, verify_server_history
from .validation import validate_model


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for average, current in zip(self.model.parameters(), model.parameters(), strict=True):
            average.mul_(self.decay).add_(current.detach(), alpha=1.0 - self.decay)
        for average, current in zip(self.model.buffers(), model.buffers(), strict=True):
            average.copy_(current)


def _model_for(config: TrainingConfig) -> DiffWavePrior:
    model = config.raw["model"]
    return DiffWavePrior(
        residual_channels=int(model["residual_channels"]),
        residual_layers=int(model["residual_layers"]),
        dilation_cycle=int(model["dilation_cycle"]),
        activation_checkpointing=bool(config.raw["runtime"]["activation_checkpointing"]),
    )


def _checkpoint_payload(*, config, model, ema, optimizer, sampler, generator,
                        global_step, best_metric, cumulative_wall, wandb_info,
                        log_sequence, amp_skips, resume_count, validation_count,
                        project_root, last_metrics, stability_achieved) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "source": config.raw["source"],
        "model": model.state_dict(), "ema": ema.model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": None, "scaler": None,
        "global_step": global_step, "microbatch_size": config.raw["runtime"]["microbatch_size"],
        "accumulation_steps": config.raw["runtime"]["effective_batch_size"] //
                              config.raw["runtime"]["microbatch_size"],
        "sampler": sampler.state_dict(), "consumed_sample_cursor": sampler.cursor,
        "rng": rng_state(generator), "best_metric": best_metric,
        "cumulative_process_seconds": cumulative_wall,
        "platform_config": config.view.platform.raw,
        "platform_hash": config.view.platform.config_hash,
        "training_view_config": config.view.raw, "training_view_hash": config.view.config_hash,
        "training_config": config.raw, "training_config_hash": config.config_hash,
        "git": git_state(project_root),
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "host": platform.node(),
                        "cuda": torch.version.cuda},
        "wandb": wandb_info, "log_sequence": log_sequence, "amp_skipped_updates": amp_skips,
        "resume_count": resume_count, "validation_count": validation_count,
        "last_metrics": last_metrics, "stability_achieved": stability_achieved,
    }


def train(config: TrainingConfig, run_dir: Path, *, device_name: str,
          stop_after: int | None = None, verify_wandb: bool = True) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    source = config.raw["source"]
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual pilot requires an available CUDA GPU")
    runtime = config.raw["runtime"]
    microbatch = int(runtime["microbatch_size"])
    effective_batch = int(runtime["effective_batch_size"])
    if effective_batch % microbatch:
        raise ValueError("effective batch must be divisible by microbatch")
    accumulation = effective_batch // microbatch
    precision = str(runtime["precision"])
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("configured BF16 is not supported by this GPU")
    seed = int(runtime["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sampler_generator = torch.Generator(device=device)
    sampler_generator.manual_seed(seed + 1)
    model = _model_for(config).to(device)
    ema = ExponentialMovingAverage(model, float(config.raw["optimizer"]["ema_decay"]))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.raw["optimizer"]["learning_rate"]),
        betas=tuple(config.raw["optimizer"]["betas"]),
        weight_decay=float(config.raw["optimizer"]["weight_decay"]),
    )
    schedule = VPSchedule(**{key: config.raw["diffusion"][key]
                             for key in ("steps", "beta_start", "beta_end")})
    sampler = CursorSampler(source)
    dataset = SingleSourceDataset(config.view, source, "train")
    validation_dataset = SingleSourceDataset(config.view, source, "validation")
    global_step = 0
    best_metric = math.inf
    cumulative_before = 0.0
    log_sequence = 0
    amp_skips = 0
    resume_count = 0
    validation_count = 0
    stability_achieved = False
    last_metrics = {}
    wandb_info = {"project": config.raw["tracking"]["project"], "run_id": None,
                  "run_path": None, "url": None}
    resumed = last_path.is_file()
    if resumed:
        payload = load_checkpoint(last_path, training_hash=config.config_hash, source=source,
                                  map_location=device)
        model.load_state_dict(payload["model"])
        ema.model.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        sampler.load_state_dict(payload["sampler"])
        global_step = int(payload["global_step"])
        best_metric = float(payload["best_metric"])
        cumulative_before = float(payload["cumulative_process_seconds"])
        log_sequence = int(payload["log_sequence"])
        amp_skips = int(payload["amp_skipped_updates"])
        resume_count = int(payload.get("resume_count", 0)) + 1
        validation_count = int(payload.get("validation_count", 0))
        stability_achieved = bool(payload.get("stability_achieved", False))
        last_metrics = dict(payload.get("last_metrics", {}))
        wandb_info = dict(payload["wandb"])
        restore_rng_state(payload["rng"], sampler_generator)
    metadata = {
        "source": source, "stage": "pilot", "window_ms": 0.5,
        "parameter_count": parameter_count(model),
        "theoretical_receptive_field": model.theoretical_receptive_field,
        "platform_hash": config.view.platform.config_hash,
        "training_view_hash": config.view.config_hash,
        "training_config_hash": config.config_hash,
        "config": config.raw, "device": torch.cuda.get_device_name(device),
        "precision": precision, "microbatch_size": microbatch,
        "accumulation_steps": accumulation,
    }
    wandb_run = init_wandb(config, wandb_info.get("run_id"), resumed, metadata)
    wandb_info.update({"run_id": wandb_run.id, "run_path": wandb_run.path,
                       "url": wandb_run.url, "project": wandb_run.project,
                       "entity": wandb_run.entity})
    started = time.monotonic()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    update_status(status_path, state="running", source=source, pid=os.getpid(), device=str(device),
                  global_step=global_step, consumed_sample_cursor=sampler.cursor,
                  budget=config.raw["budget"], wandb=wandb_info, resumed=resumed,
                  resume_count=resume_count, log_path=str(metrics_path.resolve()))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    autocast_enabled = precision in {"bf16", "fp16"}
    recent_finite = []
    memory_samples = []
    benchmark_step_times = []
    benchmark_data_times = []
    max_updates = int(config.raw["budget"]["max_updates"])
    max_seconds = float(config.raw["budget"]["max_process_seconds"])
    requested_limit = min(max_updates, stop_after) if stop_after is not None else max_updates

    def elapsed_total() -> float:
        return cumulative_before + (time.monotonic() - started)

    def save(path: Path) -> str:
        payload = _checkpoint_payload(
            config=config, model=model, ema=ema, optimizer=optimizer, sampler=sampler,
            generator=sampler_generator, global_step=global_step, best_metric=best_metric,
            cumulative_wall=elapsed_total(), wandb_info=wandb_info,
            log_sequence=log_sequence, amp_skips=amp_skips, resume_count=resume_count,
            validation_count=validation_count, project_root=project_root,
            last_metrics=last_metrics, stability_achieved=stability_achieved,
        )
        return atomic_torch_save(payload, path)

    def run_validation(full: bool) -> dict:
        nonlocal best_metric, validation_count, log_sequence, last_metrics
        count = None if full else int(config.raw["validation"]["quick_parents"])
        combined = {}
        for label, validation_model in (("raw", model), ("ema", ema.model)):
            result = validate_model(
                validation_model, validation_dataset, schedule,
                list(config.raw["validation"]["noise_levels"]), device, precision,
                parent_limit=count,
            )
            combined.update({f"validation/{label}/{key.removeprefix('validation/')}": value
                             for key, value in result.items()})
        combined.update({"optimizer/global_step": global_step,
                         "validation/full": int(full), "validation/source": source})
        validation_count += 1
        last_metrics = combined
        append_jsonl(metrics_path, {"kind": "validation", "log_sequence": log_sequence,
                                    **combined})
        log_sequence += 1
        wandb_run.log(combined, step=global_step)
        best_key = "validation/ema/t24/noise_mse"
        score = float(combined[best_key])
        if full and score < best_metric:
            best_metric = score
            digest = save(checkpoint_dir / "best.pt")
            wandb_run.summary["best_checkpoint_sha256"] = digest
            wandb_run.summary["best_ema_fixed_validation_noise_mse_t24"] = best_metric
        model.train()
        return combined

    try:
        if global_step == 0:
            run_validation(full=True)
            save(last_path)
        model.train()
        while global_step < requested_limit and elapsed_total() < max_seconds and not stop_requested:
            step_tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            parent_ids = sampler.peek(effective_batch)
            update_data_seconds = 0.0
            weighted_loss = 0.0
            valid_update = True
            for accumulation_index in range(accumulation):
                selected = parent_ids[accumulation_index * microbatch:(accumulation_index + 1) * microbatch]
                data_tick = time.perf_counter()
                batch_np = np.stack([dataset.window(parent_id).network for parent_id in selected])
                update_data_seconds += time.perf_counter() - data_tick
                x0 = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
                timestep = torch.randint(0, schedule.steps, (microbatch,), generator=sampler_generator,
                                         device=device)
                epsilon = torch.randn(x0.shape, generator=sampler_generator, device=device,
                                      dtype=torch.float32)
                x_t = schedule.add_noise(x0, timestep, epsilon)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype,
                                    enabled=autocast_enabled):
                    epsilon_hat = model.predict_noise(x_t, timestep)
                    loss = F.mse_loss(epsilon_hat.float(), epsilon) / accumulation
                if not torch.isfinite(loss):
                    valid_update = False
                    break
                loss.backward()
                weighted_loss += float(loss.detach())
            if not valid_update:
                raise FloatingPointError("non-finite training loss")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(config.raw["optimizer"]["grad_clip"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("non-finite model parameter after optimizer step")
            ema.update(model)
            sampler.commit(effective_batch)
            global_step += 1
            torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_tick
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            memory_samples.append(reserved)
            recent_finite.append(True)
            recent_finite = recent_finite[-100:]
            if global_step > 20:
                benchmark_step_times.append(step_seconds)
                benchmark_data_times.append(update_data_seconds)
                benchmark_step_times = benchmark_step_times[-20:]
                benchmark_data_times = benchmark_data_times[-20:]
            train_metrics = {
                "optimizer/global_step": global_step, "train/noise_mse": weighted_loss,
                "train/grad_norm": float(grad_norm),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/consumed_samples": sampler.cursor,
                "train/samples_per_second": effective_batch / step_seconds,
                "train/step_seconds": step_seconds, "train/data_seconds": update_data_seconds,
                "train/gpu_peak_allocated_bytes": allocated,
                "train/gpu_peak_reserved_bytes": reserved,
                "train/amp_skipped_updates": amp_skips,
                "train/cumulative_process_seconds": elapsed_total(),
            }
            if global_step % int(config.raw["tracking"]["log_every"]) == 0:
                append_jsonl(metrics_path, {"kind": "train", "log_sequence": log_sequence,
                                            **train_metrics})
                log_sequence += 1
                wandb_run.log(train_metrics, step=global_step)
                update_status(status_path, state="stable_running" if stability_achieved else "running",
                              global_step=global_step,
                              consumed_sample_cursor=sampler.cursor,
                              cumulative_process_seconds=elapsed_total(),
                              last_train=train_metrics, resume_count=resume_count)
            quick_schedule = set(int(x) for x in config.raw["validation"]["schedule"])
            full_schedule = set(int(x) for x in config.raw["validation"]["full_schedule"])
            if global_step in quick_schedule or (global_step > 200 and global_step % 250 == 0):
                run_validation(full=global_step in full_schedule)
            if global_step in {100, 200} or global_step % int(config.raw["budget"]["milestone_every"]) == 0:
                digest = save(last_path)
                if global_step % int(config.raw["budget"]["milestone_every"]) == 0:
                    save(checkpoint_dir / f"step_{global_step:06d}.pt")
                update_status(status_path, state="stable_running" if stability_achieved else "running",
                              global_step=global_step,
                              consumed_sample_cursor=sampler.cursor, checkpoint=str(last_path.resolve()),
                              checkpoint_sha256=digest, last_validation=last_metrics,
                              cumulative_process_seconds=elapsed_total())
            stable = (global_step >= 200 and len(recent_finite) == 100 and validation_count >= 3
                      and resume_count >= 1 and len(memory_samples) >= 100
                      and max(memory_samples[-20:]) <= max(memory_samples[-100:-80]) * 1.02)
            if stable and not stability_achieved:
                digest = save(last_path)
                import wandb
                artifact = wandb.Artifact(f"{source}-prior-resume", type="checkpoint",
                                          metadata={"global_step": global_step,
                                                    "sha256": digest})
                artifact.add_file(str(last_path), name="last.pt")
                wandb_run.log_artifact(artifact)
                remote = verify_server_history(wandb_run.path) if verify_wandb else {
                    "train_visible": False, "validation_visible": False,
                    "skipped": True, "run_path": wandb_run.path,
                }
                if verify_wandb and not (remote.get("train_visible") and remote.get("validation_visible")):
                    raise RuntimeError(f"W&B server history verification failed: {remote}")
                stability_achieved = True
                digest = save(last_path)
                update_status(status_path, state="stable_running", global_step=global_step,
                              consumed_sample_cursor=sampler.cursor, checkpoint=str(last_path.resolve()),
                              checkpoint_sha256=digest, wandb_server_verification=remote,
                              benchmark={"updates": len(benchmark_step_times),
                                         "mean_step_seconds": float(np.mean(benchmark_step_times)),
                                         "mean_data_seconds": float(np.mean(benchmark_data_times)),
                                         "samples_per_second": effective_batch /
                                                               float(np.mean(benchmark_step_times)),
                                         "peak_reserved_bytes": max(memory_samples)},
                              cumulative_process_seconds=elapsed_total())
        budget_end = global_step >= max_updates or elapsed_total() >= max_seconds
        stage_end = stop_after is not None and global_step >= stop_after and not budget_end
        if budget_end:
            run_validation(full=True)
        digest = save(last_path)
        if budget_end:
            import wandb
            artifact = wandb.Artifact(f"{source}-prior-final", type="checkpoint",
                                      metadata={"global_step": global_step,
                                                "sha256": digest})
            artifact.add_file(str(last_path), name="last.pt")
            wandb_run.log_artifact(artifact)
        state = "completed_budget" if budget_end else "running"
        reason = ("max_updates" if global_step >= max_updates else
                  "max_process_seconds" if elapsed_total() >= max_seconds else
                  "signal" if stop_requested else "stage_boundary")
        update_status(status_path, state=state, global_step=global_step,
                      consumed_sample_cursor=sampler.cursor, checkpoint=str(last_path.resolve()),
                      checkpoint_sha256=digest, exit_reason=reason, exit_code=0,
                      cumulative_process_seconds=elapsed_total(), last_validation=last_metrics,
                      wandb=wandb_info)
        wandb_run.summary.update({"final_global_step": global_step, "exit_reason": reason,
                                  "last_checkpoint_sha256": digest,
                                  "cumulative_process_seconds": elapsed_total()})
        wandb_run.finish(exit_code=0)
        return json.loads(status_path.read_text())
    except Exception as exc:
        try:
            digest = save(last_path) if global_step > 0 else None
        except Exception:
            digest = None
        update_status(status_path, state="failed", global_step=global_step,
                      consumed_sample_cursor=sampler.cursor, failure=f"{type(exc).__name__}: {exc}",
                      checkpoint=str(last_path.resolve()) if last_path.exists() else None,
                      checkpoint_sha256=digest, exit_code=1,
                      cumulative_process_seconds=elapsed_total(), wandb=wandb_info)
        wandb_run.finish(exit_code=1)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
