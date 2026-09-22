from __future__ import annotations

import copy
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
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import load_training_view
from ..data.materialized import MaterializedPriorDataset, RecoverableEpochBatchSampler, sha256_file
from ..models.diffwave import DiffWavePrior, parameter_count
from ..priors.vp import VPSchedule
from .checkpoint import atomic_torch_save, git_state, restore_rng_state, rng_state
from .status import append_jsonl, update_status
from .tracking import verify_server_history
from .trainer import ExponentialMovingAverage


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load_formal_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if raw.get("version") != f"lte-{raw.get('source')}-prior-formal-v1":
        raise ValueError("formal config version/source mismatch")
    source = raw["source"]
    if source not in {"target", "interference"}:
        raise ValueError("invalid source")
    view = load_training_view((path.parent / raw["view_config"]).resolve())
    expected_channels = 64 if source == "target" else 128
    expected_model = {"residual_layers": 24, "residual_channels": expected_channels,
                      "dilation_cycle": 12, "kernel_size": 3, "timestep_dim": 128,
                      "timestep_hidden": 512, "in_channels": 2, "out_channels": 2}
    if raw["model"] != expected_model:
        raise ValueError("formal model semantics mismatch")
    if raw["diffusion"] != {"steps": 50, "beta_start": 0.0001, "beta_end": 0.05,
                            "prediction": "epsilon"}:
        raise ValueError("formal diffusion semantics mismatch")
    runtime = raw["runtime"]
    effective, micro = int(runtime["effective_batch_size"]), int(runtime["microbatch_size"])
    if effective != 16 or micro not in {2, 4, 8, 16} or effective % micro:
        raise ValueError("formal batch contract mismatch")
    if runtime["precision"] == "fp16":
        raise ValueError("FP16 is unsupported without a persisted GradScaler; use BF16")
    if runtime["precision"] != "bf16" or runtime["activation_checkpointing"]:
        raise ValueError("formal v1 requires BF16 without activation checkpointing")
    if int(runtime["num_workers"]) not in {2, 4, 8}:
        raise ValueError("formal num_workers must be a measured candidate")
    budget = int(raw["budget"]["max_cumulative_updates"])
    if budget != (200000 if source == "target" else 300000):
        raise ValueError("formal update budget mismatch")
    resolved = copy.deepcopy(raw)
    resolved["_config_path"] = str(path)
    resolved["_config_hash"] = _hash(raw)
    resolved["_view"] = view
    return resolved


def _model(raw: dict) -> DiffWavePrior:
    cfg = raw["model"]
    return DiffWavePrior(cfg["residual_channels"], cfg["residual_layers"],
                         cfg["dilation_cycle"], False)


def _audit_pilot(payload: dict, raw: dict, view) -> None:
    source = raw["source"]
    if payload.get("source") != source or int(payload.get("global_step", -1)) != 2000:
        raise ValueError("pilot source/global_step mismatch")
    if int(payload.get("consumed_sample_cursor", -1)) != 32000:
        raise ValueError("pilot cumulative exposure/cursor mismatch")
    old = payload.get("training_config", {})
    for key in ("model", "diffusion"):
        if old.get(key) != raw.get(key):
            raise ValueError(f"pilot to formal {key} semantics mismatch")
    optimizer_keys = ("name", "learning_rate", "betas", "weight_decay", "grad_clip", "ema_decay")
    if any(old.get("optimizer", {}).get(key) != raw.get("optimizer", {}).get(key)
           for key in optimizer_keys):
        raise ValueError("pilot to formal optimizer semantics mismatch")
    if payload.get("platform_hash") != view.platform.config_hash:
        raise ValueError("pilot platform fingerprint mismatch")
    old_view = payload.get("training_view_config", {})
    semantic = ("parent_window_length", "window_length", "slot_offsets", "selection",
                "network_layout", "coordinate_scale", "dtype")
    if any(old_view.get(key) != view.raw.get(key) for key in semantic):
        raise ValueError("pilot training-view semantics mismatch")
    if not all(key in payload for key in ("model", "ema", "optimizer", "rng")):
        raise ValueError("pilot checkpoint lacks continuation state")


@torch.no_grad()
def _validate(raw: dict, models: tuple[tuple[str, torch.nn.Module], ...],
              dataset: MaterializedPriorDataset, schedule: VPSchedule,
              device: torch.device, *, full: bool) -> dict:
    levels = list(raw["validation"]["noise_levels"])
    parents = int(raw["validation"]["full_parents"] if full else raw["validation"]["quick_parents"])
    rows = min(len(dataset), parents * 2)
    batch_size = int(raw["validation"]["batch_size"])
    results: dict[str, float | int] = {"validation/windows": rows,
                                      "validation/full": int(full)}
    sums = {(name, level, metric): torch.zeros((), device=device)
            for name, _ in models for level in levels for metric in ("noise_mse", "real_mse", "power")}
    for name, model in models:
        model.eval()
    for begin in range(0, rows, batch_size):
        x0 = torch.stack([dataset[index] for index in range(begin, min(rows, begin + batch_size))])
        x0 = x0.pin_memory().to(device, non_blocking=True)
        for level in levels:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(raw["validation"]["noise_seed"]) + begin * 53 + level)
            epsilon = torch.randn(x0.shape, generator=generator, device=device, dtype=torch.float32)
            timestep = torch.full((x0.shape[0],), level, device=device, dtype=torch.long)
            x_t = schedule.add_noise(x0, timestep, epsilon, validate=False)
            for name, model in models:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model(x_t, timestep).float()
                estimate = schedule.denoise(x_t, timestep, predicted)
                sums[(name, level, "noise_mse")] += torch.sum((predicted - epsilon) ** 2)
                sums[(name, level, "real_mse")] += torch.sum((estimate - x0) ** 2)
                sums[(name, level, "power")] += torch.sum(x0 ** 2)
    packed = torch.stack(tuple(sums.values()))
    if not bool(torch.isfinite(packed).all()):
        raise FloatingPointError("formal validation contains non-finite values")
    for name, _ in models:
        for level in levels:
            count = rows * 2 * 15360
            results[f"validation/{name}/t{level}/noise_mse"] = float(sums[(name, level, "noise_mse")] / count)
            results[f"validation/{name}/t{level}/real_mse"] = float(sums[(name, level, "real_mse")] / count)
            results[f"validation/{name}/t{level}/nmse"] = float(
                sums[(name, level, "real_mse")] / sums[(name, level, "power")])
        results[f"validation/{name}/multi_noise_mean_noise_mse"] = float(np.mean([
            results[f"validation/{name}/t{level}/noise_mse"] for level in levels]))
    return results


def train_formal(raw: dict, run_dir: Path, *, device_name: str = "cuda:0",
                 stop_after: int | None = None) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") or torch.cuda.device_count() != 1:
        raise RuntimeError("formal worker must see exactly one physical GPU")
    device = torch.device(device_name)
    if not torch.cuda.is_available() or device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal training requires a BF16 CUDA GPU")
    source, view = raw["source"], raw.pop("_view")
    config_path, config_hash = Path(raw.pop("_config_path")), raw.pop("_config_hash")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.exists():
        resolved_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    elif yaml.safe_load(resolved_path.read_text()) != raw:
        raise ValueError("immutable resolved formal config changed")
    train_manifest = (config_path.parent / raw["data"]["train_manifest"]).resolve()
    validation_manifest = (config_path.parent / raw["data"]["validation_manifest"]).resolve()
    train_data = MaterializedPriorDataset(train_manifest)
    validation_data = MaterializedPriorDataset(validation_manifest)
    if train_data.manifest["source"] != source or train_data.manifest["count"] != 200000:
        raise ValueError("formal train manifest source/count mismatch")
    if validation_data.manifest["source"] != source or validation_data.manifest["count"] != 2048:
        raise ValueError("formal validation manifest source/count mismatch")
    model = _model(raw).to(device)
    ema = ExponentialMovingAverage(model, float(raw["optimizer"]["ema_decay"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(raw["optimizer"]["learning_rate"]),
                                 betas=tuple(raw["optimizer"]["betas"]),
                                 weight_decay=float(raw["optimizer"]["weight_decay"]),
                                 foreach=True)
    schedule = VPSchedule(50, 0.0001, 0.05).to(device)
    noise_generator = torch.Generator(device=device)
    sampler = RecoverableEpochBatchSampler(len(train_data), 16,
                                            int(raw["runtime"]["sampler_seed"]))
    pilot_path = (config_path.parent / raw["continue_from"]["checkpoint"]).resolve()
    parent_sha = sha256_file(pilot_path)
    if parent_sha != raw["continue_from"]["sha256"]:
        raise ValueError("parent pilot checkpoint SHA256 mismatch")
    formal_last = checkpoint_dir / "last.pt"
    resumed = formal_last.exists()
    if resumed:
        payload = torch.load(formal_last, map_location="cpu", weights_only=False)
        if payload.get("formal_config_hash") != config_hash or payload.get("source") != source:
            raise ValueError("strict formal resume config/source mismatch")
    else:
        payload = torch.load(pilot_path, map_location="cpu", weights_only=False)
        _audit_pilot(payload, raw, view)
    model.load_state_dict(payload["model"])
    ema.model.load_state_dict(payload["ema"])
    optimizer.load_state_dict(payload["optimizer"])
    global_step = int(payload["global_step"])
    historical_exposures = int(payload.get("historical_exposures", payload.get("consumed_sample_cursor", 0)))
    if resumed:
        sampler.load_state_dict(payload["dataset_sampler"])
    restore_rng_state(payload["rng"], noise_generator)
    start_step = global_step
    formal_run_id = (payload.get("wandb", {}).get("run_id") if resumed
                     else raw["tracking"].get("run_id"))
    import wandb
    tracking = raw["tracking"]
    wandb_run = wandb.init(project=tracking["project"], group=tracking["group"],
                           name=tracking["name"], id=formal_run_id,
                           resume="must" if resumed else ("allow" if formal_run_id else "never"),
                           mode="online",
                           config={"phase": "formal", "source": source,
                                   "formal_config_hash": config_hash,
                                   "model_distribution_fingerprint": raw["fingerprints"]["model_distribution"],
                                   "execution_config_hash": config_hash,
                                   "evaluation_protocol_hash": raw["fingerprints"]["evaluation_protocol"],
                                   "parent_checkpoint_sha256": parent_sha,
                                   "parent_wandb_run": raw["continue_from"]["wandb_run"],
                                   "gpu_uuid": torch.cuda.get_device_properties(0).uuid,
                                   "dataset_id": train_data.manifest["dataset_id"]},
                           tags=[source, "stage-formal", "materialized-data-v1"])
    wandb_run.define_metric("optimizer/global_step")
    wandb_run.define_metric("train/*", step_metric="optimizer/global_step")
    wandb_run.define_metric("validation/*", step_metric="optimizer/global_step")
    status_path, metrics_path = run_dir / "status.json", run_dir / "metrics.jsonl"
    log_sequence = int(payload.get("log_sequence", 0))
    validation_count = int(payload.get("validation_count", 0)) if resumed else 0
    best_metric = float(payload.get("best_denoising_metric", math.inf)) if resumed else math.inf
    last_validation = dict(payload.get("last_validation", {})) if resumed else {}
    started, stop_requested = time.monotonic(), False
    initial_step = 2000
    stable_verified = bool(payload.get("stability_achieved", False)) if resumed else False

    def on_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}

    def learning_rate(step: int) -> float:
        maximum, minimum = float(raw["optimizer"]["learning_rate"]), float(raw["optimizer"]["min_learning_rate"])
        maximum_step = int(raw["budget"]["max_cumulative_updates"])
        decay_start = int(maximum_step * 0.8)
        if step <= decay_start:
            return maximum
        ratio = (step - decay_start) / (maximum_step - decay_start)
        return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * ratio))

    def state_payload() -> dict:
        return {"schema_version": 2, "phase": "formal", "source": source,
                "model": model.state_dict(), "ema": ema.model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": "cumulative-80pct-cosine-v1",
                "scaler": None, "global_step": global_step,
                "formal_updates": global_step - initial_step,
                "historical_exposures": historical_exposures,
                "formal_exposures": sampler.exposures,
                "cumulative_exposures": historical_exposures + sampler.exposures,
                "dataset_sampler": sampler.state_dict(), "rng": rng_state(noise_generator),
                "platform_config": view.platform.raw, "platform_hash": view.platform.config_hash,
                "training_view_config": view.raw, "training_view_hash": view.config_hash,
                "training_config": raw, "training_config_hash": config_hash,
                "formal_config_hash": config_hash, "dataset_id": train_data.manifest["dataset_id"],
                "dataset_manifest_sha256": sha256_file(train_manifest),
                "parent_checkpoint": str(pilot_path), "parent_checkpoint_sha256": parent_sha,
                "parent_wandb_run": raw["continue_from"]["wandb_run"],
                "best_denoising_metric": best_metric, "last_validation": last_validation,
                "validation_count": validation_count, "log_sequence": log_sequence,
                "stability_achieved": stable_verified,
                "wandb": {"run_id": wandb_run.id, "run_path": wandb_run.path,
                          "url": wandb_run.url},
                "environment": {"python": sys.version, "torch": torch.__version__,
                                "numpy": np.__version__, "host": platform.node(),
                                "cuda": torch.version.cuda,
                                "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)},
                "git": git_state(Path(__file__).resolve().parents[3])}

    def save(path: Path) -> str:
        finite = torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all()
        if not bool(finite):
            raise FloatingPointError("refusing to overwrite checkpoint with non-finite parameters")
        return atomic_torch_save(state_payload(), path)

    def validate(full: bool) -> None:
        nonlocal validation_count, last_validation, best_metric, log_sequence
        last_validation = _validate(raw, (("raw", model), ("ema", ema.model)), validation_data,
                                    schedule, device, full=full)
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

    update_status(status_path, state="validation", source=source, pid=os.getpid(),
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), global_step=global_step,
                  formal_exposures=sampler.exposures, historical_exposures=historical_exposures,
                  wandb={"run_id": wandb_run.id, "run_path": wandb_run.path, "url": wandb_run.url})
    if not resumed:
        validate(full=False)
        save(formal_last)
    loader_kwargs = dict(num_workers=int(raw["runtime"]["num_workers"]), pin_memory=True,
                         persistent_workers=True, prefetch_factor=int(raw["runtime"]["prefetch_factor"]))
    loader = DataLoader(train_data, batch_sampler=sampler, **loader_kwargs)
    iterator = iter(loader)
    micro = int(raw["runtime"]["microbatch_size"])
    accumulation = 16 // micro
    benchmark_times = []
    max_updates = int(raw["budget"]["max_cumulative_updates"])
    limit = min(max_updates, stop_after) if stop_after is not None else max_updates
    try:
        model.train()
        while global_step < limit and not stop_requested:
            tick = time.perf_counter()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                loader = DataLoader(train_data, batch_sampler=sampler, **loader_kwargs)
                iterator = iter(loader)
                cpu_batch = next(iterator)
            data_ready = time.perf_counter()
            batch = cpu_batch.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss_accumulator = torch.zeros((), device=device)
            finite = torch.ones((), dtype=torch.bool, device=device)
            for begin in range(0, 16, micro):
                x0 = batch[begin:begin + micro]
                timestep = torch.randint(0, 50, (micro,), generator=noise_generator, device=device)
                epsilon = torch.randn(x0.shape, generator=noise_generator, device=device)
                x_t = schedule.add_noise(x0, timestep, epsilon, validate=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model(x_t, timestep)
                    loss = F.mse_loss(predicted.float(), epsilon) / accumulation
                finite.logical_and_(torch.isfinite(loss))
                loss.backward()
                loss_accumulator += loss.detach()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(raw["optimizer"]["grad_clip"]),
                                                       foreach=True)
            finite.logical_and_(torch.isfinite(grad_norm))
            if not bool(finite):
                raise FloatingPointError("non-finite loss or gradient before optimizer commit")
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
                metrics = {"optimizer/global_step": global_step,
                           "tracking/event_sequence": log_sequence,
                           "train/noise_mse": float(loss_accumulator),
                           "train/grad_norm": float(grad_norm), "train/learning_rate": lr,
                           "train/formal_exposures": sampler.exposures,
                           "train/cumulative_exposures": historical_exposures + sampler.exposures,
                           "train/epoch": sampler.epoch, "train/epoch_offset": sampler.offset,
                           "train/step_seconds": step_seconds,
                           "train/data_wait_seconds": data_ready - tick,
                           "train/samples_per_second": 16 / step_seconds,
                           "train/gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                append_jsonl(metrics_path, {"kind": "train", **metrics})
                wandb_run.log(metrics)
                log_sequence += 1
                update_status(status_path, state="stable_running" if stable_verified else "training",
                              global_step=global_step,
                              formal_exposures=sampler.exposures, epoch=sampler.epoch,
                              epoch_offset=sampler.offset, last_train=metrics)
            if global_step % int(raw["validation"]["quick_every"]) == 0:
                update_status(status_path, state="validation", global_step=global_step)
                validate(full=global_step % int(raw["validation"]["full_every"]) == 0)
            if global_step % int(raw["checkpoint"]["last_every"]) == 0:
                digest = save(formal_last)
                numbered = checkpoint_dir / f"step_{global_step:06d}.pt"
                save(numbered)
                numbered_paths = sorted(checkpoint_dir.glob("step_*.pt"))
                for stale in numbered_paths[:-int(raw["checkpoint"]["keep_last"])]:
                    stale.unlink()
                update_status(status_path, state="stable_running" if stable_verified else "training",
                              checkpoint=str(formal_last),
                              checkpoint_sha256=digest, global_step=global_step)
            if not stable_verified and global_step >= start_step + 200:
                update_status(status_path, state="validation", global_step=global_step)
                validate(full=False)
                digest = save(formal_last)
                wandb_run.summary.update({"stable_handoff_step": global_step,
                                          "stable_checkpoint_sha256": digest})
                wandb_run.log({"optimizer/global_step": global_step,
                               "tracking/event_sequence": log_sequence,
                               "training/stable_handoff": 1})
                remote = verify_server_history(wandb_run.path, timeout_seconds=45.0)
                train_visible = bool(remote.get("train_visible"))
                validation_visible = bool(remote.get("validation_visible"))
                if not (train_visible and validation_visible):
                    raise RuntimeError("formal W&B remote history is not visible")
                stable_verified = True
                digest = save(formal_last)
                update_status(status_path, state="stable_running", global_step=global_step,
                              formal_exposures=sampler.exposures, checkpoint=str(formal_last),
                              checkpoint_sha256=digest,
                              benchmark={"updates": len(benchmark_times),
                                         "mean_step_seconds": float(np.mean(benchmark_times)),
                                         "samples_per_second": 16 / float(np.mean(benchmark_times)),
                                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
                              wandb_server_verification={"train_visible": train_visible,
                                                         "validation_visible": validation_visible,
                                                         "history_rows": remote.get("history_rows", 0)})
        if global_step >= max_updates:
            validate(full=True)
        digest = save(formal_last)
        state = "completed" if global_step >= max_updates else "paused"
        update_status(status_path, state=state, global_step=global_step,
                      formal_exposures=sampler.exposures, checkpoint=str(formal_last),
                      checkpoint_sha256=digest,
                      exit_reason="budget" if state == "completed" else "signal_or_stage")
        wandb_run.finish(exit_code=0)
        return json.loads(status_path.read_text())
    except Exception as exc:
        # Never save here: optimizer may already have corrupted state. Last remains known-good.
        update_status(status_path, state="failed", global_step=global_step,
                      failure=f"{type(exc).__name__}: {exc}", checkpoint=str(formal_last),
                      checkpoint_sha256=sha256_file(formal_last) if formal_last.exists() else None)
        wandb_run.finish(exit_code=1)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
