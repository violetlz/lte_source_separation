#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfsep.config import load_training  # noqa: E402
from rfsep.models.diffwave import DiffWavePrior, parameter_count  # noqa: E402
from rfsep.training.status import update_status, write_json_atomic  # noqa: E402
from rfsep.training.trainer import train  # noqa: E402


def worker(args) -> int:
    config = load_training(args.config)
    source_dir = Path(args.output).resolve() / config.raw["source"]
    try:
        result = train(config, source_dir, device_name=args.device,
                       stop_after=args.stop_after, verify_wandb=not args.skip_wandb_verify)
    except Exception as exc:
        status_path = source_dir / "status.json"
        current = json.loads(status_path.read_text()) if status_path.is_file() else {}
        update_status(status_path, state="failed", **{
            key: value for key, value in current.items()
            if key not in {"state", "updated_utc", "failure", "exit_code"}
        }, failure=f"{type(exc).__name__}: {exc}", exit_code=1)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def inspect(args) -> int:
    config = load_training(args.config)
    model = DiffWavePrior(config.raw["model"]["residual_channels"])
    payload = {
        "source": config.raw["source"], "platform_hash": config.view.platform.config_hash,
        "training_view_hash": config.view.config_hash, "training_config_hash": config.config_hash,
        "parameters": parameter_count(model),
        "theoretical_receptive_field": model.theoretical_receptive_field,
        "theoretical_radius": (model.theoretical_receptive_field - 1) // 2,
        "window_length": config.view.raw["window_length"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _launch(config: Path, output: Path, gpu: int, stop_after: int | None, log_path: Path):
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(config),
               "--output", str(output), "--device", "cuda:0"]
    if stop_after is not None:
        command.extend(["--stop-after", str(stop_after)])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    handle = log_path.open("a")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return process, handle, command


def orchestrate(args) -> int:
    target = Path(args.target_config).resolve()
    interference = Path(args.interference_config).resolve()
    target_config = load_training(target)
    interference_config = load_training(interference)
    if target_config.raw["source"] != "target" or interference_config.raw["source"] != "interference":
        raise ValueError("orchestrator configs have wrong sources")
    group = args.group or target_config.raw["tracking"]["group"]
    output = ROOT / "artifacts" / "prior_pilot" / group
    output.mkdir(parents=True, exist_ok=True)
    group_status = output / "status.json"
    write_json_atomic(group_status, {
        "state": "running", "group": group, "orchestrator_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output.resolve()), "stage": "first_200",
        "gpu_assignment": {"target": args.target_gpu, "interference": args.interference_gpu},
    })
    configs = {"target": target, "interference": interference}
    gpus = {"target": args.target_gpu, "interference": args.interference_gpu}
    try:
        stages = (("resume_to_budget", None),) if args.resume_only else (
            ("first_200", 200), ("resume_to_budget", None))
        for stage, stop_after in stages:
            processes = {}
            handles = {}
            commands = {}
            for source in ("target", "interference"):
                log_path = output / source / f"{stage}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                process, handle, command = _launch(configs[source], output, gpus[source],
                                                    stop_after, log_path)
                processes[source], handles[source], commands[source] = process, handle, command
            write_json_atomic(group_status, {
                **json.loads(group_status.read_text()), "stage": stage,
                "worker_pids": {key: value.pid for key, value in processes.items()},
                "worker_commands": commands,
            })
            return_codes = {source: process.wait() for source, process in processes.items()}
            for handle in handles.values():
                handle.close()
            if any(code != 0 for code in return_codes.values()):
                update_status(group_status, state="failed", stage=stage,
                              worker_exit_codes=return_codes,
                              failure="one or more bounded worker attempts failed")
                return 1
        sources = {source: json.loads((output / source / "status.json").read_text())
                   for source in ("target", "interference")}
        completed = all(payload.get("state") == "completed_budget" for payload in sources.values())
        result = {
            "state": "completed_budget" if completed else "failed", "group": group,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
        }
        write_json_atomic(output / "pilot_result.json", result)
        update_status(group_status, state=result["state"], stage="done", worker_exit_codes={
            "target": 0, "interference": 0},
                      failure=None if completed else "workers exited before budget completion")
        return 0 if completed else 1
    except Exception as exc:
        update_status(group_status, state="failed", failure=f"{type(exc).__name__}: {exc}")
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--config", required=True)
    worker_parser.add_argument("--output", required=True)
    worker_parser.add_argument("--device", default="cuda:0")
    worker_parser.add_argument("--stop-after", type=int)
    worker_parser.add_argument("--skip-wandb-verify", action="store_true")
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--config", required=True)
    orchestrator = sub.add_parser("orchestrate")
    orchestrator.add_argument("--target-config", default=str(ROOT / "configs/training/target_prior_pilot.yaml"))
    orchestrator.add_argument("--interference-config", default=str(ROOT / "configs/training/interference_prior_pilot.yaml"))
    orchestrator.add_argument("--target-gpu", type=int, default=0)
    orchestrator.add_argument("--interference-gpu", type=int, default=1)
    orchestrator.add_argument("--group")
    orchestrator.add_argument("--resume-only", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    raise SystemExit({"worker": worker, "inspect": inspect, "orchestrate": orchestrate}[args.command](args))
