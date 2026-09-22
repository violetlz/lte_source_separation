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

from rfsep.training.formal import load_formal_config, train_formal  # noqa: E402
from rfsep.training.status import write_json_atomic  # noqa: E402


def _gpu_uuid(index: int) -> str:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], text=True,
                            capture_output=True, check=True)
    rows = [line.split(", ") for line in result.stdout.strip().splitlines()]
    matches = [row for row in rows if int(row[0]) == index]
    if len(matches) != 1:
        raise ValueError(f"physical GPU {index} not found")
    _, uuid, used, utilization = matches[0]
    if int(used) > 1024 or int(utilization) > 10:
        raise RuntimeError(f"GPU {index} ({uuid}) is not free: used={used} MiB util={utilization}%")
    return uuid


def worker(args) -> int:
    raw = load_formal_config(args.config)
    result = train_formal(raw, Path(args.run_dir).resolve(), device_name="cuda:0",
                          stop_after=args.stop_after)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def orchestrate(args) -> int:
    if args.target_gpu == args.interference_gpu:
        raise ValueError("target and interference cannot share a physical GPU index")
    uuids = {"target": _gpu_uuid(args.target_gpu),
             "interference": _gpu_uuid(args.interference_gpu)}
    if len(set(uuids.values())) != 2:
        raise ValueError("target and interference resolved to the same physical GPU UUID")
    configs = {"target": Path(args.target_config).resolve(),
               "interference": Path(args.interference_config).resolve()}
    for source, path in configs.items():
        raw = load_formal_config(path)
        if raw["source"] != source:
            raise ValueError(f"{source} config has wrong source")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    processes, commands, logs = {}, {}, {}
    for source, gpu in (("target", args.target_gpu), ("interference", args.interference_gpu)):
        run_dir = output / source
        run_dir.mkdir(exist_ok=True)
        status = run_dir / "status.json"
        if status.exists():
            old = json.loads(status.read_text())
            pid = int(old.get("pid", -1))
            if pid > 1:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f"duplicate live {source} worker pid={pid}")
        command = [sys.executable, str(Path(__file__).resolve()), "worker",
                   "--config", str(configs[source]), "--run-dir", str(run_dir)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        log_path = run_dir / "formal.log"
        handle = log_path.open("a")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes[source], commands[source], logs[source] = process, command, str(log_path)
        handle.close()
    manifest = {"state": "training", "group": "prior-formal-20260923",
                "launched_utc": datetime.now(timezone.utc).isoformat(),
                "orchestrator_pid": os.getpid(),
                "workers": {source: {"pid": process.pid, "physical_gpu": gpu,
                                      "gpu_uuid": uuids[source], "command": commands[source],
                                      "log": logs[source]}
                            for source, process, gpu in (("target", processes["target"], args.target_gpu),
                                                         ("interference", processes["interference"], args.interference_gpu))}}
    write_json_atomic(output / "orchestrator_status.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    work = sub.add_parser("worker")
    work.add_argument("--config", required=True)
    work.add_argument("--run-dir", required=True)
    work.add_argument("--stop-after", type=int)
    launch = sub.add_parser("orchestrate")
    launch.add_argument("--target-config", default=str(ROOT / "configs/training/target_prior_formal.yaml"))
    launch.add_argument("--interference-config", default=str(ROOT / "configs/training/interference_prior_formal.yaml"))
    launch.add_argument("--target-gpu", type=int, default=0)
    launch.add_argument("--interference-gpu", type=int, default=1)
    launch.add_argument("--output", default=str(ROOT / "artifacts/prior_formal/prior-formal-20260923"))
    args = parser.parse_args()
    return worker(args) if args.command == "worker" else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
