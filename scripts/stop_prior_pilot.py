#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path


def command_line(pid: int) -> str:
    return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop only the exact bounded prior-pilot processes")
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.status.read_text())
    output = str(Path(payload["output"]).resolve())
    targets = []
    orchestrator = int(payload["orchestrator_pid"])
    orchestrator_command = command_line(orchestrator)
    if "run_prior_pilot.py orchestrate" not in orchestrator_command:
        raise RuntimeError(f"PID {orchestrator} is not the recorded orchestrator")
    targets.append((orchestrator, orchestrator_command))
    for source, value in payload["worker_pids"].items():
        pid = int(value)
        command = command_line(pid)
        if ("run_prior_pilot.py worker" not in command or output not in command
                or f"{source}_prior_pilot.yaml" not in command):
            raise RuntimeError(f"PID {pid} does not match recorded {source} worker")
        targets.append((pid, command))
    # Stop the coordinator first so graceful worker exits are not misinterpreted as budget completion.
    for pid, _ in targets:
        os.kill(pid, signal.SIGTERM)
    for pid, command in targets:
        print(f"sent SIGTERM to {pid}: {command}")


if __name__ == "__main__":
    main()
