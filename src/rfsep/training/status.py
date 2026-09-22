from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


VALID_STATES = {
    "not_started", "running", "training", "validation", "rgs_eval",
    "stable_running", "paused", "completed", "completed_budget", "failed", "blocked",
}


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def update_status(path: Path, *, state: str, **fields) -> None:
    if state not in VALID_STATES:
        raise ValueError(f"invalid state {state}")
    current = json.loads(path.read_text()) if path.is_file() else {}
    current.update(fields)
    current["state"] = state
    write_json_atomic(path, current)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
