from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


SCHEMA_VERSION = 1


def git_state(root: Path) -> dict:
    def run(*args):
        result = subprocess.run(args, cwd=root, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "status_porcelain": run("git", "status", "--short")}


def rng_state(sampler_generator: torch.Generator) -> dict:
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(), "sampler_generator": sampler_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, sampler_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    sampler_generator.set_state(state["sampler_generator"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def atomic_torch_save(payload: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_checkpoint(path: Path, *, training_hash: str, source: str,
                    map_location="cpu") -> dict:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError(f"checkpoint is unreadable: {path}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if payload.get("training_config_hash") != training_hash:
        raise ValueError("checkpoint training config hash mismatch")
    if payload.get("source") != source:
        raise ValueError("checkpoint source mismatch")
    return payload
