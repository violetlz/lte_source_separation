from __future__ import annotations

import hashlib

import numpy as np


SPLITS = ("train", "validation", "test", "calibration", "audit")


def derive_seed(root_seed: int, purpose: str, split: str, record_id: int) -> int:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    payload = f"rfsep-v1|{root_seed}|{purpose}|{split}|{record_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def rng_for(root_seed: int, purpose: str, split: str, record_id: int) -> np.random.Generator:
    return np.random.default_rng(derive_seed(root_seed, purpose, split, record_id))


def record_identity(source: str, split: str, record_id: int) -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return f"{source}:{split}:{record_id:08d}"
