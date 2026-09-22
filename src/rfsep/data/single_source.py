from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TrainingViewConfig
from ..contracts import complex_to_network
from ..randomness import derive_seed, record_identity
from ..waveforms.sources import generate_interference, generate_target


SOURCES = ("target", "interference")


@dataclass(frozen=True)
class SourceWindow:
    source: str
    split: str
    parent_id: int
    parent_record_id: str
    slot_index: int
    offset: int
    waveform: np.ndarray
    network: np.ndarray
    audit: dict


def selected_slot(root_seed: int, source: str, split: str, parent_id: int) -> int:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    rng = np.random.default_rng(derive_seed(root_seed, f"{source}-training-slot", split, parent_id))
    return int(rng.integers(0, 2))


class SingleSourceDataset:
    """Online deterministic access to clean, independent 0.5 ms source windows."""

    def __init__(self, view: TrainingViewConfig, source: str, split: str):
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        if split not in {"train", "validation"}:
            raise ValueError("prior datasets expose only train or validation")
        self.view = view
        self.platform = view.platform
        self.source = source
        self.split = split
        self.window_length = int(view.raw["window_length"])

    def parent(self, parent_id: int) -> tuple[np.ndarray, dict]:
        if parent_id < 0:
            raise ValueError("parent_id must be non-negative")
        if self.source == "target":
            record, _, _ = generate_target(self.platform, self.split, parent_id, dtype=np.complex64)
        else:
            record, _, _ = generate_interference(self.platform, self.split, parent_id,
                                                 dtype=np.complex64)
        if record.waveform.shape != (self.platform.window_length,):
            raise RuntimeError("generator did not return one 1 ms parent record")
        return record.waveform, dict(record.audit)

    def window(self, parent_id: int, slot_index: int | None = None) -> SourceWindow:
        waveform, audit = self.parent(parent_id)
        slot_index = (selected_slot(self.platform.root_seed, self.source, self.split, parent_id)
                      if slot_index is None else int(slot_index))
        if slot_index not in (0, 1):
            raise ValueError("slot_index must be 0 or 1")
        offset = slot_index * self.window_length
        crop = waveform[offset:offset + self.window_length]
        if crop.shape != (self.window_length,):
            raise RuntimeError("invalid training crop")
        identity = record_identity(self.source, self.split, parent_id)
        audit.update({"training_view_hash": self.view.config_hash, "slot_index": slot_index,
                      "slot_offset": offset})
        return SourceWindow(self.source, self.split, parent_id, identity, slot_index, offset,
                            crop, complex_to_network(crop), audit)

    def validation_windows(self):
        count = int(self.view.raw["validation_parents"])
        both = bool(self.view.raw["validation_both_slots"])
        for parent_id in range(count):
            slots = (0, 1) if both else (selected_slot(self.platform.root_seed, self.source,
                                                       self.split, parent_id),)
            for slot in slots:
                yield self.window(parent_id, slot)


@dataclass
class CursorSampler:
    """Tracks only windows committed to successful optimizer updates."""

    source: str
    cursor: int = 0

    def peek(self, count: int) -> tuple[int, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        return tuple(range(self.cursor, self.cursor + count))

    def commit(self, count: int) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self.cursor += count

    def state_dict(self) -> dict:
        return {"source": self.source, "cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        if state != {"source": self.source, "cursor": int(state.get("cursor", -1))}:
            raise ValueError("incompatible sampler state")
        self.cursor = int(state["cursor"])
