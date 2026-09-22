from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    long_record_id: str
    split: str
    waveform: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    example_id: str
    y: np.ndarray


@dataclass(frozen=True)
class ReceiverInfo:
    platform_version: str
    platform_hash: str
    calibration_id: str
    kappa: float
    sir_re_db: float
    noise: str = "none"
    noise_variance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruth:
    example_id: str
    target_waveform: np.ndarray
    interference_waveform: np.ndarray
    target_bits: np.ndarray
    target_grid: np.ndarray
    interference_modulation_map: tuple[tuple[str, ...], ...]
    tau: int
    phi: float
    source_record_ids: tuple[str, str]
    generation_audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReceiverOutput:
    example_id: str
    target_grid: np.ndarray
    target_bits: np.ndarray
    target_waveform_estimate: np.ndarray | None = None


def assert_receiver_info_public(info: ReceiverInfo) -> None:
    forbidden = {"bits", "truth", "target_waveform", "interference_waveform", "tau", "phi", "modulation"}
    leaked = forbidden.intersection(info.to_dict())
    if leaked:
        raise ValueError(f"ReceiverInfo contains forbidden truth fields: {sorted(leaked)}")


def complex_to_network(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.sqrt(2.0) * np.stack((x.real, x.imag), axis=-1)


def network_to_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[-1] != 2:
        raise ValueError("network coordinates require final dimension 2")
    return (x[..., 0] + 1j * x[..., 1]) / np.sqrt(2.0)
