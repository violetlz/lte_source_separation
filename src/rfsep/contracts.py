from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WaveformSpec:
    """Public waveform geometry; deliberately contains no generation seed."""

    platform_version: str
    platform_hash: str
    sample_rate_hz: int
    subcarrier_spacing_hz: int
    fft_size: int
    num_symbols: int
    cp_lengths: tuple[int, ...]
    window_length: int
    target_subcarriers: tuple[int, ...]


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
    if x.ndim not in (1, 2) or not np.iscomplexobj(x):
        raise ValueError("expected complex [L] or [B,L]")
    axis = 0 if x.ndim == 1 else 1
    return (np.sqrt(np.float32(2.0)) * np.stack((x.real, x.imag), axis=axis)).astype(
        np.float32, copy=False
    )


def network_to_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim not in (2, 3):
        raise ValueError("expected real [2,L] or [B,2,L]")
    axis = 0 if x.ndim == 2 else 1
    if x.shape[axis] != 2 or np.iscomplexobj(x):
        raise ValueError("network coordinates require a two-channel real axis")
    real = x[0] if x.ndim == 2 else x[:, 0]
    imag = x[1] if x.ndim == 2 else x[:, 1]
    return ((real + 1j * imag) / np.sqrt(np.float32(2.0))).astype(np.complex64, copy=False)
