from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PLATFORM_FIELDS = {
    "version", "root_seed", "sample_rate_hz", "subcarrier_spacing_hz", "fft_size",
    "num_symbols", "cp_lengths", "window_length", "windowing", "noise",
    "noise_variance", "channel", "target", "interference",
}
TARGET_FIELDS = {
    "nominal_bandwidth_mhz", "num_rbs", "active_subcarriers", "modulation",
    "signed_subcarriers",
}
INTERFERENCE_FIELDS = {
    "nominal_bandwidth_mhz", "num_rbs", "active_subcarriers",
    "block_size_subcarriers", "modulations", "signed_subcarrier_min",
    "signed_subcarrier_max",
}
EXPERIMENT_FIELDS = {
    "platform_config", "name", "split", "num_examples", "sir_re_db",
    "calibration_path", "precision",
}


def _exact_fields(data: dict[str, Any], expected: set[str], context: str) -> None:
    unknown = sorted(set(data) - expected)
    missing = sorted(expected - set(data))
    if unknown or missing:
        raise ValueError(f"{context} fields invalid: unknown={unknown}, missing={missing}")


@dataclass(frozen=True)
class PlatformConfig:
    path: Path
    raw: dict[str, Any]
    config_hash: str

    @property
    def fft_size(self) -> int:
        return int(self.raw["fft_size"])

    @property
    def cp_lengths(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.raw["cp_lengths"])

    @property
    def window_length(self) -> int:
        return int(self.raw["window_length"])

    @property
    def num_symbols(self) -> int:
        return int(self.raw["num_symbols"])

    @property
    def root_seed(self) -> int:
        return int(self.raw["root_seed"])

    @property
    def target_subcarriers(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.raw["target"]["signed_subcarriers"])

    @property
    def interference_subcarriers(self) -> tuple[int, ...]:
        spec = self.raw["interference"]
        lo, hi = int(spec["signed_subcarrier_min"]), int(spec["signed_subcarrier_max"])
        return tuple(k for k in range(lo, hi + 1) if k != 0)


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    platform: PlatformConfig
    calibration_path: Path


def _canonical_hash(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_platform(path: str | Path) -> PlatformConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("platform config must be a mapping")
    _exact_fields(raw, PLATFORM_FIELDS, "platform")
    _exact_fields(raw["target"], TARGET_FIELDS, "target")
    _exact_fields(raw["interference"], INTERFERENCE_FIELDS, "interference")
    n = int(raw["fft_size"])
    delta_f = float(raw["subcarrier_spacing_hz"])
    if float(raw["sample_rate_hz"]) != n * delta_f:
        raise ValueError("sample_rate_hz must equal fft_size * subcarrier_spacing_hz")
    cps = [int(x) for x in raw["cp_lengths"]]
    if len(cps) != int(raw["num_symbols"]):
        raise ValueError("cp_lengths must contain one entry per symbol")
    if cps != [160, 144, 144, 144, 144, 144, 144] * 2:
        raise ValueError("v1 requires LTE normal-CP lengths repeated for two slots")
    if int(raw["window_length"]) != int(raw["num_symbols"]) * n + sum(cps):
        raise ValueError("window_length is inconsistent with FFT and CP lengths")
    if raw["windowing"] is not False or raw["noise"] != "none" or float(raw["noise_variance"]) != 0:
        raise NotImplementedError("v1 supports only windowing=false and zero physical noise")
    if raw["channel"] != "flat_constant":
        raise NotImplementedError("v1 supports only a flat constant channel")
    target = tuple(int(k) for k in raw["target"]["signed_subcarriers"])
    interference = tuple(k for k in range(int(raw["interference"]["signed_subcarrier_min"]),
                                          int(raw["interference"]["signed_subcarrier_max"]) + 1)
                         if k != 0)
    if target != tuple(range(-150, 0)) + tuple(range(1, 151)):
        raise ValueError("target subcarriers must be signed ascending [-150..-1,1..150]")
    if interference != tuple(range(-600, 0)) + tuple(range(1, 601)):
        raise ValueError("interference subcarriers must be [-600..-1,1..600]")
    for name, tones, expected_count in (("target", target, 300), ("interference", interference, 1200)):
        if len(tones) != expected_count or len(set(tones)) != len(tones) or 0 in tones:
            raise ValueError(f"invalid {name} active subcarriers")
        if any(abs(k) >= n // 2 for k in tones):
            raise ValueError(f"{name} subcarrier outside FFT range")
    if not set(target).issubset(interference):
        raise ValueError("target resources must lie inside interference resources")
    if raw["target"]["modulation"] != "QPSK":
        raise ValueError("v1 target modulation must be QPSK")
    if raw["interference"]["modulations"] != ["QPSK", "16QAM", "64QAM"]:
        raise ValueError("v1 interference modulation choices are fixed")
    return PlatformConfig(path=path, raw=raw, config_hash=_canonical_hash(raw))


def load_experiment(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a mapping")
    _exact_fields(raw, EXPERIMENT_FIELDS, "experiment")
    if raw["split"] not in {"audit", "test"}:
        raise ValueError("experiment split must be audit or test")
    if int(raw["num_examples"]) <= 0:
        raise ValueError("num_examples must be positive")
    if raw["precision"] not in {"complex128", "complex64"}:
        raise ValueError("precision must be complex128 or complex64")
    platform = load_platform((path.parent / raw["platform_config"]).resolve())
    calibration = (path.parent / raw["calibration_path"]).resolve()
    return ExperimentConfig(path=path, raw=raw, platform=platform,
                            calibration_path=calibration)
