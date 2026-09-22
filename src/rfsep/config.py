from __future__ import annotations

import hashlib
import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import WaveformSpec


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
TRAINING_VIEW_FIELDS = {
    "version", "platform_config", "parent_window_length", "window_length",
    "slot_offsets", "selection", "network_layout", "coordinate_scale",
    "dtype", "validation_parents", "validation_both_slots",
}
TRAINING_FIELDS = {
    "version", "view_config", "source", "model", "diffusion", "optimizer",
    "runtime", "validation", "tracking", "budget",
}
MODEL_FIELDS = {"residual_layers", "residual_channels", "dilation_cycle", "kernel_size",
                "timestep_dim", "timestep_hidden", "in_channels", "out_channels"}
DIFFUSION_FIELDS = {"steps", "beta_start", "beta_end", "prediction"}
OPTIMIZER_FIELDS = {"name", "learning_rate", "betas", "weight_decay", "grad_clip", "ema_decay"}
RUNTIME_FIELDS = {"effective_batch_size", "microbatch_size", "precision", "seed",
                  "activation_checkpointing", "num_workers"}
VALIDATION_FIELDS = {"noise_levels", "quick_parents", "schedule", "full_schedule"}
TRACKING_FIELDS = {"project", "group", "name", "mode", "log_every"}
BUDGET_FIELDS = {"max_updates", "max_process_seconds", "milestone_every"}


def _exact_fields(data: dict[str, Any], expected: set[str], context: str) -> None:
    unknown = sorted(set(data) - expected)
    missing = sorted(expected - set(data))
    if unknown or missing:
        raise ValueError(f"{context} fields invalid: unknown={unknown}, missing={missing}")


@dataclass(frozen=True)
class PlatformConfig:
    path: Path
    _raw: dict[str, Any]
    config_hash: str

    @property
    def raw(self) -> dict[str, Any]:
        """Return a defensive copy so the hashed effective config cannot be mutated."""
        return copy.deepcopy(self._raw)

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

    @property
    def waveform_spec(self) -> WaveformSpec:
        return WaveformSpec(
            platform_version=str(self.raw["version"]), platform_hash=self.config_hash,
            sample_rate_hz=int(self.raw["sample_rate_hz"]),
            subcarrier_spacing_hz=int(self.raw["subcarrier_spacing_hz"]),
            fft_size=self.fft_size, num_symbols=self.num_symbols, cp_lengths=self.cp_lengths,
            window_length=self.window_length, target_subcarriers=self.target_subcarriers,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    platform: PlatformConfig
    calibration_path: Path


@dataclass(frozen=True)
class TrainingViewConfig:
    path: Path
    _raw: dict[str, Any]
    platform: PlatformConfig
    config_hash: str

    @property
    def raw(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)


@dataclass(frozen=True)
class TrainingConfig:
    path: Path
    _raw: dict[str, Any]
    view: TrainingViewConfig
    config_hash: str

    @property
    def raw(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)


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
    if raw["version"] != "lte-inspired-5mhz-20mhz-v1":
        raise ValueError("unsupported platform version")
    if int(raw["root_seed"]) != 20260922:
        raise ValueError("v1 requires root_seed=20260922")
    fixed_scalars = {
        "sample_rate_hz": 30_720_000, "subcarrier_spacing_hz": 15_000,
        "fft_size": 2048, "num_symbols": 14, "window_length": 30_720,
    }
    for key, expected in fixed_scalars.items():
        if int(raw[key]) != expected:
            raise ValueError(f"v1 requires {key}={expected}")
    if int(raw["target"]["nominal_bandwidth_mhz"]) != 5 or int(raw["target"]["num_rbs"]) != 25:
        raise ValueError("v1 target requires 5 MHz and 25 RB")
    if int(raw["target"]["active_subcarriers"]) != 300:
        raise ValueError("v1 target requires 300 active subcarriers")
    if (int(raw["interference"]["nominal_bandwidth_mhz"]) != 20
            or int(raw["interference"]["num_rbs"]) != 100):
        raise ValueError("v1 interference requires 20 MHz and 100 RB")
    if (int(raw["interference"]["active_subcarriers"]) != 1200
            or int(raw["interference"]["block_size_subcarriers"]) != 300):
        raise ValueError("v1 interference requires 1200 active tones in four 300-tone blocks")
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
    return PlatformConfig(path=path, _raw=copy.deepcopy(raw), config_hash=_canonical_hash(raw))


def load_training_view(path: str | Path) -> TrainingViewConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("training view config must be a mapping")
    _exact_fields(raw, TRAINING_VIEW_FIELDS, "training view")
    platform = load_platform((path.parent / raw["platform_config"]).resolve())
    required = {
        "version": "lte-prior-slot-view-v1",
        "parent_window_length": platform.window_length, "window_length": 15360,
        "slot_offsets": [0, 15360], "selection": "sha256_parent_deterministic",
        "network_layout": "channels_first", "coordinate_scale": "sqrt2_ri",
        "dtype": "float32", "validation_parents": 64,
    }
    for key, expected in required.items():
        if raw[key] != expected:
            raise ValueError(f"v1 training view requires {key}={expected!r}")
    if raw["validation_both_slots"] is not True:
        raise ValueError("v1 validation must evaluate both slots")
    hashed = {key: value for key, value in raw.items() if key != "platform_config"}
    hashed["platform_hash"] = platform.config_hash
    return TrainingViewConfig(path, raw, platform, _canonical_hash(hashed))


def load_training(path: str | Path) -> TrainingConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("training config must be a mapping")
    _exact_fields(raw, TRAINING_FIELDS, "training")
    for key, fields in (("model", MODEL_FIELDS), ("diffusion", DIFFUSION_FIELDS),
                        ("optimizer", OPTIMIZER_FIELDS), ("runtime", RUNTIME_FIELDS),
                        ("validation", VALIDATION_FIELDS), ("tracking", TRACKING_FIELDS),
                        ("budget", BUDGET_FIELDS)):
        if not isinstance(raw[key], dict):
            raise ValueError(f"training.{key} must be a mapping")
        _exact_fields(raw[key], fields, f"training.{key}")
    view = load_training_view((path.parent / raw["view_config"]).resolve())
    source = raw["source"]
    if source not in {"target", "interference"}:
        raise ValueError("source must be target or interference")
    if raw["version"] != f"lte-{source}-prior-pilot-v1":
        raise ValueError("training version does not match source")
    expected_channels = 64 if source == "target" else 128
    model = raw["model"]
    expected_model = {"residual_layers": 24, "residual_channels": expected_channels,
                      "dilation_cycle": 12, "kernel_size": 3, "timestep_dim": 128,
                      "timestep_hidden": 512, "in_channels": 2, "out_channels": 2}
    if any(model[key] != value for key, value in expected_model.items()):
        raise ValueError(f"{source} v1 model contract mismatch")
    diffusion = raw["diffusion"]
    if diffusion != {"steps": 50, "beta_start": 0.0001, "beta_end": 0.05,
                     "prediction": "epsilon"}:
        raise ValueError("v1 diffusion contract mismatch")
    optimizer = raw["optimizer"]
    if (optimizer["name"] != "adam" or optimizer["learning_rate"] != 0.0002
            or optimizer["betas"] != [0.9, 0.999] or optimizer["weight_decay"] != 0
            or optimizer["grad_clip"] != 1.0 or optimizer["ema_decay"] != 0.999):
        raise ValueError("v1 optimizer contract mismatch")
    if raw["runtime"]["effective_batch_size"] != 16:
        raise ValueError("pilot effective batch size must be 16")
    runtime = raw["runtime"]
    if (runtime["precision"] != "bf16" or int(runtime["microbatch_size"]) <= 0
            or 16 % int(runtime["microbatch_size"]) != 0 or runtime["num_workers"] != 0):
        raise ValueError("pilot runtime contract mismatch")
    if raw["validation"]["noise_levels"] != [0, 9, 24, 39, 49]:
        raise ValueError("validation noise levels are frozen")
    validation = raw["validation"]
    if (validation["schedule"] != [0, 100, 200]
            or validation["full_schedule"] != [0, 200]
            or int(validation["quick_parents"]) <= 0
            or int(validation["quick_parents"]) > int(view.raw["validation_parents"])):
        raise ValueError("pilot validation schedule mismatch")
    tracking = raw["tracking"]
    if tracking["mode"] != "online" or int(tracking["log_every"]) != 10:
        raise ValueError("pilot tracking must be online at 10-update intervals")
    budget = raw["budget"]
    if (not 0 < int(budget["max_updates"]) <= 2000
            or not 0 < float(budget["max_process_seconds"]) <= 7200
            or int(budget["milestone_every"]) <= 0):
        raise ValueError("pilot budget exceeds authorization")
    hashed = {key: value for key, value in raw.items() if key != "view_config"}
    hashed["training_view_hash"] = view.config_hash
    return TrainingConfig(path, raw, view, _canonical_hash(hashed))


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
