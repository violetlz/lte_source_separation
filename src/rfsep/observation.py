from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PlatformConfig
from .contracts import GroundTruth, Observation, ReceiverInfo
from .waveforms.receiver import target_demodulate
from .waveforms.sources import generate_interference, generate_target


def calibration_identity(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "calibration_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calibrate(config: PlatformConfig, num_records: int, output_path: str | Path,
              *, provisional: bool = False, target_check_records: int = 64) -> dict[str, Any]:
    if num_records <= 1:
        raise ValueError("calibration requires at least two records")
    qb_values = np.empty(num_records, dtype=np.float64)
    interference_power = np.empty(num_records, dtype=np.float64)
    ids = []
    for record_id in range(num_records):
        record, _, _ = generate_interference(config, "calibration", record_id)
        detected = target_demodulate(record.waveform, config.waveform_spec)
        qb_values[record_id] = np.mean(np.abs(detected) ** 2)
        interference_power[record_id] = np.mean(np.abs(record.waveform) ** 2)
        ids.append(record.long_record_id)
    qs_values = np.empty(target_check_records, dtype=np.float64)
    target_power = np.empty(target_check_records, dtype=np.float64)
    for record_id in range(target_check_records):
        record, _, _ = generate_target(config, "calibration", record_id)
        detected = target_demodulate(record.waveform, config.waveform_spec)
        qs_values[record_id] = np.mean(np.abs(detected) ** 2)
        target_power[record_id] = np.mean(np.abs(record.waveform) ** 2)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "platform_version": config.raw["version"],
        "platform_hash": config.config_hash,
        "num_records": int(num_records),
        "target_check_records": int(target_check_records),
        "split": "calibration",
        "provisional": bool(provisional),
        "q_b": float(qb_values.mean()),
        "q_b_standard_error": float(qb_values.std(ddof=1) / np.sqrt(num_records)),
        "q_s": float(qs_values.mean()),
        "q_s_standard_error": float(qs_values.std(ddof=1) / np.sqrt(target_check_records)),
        "interference_time_power_mean": float(interference_power.mean()),
        "interference_time_power_standard_error": float(interference_power.std(ddof=1) / np.sqrt(num_records)),
        "target_time_power_mean": float(target_power.mean()),
        "target_time_power_standard_error": float(target_power.std(ddof=1) / np.sqrt(target_check_records)),
        "dtype": "complex128",
        "fft_compute_precision": "numpy complex128",
        "seed_scheme": "sha256(rfsep-v1|root|purpose|split|record_id)",
        "record_manifest_digest": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "record_id_first": ids[0],
        "record_id_last": ids[-1],
    }
    payload["calibration_id"] = calibration_identity(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_calibration(path: str | Path, config: PlatformConfig,
                     *, allow_provisional: bool = False) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"calibration does not exist: {path}")
    payload = json.loads(path.read_text())
    if payload.get("platform_hash") != config.config_hash:
        raise ValueError("calibration platform hash mismatch")
    if payload.get("platform_version") != config.raw["version"]:
        raise ValueError("calibration platform version mismatch")
    if calibration_identity(payload) != payload.get("calibration_id"):
        raise ValueError("calibration identity/hash mismatch")
    if payload.get("provisional") and not allow_provisional:
        raise ValueError("formal evaluation refuses provisional calibration")
    if not np.isfinite(payload.get("q_b", np.nan)) or payload["q_b"] <= 0:
        raise ValueError("calibration q_b must be finite and positive")
    return payload


def kappa_for_sir(q_b: float, sir_re_db: float) -> float:
    return float(np.sqrt(1.0 / (q_b * 10.0 ** (sir_re_db / 10.0))))


def mix_time_domain(target: np.ndarray, interference: np.ndarray, kappa: float,
                    dtype=np.complex128) -> np.ndarray:
    target, interference = np.asarray(target), np.asarray(interference)
    if (target.ndim != 1 or interference.ndim != 1 or target.shape != interference.shape
            or not np.iscomplexobj(target) or not np.iscomplexobj(interference)):
        raise ValueError("mixing requires equal-length 1-D complex time-domain arrays")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    return (target + kappa * interference).astype(dtype, copy=False)


def build_example(config: PlatformConfig, split: str, record_id: int,
                  sir_re_db: float, calibration: dict[str, Any],
                  dtype=np.complex128) -> tuple[Observation, ReceiverInfo, GroundTruth]:
    target, grid, bits = generate_target(config, split, record_id, dtype=dtype)
    interference, _, modulation_map = generate_interference(config, split, record_id, dtype=dtype)
    kappa = kappa_for_sir(float(calibration["q_b"]), sir_re_db)
    y = mix_time_domain(target.waveform, interference.waveform, kappa, dtype=dtype)
    example_id = f"pair:{split}:{record_id:08d}"
    observation = Observation(example_id, y)
    info = ReceiverInfo(
        platform_version=config.raw["version"], platform_hash=config.config_hash,
        calibration_id=calibration["calibration_id"], kappa=kappa,
        sir_re_db=float(sir_re_db),
    )
    truth = GroundTruth(
        example_id=example_id, target_waveform=target.waveform,
        interference_waveform=interference.waveform, target_bits=bits,
        target_grid=grid, interference_modulation_map=modulation_map,
        tau=int(interference.audit["tau"]), phi=float(interference.audit["phi"]),
        source_record_ids=(target.long_record_id, interference.long_record_id),
        generation_audit={"target": target.audit, "interference": interference.audit},
    )
    return observation, info, truth
