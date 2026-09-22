from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import PlatformConfig
from ..contracts import ReceiverInfo, complex_to_network, network_to_complex
from ..data.manifests import assert_split_disjoint, build_manifest
from ..randomness import derive_seed
from ..waveforms.constellations import constellation_table, demap_symbols, map_bits
from ..waveforms.ofdm import audit_cp, demodulate_waveform, fixed_power_scale, modulate_grid
from ..waveforms.receiver import target_demodulate
from ..waveforms.sources import generate_interference, generate_target


def run_waveform_audit(config: PlatformConfig, output_path: str | Path | None = None) -> dict:
    checks: dict[str, dict] = {}
    anchors = {
        "QPSK": (np.array([[0, 0]], dtype=np.uint8), (1 + 1j) / np.sqrt(2)),
        "16QAM": (np.array([[0, 0, 0, 0]], dtype=np.uint8), (1 + 1j) / np.sqrt(10)),
        "64QAM": (np.array([[0, 0, 0, 0, 0, 0]], dtype=np.uint8), (3 + 3j) / np.sqrt(42)),
    }
    for modulation, (bits, expected) in anchors.items():
        symbols, labels = constellation_table(modulation)
        recovered = demap_symbols(symbols, modulation)
        anchor_error = float(abs(map_bits(bits, modulation)[0] - expected))
        checks[f"constellation_{modulation}"] = {
            "passed": bool(np.array_equal(recovered, labels) and anchor_error < 1e-15
                           and abs(np.mean(np.abs(symbols) ** 2) - 1) < 1e-15),
            "num_labels": int(labels.shape[0]), "anchor_error": anchor_error,
            "mean_energy": float(np.mean(np.abs(symbols) ** 2)),
        }

    target, grid, bits = generate_target(config, "audit", 0)
    reconstructed = target_demodulate(target.waveform, config)
    recovered_bits = demap_symbols(reconstructed, "QPSK").reshape(-1)
    relative_error = float(np.linalg.norm(reconstructed - grid) / np.linalg.norm(grid))
    cp_result = audit_cp(target.waveform, config.fft_size, config.cp_lengths)
    checks["target_roundtrip"] = {
        "passed": bool(np.array_equal(recovered_bits, bits) and relative_error < 1e-10),
        "num_bits": int(bits.size), "num_errors": int(np.count_nonzero(recovered_bits != bits)),
        "relative_grid_error": relative_error,
    }
    checks["cp"] = {"passed": cp_result["cp_max_abs_error"] == 0.0, **cp_result}
    damaged = target.waveform.copy()
    damaged[:config.cp_lengths[0]] = 0
    damaged_cp = audit_cp(damaged, config.fft_size, config.cp_lengths)
    checks["cp_damage_counterexample"] = {
        "passed": damaged_cp["cp_max_abs_error"] > 0,
        "damaged_cp_max_abs_error": damaged_cp["cp_max_abs_error"],
        "post_cp_removal_grid_relative_error": float(
            np.linalg.norm(target_demodulate(damaged, config) - grid) / np.linalg.norm(grid)
        ),
    }

    tone_checks = []
    for signed_tone in (-150, 150, -600, 600):
        tones = config.target_subcarriers if abs(signed_tone) <= 150 else config.interference_subcarriers
        index = tones.index(signed_tone)
        one = np.zeros((config.num_symbols, len(tones)), dtype=np.complex128)
        one[0, index] = 1
        waveform = modulate_grid(one, tones, config.fft_size, config.cp_lengths)
        cp = config.cp_lengths[0]
        sample_indices = np.array([0, 1, 17, config.fft_size - 1])
        expected = (fixed_power_scale(config.fft_size, len(tones)) / np.sqrt(config.fft_size)
                    * np.exp(2j * np.pi * signed_tone * sample_indices / config.fft_size))
        error = float(np.max(np.abs(waveform[cp + sample_indices] - expected)))
        tone_checks.append({"signed_tone": signed_tone, "max_abs_error": error})
    checks["independent_single_tone_formula"] = {
        "passed": all(row["max_abs_error"] < 1e-12 for row in tone_checks), "tones": tone_checks,
    }

    edge_crops = []
    for tau in (0, config.window_length - 1):
        record, long_waveform, _ = generate_interference(
            config, "audit", 100 + tau, force_tau=tau, force_phi=0.0)
        expected = long_waveform[config.window_length + tau:2 * config.window_length + tau]
        edge_crops.append({"tau": tau, "max_abs_error": float(np.max(np.abs(record.waveform - expected)))})
    checks["interference_crop_edges"] = {
        "passed": all(row["max_abs_error"] == 0 for row in edge_crops), "cases": edge_crops,
    }

    forced = tuple(("QPSK", "16QAM", "64QAM", "QPSK") for _ in range(3))
    _, reference_long, reference_maps = generate_interference(
        config, "audit", 31337, force_tau=0, force_phi=0.0, force_modulations=forced)
    modulation_errors = []
    reference_cp_errors = []
    for frame_index, frame_modulations in enumerate(reference_maps):
        frame = reference_long[frame_index * config.window_length:(frame_index + 1) * config.window_length]
        reference_cp_errors.append(audit_cp(frame, config.fft_size, config.cp_lengths)["cp_max_abs_error"])
        recovered_grid = demodulate_waveform(frame, config.interference_subcarriers,
                                             config.fft_size, config.cp_lengths)
        for block_index, modulation in enumerate(frame_modulations):
            block_grid = recovered_grid[:, block_index * 300:(block_index + 1) * 300]
            decided = map_bits(demap_symbols(block_grid, modulation), modulation)
            modulation_errors.append({
                "frame": frame_index, "block": block_index, "modulation": modulation,
                "max_constellation_error": float(np.max(np.abs(decided - block_grid))),
            })
    checks["interference_reference_frame"] = {
        "passed": max(reference_cp_errors) == 0 and all(row["max_constellation_error"] < 1e-12
                                                        for row in modulation_errors),
        "modulations_covered": sorted(set(row["modulation"] for row in modulation_errors)),
        "cp_max_abs_error": max(reference_cp_errors), "blocks": modulation_errors,
    }

    net = complex_to_network(target.waveform)
    net_error = float(np.max(np.abs(network_to_complex(net) - target.waveform)))
    target64, grid64, bits64 = generate_target(config, "audit", 1, dtype=np.complex64)
    reconstructed64 = target_demodulate(target64.waveform, config)
    rel64 = float(np.linalg.norm(reconstructed64 - grid64) / np.linalg.norm(grid64))
    bit_errors64 = int(np.count_nonzero(demap_symbols(reconstructed64, "QPSK").reshape(-1) != bits64))
    checks["precision_and_network_coordinates"] = {
        "passed": net_error < 1e-12 and rel64 < 1e-5 and bit_errors64 == 0,
        "network_roundtrip_max_abs_error": net_error,
        "complex64_relative_grid_error": rel64,
        "complex64_bit_errors": bit_errors64,
        "fft_compute_precision": "numpy promotes complex64 FFT output to complex128",
    }

    ordered = [generate_target(config, "test", i)[0].waveform for i in (1, 7, 3)]
    reordered = {i: generate_target(config, "test", i)[0].waveform for i in (3, 1, 7)}
    order_ok = all(np.array_equal(wave, reordered[i]) for i, wave in zip((1, 7, 3), ordered))
    seed_before = derive_seed(config.root_seed, "target-data-frame-1", "test", 9)
    generate_interference(config, "test", 9, data_seed_offset=12345)
    seed_after = derive_seed(config.root_seed, "target-data-frame-1", "test", 9)
    checks["randomness"] = {
        "passed": bool(order_ok and seed_before == seed_after),
        "order_invariant": bool(order_ok), "source_stream_independent": seed_before == seed_after,
    }

    manifests = [build_manifest(config, split, 3)
                 for split in ("train", "validation", "test", "calibration", "audit")]
    assert_split_disjoint(manifests)
    collision_detected = False
    bad = json.loads(json.dumps(manifests[1]))
    bad["records"][0]["target_parent"] = manifests[0]["records"][0]["target_parent"]
    try:
        assert_split_disjoint([manifests[0], bad])
    except ValueError:
        collision_detected = True
    checks["split_disjointness"] = {
        "passed": collision_detected, "normal_manifests_passed": True,
        "deliberate_collision_rejected": collision_detected,
    }

    info_fields = set(ReceiverInfo.__dataclass_fields__)
    forbidden = {"target_bits", "target_waveform", "interference_waveform", "tau", "phi", "modulation_map"}
    checks["information_boundary"] = {
        "passed": not bool(info_fields & forbidden),
        "receiver_info_fields": sorted(info_fields), "forbidden_fields_present": sorted(info_fields & forbidden),
    }
    checks["resource_map"] = {
        "passed": (len(config.target_subcarriers) == 300 and len(config.interference_subcarriers) == 1200
                   and 0 not in config.target_subcarriers and 0 not in config.interference_subcarriers
                   and set(config.target_subcarriers).issubset(config.interference_subcarriers)),
        "target_count": len(config.target_subcarriers), "interference_count": len(config.interference_subcarriers),
        "target_edges": [config.target_subcarriers[0], config.target_subcarriers[-1]],
        "interference_edges": [config.interference_subcarriers[0], config.interference_subcarriers[-1]],
    }
    result = {
        "platform_hash": config.config_hash, "all_passed": all(row["passed"] for row in checks.values()),
        "passed": sum(bool(row["passed"]) for row in checks.values()),
        "failed": sum(not bool(row["passed"]) for row in checks.values()), "checks": checks,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
