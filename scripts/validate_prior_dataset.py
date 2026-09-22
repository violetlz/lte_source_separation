#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfsep.config import load_training_view  # noqa: E402
from rfsep.data.materialized import sha256_file  # noqa: E402
from rfsep.data.single_source import SingleSourceDataset  # noqa: E402
from rfsep.contracts import network_to_complex  # noqa: E402
from rfsep.waveforms.constellations import demap_symbols  # noqa: E402
from rfsep.waveforms.ofdm import audit_cp, demodulate_waveform, symbol_boundaries  # noqa: E402
from rfsep.waveforms.sources import generate_target  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--view", default=str(ROOT / "configs/training/prior_view_v1.yaml"))
    parser.add_argument("--regenerate", type=int, default=8)
    args = parser.parse_args()
    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text())
    view = load_training_view(args.view)
    online = SingleSourceDataset(view, manifest["source"], manifest["split"])
    parent_ids, slots, taus, phis, mods = [], [], [], [], []
    checks = []
    square_sum, scalar_count = 0.0, 0
    for item in manifest["shards"]:
        shard_path = path.parent / item["path"]
        audit_path = path.parent / item["audit_path"]
        array = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        if array.shape != (item["count"], 2, 15360) or array.dtype != np.float32:
            raise ValueError(f"shard ABI mismatch: {shard_path}")
        if not np.isfinite(array).all():
            raise FloatingPointError(f"nonfinite shard: {shard_path}")
        square_sum += float(np.sum(np.square(array, dtype=np.float64), dtype=np.float64))
        scalar_count += int(array.size)
        if sha256_file(shard_path) != item["sha256"] or sha256_file(audit_path) != item["audit_sha256"]:
            raise ValueError(f"shard digest mismatch: {shard_path}")
        audit = np.load(audit_path, allow_pickle=False)
        parent_ids.extend(audit["parent_id"].tolist())
        slots.extend(audit["slot"].tolist())
        taus.extend(audit["tau"].tolist())
        phis.extend(audit["phi"].tolist())
        mods.append(audit["modulation"])
        checks.append((array, audit))
    if len(parent_ids) != manifest["count"]:
        raise ValueError("manifest row count mismatch")
    expected_multiplicity = 2 if manifest["both_slots"] else 1
    counts = Counter(parent_ids)
    expected = set(range(manifest["parent_id_start"], manifest["parent_id_stop"]))
    if set(counts) != expected or any(value != expected_multiplicity for value in counts.values()):
        raise ValueError("parent IDs are missing, duplicated, or outside the declared range")
    # Deterministic samples span first/middle/last shards.
    sample_rows = np.linspace(0, manifest["count"] - 1,
                              min(args.regenerate, manifest["count"]), dtype=int)
    cp_max, cp_pairs, target_bit_errors, target_bits = 0.0, 0, 0, 0
    psd_accumulator = None
    for row in sample_rows:
        shard_index = max(index for index, item in enumerate(manifest["shards"])
                          if item["row_start"] <= row)
        item = manifest["shards"][shard_index]
        local = row - item["row_start"]
        audit = checks[shard_index][1]
        parent_id, slot = int(audit["parent_id"][local]), int(audit["slot"][local])
        materialized = checks[shard_index][0][local]
        reference_window = online.window(parent_id, slot)
        reference = reference_window.network
        if not np.array_equal(materialized, reference):
            raise ValueError(f"generator regression at row {row}")
        waveform = network_to_complex(materialized)
        if manifest["source"] == "target":
            cp_max = max(cp_max, audit_cp(waveform, view.platform.fft_size,
                                          view.platform.cp_lengths[:7])["cp_max_abs_error"])
            cp_pairs += 7
        else:
            crop_start = int(reference_window.audit["crop_start"]) + reference_window.offset
            for frame in range(3):
                for begin, useful, stop in symbol_boundaries(view.platform.fft_size,
                                                              view.platform.cp_lengths):
                    begin += frame * 30720 - crop_start
                    useful += frame * 30720 - crop_start
                    stop += frame * 30720 - crop_start
                    if begin >= 0 and stop <= waveform.size:
                        cp = useful - begin
                        cp_max = max(cp_max, float(np.max(np.abs(
                            waveform[begin:useful] - waveform[stop-cp:stop]))))
                        cp_pairs += 1
        spectrum = np.abs(np.fft.fft(waveform, norm="ortho")) ** 2
        psd_accumulator = spectrum if psd_accumulator is None else psd_accumulator + spectrum
        if manifest["source"] == "target":
            _, _, bits = generate_target(view.platform, manifest["split"], parent_id,
                                         dtype=np.complex64)
            truth = bits.reshape(14, 300, 2)[slot * 7:(slot + 1) * 7].reshape(-1)
            grid = demodulate_waveform(waveform, view.platform.target_subcarriers,
                                       view.platform.fft_size, view.platform.cp_lengths[:7])
            detected = demap_symbols(grid, "QPSK").reshape(-1)
            target_bit_errors += int(np.count_nonzero(detected != truth))
            target_bits += int(truth.size)
    tau_valid = np.asarray(taus) >= 0
    summary = {
        "state": "validated", "rows": len(parent_ids), "unique_parents": len(counts),
        "parent_id_min": min(parent_ids), "parent_id_max": max(parent_ids),
        "slot_counts": {str(key): int(value) for key, value in Counter(slots).items()},
        "regenerated_rows": sample_rows.tolist(), "all_finite": True,
        "total_bytes": sum(item["bytes"] for item in manifest["shards"]),
        "complex_power_mean_full": square_sum / scalar_count,
        "regenerated_cp_max_abs_error": cp_max,
        "regenerated_complete_cp_pairs": cp_pairs,
        "regenerated_psd_min": float((psd_accumulator / len(sample_rows)).min()),
        "regenerated_psd_max": float((psd_accumulator / len(sample_rows)).max()),
    }
    if manifest["source"] == "target":
        summary.update({"regenerated_target_bit_errors": target_bit_errors,
                        "regenerated_target_bits": target_bits})
        if target_bit_errors:
            raise ValueError("clean target materialization has nonzero bit errors")
    if tau_valid.any():
        tau_array, phi_array = np.asarray(taus)[tau_valid], np.asarray(phis)[tau_valid]
        mod_array = np.concatenate(mods)[tau_valid]
        summary.update({
            "tau_min": int(tau_array.min()), "tau_max": int(tau_array.max()),
            "tau_bins_32": np.histogram(tau_array, bins=32, range=(0, 30720))[0].tolist(),
            "phi_bins_32": np.histogram(phi_array, bins=32, range=(0, 2*np.pi))[0].tolist(),
            "modulation_counts": {str(index): int((mod_array == index).sum()) for index in range(3)},
        })
    manifest["state"] = "validated"
    manifest["validation"] = summary
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    report = path.parent / "validation_summary.json"
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
