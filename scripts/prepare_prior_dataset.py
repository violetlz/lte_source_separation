#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfsep.config import load_training_view  # noqa: E402
from rfsep.data.materialized import expected_shards, sha256_file  # noqa: E402
from rfsep.data.single_source import SingleSourceDataset, selected_slot  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _make_shard(task: tuple) -> dict:
    (view_path, source, split, id_start, parent_count, both_slots, output_dir,
     shard_index, shard_size) = task
    view = load_training_view(view_path)
    dataset = SingleSourceDataset(view, source, split)
    rows_per_parent = 2 if both_slots else 1
    total_rows = parent_count * rows_per_parent
    row_start = shard_index * shard_size
    row_stop = min(total_rows, row_start + shard_size)
    output_dir = Path(output_dir)
    path = output_dir / f"shard_{shard_index:04d}.npy"
    tmp = output_dir / f"shard_{shard_index:04d}.npy.tmp.{os.getpid()}"
    audit_path = output_dir / f"shard_{shard_index:04d}.audit.npz"
    audit_tmp = output_dir / f"shard_{shard_index:04d}.audit.npz.tmp.{os.getpid()}"
    array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                      shape=(row_stop - row_start, 2, 15360))
    parent_ids = np.empty(row_stop - row_start, dtype=np.int64)
    slots = np.empty(row_stop - row_start, dtype=np.uint8)
    tau = np.full(row_stop - row_start, -1, dtype=np.int32)
    phi = np.full(row_stop - row_start, np.nan, dtype=np.float32)
    modulation = np.full((row_stop - row_start, 3, 4), -1, dtype=np.int8)
    modulation_names = {"QPSK": 0, "16QAM": 1, "64QAM": 2}
    for local, row in enumerate(range(row_start, row_stop)):
        parent_offset, slot_offset = divmod(row, rows_per_parent)
        parent_id = id_start + parent_offset
        slot = slot_offset if both_slots else selected_slot(view.platform.root_seed, source,
                                                              split, parent_id)
        window = dataset.window(parent_id, slot)
        if window.network.shape != (2, 15360) or not np.isfinite(window.network).all():
            raise FloatingPointError(f"invalid generated window parent={parent_id} slot={slot}")
        array[local] = window.network
        parent_ids[local], slots[local] = parent_id, slot
        if source == "interference":
            tau[local], phi[local] = int(window.audit["tau"]), float(window.audit["phi"])
            modulation[local] = [[modulation_names[name] for name in frame]
                                 for frame in window.audit["modulation_map"]]
    array.flush()
    del array
    with audit_tmp.open("wb") as handle:
        np.savez(handle, parent_id=parent_ids, slot=slots, tau=tau, phi=phi,
                 modulation=modulation)
    os.replace(tmp, path)
    os.replace(audit_tmp, audit_path)
    return {"index": shard_index, "path": path.name, "audit_path": audit_path.name,
            "row_start": row_start, "row_stop": row_stop, "count": row_stop - row_start,
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "audit_bytes": audit_path.stat().st_size, "audit_sha256": sha256_file(audit_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/data/prior_dataset_v1.yaml"))
    parser.add_argument("--source", choices=("target", "interference"), required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(config_path.read_text())
    view_path = (config_path.parent / raw["view_config"]).resolve()
    view = load_training_view(view_path)
    split_cfg = raw[args.split]
    count = int(split_cfg["parents_per_source"])
    id_start = int(split_cfg["parent_id_start"])
    both_slots = bool(split_cfg["both_slots"])
    shard_size = int(raw["storage"]["shard_rows"])
    workers = args.workers or int(raw["generation"]["workers"])
    output_dir = (config_path.parent / raw["storage"]["root"]).resolve() / raw["dataset_id"] / args.source / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = count * (2 if both_slots else 1)
    tasks, complete = [], []
    for index in range(expected_shards(rows, shard_size)):
        path = output_dir / f"shard_{index:04d}.npy"
        audit = output_dir / f"shard_{index:04d}.audit.npz"
        if path.is_file() and audit.is_file():
            expected = min(shard_size, rows - index * shard_size)
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            if loaded.shape == (expected, 2, 15360) and loaded.dtype == np.float32:
                complete.append({"index": index, "path": path.name, "audit_path": audit.name,
                                 "row_start": index * shard_size,
                                 "row_stop": index * shard_size + expected, "count": expected,
                                 "bytes": path.stat().st_size, "sha256": sha256_file(path),
                                 "audit_bytes": audit.stat().st_size,
                                 "audit_sha256": sha256_file(audit)})
                continue
        tasks.append((str(view_path), args.source, args.split, id_start, count, both_slots,
                      str(output_dir), index, shard_size))
    started = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_make_shard, task) for task in tasks]
            for future in as_completed(futures):
                item = future.result()
                complete.append(item)
                print(json.dumps({"completed_shard": item["index"], "source": args.source,
                                  "split": args.split}), flush=True)
    complete.sort(key=lambda item: item["index"])
    manifest = {
        "schema_version": 1, "dataset_id": raw["dataset_id"], "version": raw["version"],
        "state": "generated", "source": args.source, "split": args.split,
        "parent_id_start": id_start, "parent_id_stop": id_start + count,
        "parents": count, "both_slots": both_slots, "count": rows,
        "shape_tail": [2, 15360], "dtype": "float32", "coordinate_scale": "sqrt2_ri",
        "platform_hash": view.platform.config_hash, "training_view_hash": view.config_hash,
        "slot_rule": "both" if both_slots else "sha256_parent_deterministic",
        "generation_seconds_this_invocation": time.time() - started,
        "generator": "rfsep.data.single_source.SingleSourceDataset",
        "shards": complete,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str((output_dir / 'manifest.json').resolve()),
                      "rows": rows, "bytes": sum(item["bytes"] for item in complete)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
