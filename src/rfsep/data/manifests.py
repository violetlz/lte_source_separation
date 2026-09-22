from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ..config import PlatformConfig
from ..randomness import record_identity


def build_manifest(config: PlatformConfig, split: str, num_records: int) -> dict:
    records = []
    for record_id in range(num_records):
        records.append({
            "example_id": f"pair:{split}:{record_id:08d}",
            "split": split,
            "record_id": record_id,
            "target_parent": record_identity("target", split, record_id),
            "interference_parent": record_identity("interference", split, record_id),
        })
    payload = {
        "schema_version": 1, "platform_hash": config.config_hash,
        "root_seed": config.root_seed, "split": split, "records": records,
    }
    payload["manifest_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def write_manifest(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def assert_split_disjoint(manifests: Iterable[dict]) -> None:
    owners: dict[str, str] = {}
    for manifest in manifests:
        split = str(manifest["split"])
        for record in manifest["records"]:
            for key in ("target_parent", "interference_parent"):
                identity = str(record[key])
                previous = owners.get(identity)
                if previous is not None and previous != split:
                    raise ValueError(f"underlying long record {identity} appears in {previous} and {split}")
                owners[identity] = split
