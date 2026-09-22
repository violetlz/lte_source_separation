from __future__ import annotations

from collections.abc import Iterator

from ..config import PlatformConfig
from ..observation import build_example


def iter_examples(config: PlatformConfig, split: str, record_ids, sir_re_db: float,
                  calibration: dict, dtype):
    for record_id in record_ids:
        yield build_example(config, split, int(record_id), sir_re_db, calibration, dtype=dtype)
