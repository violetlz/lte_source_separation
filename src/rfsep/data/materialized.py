from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


class MaterializedPriorDataset(Dataset):
    """Read-only map dataset backed by independently reopenable NPY shards."""

    def __init__(self, manifest_path: str | Path, *, max_open_shards: int = 4):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("state") != "validated":
            raise ValueError("training refuses a dataset that is not validated")
        if self.manifest.get("dtype") != "float32" or self.manifest.get("shape_tail") != [2, 15360]:
            raise ValueError("materialized prior dataset ABI mismatch")
        self.shards = tuple(self.manifest["shards"])
        self.starts = np.asarray([int(item["row_start"]) for item in self.shards], dtype=np.int64)
        self.count = int(self.manifest["count"])
        self.max_open_shards = int(max_open_shards)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return self.count

    def _array(self, shard_index: int) -> np.ndarray:
        array = self._cache.pop(shard_index, None)
        if array is None:
            path = (self.manifest_path.parent / self.shards[shard_index]["path"]).resolve()
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._cache[shard_index] = array
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return array

    def __getitem__(self, index: int) -> torch.Tensor:
        index = int(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        shard_index = int(np.searchsorted(self.starts, index, side="right") - 1)
        local = index - int(self.shards[shard_index]["row_start"])
        # Copy is intentional: mmap is read-only and DataLoader can pin this storage safely.
        return torch.from_numpy(np.array(self._array(shard_index)[local], copy=True,
                                         order="C"))


class RecoverableEpochBatchSampler(Sampler[list[int]]):
    """Deterministic epoch shuffle whose committed position is checkpointable.

    DataLoader may request future batches, but ``commit`` is the only operation that
    advances durable state. Recreating the iterator discards prefetched batches.
    """

    algorithm = "torch-randperm-v1"

    def __init__(self, size: int, batch_size: int, seed: int, *, epoch: int = 0,
                 offset: int = 0):
        if size <= 0 or batch_size <= 0 or size % batch_size:
            raise ValueError("dataset size must be positive and divisible by batch size")
        self.size, self.batch_size, self.seed = int(size), int(batch_size), int(seed)
        self.epoch, self.offset = int(epoch), int(offset)
        self._check()

    def _check(self) -> None:
        if self.epoch < 0 or not 0 <= self.offset < self.size or self.offset % self.batch_size:
            raise ValueError("invalid recoverable sampler position")

    def permutation(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(self.size, generator=generator)

    def __iter__(self):
        permutation = self.permutation()
        for begin in range(self.offset, self.size, self.batch_size):
            yield permutation[begin:begin + self.batch_size].tolist()

    def __len__(self) -> int:
        return (self.size - self.offset) // self.batch_size

    def commit(self) -> None:
        self.offset += self.batch_size
        if self.offset == self.size:
            self.epoch += 1
            self.offset = 0

    @property
    def exposures(self) -> int:
        return self.epoch * self.size + self.offset

    def state_dict(self) -> dict:
        return {"algorithm": self.algorithm, "size": self.size,
                "batch_size": self.batch_size, "seed": self.seed,
                "epoch": self.epoch, "offset": self.offset,
                "exposures": self.exposures}

    def load_state_dict(self, state: dict) -> None:
        fixed = {"algorithm": self.algorithm, "size": self.size,
                 "batch_size": self.batch_size, "seed": self.seed}
        for key, value in fixed.items():
            if state.get(key) != value:
                raise ValueError(f"sampler {key} mismatch")
        self.epoch, self.offset = int(state["epoch"]), int(state["offset"])
        self._check()
        if int(state.get("exposures", self.exposures)) != self.exposures:
            raise ValueError("sampler exposure count mismatch")


def expected_shards(count: int, shard_size: int) -> int:
    return int(math.ceil(count / shard_size))
