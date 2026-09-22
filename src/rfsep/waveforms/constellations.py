from __future__ import annotations

import itertools

import numpy as np


BITS_PER_SYMBOL = {"QPSK": 2, "16QAM": 4, "64QAM": 6}


def _validated_bits(bits: np.ndarray, width: int) -> np.ndarray:
    bits = np.asarray(bits)
    if bits.shape[-1] != width:
        raise ValueError(f"expected final bit dimension {width}, got {bits.shape}")
    if not np.isin(bits, (0, 1)).all():
        raise ValueError("bits must contain only 0 or 1")
    return bits.astype(np.int8, copy=False)


def map_bits(bits: np.ndarray, modulation: str) -> np.ndarray:
    width = BITS_PER_SYMBOL.get(modulation)
    if width is None:
        raise ValueError(f"unsupported modulation {modulation!r}")
    b = _validated_bits(bits, width)
    signs_i = 1 - 2 * b[..., 0]
    signs_q = 1 - 2 * b[..., 1]
    if modulation == "QPSK":
        i, q, scale = signs_i, signs_q, np.sqrt(2.0)
    elif modulation == "16QAM":
        i = signs_i * (2 - (1 - 2 * b[..., 2]))
        q = signs_q * (2 - (1 - 2 * b[..., 3]))
        scale = np.sqrt(10.0)
    else:
        i = signs_i * (4 - (1 - 2 * b[..., 2]) * (2 - (1 - 2 * b[..., 4])))
        q = signs_q * (4 - (1 - 2 * b[..., 3]) * (2 - (1 - 2 * b[..., 5])))
        scale = np.sqrt(42.0)
    return (i.astype(np.float64) + 1j * q.astype(np.float64)) / scale


def constellation_table(modulation: str) -> tuple[np.ndarray, np.ndarray]:
    width = BITS_PER_SYMBOL.get(modulation)
    if width is None:
        raise ValueError(f"unsupported modulation {modulation!r}")
    labels = np.array(list(itertools.product((0, 1), repeat=width)), dtype=np.uint8)
    return map_bits(labels, modulation), labels


def demap_symbols(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """Nearest-neighbour hard decision; exact ties choose the first binary label."""
    symbols = np.asarray(symbols)
    table, labels = constellation_table(modulation)
    indices = np.argmin(np.abs(symbols[..., None] - table) ** 2, axis=-1)
    return labels[indices]
