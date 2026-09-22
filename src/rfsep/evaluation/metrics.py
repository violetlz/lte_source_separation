from __future__ import annotations

import numpy as np


def bit_errors(predicted: np.ndarray, truth: np.ndarray) -> tuple[int, int]:
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    if predicted.shape != truth.shape:
        raise ValueError(f"bit shapes differ: {predicted.shape} versus {truth.shape}")
    if not np.isin(predicted, (0, 1)).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("bit arrays must contain only 0/1")
    return int(np.count_nonzero(predicted != truth)), int(truth.size)


def nmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    if estimate.shape != truth.shape:
        raise ValueError("NMSE inputs must have identical shape")
    denominator = np.sum(np.abs(truth) ** 2)
    return float(np.sum(np.abs(estimate - truth) ** 2) / denominator)


def evm_rms(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    if estimate.shape != truth.shape:
        raise ValueError("EVM inputs must have identical shape")
    return float(np.sqrt(np.sum(np.abs(estimate - truth) ** 2) / np.sum(np.abs(truth) ** 2)))
