from __future__ import annotations

import numpy as np


def fixed_power_scale(fft_size: int, num_active: int) -> float:
    return float(np.sqrt(fft_size / num_active))


def signed_to_bins(subcarriers: tuple[int, ...] | list[int], fft_size: int) -> np.ndarray:
    tones = np.asarray(subcarriers, dtype=np.int64)
    if tones.ndim != 1 or tones.size == 0 or np.any(tones == 0):
        raise ValueError("signed subcarriers must be a non-empty 1-D array without DC")
    if np.unique(tones).size != tones.size or np.any(np.abs(tones) >= fft_size // 2):
        raise ValueError("subcarriers must be unique and inside the FFT range")
    return tones % fft_size


def modulate_grid(active_grid: np.ndarray, subcarriers: tuple[int, ...], fft_size: int,
                  cp_lengths: tuple[int, ...], dtype=np.complex128) -> np.ndarray:
    active_grid = np.asarray(active_grid)
    expected = (len(cp_lengths), len(subcarriers))
    if active_grid.shape != expected:
        raise ValueError(f"active_grid must have shape {expected}, got {active_grid.shape}")
    bins = signed_to_bins(subcarriers, fft_size)
    full = np.zeros((len(cp_lengths), fft_size), dtype=np.complex128)
    full[:, bins] = active_grid
    useful = np.fft.ifft(full, axis=1, norm="ortho") * fixed_power_scale(fft_size, len(subcarriers))
    parts = [np.concatenate((row[-cp:], row)) for row, cp in zip(useful, cp_lengths)]
    return np.concatenate(parts).astype(dtype, copy=False)


def demodulate_waveform(waveform: np.ndarray, subcarriers: tuple[int, ...], fft_size: int,
                        cp_lengths: tuple[int, ...]) -> np.ndarray:
    waveform = np.asarray(waveform)
    expected = len(cp_lengths) * fft_size + sum(cp_lengths)
    if waveform.ndim != 1 or waveform.size != expected or not np.iscomplexobj(waveform):
        raise ValueError(f"waveform must be a complex vector of length {expected}")
    bins = signed_to_bins(subcarriers, fft_size)
    result = np.empty((len(cp_lengths), len(subcarriers)), dtype=np.complex128)
    cursor = 0
    for index, cp in enumerate(cp_lengths):
        useful = waveform[cursor + cp:cursor + cp + fft_size]
        result[index] = np.fft.fft(useful, norm="ortho")[bins]
        cursor += cp + fft_size
    return result / fixed_power_scale(fft_size, len(subcarriers))


def symbol_boundaries(fft_size: int, cp_lengths: tuple[int, ...]) -> list[tuple[int, int, int]]:
    result = []
    cursor = 0
    for cp in cp_lengths:
        result.append((cursor, cursor + cp, cursor + cp + fft_size))
        cursor += cp + fft_size
    return result


def audit_cp(waveform: np.ndarray, fft_size: int, cp_lengths: tuple[int, ...]) -> dict[str, float]:
    waveform = np.asarray(waveform)
    errors = []
    for start, useful_start, stop in symbol_boundaries(fft_size, cp_lengths):
        cp = useful_start - start
        errors.append(float(np.max(np.abs(waveform[start:useful_start] - waveform[stop-cp:stop]))))
    return {"cp_max_abs_error": max(errors), "cp_mean_symbol_max_abs_error": float(np.mean(errors))}
