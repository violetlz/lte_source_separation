"""Train-only estimated LTE baselines; neither is a full-waveform Bayes optimum.

The FFT-diagonal filter is an approximate Wiener baseline, not the exact
two-source LMMSE for the CP/nonstationary waveform. The bit detector is a
single-RE marginal bit-MAP with a smoothed empirical radial likelihood.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

from ..config import PlatformConfig
from ..waveforms.constellations import constellation_table
from ..waveforms.receiver import target_demodulate
from ..waveforms.sources import generate_interference, generate_target


def estimate_train_statistics(platform: PlatformConfig, parent_ids: range,
                              *, bins: int = 4096, smooth_bins: float = 3.0) -> dict:
    if len(parent_ids) < 2:
        raise ValueError("need independent train parents")
    n = platform.window_length
    psd_s = np.zeros(n, dtype=np.float64)
    psd_b = np.zeros(n, dtype=np.float64)
    squared_radii = []
    per_symbol_power = np.zeros(platform.num_symbols)
    slot_power = np.zeros(2)
    signed_power = np.zeros(2)
    complex_sum = 0j
    for record_id in parent_ids:
        s, _, _ = generate_target(platform, "train", record_id)
        b, _, _ = generate_interference(platform, "train", record_id)
        psd_s += np.abs(np.fft.fft(s.waveform, norm="ortho")) ** 2
        psd_b += np.abs(np.fft.fft(b.waveform, norm="ortho")) ** 2
        grid = target_demodulate(b.waveform, platform.waveform_spec)
        squared = np.abs(grid) ** 2
        squared_radii.append(squared.reshape(-1))
        per_symbol_power += squared.mean(axis=1)
        slot_power += (squared[:7].mean(), squared[7:].mean())
        signed_power += (squared[:, :150].mean(), squared[:, 150:].mean())
        complex_sum += grid.sum()
    count = len(parent_ids)
    r2 = np.concatenate(squared_radii)
    upper = float(np.quantile(r2, 0.99999) * 1.2)
    edges = np.linspace(0, upper, bins + 1)
    hist, _ = np.histogram(r2, edges)
    density = gaussian_filter1d(hist.astype(np.float64), smooth_bins, mode="nearest")
    density = np.maximum(density, 1e-12) / (density.sum() * (edges[1] - edges[0]))
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "schema": "lte-train-estimated-baselines-v1", "platform_hash": platform.config_hash,
        "split": "train", "parent_first": parent_ids.start, "parent_last": parent_ids.stop - 1,
        "num_parents": count, "num_re_samples": int(r2.size), "bins": bins,
        "smooth_bins": smooth_bins, "r2_centers": centers.tolist(),
        "r2_density": density.tolist(), "psd_s": (psd_s / count).tolist(),
        "psd_b": (psd_b / count).tolist(),
        "diagnostics": {
            "per_symbol_power": (per_symbol_power / count).tolist(),
            "slot_power": (slot_power / count).tolist(),
            "signed_carrier_power": (signed_power / count).tolist(),
            "mean_complex_re": [float(complex_sum.real / r2.size),
                                float(complex_sum.imag / r2.size)],
            "mean_r2": float(r2.mean()), "r2_upper": upper,
            "tail_fraction_beyond_grid": float(np.mean(r2 > upper)),
        },
    }


def load_statistics(path: str | Path, platform: PlatformConfig) -> dict:
    stats = json.loads(Path(path).read_text())
    if stats["platform_hash"] != platform.config_hash or stats["split"] != "train":
        raise ValueError("baseline statistics are not from this platform/train split")
    return stats


def approximate_wiener(y: np.ndarray, kappa: float, stats: dict) -> np.ndarray:
    psd_s = np.asarray(stats["psd_s"])
    psd_b = np.asarray(stats["psd_b"])
    transfer = psd_s / np.maximum(psd_s + kappa * kappa * psd_b, 1e-15)
    return np.fft.ifft(np.fft.fft(y, norm="ortho") * transfer, norm="ortho")


def marginal_bit_map(grid: np.ndarray, kappa: float, stats: dict) -> np.ndarray:
    """Bit-wise posterior decisions, using only the observed target RE grid."""
    grid = np.asarray(grid)
    symbols, labels = constellation_table("QPSK")
    r2 = np.abs((grid[..., None] - symbols) / kappa) ** 2
    centers = np.asarray(stats["r2_centers"])
    density = np.asarray(stats["r2_density"])
    # The terminal density is a conservative positive floor for out-of-grid
    # observations. Such events are measured on independent audit data.
    likelihood = np.interp(r2, centers, density, left=density[0], right=max(density[-1], 1e-300))
    log_likelihood = np.log(np.maximum(likelihood, 1e-300))
    output = np.empty((*grid.shape, 2), dtype=np.uint8)
    for bit in range(2):
        log_zero = logsumexp(log_likelihood[..., labels[:, bit] == 0], axis=-1)
        log_one = logsumexp(log_likelihood[..., labels[:, bit] == 1], axis=-1)
        output[..., bit] = log_one > log_zero
    return output.reshape(-1)
