from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PlatformConfig
from ..priors.vp import VPSchedule
from ..waveforms.ofdm import fixed_power_scale, signed_to_bins, symbol_boundaries


class TargetOFDMLMMSE:
    """Exact target second-order LMMSE using 300-dimensional symbol solves."""

    def __init__(self, platform: PlatformConfig, schedule: VPSchedule):
        self.platform = platform
        self.schedule = schedule
        self._systems: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for cp in sorted(set(platform.cp_lengths)):
            mapping = self._mapping(cp)
            gram = mapping.conj().T @ mapping
            for level in range(schedule.steps):
                alpha_bar = float(schedule.alpha_bar[level])
                sigma_sq = 1.0 - alpha_bar
                system = alpha_bar * gram + sigma_sq * np.eye(gram.shape[0])
                self._systems[(cp, level)] = (mapping, np.linalg.cholesky(system))

    def _mapping(self, cp: int) -> np.ndarray:
        n = self.platform.fft_size
        bins = signed_to_bins(self.platform.target_subcarriers, n)
        frequency = np.zeros((n, len(bins)), dtype=np.complex128)
        frequency[bins, np.arange(len(bins))] = 1.0
        useful = np.fft.ifft(frequency, axis=0, norm="ortho")
        useful *= fixed_power_scale(n, len(bins))
        return np.concatenate((useful[-cp:], useful), axis=0)

    def estimate(self, x_t: np.ndarray, level: int,
                 cp_lengths: tuple[int, ...] | None = None) -> np.ndarray:
        x_t = np.asarray(x_t)
        cp_lengths = cp_lengths or self.platform.cp_lengths[:7]
        expected = len(cp_lengths) * self.platform.fft_size + sum(cp_lengths)
        if x_t.shape != (expected,) or not np.iscomplexobj(x_t):
            raise ValueError("target LMMSE input waveform has the wrong ABI")
        alpha_bar = float(self.schedule.alpha_bar[level])
        a = np.sqrt(alpha_bar)
        output, cursor = [], 0
        for cp in cp_lengths:
            mapping, cholesky = self._systems[(cp, int(level))]
            observed = x_t[cursor:cursor + cp + self.platform.fft_size]
            rhs = a * mapping.conj().T @ observed
            coefficients = np.linalg.solve(cholesky.conj().T,
                                           np.linalg.solve(cholesky, rhs))
            output.append(mapping @ coefficients)
            cursor += observed.size
        estimate = np.concatenate(output)
        if not np.isfinite(estimate).all():
            raise FloatingPointError("target LMMSE produced non-finite values")
        return estimate

    def theoretical_nmse(self, level: int,
                         cp_lengths: tuple[int, ...] | None = None) -> float:
        cp_lengths = cp_lengths or self.platform.cp_lengths[:7]
        alpha_bar = float(self.schedule.alpha_bar[level])
        sigma_sq = 1.0 - alpha_bar
        error, power = 0.0, 0.0
        for cp in cp_lengths:
            mapping, cholesky = self._systems[(cp, int(level))]
            gram = mapping.conj().T @ mapping
            inverse = np.linalg.solve(cholesky.conj().T,
                                      np.linalg.solve(cholesky, np.eye(gram.shape[0])))
            error += float(np.trace(gram @ (sigma_sq * inverse)).real)
            power += float(np.trace(gram).real)
        return error / power


@dataclass(frozen=True)
class InterferencePSDWiener:
    """Stationary diagonal-frequency approximation fitted only on clean train data."""

    psd: np.ndarray

    @classmethod
    def fit(cls, clean_windows, *, max_windows: int | None = None):
        total = None
        count = 0
        for waveform in clean_windows:
            waveform = np.asarray(waveform)
            spectrum = np.fft.fft(waveform, norm="ortho")
            power = np.abs(spectrum) ** 2
            total = power if total is None else total + power
            count += 1
            if max_windows is not None and count >= max_windows:
                break
        if count == 0 or not np.isfinite(total).all():
            raise ValueError("PSD fitting requires finite independent train windows")
        return cls((total / count).astype(np.float64))

    def estimate(self, x_t: np.ndarray, signal: float, noise: float) -> np.ndarray:
        spectrum = np.fft.fft(np.asarray(x_t), norm="ortho")
        gain = signal * self.psd / (signal * signal * self.psd + noise * noise)
        result = np.fft.ifft(gain * spectrum, norm="ortho")
        if not np.isfinite(result).all():
            raise FloatingPointError("PSD Wiener estimate is non-finite")
        return result
