from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import PlatformConfig


def plot_spectrum(target: np.ndarray, interference: np.ndarray, mixture: np.ndarray,
                  config: PlatformConfig, path: str | Path) -> None:
    path = Path(path)
    nfft = 65536
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / float(config.raw["sample_rate_hz"]))) / 1e6
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for values, label in ((target, "target (5 MHz nominal)"),
                          (interference, "interference (20 MHz nominal)"),
                          (mixture, "mixture")):
        spectrum = np.fft.fftshift(np.fft.fft(values, n=nfft))
        power = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
        power -= power.max()
        ax.plot(freq, power, lw=0.8, label=label)
    ax.set(xlabel="Frequency (MHz)", ylabel="Relative magnitude (dB)",
           title="One 1 ms asynchronous OFDM example (finite-window sidelobes expected)", xlim=(-15.36, 15.36), ylim=(-100, 3))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cp(target: np.ndarray, config: PlatformConfig, path: str | Path) -> None:
    cp = config.cp_lengths[0]
    useful = target[cp:cp + config.fft_size]
    prefix = target[:cp]
    tail = useful[-cp:]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(prefix.real, label="CP real")
    axes[0].plot(tail.real, "--", label="symbol-tail real")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].semilogy(np.maximum(np.abs(prefix - tail), 1e-18))
    axes[1].set(xlabel="Sample in first CP", ylabel="Absolute error",
                title=f"First-symbol CP comparison (length {cp})")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ber(summary_rows: list[dict], path: str | Path) -> None:
    direct = sorted((row for row in summary_rows if row["method"] == "direct_mixture"),
                    key=lambda row: float(row["sir_re_db"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = [float(row["sir_re_db"]) for row in direct]
    y = [max(float(row["ber"]), 0.5 / int(row["num_bits"])) for row in direct]
    ax.semilogy(x, y, marker="o", label="Direct mixture (no separation)")
    ax.set(xlabel="Nominal target in-band SIR (dB)", ylabel="BER",
           title="Platform baseline only; no source-separation method was run")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
