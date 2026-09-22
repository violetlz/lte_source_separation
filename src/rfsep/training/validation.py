from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
import torch

from ..contracts import network_to_complex
from ..data.single_source import SingleSourceDataset
from ..evaluation.metrics import bit_errors, evm_rms, nmse
from ..priors.vp import VPSchedule
from ..waveforms.constellations import demap_symbols
from ..waveforms.ofdm import demodulate_waveform, symbol_boundaries


def _validation_seed(source: str, parent_id: int, slot: int, timestep: int) -> int:
    body = f"prior-validation-v1|{source}|{parent_id}|{slot}|{timestep}".encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "little")


def _cp_error(estimate: np.ndarray, window, dataset: SingleSourceDataset) -> tuple[float, int]:
    spec = dataset.platform.waveform_spec
    if dataset.source == "target":
        starts = symbol_boundaries(spec.fft_size, spec.cp_lengths[:7])
    else:
        crop_start = int(window.audit["crop_start"]) + window.offset
        frame = spec.window_length
        base = symbol_boundaries(spec.fft_size, spec.cp_lengths)
        starts = []
        for frame_index in range(3):
            for begin, useful, stop in base:
                begin += frame_index * frame - crop_start
                useful += frame_index * frame - crop_start
                stop += frame_index * frame - crop_start
                if begin >= 0 and stop <= estimate.size:
                    starts.append((begin, useful, stop))
    errors = []
    powers = []
    for begin, useful, stop in starts:
        cp = useful - begin
        errors.append(np.mean(np.abs(estimate[begin:useful] - estimate[stop-cp:stop]) ** 2))
        powers.append(np.mean(np.abs(estimate[stop-cp:stop]) ** 2))
    if not errors:
        return math.nan, 0
    return float(np.sum(errors) / max(np.sum(powers), 1e-12)), len(errors)


@torch.no_grad()
def validate_model(model, dataset: SingleSourceDataset, schedule: VPSchedule,
                   noise_levels: list[int], device: torch.device, precision: str,
                   parent_limit: int | None = None) -> dict:
    model.eval()
    values = defaultdict(list)
    windows = dataset.validation_windows()
    if parent_limit is not None:
        windows = (window for window in windows if window.parent_id < parent_limit)
    autocast_enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for window in windows:
        x0 = torch.from_numpy(window.network).unsqueeze(0).to(device=device, dtype=torch.float32)
        truth_complex = window.waveform
        target_truth_grid = None
        target_truth_bits = None
        if dataset.source == "target":
            spec = dataset.platform.waveform_spec
            target_truth_grid = demodulate_waveform(truth_complex, spec.target_subcarriers,
                                                    spec.fft_size, spec.cp_lengths[:7])
            target_truth_bits = demap_symbols(target_truth_grid, "QPSK").reshape(-1)
        for level in noise_levels:
            generator = torch.Generator(device=device)
            generator.manual_seed(_validation_seed(dataset.source, window.parent_id,
                                                   window.slot_index, level))
            epsilon = torch.randn(x0.shape, generator=generator, device=device, dtype=torch.float32)
            timestep = torch.tensor([level], device=device, dtype=torch.long)
            x_t = schedule.add_noise(x0, timestep, epsilon)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_enabled):
                epsilon_hat = model.predict_noise(x_t, timestep)
            epsilon_hat = epsilon_hat.float()
            x0_hat = schedule.denoise(x_t, timestep, epsilon_hat)
            signal, _ = schedule._coefficients(timestep, x_t)
            direct = x_t / signal
            gaussian = signal * x_t
            prefix = f"t{level}"
            values[(prefix, "noise_mse")].append(torch.mean((epsilon_hat - epsilon) ** 2).item())
            values[(prefix, "zero_noise_mse")].append(torch.mean(epsilon ** 2).item())
            values[(prefix, "real_mse")].append(torch.mean((x0_hat - x0) ** 2).item())
            for name, estimate in (("model", x0_hat), ("direct", direct), ("gaussian", gaussian)):
                estimated_complex = network_to_complex(estimate[0].cpu().numpy())
                values[(prefix, f"{name}_complex_nmse")].append(nmse(estimated_complex, truth_complex))
                cp_value, cp_pairs = _cp_error(estimated_complex, window, dataset)
                values[(prefix, f"{name}_cp_error")].append(cp_value)
                values[(prefix, f"{name}_cp_pairs")].append(cp_pairs)
                if target_truth_grid is not None:
                    spec = dataset.platform.waveform_spec
                    grid = demodulate_waveform(estimated_complex, spec.target_subcarriers,
                                               spec.fft_size, spec.cp_lengths[:7])
                    bits = demap_symbols(grid, "QPSK").reshape(-1)
                    errors, count = bit_errors(bits, target_truth_bits)
                    values[(prefix, f"{name}_bit_errors")].append(errors)
                    values[(prefix, f"{name}_bits")].append(count)
                    values[(prefix, f"{name}_evm")].append(evm_rms(grid, target_truth_grid))
    output = {}
    for (level, metric), items in values.items():
        key = f"validation/{level}/{metric}"
        if metric.endswith("bit_errors") or metric.endswith("bits") or metric.endswith("cp_pairs"):
            output[key] = int(np.sum(items))
        else:
            finite = [value for value in items if np.isfinite(value)]
            output[key] = float(np.mean(finite)) if finite else math.nan
    output["validation/windows"] = sum(1 for window in dataset.validation_windows()
                                       if parent_limit is None or window.parent_id < parent_limit)
    for level in noise_levels:
        for name in ("model", "direct", "gaussian"):
            errors = output.get(f"validation/t{level}/{name}_bit_errors")
            bits = output.get(f"validation/t{level}/{name}_bits")
            if bits:
                output[f"validation/t{level}/{name}_ber"] = errors / bits
    return output
