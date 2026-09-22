from __future__ import annotations

import numpy as np

from ..config import PlatformConfig
from ..contracts import SourceRecord
from ..randomness import record_identity, rng_for
from .constellations import BITS_PER_SYMBOL, map_bits
from .ofdm import modulate_grid


def _random_grid(rng: np.random.Generator, num_symbols: int, num_subcarriers: int,
                 modulation: str) -> tuple[np.ndarray, np.ndarray]:
    width = BITS_PER_SYMBOL[modulation]
    bits = rng.integers(0, 2, size=(num_symbols, num_subcarriers, width), dtype=np.uint8)
    return map_bits(bits, modulation), bits


def generate_target(config: PlatformConfig, split: str, record_id: int,
                    dtype=np.complex128) -> tuple[SourceRecord, np.ndarray, np.ndarray]:
    grids, bits_by_frame, waves = [], [], []
    for frame in range(3):
        rng = rng_for(config.root_seed, f"target-data-frame-{frame}", split, record_id)
        grid, bits = _random_grid(rng, config.num_symbols, len(config.target_subcarriers), "QPSK")
        grids.append(grid)
        bits_by_frame.append(bits)
        waves.append(modulate_grid(grid, config.target_subcarriers, config.fft_size,
                                   config.cp_lengths, dtype=dtype))
    long_waveform = np.concatenate(waves)
    identity = record_identity("target", split, record_id)
    audit = {
        "parent_record_id": identity, "selected_subframe": 1,
        "long_waveform_length": int(long_waveform.size), "phase": 0.0,
    }
    record = SourceRecord(identity, identity, split,
                          long_waveform[config.window_length:2 * config.window_length], audit)
    return record, grids[1], bits_by_frame[1].reshape(-1)


def generate_interference(config: PlatformConfig, split: str, record_id: int,
                          dtype=np.complex128, *, force_tau: int | None = None,
                          force_phi: float | None = None,
                          force_modulations: tuple[tuple[str, ...], ...] | None = None,
                          data_seed_offset: int = 0) -> tuple[SourceRecord, np.ndarray, tuple[tuple[str, ...], ...]]:
    tones = config.interference_subcarriers
    block = int(config.raw["interference"]["block_size_subcarriers"])
    choices = tuple(config.raw["interference"]["modulations"])
    long_parts, maps = [], []
    for frame in range(3):
        modulation_rng = rng_for(config.root_seed, f"interference-modulation-frame-{frame}", split,
                                 record_id + data_seed_offset)
        mods = (tuple(force_modulations[frame]) if force_modulations is not None
                else tuple(str(x) for x in modulation_rng.choice(choices, size=4)))
        if len(mods) != 4 or any(m not in choices for m in mods):
            raise ValueError("each interference subframe requires four valid modulation labels")
        grid = np.empty((config.num_symbols, len(tones)), dtype=np.complex128)
        for block_index, modulation in enumerate(mods):
            data_rng = rng_for(config.root_seed,
                               f"interference-data-frame-{frame}-block-{block_index}", split,
                               record_id + data_seed_offset)
            values, _ = _random_grid(data_rng, config.num_symbols, block, modulation)
            grid[:, block_index * block:(block_index + 1) * block] = values
        long_parts.append(modulate_grid(grid, tones, config.fft_size, config.cp_lengths, dtype=dtype))
        maps.append(mods)
    long_waveform = np.concatenate(long_parts)
    tau_rng = rng_for(config.root_seed, "interference-tau", split, record_id + data_seed_offset)
    phi_rng = rng_for(config.root_seed, "interference-phi", split, record_id + data_seed_offset)
    tau = int(tau_rng.integers(0, config.window_length)) if force_tau is None else int(force_tau)
    phi = float(phi_rng.uniform(0, 2 * np.pi)) if force_phi is None else float(force_phi)
    if not 0 <= tau < config.window_length:
        raise ValueError("tau must be in [0, window_length-1]")
    start = config.window_length + tau
    stop = 2 * config.window_length + tau
    if stop > long_waveform.size:
        raise ValueError("long interference record is too short for requested crop")
    crop = np.exp(1j * phi) * long_waveform[start:stop]
    crop = crop.astype(dtype, copy=False)
    identity = record_identity("interference", split, record_id)
    audit = {
        "parent_record_id": identity, "tau": tau, "phi": phi,
        "modulation_map": maps, "crop_start": start, "crop_stop": stop,
        "long_waveform_length": int(long_waveform.size),
    }
    return SourceRecord(identity, identity, split, crop, audit), long_waveform, tuple(maps)
