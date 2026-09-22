from __future__ import annotations

import numpy as np

from ..contracts import Observation, ReceiverInfo, ReceiverOutput, WaveformSpec, assert_receiver_info_public
from .constellations import demap_symbols
from .ofdm import demodulate_waveform


def target_demodulate(waveform: np.ndarray, spec: WaveformSpec) -> np.ndarray:
    return demodulate_waveform(waveform, spec.target_subcarriers,
                               spec.fft_size, spec.cp_lengths)


def receive_target(observation: Observation, info: ReceiverInfo,
                   spec: WaveformSpec) -> ReceiverOutput:
    assert_receiver_info_public(info)
    if info.platform_hash != spec.platform_hash:
        raise ValueError("ReceiverInfo platform hash does not match active platform")
    grid = target_demodulate(observation.y, spec)
    bits = demap_symbols(grid, "QPSK").reshape(-1)
    return ReceiverOutput(observation.example_id, grid, bits)
