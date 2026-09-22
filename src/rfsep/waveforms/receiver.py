from __future__ import annotations

import numpy as np

from ..config import PlatformConfig
from ..contracts import Observation, ReceiverInfo, ReceiverOutput, assert_receiver_info_public
from .constellations import demap_symbols
from .ofdm import demodulate_waveform


def target_demodulate(waveform: np.ndarray, config: PlatformConfig) -> np.ndarray:
    return demodulate_waveform(waveform, config.target_subcarriers,
                               config.fft_size, config.cp_lengths)


def receive_target(observation: Observation, info: ReceiverInfo,
                   config: PlatformConfig) -> ReceiverOutput:
    assert_receiver_info_public(info)
    if info.platform_hash != config.config_hash:
        raise ValueError("ReceiverInfo platform hash does not match active platform")
    grid = target_demodulate(observation.y, config)
    bits = demap_symbols(grid, "QPSK").reshape(-1)
    return ReceiverOutput(observation.example_id, grid, bits)
