from __future__ import annotations

from ..config import PlatformConfig
from ..contracts import Observation, ReceiverInfo, ReceiverOutput
from ..waveforms.receiver import receive_target


def direct_receive(observation: Observation, info: ReceiverInfo,
                   config: PlatformConfig) -> ReceiverOutput:
    return receive_target(observation, info, config)
