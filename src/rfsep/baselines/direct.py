from __future__ import annotations

from ..contracts import Observation, ReceiverInfo, ReceiverOutput, WaveformSpec
from ..waveforms.receiver import receive_target


def direct_receive(observation: Observation, info: ReceiverInfo,
                   spec: WaveformSpec) -> ReceiverOutput:
    return receive_target(observation, info, spec)
