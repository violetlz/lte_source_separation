from dataclasses import replace

import numpy as np

from rfsep.baselines.direct import direct_receive
from rfsep.contracts import Observation, ReceiverInfo, complex_to_network, network_to_complex
from rfsep.observation import build_example, calibration_identity


def _calibration(platform_config):
    payload = {"platform_version": platform_config.raw["version"],
               "platform_hash": platform_config.config_hash, "q_b": 0.25,
               "provisional": True}
    payload["calibration_id"] = calibration_identity(payload)
    return payload


def test_receiver_info_has_no_truth_and_ground_truth_mutation_cannot_change_output(platform_config):
    observation, info, truth = build_example(platform_config, "audit", 4, 0, _calibration(platform_config))
    forbidden = {"target_bits", "target_waveform", "interference_waveform", "tau", "phi", "modulation_map"}
    assert not (set(ReceiverInfo.__dataclass_fields__) & forbidden)
    first = direct_receive(observation, info, platform_config.waveform_spec)
    altered_truth = replace(truth, target_bits=1 - truth.target_bits, tau=(truth.tau + 1) % 30720)
    second = direct_receive(observation, info, platform_config.waveform_spec)
    assert altered_truth.target_bits.shape == truth.target_bits.shape
    assert np.array_equal(first.target_bits, second.target_bits)
    assert np.array_equal(first.target_grid, second.target_grid)


def test_network_coordinate_roundtrip():
    x = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex128)
    assert np.allclose(network_to_complex(complex_to_network(x)), x, atol=1e-15)


def test_receiver_operates_without_truth(platform_config):
    observation, info, _ = build_example(platform_config, "audit", 6, 3, _calibration(platform_config))
    del _
    output = direct_receive(Observation(observation.example_id, observation.y.copy()), info,
                            platform_config.waveform_spec)
    assert output.target_bits.shape == (8400,)
