import numpy as np

from rfsep.baselines.direct import direct_receive
from rfsep.contracts import Observation
from rfsep.observation import build_example, calibration_identity, mix_time_domain
from rfsep.waveforms.receiver import receive_target


def test_clean_and_oracle_noiseless_closed_loop(platform_config):
    calibration = {"platform_version": platform_config.raw["version"],
                   "platform_hash": platform_config.config_hash, "q_b": 0.25,
                   "provisional": True}
    calibration["calibration_id"] = calibration_identity(calibration)
    observation, info, truth = build_example(platform_config, "audit", 2, -6, calibration)
    expected = truth.target_waveform + info.kappa * truth.interference_waveform
    assert np.max(np.abs(observation.y - expected)) < 1e-12
    clean = receive_target(Observation(observation.example_id, truth.target_waveform), info,
                           platform_config.waveform_spec)
    oracle_waveform = observation.y - info.kappa * truth.interference_waveform
    oracle = receive_target(Observation(observation.example_id, oracle_waveform), info,
                            platform_config.waveform_spec)
    direct = direct_receive(observation, info, platform_config.waveform_spec)
    assert np.array_equal(clean.target_bits, truth.target_bits)
    assert np.array_equal(oracle.target_bits, truth.target_bits)
    assert direct.target_bits.shape == truth.target_bits.shape


def test_mixing_rejects_grid_shaped_inputs(platform_config):
    target = np.ones((14, 300), dtype=np.complex128)
    interference = np.ones_like(target)
    import pytest
    with pytest.raises(ValueError, match="time-domain arrays"):
        mix_time_domain(target, interference, 1.0)
