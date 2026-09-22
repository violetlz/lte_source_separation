import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from rfsep.data.manifests import assert_split_disjoint, build_manifest
from rfsep.waveforms.sources import generate_interference, generate_target
from rfsep.observation import build_example, calibration_identity


def test_access_order_and_source_stream_independence(platform_config):
    expected = {i: generate_target(platform_config, "test", i)[0].waveform for i in (0, 1, 7)}
    generate_interference(platform_config, "test", 1, data_seed_offset=999)
    for i in (7, 0, 1):
        assert np.array_equal(generate_target(platform_config, "test", i)[0].waveform, expected[i])


def test_process_restart_reproducibility(platform_config, project_root):
    expected = hashlib.sha256(generate_target(platform_config, "test", 17)[0].waveform.tobytes()).hexdigest()
    code = """import hashlib
from rfsep.config import load_platform
from rfsep.waveforms.sources import generate_target
c=load_platform('configs/platform/lte_v1.yaml')
print(hashlib.sha256(generate_target(c,'test',17)[0].waveform.tobytes()).hexdigest())
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=project_root, text=True,
                            capture_output=True, check=True)
    assert result.stdout.strip() == expected


def test_split_disjointness_and_collision_detection(platform_config):
    manifests = [build_manifest(platform_config, split, 4)
                 for split in ("train", "validation", "test", "calibration", "audit")]
    assert_split_disjoint(manifests)
    bad = build_manifest(platform_config, "validation", 1)
    bad["records"][0]["target_parent"] = "target:train:00000000"
    with pytest.raises(ValueError, match="underlying long record"):
        assert_split_disjoint([build_manifest(platform_config, "train", 1), bad])


def test_changing_sir_changes_only_kappa_and_observation(platform_config):
    calibration = {"platform_version": platform_config.raw["version"],
                   "platform_hash": platform_config.config_hash, "q_b": 0.25,
                   "provisional": True}
    calibration["calibration_id"] = calibration_identity(calibration)
    obs_a, info_a, truth_a = build_example(platform_config, "audit", 12, -3, calibration)
    obs_b, info_b, truth_b = build_example(platform_config, "audit", 12, 6, calibration)
    assert info_a.kappa != info_b.kappa
    assert np.array_equal(truth_a.target_waveform, truth_b.target_waveform)
    assert np.array_equal(truth_a.interference_waveform, truth_b.interference_waveform)
    assert np.array_equal(truth_a.target_bits, truth_b.target_bits)
    assert not np.array_equal(obs_a.y, obs_b.y)
