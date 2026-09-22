import json

import numpy as np
import pytest

from rfsep.observation import calibrate, load_calibration
from rfsep.waveforms.sources import generate_interference, generate_target


def test_fixed_ensemble_power_has_natural_record_variation(platform_config):
    target = np.array([np.mean(np.abs(generate_target(platform_config, "audit", i)[0].waveform) ** 2)
                       for i in range(24)])
    interference = np.array([np.mean(np.abs(generate_interference(platform_config, "audit", i)[0].waveform) ** 2)
                             for i in range(24)])
    for values in (target, interference):
        se = values.std(ddof=1) / np.sqrt(values.size)
        assert abs(values.mean() - 1) <= max(0.02, 5 * se)
        assert values.std() > 0


def test_calibration_and_hash_mismatch_rejected(platform_config, tmp_path):
    path = tmp_path / "calibration.json"
    payload = calibrate(platform_config, 64, path, provisional=True, target_check_records=8)
    assert abs(payload["q_b"] - 0.25) < 0.04
    assert load_calibration(path, platform_config, allow_provisional=True)["calibration_id"] == payload["calibration_id"]
    changed = json.loads(path.read_text())
    changed["platform_hash"] = "0" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="platform hash mismatch"):
        load_calibration(path, platform_config, allow_provisional=True)
