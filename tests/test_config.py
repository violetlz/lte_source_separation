import copy

import pytest
import yaml

from rfsep.config import load_platform


def test_unknown_and_unsupported_platform_fields_rejected(platform_config, tmp_path):
    raw = copy.deepcopy(platform_config.raw)
    raw["silent_option"] = 1
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown"):
        load_platform(path)
    raw = copy.deepcopy(platform_config.raw)
    raw["noise"] = "awgn"
    raw["noise_variance"] = 0.1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(NotImplementedError, match="zero physical noise"):
        load_platform(path)
