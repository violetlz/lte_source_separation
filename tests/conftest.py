from pathlib import Path

import pytest

from rfsep.config import load_platform


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def platform_config():
    return load_platform(ROOT / "configs/platform/lte_v1.yaml")
