from __future__ import annotations

import pytest

from rfsep.baselines.structured import TargetOFDMLMMSE
from rfsep.config import load_platform
from rfsep.priors.vp import VPSchedule


def test_target_lmmse_t24_theory_scale(project_root):
    platform = load_platform(project_root / "configs/platform/lte_v1.yaml")
    baseline = TargetOFDMLMMSE(platform, VPSchedule())
    assert baseline.theoretical_nmse(24) == pytest.approx(0.04737, abs=2e-4)
