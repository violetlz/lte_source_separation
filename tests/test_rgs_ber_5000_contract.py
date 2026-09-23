from pathlib import Path

import numpy as np
import torch

from rfsep.baselines.lte_estimated import marginal_bit_map
from rfsep.priors.inference import FrozenPrior
from rfsep.priors.vp import VPSchedule
from rfsep.separation.rgs import RGSConfig, separate_rgs


class _ZeroModel:
    def predict_noise(self, x, timestep):
        return torch.zeros_like(x)


def _prior(source):
    return FrozenPrior(source, _ZeroModel(), VPSchedule(50, 0.0001, 0.05),
                       Path("none"), 0, "ema", {"platform_hash": "same"})


def test_sample_seed_invariant_to_batch_partition():
    y = torch.full((2, 2, 15360), 0.3)
    x0 = torch.zeros_like(y)
    target, interference = _prior("target"), _prior("interference")
    both, _ = separate_rgs(y, 2.0, x0, target, interference,
                           RGSConfig(3, 0.001, sample_seeds=(11, 22), diagnostic_interval=3))
    alone = []
    for index, seed in enumerate((11, 22)):
        result, _ = separate_rgs(y[index:index + 1], 2.0, x0[index:index + 1],
                                 target, interference,
                                 RGSConfig(3, 0.001, sample_seeds=(seed,), diagnostic_interval=3))
        alone.append(result)
    assert torch.equal(both, torch.cat(alone))


def test_bit_map_is_bitwise_and_finite_at_extreme_kappa():
    centers = np.linspace(0.001, 4, 2048)
    stats = {"r2_centers": centers, "r2_density": np.exp(-centers)}
    grid = np.array([[1 + 1j, -1 - 1j]])
    for kappa in (0.01, 100.0):
        result = marginal_bit_map(grid, kappa, stats)
        assert result.shape == (4,)
        assert np.isin(result, (0, 1)).all()


def test_snapshot_is_same_trajectory_and_does_not_change_final_state():
    y = torch.full((1, 2, 15360), 0.2)
    initial = torch.zeros_like(y)
    target, interference = _prior("target"), _prior("interference")
    plain, _ = separate_rgs(y, 1.8, initial, target, interference,
                            RGSConfig(3, 0.001, sample_seeds=(47,), diagnostic_interval=3))
    captured, audit = separate_rgs(y, 1.8, initial, target, interference,
                                   RGSConfig(3, 0.001, sample_seeds=(47,),
                                             diagnostic_interval=3, snapshot_iterations=(1, 2)))
    assert torch.equal(plain, captured)
    assert set(audit["snapshots"]) == {1, 2}
    assert all(state.shape == y.shape for state in audit["snapshots"].values())
    assert audit["target_prior_calls"] == audit["interference_prior_calls"] == 3
