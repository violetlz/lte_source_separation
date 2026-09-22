from __future__ import annotations

from dataclasses import dataclass

import torch

from rfsep.priors.vp import VPSchedule
from rfsep.separation.rgs import RGSConfig, separate_rgs


@dataclass
class GaussianPrior:
    source: str
    mean: float
    variance: float
    schedule: VPSchedule

    def __post_init__(self):
        self.metadata = {"platform_hash": "analytic"}

    def epsilon(self, x_t, timestep):
        a, sigma = self.schedule._coefficients(timestep, x_t, validate=False)
        variance_t = a * a * self.variance + sigma * sigma
        return sigma / variance_t * (x_t - a * self.mean)


def test_rgs_gaussian_expected_direction_and_optimum():
    schedule = VPSchedule()
    target = GaussianPrior("target", 0.7, 0.4, schedule)
    background = GaussianPrior("interference", -0.2, 1.1, schedule)
    kappa, omega = 1.7, 1.0
    y = torch.full((1, 2, 15360), 0.9)
    # Gaussian MAP optimum in unsmoothed coordinates.
    optimum = ((target.mean / target.variance)
               + omega * (y[0, 0, 0].item() / kappa - background.mean)
               / (kappa * background.variance)) / (
                   1 / target.variance + omega / (kappa * kappa * background.variance))
    initial = torch.zeros_like(y)
    estimate, audit = separate_rgs(
        y, kappa, initial, target, background,
        RGSConfig(iterations=2000, max_lr=2e-3, min_lr=2e-4,
                  omega=omega, seed=13),
    )
    assert audit["reconstruction_max_abs"] < 1e-6
    assert abs(float(estimate.mean()) - optimum) < 0.08


def test_rgs_supports_kappa_squared_weight_and_is_deterministic():
    schedule = VPSchedule()
    target = GaussianPrior("target", 0.1, 0.8, schedule)
    background = GaussianPrior("interference", 0.4, 0.5, schedule)
    y = torch.full((1, 2, 15360), -0.3)
    config = RGSConfig(iterations=12, max_lr=1e-3, omega=None, seed=99)
    first, first_audit = separate_rgs(y, 2.0, torch.zeros_like(y), target, background, config)
    second, second_audit = separate_rgs(y, 2.0, torch.zeros_like(y), target, background, config)
    assert torch.equal(first, second)
    assert first_audit["omega"] == 4.0
    assert second_audit["reconstruction_max_abs"] < 1e-6
