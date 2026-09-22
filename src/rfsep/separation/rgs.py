from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..priors.inference import FrozenPrior


@dataclass(frozen=True)
class RGSConfig:
    iterations: int
    max_lr: float
    min_lr: float = 1e-6
    omega: float | None = None
    timestep_min: int = 1
    timestep_max: int = 49
    seed: int = 0


@torch.no_grad()
def separate_rgs(y: torch.Tensor, kappa: float, initial_s: torch.Tensor,
                 target_prior: FrozenPrior, interference_prior: FrozenPrior,
                 config: RGSConfig) -> tuple[torch.Tensor, dict]:
    """Waveform alpha-RGS in sqrt(2)[Re,Im] coordinates.

    Truth is deliberately absent from this API. Both priors are evaluated under
    ``no_grad``; the update is the explicitly derived score direction rather than
    denoiser-Jacobian backpropagation.
    """
    if y.shape != initial_s.shape or y.ndim != 3 or y.shape[1:] != (2, 15360):
        raise ValueError("RGS expects equal [B,2,15360] mixture and initialization")
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if target_prior.source != "target" or interference_prior.source != "interference":
        raise ValueError("RGS priors are assigned to the wrong sources")
    if target_prior.metadata["platform_hash"] != interference_prior.metadata["platform_hash"]:
        raise ValueError("RGS prior platform hashes differ")
    if not 1 <= config.timestep_min <= config.timestep_max <= 49:
        raise ValueError("RGS timesteps must be inside trained levels 1..49")
    omega = kappa * kappa if config.omega is None else float(config.omega)
    generator = torch.Generator(device=y.device).manual_seed(config.seed)
    s = initial_s.detach().float().clone()
    update_norms = []
    for iteration in range(config.iterations):
        batch = y.shape[0]
        t = torch.randint(config.timestep_min, config.timestep_max + 1, (batch,),
                          device=y.device, generator=generator)
        u = torch.randint(config.timestep_min, config.timestep_max + 1, (batch,),
                          device=y.device, generator=generator)
        z_s = torch.randn(s.shape, device=s.device, dtype=torch.float32, generator=generator)
        z_b = torch.randn(s.shape, device=s.device, dtype=torch.float32, generator=generator)
        a_t, sigma_t = target_prior.schedule._coefficients(t, s, validate=False)
        a_u, sigma_u = interference_prior.schedule._coefficients(u, s, validate=False)
        b = (y - s) / kappa
        s_t, b_u = a_t * s + sigma_t * z_s, a_u * b + sigma_u * z_b
        eps_s = target_prior.epsilon(s_t, t).float()
        eps_b = interference_prior.epsilon(b_u, u).float()
        gradient = ((a_t / sigma_t) * (eps_s - z_s)
                    - (omega / kappa) * (a_u / sigma_u) * (eps_b - z_b))
        phase = iteration / max(config.iterations - 1, 1)
        lr = config.min_lr + 0.5 * (config.max_lr - config.min_lr) * (1.0 + math.cos(math.pi * phase))
        s = s - lr * gradient
        if not torch.isfinite(s).all():
            raise FloatingPointError(f"non-finite RGS state at iteration {iteration}")
        update_norms.append(float(torch.sqrt(torch.mean((lr * gradient) ** 2))))
    b = (y - s) / kappa
    reconstruction = torch.max(torch.abs(y - (s + kappa * b))).item()
    return s, {"iterations": config.iterations, "omega": omega,
               "max_lr": config.max_lr, "min_lr": config.min_lr,
               "mean_update_rms": sum(update_norms) / len(update_norms),
               "reconstruction_max_abs": reconstruction}
