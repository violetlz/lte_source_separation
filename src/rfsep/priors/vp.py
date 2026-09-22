from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VPSchedule:
    steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.05

    def tensors(self, device=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta = torch.linspace(self.beta_start, self.beta_end, self.steps,
                              dtype=torch.float32, device=device)
        alpha = 1.0 - beta
        return beta, alpha, torch.cumprod(alpha, dim=0)

    def _coefficients(self, timestep: torch.Tensor, like: torch.Tensor):
        if timestep.ndim != 1 or timestep.shape[0] != like.shape[0]:
            raise ValueError("timestep must have one entry per batch sample")
        if torch.any(timestep < 0) or torch.any(timestep >= self.steps):
            raise ValueError("timestep outside diffusion schedule")
        alpha_bar = self.tensors(like.device)[2][timestep.long()].view(-1, 1, 1)
        return torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)

    def add_noise(self, x0: torch.Tensor, timestep: torch.Tensor,
                  epsilon: torch.Tensor) -> torch.Tensor:
        if x0.shape != epsilon.shape:
            raise ValueError("x0 and epsilon shapes differ")
        signal, noise = self._coefficients(timestep, x0)
        return signal * x0 + noise * epsilon

    def denoise(self, x_t: torch.Tensor, timestep: torch.Tensor,
                epsilon_hat: torch.Tensor) -> torch.Tensor:
        signal, noise = self._coefficients(timestep, x_t)
        return (x_t - noise * epsilon_hat) / signal

    def score_xt(self, epsilon_hat: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        _, noise = self._coefficients(timestep, epsilon_hat)
        return -epsilon_hat / noise

    def predict_noise(self, model, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return model.predict_noise(x_t, timestep)
