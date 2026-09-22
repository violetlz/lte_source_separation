from __future__ import annotations

import torch
from torch import nn


class VPSchedule(nn.Module):
    """Variance-preserving schedule with device-cached FP32 coefficients."""

    def __init__(self, steps: int = 50, beta_start: float = 1e-4,
                 beta_end: float = 0.05):
        super().__init__()
        self.steps = int(steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        beta = torch.linspace(self.beta_start, self.beta_end, self.steps,
                              dtype=torch.float32)
        alpha = 1.0 - beta
        self.register_buffer("beta", beta, persistent=True)
        self.register_buffer("alpha", alpha, persistent=True)
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0), persistent=True)

    def tensors(self, device=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None or self.beta.device == torch.device(device):
            return self.beta, self.alpha, self.alpha_bar
        return self.beta.to(device), self.alpha.to(device), self.alpha_bar.to(device)

    def _coefficients(self, timestep: torch.Tensor, like: torch.Tensor,
                      *, validate: bool = True):
        if timestep.ndim != 1 or timestep.shape[0] != like.shape[0]:
            raise ValueError("timestep must have one entry per batch sample")
        if validate and (int(timestep.min()) < 0 or int(timestep.max()) >= self.steps):
            raise ValueError("timestep outside diffusion schedule")
        alpha_bar = self.alpha_bar.to(like.device)[timestep.long()].view(-1, 1, 1)
        return torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)

    def add_noise(self, x0: torch.Tensor, timestep: torch.Tensor,
                  epsilon: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
        if x0.shape != epsilon.shape:
            raise ValueError("x0 and epsilon shapes differ")
        signal, noise = self._coefficients(timestep, x0, validate=validate)
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
