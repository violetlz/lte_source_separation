from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 512):
        super().__init__()
        if dim % 2:
            raise ValueError("sinusoidal dimension must be even")
        self.dim = dim
        half = dim // 2
        self.register_buffer(
            "frequencies",
            torch.exp(torch.arange(half, dtype=torch.float32)
                      * (-math.log(10_000.0) / max(half - 1, 1))),
            persistent=False,
        )
        self.projection1 = nn.Linear(dim, hidden)
        self.projection2 = nn.Linear(hidden, hidden)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = timestep.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        encoding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        return F.silu(self.projection2(F.silu(self.projection1(encoding))))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, timestep_hidden: int):
        super().__init__()
        self.dilated = nn.Conv1d(channels, 2 * channels, 3, padding=dilation,
                                 dilation=dilation)
        self.diffusion_projection = nn.Linear(timestep_hidden, channels)
        self.output_projection = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, x: torch.Tensor, diffusion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x + self.diffusion_projection(diffusion).unsqueeze(-1)
        gate, filt = self.dilated(y).chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        residual, skip = self.output_projection(y).chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class DiffWavePrior(nn.Module):
    def __init__(self, residual_channels: int, residual_layers: int = 24,
                 dilation_cycle: int = 12, activation_checkpointing: bool = False):
        super().__init__()
        self.residual_channels = residual_channels
        self.residual_layers = residual_layers
        self.dilation_cycle = dilation_cycle
        self.activation_checkpointing = activation_checkpointing
        self.input_projection = nn.Conv1d(2, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(128, 512)
        self.blocks = nn.ModuleList([
            ResidualBlock(residual_channels, 2 ** (index % dilation_cycle), 512)
            for index in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def theoretical_receptive_field(self) -> int:
        return 1 + 2 * sum(2 ** (index % self.dilation_cycle)
                           for index in range(self.residual_layers))

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError("DiffWave input must have shape [B,2,L]")
        if timestep.shape != (x.shape[0],):
            raise ValueError("timestep must have shape [B]")
        h = F.relu(self.input_projection(x))
        embedding = self.diffusion_embedding(timestep)
        skip_sum = None
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                h, skip = checkpoint(block, h, embedding, use_reentrant=False)
            else:
                h, skip = block(h, embedding)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        h = skip_sum / math.sqrt(len(self.blocks))
        h = F.relu(self.skip_projection(h))
        return self.output_projection(h)

    def predict_noise(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self(x_t, timestep)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
