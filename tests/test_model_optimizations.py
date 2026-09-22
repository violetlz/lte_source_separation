from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from rfsep.models.diffwave import DiffWavePrior


def _stack_forward(model, x, timestep):
    h = F.relu(model.input_projection(x))
    embedding = model.diffusion_embedding(timestep)
    skips = []
    for block in model.blocks:
        h, skip = block(h, embedding)
        skips.append(skip)
    h = torch.stack(skips).sum(dim=0) / math.sqrt(len(skips))
    return model.output_projection(F.relu(model.skip_projection(h)))


def test_cached_embedding_and_running_skip_match_old_computation():
    torch.manual_seed(5)
    model = DiffWavePrior(8, residual_layers=4, dilation_cycle=2)
    torch.nn.init.normal_(model.output_projection.weight, std=0.01)
    x_new = torch.randn(2, 2, 64, requires_grad=True)
    x_old = x_new.detach().clone().requires_grad_(True)
    timestep = torch.tensor([3, 27])
    new = model(x_new, timestep)
    old = _stack_forward(model, x_old, timestep)
    torch.testing.assert_close(new, old, rtol=2e-6, atol=2e-7)
    new.square().mean().backward()
    old.square().mean().backward()
    torch.testing.assert_close(x_new.grad, x_old.grad, rtol=3e-6, atol=3e-7)

    half = model.diffusion_embedding.dim // 2
    legacy_frequency = torch.exp(torch.arange(half, dtype=torch.float32)
                                 * (-math.log(10_000.0) / (half - 1)))
    torch.testing.assert_close(model.diffusion_embedding.frequencies, legacy_frequency,
                               rtol=0, atol=0)
