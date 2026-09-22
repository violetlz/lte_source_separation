from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..models.diffwave import DiffWavePrior
from ..training.checkpoint import atomic_torch_save
from .vp import VPSchedule


@dataclass(frozen=True)
class FrozenPrior:
    source: str
    model: DiffWavePrior
    schedule: VPSchedule
    checkpoint_path: Path
    global_step: int
    weights: str
    metadata: dict

    @torch.no_grad()
    def epsilon(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.model.predict_noise(x_t, timestep)

    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.schedule.denoise(x_t, timestep, self.epsilon(x_t, timestep))

    @torch.no_grad()
    def score_xt(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.schedule.score_xt(self.epsilon(x_t, timestep), timestep)


def load_prior_for_inference(path: str | Path, *, source: str, device: str | torch.device,
                             weights: str = "ema") -> FrozenPrior:
    """Load only frozen inference state; no optimizer, dataset, W&B, or trainer."""
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("source") != source or weights not in {"raw", "ema"}:
        raise ValueError("checkpoint source or requested weights mismatch")
    exported = payload.get("schema") == "rfsep-frozen-prior-v1"
    platform = payload.get("platform_config", {})
    view = payload.get("training_view_config", {})
    training = payload.get("training_config", {})
    model_cfg = training.get("model", {})
    diffusion = training.get("diffusion", {})
    expected_channels = 64 if source == "target" else 128
    required_model = {"residual_layers": 24, "residual_channels": expected_channels,
                      "dilation_cycle": 12, "in_channels": 2, "out_channels": 2,
                      "timestep_dim": 128, "timestep_hidden": 512, "kernel_size": 3}
    if any(model_cfg.get(key) != value for key, value in required_model.items()):
        raise ValueError("checkpoint model semantics mismatch")
    if (view.get("window_length") != 15360 or view.get("coordinate_scale") != "sqrt2_ri"
            or platform.get("sample_rate_hz") != 30720000
            or diffusion != {"steps": 50, "beta_start": 0.0001, "beta_end": 0.05,
                             "prediction": "epsilon"}):
        raise ValueError("checkpoint physical/view/diffusion semantics mismatch")
    model = DiffWavePrior(expected_channels, 24, 12, False)
    if exported and weights != payload.get("weights"):
        raise ValueError("export contains only the declared frozen weight variant")
    state = payload["state_dict"] if exported else (
        payload["model"] if weights == "raw" else payload["ema"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    schedule = VPSchedule(50, 0.0001, 0.05).to(device).eval()
    return FrozenPrior(source, model, schedule, path, int(payload["global_step"]), weights,
                       {"platform_hash": payload["platform_hash"],
                        "training_view_hash": payload["training_view_hash"],
                        "training_config_hash": payload["training_config_hash"]})


def export_prior_for_inference(checkpoint: str | Path, output: str | Path, *, source: str,
                               weights: str = "ema") -> str:
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Reuse the full loader as the semantic gate; it creates no trainer/optimizer/W&B/data.
    load_prior_for_inference(checkpoint, source=source, device="cpu", weights=weights)
    export = {"schema": "rfsep-frozen-prior-v1", "source": source, "weights": weights,
              "global_step": int(payload["global_step"]),
              "state_dict": payload["model"] if weights == "raw" else payload["ema"],
              "platform_config": payload["platform_config"],
              "platform_hash": payload["platform_hash"],
              "training_view_config": payload["training_view_config"],
              "training_view_hash": payload["training_view_hash"],
              "training_config": payload["training_config"],
              "training_config_hash": payload["training_config_hash"],
              "parent_checkpoint": str(checkpoint)}
    return atomic_torch_save(export, output)
