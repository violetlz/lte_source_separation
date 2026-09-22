import copy

import numpy as np
import pytest
import torch
import yaml

from rfsep.config import load_platform, load_training, load_training_view
from rfsep.contracts import complex_to_network, network_to_complex
from rfsep.data.single_source import CursorSampler, SingleSourceDataset, selected_slot
from rfsep.models.diffwave import DiffWavePrior, parameter_count
from rfsep.priors.vp import VPSchedule


def test_network_coordinates_channels_first_batch_scale_and_dtype():
    x = np.array([[1 + 2j, -3 + 0.5j], [0.25 - 1j, 2 + 0j]], dtype=np.complex128)
    network = complex_to_network(x)
    assert network.shape == (2, 2, 2)
    assert network.dtype == np.float32
    assert np.allclose(np.mean(network ** 2), np.mean(np.abs(x) ** 2), rtol=1e-6)
    assert np.allclose(network_to_complex(network), x, rtol=1e-6, atol=1e-6)
    assert network_to_complex(network).dtype == np.complex64


def test_platform_effective_config_is_defensive_and_v1_is_strict(platform_config, tmp_path):
    before = platform_config.config_hash
    external = platform_config.raw
    external["target"]["num_rbs"] = 99
    assert platform_config.raw["target"]["num_rbs"] == 25
    assert platform_config.config_hash == before
    invalid = copy.deepcopy(platform_config.raw)
    invalid["sample_rate_hz"] = 61_440_000
    invalid["subcarrier_spacing_hz"] = 30_000
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(invalid))
    with pytest.raises(ValueError, match="v1 requires sample_rate_hz"):
        load_platform(path)


def test_training_hash_boundaries(project_root):
    target = load_training(project_root / "configs/training/target_prior_pilot.yaml")
    interference = load_training(project_root / "configs/training/interference_prior_pilot.yaml")
    assert target.view.platform.config_hash == interference.view.platform.config_hash
    assert target.view.config_hash == interference.view.config_hash
    assert target.config_hash != interference.config_hash
    assert "root_seed" not in target.view.platform.waveform_spec.__dataclass_fields__


def test_independent_datasets_window_after_parent_crop_and_sampler_restore(project_root):
    view = load_training_view(project_root / "configs/training/prior_view_v1.yaml")
    for source in ("target", "interference"):
        dataset = SingleSourceDataset(view, source, "train")
        first = dataset.window(7)
        second = dataset.window(7)
        assert first.network.shape == (2, 15360)
        assert first.network.dtype == np.float32
        assert first.offset in (0, 15360)
        assert first.slot_index == selected_slot(view.platform.root_seed, source, "train", 7)
        assert np.array_equal(first.network, second.network)
        assert first.parent_record_id.startswith(f"{source}:train:")
        if source == "interference":
            assert "tau" in first.audit and "phi" in first.audit
    sampler = CursorSampler("target")
    assert sampler.peek(3) == (0, 1, 2)
    sampler.commit(3)
    restored = CursorSampler("target")
    restored.load_state_dict(sampler.state_dict())
    assert restored.peek(2) == (3, 4)


@pytest.mark.parametrize("channels,expected", [(64, 1_913_730), (128, 5_079_810)])
def test_actual_diffwave_forward_backward_update_and_receptive_field(channels, expected):
    torch.manual_seed(3)
    model = DiffWavePrior(channels)
    assert parameter_count(model) == expected
    assert model.theoretical_receptive_field == 16381
    x = torch.randn(1, 2, 64)
    timestep = torch.tensor([24])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    before = model.output_projection.weight.detach().clone()
    output = model(x, timestep)
    assert output.shape == x.shape and torch.isfinite(output).all()
    torch.mean((output - torch.randn_like(output)) ** 2).backward()
    optimizer.step()
    assert not torch.equal(before, model.output_projection.weight)
    assert model(torch.randn(1, 2, 37), timestep).shape == (1, 2, 37)


def test_vp_epsilon_denoise_and_score_contract():
    schedule = VPSchedule()
    x0 = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    epsilon = torch.tensor([[[0.25, -0.5], [1.0, -1.5]]])
    timestep = torch.tensor([9])
    x_t = schedule.add_noise(x0, timestep, epsilon)
    assert torch.allclose(schedule.denoise(x_t, timestep, epsilon), x0, atol=1e-6)
    _, sigma = schedule._coefficients(timestep, x_t)
    assert torch.allclose(schedule.score_xt(epsilon, timestep), -epsilon / sigma)
    beta, _, alpha_bar = schedule.tensors()
    assert beta.shape == (50,) and alpha_bar.shape == (50,)
    assert torch.all(alpha_bar[1:] < alpha_bar[:-1])


def test_tiny_fixed_batch_overfit_smoke():
    torch.manual_seed(11)
    model = DiffWavePrior(8, residual_layers=2, dilation_cycle=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(2, 2, 32)
    target = torch.randn_like(x)
    timestep = torch.tensor([3, 7])
    losses = []
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x, timestep) - target) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.98
