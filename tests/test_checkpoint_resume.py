import os
import subprocess
import sys

import pytest
import torch

from rfsep.training.checkpoint import load_checkpoint
from rfsep.training.checkpoint import restore_rng_state, rng_state


def _run(project_root, *arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    subprocess.run([sys.executable, str(project_root / "tests/helpers/checkpoint_determinism.py"),
                    *map(str, arguments)], cwd=project_root, env=environment, check=True)


def test_cross_process_checkpoint_resume_matches_continuous(project_root, tmp_path):
    baseline_checkpoint = tmp_path / "baseline.pt"
    resumed_checkpoint = tmp_path / "resumed.pt"
    baseline_result = tmp_path / "baseline_result.pt"
    resumed_result = tmp_path / "resumed_result.pt"
    _run(project_root, "--checkpoint", baseline_checkpoint, "--stop", 4,
         "--result", baseline_result)
    _run(project_root, "--checkpoint", resumed_checkpoint, "--stop", 2)
    _run(project_root, "--checkpoint", resumed_checkpoint, "--stop", 4, "--resume",
         "--result", resumed_result)
    baseline = torch.load(baseline_result, weights_only=False)
    resumed = torch.load(resumed_result, weights_only=False)
    assert baseline["global_step"] == resumed["global_step"] == 4
    assert torch.equal(baseline["next_sample"], resumed["next_sample"])
    for name in baseline["model"]:
        assert torch.equal(baseline["model"][name], resumed["model"][name])


def test_corrupt_and_incompatible_checkpoint_are_rejected(tmp_path):
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="unreadable"):
        load_checkpoint(corrupt, training_hash="x", source="target")
    incompatible = tmp_path / "wrong.pt"
    torch.save({"schema_version": 1, "training_config_hash": "other", "source": "target"},
               incompatible)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(incompatible, training_hash="x", source="target")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rng_restore_survives_cuda_map_location(tmp_path):
    generator = torch.Generator(device="cuda").manual_seed(44)
    path = tmp_path / "rng.pt"
    torch.save(rng_state(generator), path)
    mapped = torch.load(path, map_location="cuda", weights_only=False)
    restored = torch.Generator(device="cuda")
    restore_rng_state(mapped, restored)
    assert torch.equal(generator.get_state(), restored.get_state())
