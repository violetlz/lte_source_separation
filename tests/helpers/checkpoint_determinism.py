from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rfsep.training.checkpoint import (SCHEMA_VERSION, atomic_torch_save, load_checkpoint,
                                       restore_rng_state, rng_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result")
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(123)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(456)
    step = 0
    checkpoint = Path(args.checkpoint)
    if args.resume:
        payload = load_checkpoint(checkpoint, training_hash="test-hash", source="target")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng"], generator)
        step = payload["global_step"]
    while step < args.stop:
        x = torch.randn((5, 4), generator=generator)
        y = torch.randn((5, 3), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
        step += 1
    payload = {
        "schema_version": SCHEMA_VERSION, "source": "target",
        "training_config_hash": "test-hash", "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "rng": rng_state(generator),
        "global_step": step,
    }
    atomic_torch_save(payload, checkpoint)
    if args.result:
        next_sample = torch.randn((2, 4), generator=generator)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "next_sample": next_sample, "global_step": step}, args.result)


if __name__ == "__main__":
    main()
