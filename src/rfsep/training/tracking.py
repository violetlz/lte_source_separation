from __future__ import annotations

import time


def init_wandb(training, run_id: str | None, resume: bool, metadata: dict):
    import wandb

    settings = training.raw["tracking"]
    if settings["mode"] != "online":
        raise ValueError("actual pilot requires W&B mode=online")
    kwargs = {}
    if run_id is not None:
        kwargs["id"] = run_id
    run = wandb.init(
        project=settings["project"], group=settings["group"], name=settings["name"],
        resume="must" if resume else "never", mode="online",
        config=metadata,
        tags=[training.raw["source"], "stage-pilot", "window-0.5ms",
              training.view.platform.raw["version"]],
        **kwargs,
    )
    run.define_metric("optimizer/global_step")
    run.define_metric("train/*", step_metric="optimizer/global_step")
    run.define_metric("validation/*", step_metric="optimizer/global_step")
    return run


def verify_server_history(run_path: str, *, timeout_seconds: float = 45.0) -> dict:
    """Confirm public API can see both real train and validation history."""
    import wandb

    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        remote = wandb.Api(timeout=15).run(run_path)
        history = remote.history(samples=500, pandas=False)
        train = any("train/noise_mse" in row for row in history)
        validation = any(any(key.startswith("validation/") for key in row) for row in history)
        last = {"train_visible": train, "validation_visible": validation,
                "history_rows": len(history), "run_path": run_path}
        if train and validation:
            return last
        time.sleep(5)
    return last
