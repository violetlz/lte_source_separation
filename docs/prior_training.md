# Independent prior training

## Scope

Two models learn independent clean-source distributions. The target and interference use separate online data streams, model parameters, Adam optimizers, EMA states, checkpoints, W&B runs, and sample cursors. Neither model sees a mixture, the other source, SIR, bits, tau, phi, or modulation truth. Diffusion perturbations are a learning device and are not physical receiver AWGN.

The physical generator still emits 1 ms / 30,720 complex samples. Training uses only the versioned `prior_view_v1`: a deterministic front or back 0.5 ms slot / 15,360 samples. Target crops remain slot aligned. Interference is first asynchronously cropped and globally phase rotated by the physical generator, then halved.

## Model and objective

Both priors use 24 noncausal same-length gated residual blocks with kernel 3 and dilation `2**(i % 12)`. Target has 64 residual channels and 1,913,730 parameters; interference has 128 and 5,079,810. The timestep encoding is 128-dimensional sinusoidal input followed by 128→512→512, with an independent 512→C projection per block. Skip/residual paths use `sqrt(2)` scaling and the final two-channel projection starts at zero.

The theoretical receptive field is 16,381 samples, radius 8,190. This is a support bound: edge outputs see zero padding, not circular context; it does not prove that every output sees the full 15,360-sample window or that the effective receptive field has been learned. Variable-length forward calls are structural checks only; the learned distribution in this delivery is fixed to 15,360 samples.

VP epsilon prediction uses 50 linearly spaced beta values from 1e-4 to 0.05. `predict_noise`, `denoise`, and `score_xt` implement:

```text
x_t = sqrt(alpha_bar[t]) x_0 + sqrt(1-alpha_bar[t]) epsilon
x0_hat = (x_t - sqrt(1-alpha_bar[t]) eps_hat) / sqrt(alpha_bar[t])
score_xt = -eps_hat / sqrt(1-alpha_bar[t])
```

The score is with respect to real-coordinate `x_t`; it is not labeled as a clean-variable or RGS gradient.

## Commands

Create the verified project-local CUDA environment:

```bash
uv venv --python /usr/bin/python3.10 .venv-train
uv pip install --python .venv-train/bin/python 'torch==2.9.1' \
  --index https://download.pytorch.org/whl/cu130
uv pip install --python .venv-train/bin/python -e '.[test]' 'wandb==0.30.0'
.venv-train/bin/pytest
.venv-train/bin/python scripts/run_prior_pilot.py inspect \
  --config configs/training/target_prior_pilot.yaml
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/benchmark_prior.py \
  --config configs/training/target_prior_pilot.yaml --device cuda:0 \
  --output artifacts/prior_pilot/benchmarks/target.json
.venv-train/bin/python scripts/run_prior_pilot.py orchestrate
```

The orchestrator runs both GPUs to step 200, closes both processes, then resumes each from `last.pt` and the same online W&B run. Each source stops at 2,000 successful optimizer updates or 7,200 cumulative process seconds, whichever comes first. There is no automatic unbounded retry.

Detached launch used on the validated host:

```bash
screen -dmS lte-prior-pilot-20260922 bash -lc \
  'cd /data3/workspace/lab/lte_source_separation && \
   export PYTHONPATH=src:.training-deps && \
   exec .venv-train/bin/python \
   scripts/run_prior_pilot.py orchestrate >> artifacts/prior_pilot/orchestrator.log 2>&1'
```

Inspect or stop only this task:

```bash
screen -ls
cat artifacts/prior_pilot/prior-pilot-20260922/status.json
cat artifacts/prior_pilot/prior-pilot-20260922/target/status.json
cat artifacts/prior_pilot/prior-pilot-20260922/interference/status.json
.venv-train/bin/python scripts/stop_prior_pilot.py \
  --status artifacts/prior_pilot/prior-pilot-20260922/status.json
```

The stop helper validates every PID's live command line against this exact group before signaling the coordinator and workers. Workers handle SIGINT/SIGTERM at optimizer-step boundaries, save `last.pt`, flush W&B, and exit. A strict manual resume reuses the same worker command; the existing `last.pt` supplies the run ID, RNG states, sample cursor, optimizer, EMA, and cumulative budget.

## Validation interpretation

Fixed levels 0, 9, 24, 39, and 49 report raw and EMA epsilon MSE, zero-predictor MSE, real-coordinate MSE, complex NMSE, CP/tail discrepancy, and direct-rescaling/scalar-Gaussian baselines. Target validation additionally reports uncoded BER/EVM on the artificial diffusion perturbation; these are neither physical-noise BER nor separation BER. Best selection uses EMA fixed-validation epsilon MSE at t=24 and never uses the test split.

Step 0, 200, and budget end use all 128 windows (two slots from each of 64 parents); intermediate validation uses a fixed 16-window subset. Failures and unfavorable metrics are retained.
