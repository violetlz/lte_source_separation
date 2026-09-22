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

## Formal continuation (Task 03)

Formal training is a separate contract, not a relaxed pilot config. It explicitly
continues from each audited 2000-step checkpoint, preserving raw/EMA/Adam, cumulative
step/exposures and diffusion-noise RNG. The fixed-data sampler is a new stream and starts
at epoch 0 offset 0; the historical online cursor remains recorded separately.

Generate and validate the data before launch:

```bash
for source in target interference; do
  .venv-train/bin/python scripts/prepare_prior_dataset.py --source "$source" --split train
  .venv-train/bin/python scripts/prepare_prior_dataset.py --source "$source" --split validation
  .venv-train/bin/python scripts/validate_prior_dataset.py \
    artifacts/datasets/lte-prior-v1-parent32000-n200000-val1024/$source/train/manifest.json
  .venv-train/bin/python scripts/validate_prior_dataset.py \
    artifacts/datasets/lte-prior-v1-parent32000-n200000-val1024/$source/validation/manifest.json
done
```

Each training manifest contains 200,000 parents/rows with IDs `[32000,232000)`.
Validation contains 1,024 independent parents and both slots (2,048 rows). Training
refuses manifests not marked `validated`.

Profile one visible GPU at a time, then launch exactly one source process per physical GPU:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/profile_prior_training.py --source target
CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python scripts/profile_prior_training.py --source interference
.venv-train/bin/python scripts/run_prior_formal.py orchestrate
```

The launcher rejects identical GPU indices/UUIDs, occupied cards, duplicate live workers,
and workers that can see more than one GPU. Formal updates use effective batch 16,
microbatch 8, BF16, four persistent DataLoader workers, pinned memory and prefetch 2.
The recoverable sampler commits its 16 indices only after a successful optimizer update;
prefetched indices do not advance durable state.

Formal quick validation runs every 1,000 cumulative updates, full validation every
10,000, and `last.pt` every 2,000. The last three numbered ordinary checkpoints are
retained. Target/interference cumulative budgets are 200,000/300,000; learning rate stays
2e-4 through 80% of the cumulative budget and then cosine decays to 2e-5.

## Frozen-prior RGS

`load_prior_for_inference` loads only frozen raw/EMA inference state and validates model,
source, schedule, view and platform semantics. `separate_rgs` accepts only mixture,
public kappa, public initialization, frozen priors and an inference config; ground truth
is not in its API. Default `omega=kappa^2`, timesteps are sampled from 1..49 and the
learning rate follows a cosine schedule. Dev and functional protocols are frozen in
`configs/experiments/rgs_dev.yaml` and `rgs_functional.yaml`.

The 16-step pilot-checkpoint smoke is an engineering gate only:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python scripts/smoke_rgs.py \
  --target-checkpoint artifacts/prior_pilot/prior-pilot-20260922/target/checkpoints/last.pt \
  --interference-checkpoint artifacts/prior_pilot/prior-pilot-20260922/interference/checkpoints/last.pt \
  --output artifacts/runs/task03_rgs_smoke.json
```

It checks loading, freezing, finite scores/updates, estimate movement and the mixture
identity. It is not a claim of BER improvement. Full RGS selection remains dev-only and
formal test stays sealed until the final pair and hyperparameters are frozen.

A lightweight inference-only package (no Adam/RNG/dataset state) can be exported and is
immediately reloaded as a verification gate:

```bash
.venv-train/bin/python scripts/export_prior.py --source target --weights ema \
  --checkpoint <formal-or-pilot-checkpoint.pt> --output artifacts/runs/target_ema_export.pt
```
