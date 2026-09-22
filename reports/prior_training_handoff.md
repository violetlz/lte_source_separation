# Prior pilot handoff

Updated: 2026-09-23 Asia/Shanghai.

## Implementation acceptance

- Physical platform hash: `18b2a1bc425f217e36c9e67f9fd64f13701f45c83148ec0ccba42205f8115cb5` (unchanged).
- Training-view hash: `6074de587b1e7a46f83afb062bfa8eaf5a8948f838e08bd5cd66afaa5178d0b9`.
- Target training hash: `43e8803dc42efdad0097968ab22e7e7289e17540e754bc01c9a044b5b5ce48fe`.
- Interference training hash: `94211f9a5a03eda6ee83f1be1e2cb720c9fff72f54734585d20f26ddcf84029f`.
- Test suite: 27 CPU-visible tests passed, plus the CUDA RNG-device regression passed separately. Waveform audit: 14/14 passed.
- Cross-process deterministic checkpoint test matches continuous CPU training exactly and rejects corrupt/incompatible state.

## Engineering benchmark

These are program-throughput measurements, not model-quality conclusions.

| Source | GPU | Precision | Microbatch × accumulation | Mean update | Samples/s | Peak reserved | 2,000-step compute estimate |
|---|---|---|---:|---:|---:|---:|---:|
| target | RTX 4090 | BF16 | 2 × 8 | 0.253 s | 63.21 | 0.78 GB | 506 s |
| interference | RTX 4090 | BF16 | 1 × 16 | 0.932 s | 17.17 | 0.64 GB | 1,864 s |

Interference uses activation checkpointing. Both preserve effective batch 16 and the full 15,360-sample window.

The stable runs began with a pre-existing CUDA 13/PyTorch 2.9.1 interpreter used read-only plus a project-local `.training-deps` W&B overlay; no package in the old environment was changed. A fully independent project environment, `.venv-train`, was then created from official CUDA 13 wheels and verified as Python 3.10.12 / PyTorch 2.9.1+cu130 / W&B 0.30.0 on the RTX 4090. Future manual resumes use `.venv-train`; model/checkpoint compatibility is unchanged.

## Online runs and background state

- Target: <https://wandb.ai/violetftrt-evergarden/lte-source-separation/runs/1za8kes6>
- Interference: <https://wandb.ai/violetftrt-evergarden/lte-source-separation/runs/2xif3tr0>
- Detached session: finished; no active `screen` session remains.
- Group status: `artifacts/prior_pilot/prior-pilot-20260922/status.json`

Final budget result:

| Source | State | Step / cursor | Resume count | Cumulative process time | Stable throughput / peak reserved | Final checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---|
| target | `completed_budget` | 2,000 / 32,000 | 2 | 743.5 s | 54.13 samples/s / 0.81 GB | `ec13a8f9bea188fa584643ef8ee31a82491ed73801b886d0792129fdb62f5826` |
| interference | `completed_budget` | 2,000 / 32,000 | 2 | 2,079.9 s | 16.42 samples/s / 0.70 GB | `82f56b4257f145bc862d89b41d4b8d3220107b029bae027462511f4ebc90aec7` |

Both public W&B API checks returned `train_visible=true` and `validation_visible=true` (71 and 50 history rows respectively at stability handoff). Each source completed an actual step-200 process exit and strict reload of its model, EMA, optimizer, RNG, cursor, budget, and original W&B run. Each then completed the authorized 2,000 optimizer updates, final 128-window validation, atomic final checkpoint, and final checkpoint artifact with exit code 0. No AMP skips, NaN/Inf, OOM, or cursor discontinuity was observed.

Both stopped on `max_updates`, well before the 7,200-second per-source process-time limit. The exact status, logs, checkpoints, and automatic `pilot_result.json` remain under ignored `artifacts/prior_pilot/prior-pilot-20260922/`; no model weight or large run artifact is committed.

## Preliminary learning signal

These are final full fixed-validation results over 128 windows, not convergence or separation claims:

| Source/model | Step | t | Noise MSE (zero baseline) | Model NMSE | Direct reverse-scale NMSE | Scalar Gaussian NMSE |
|---|---:|---:|---:|---:|---:|---:|
| target raw | 2,000 | 24 | 0.1464 (0.9999) | 0.0534 | 0.3645 | 0.2673 |
| target EMA | 2,000 | 24 | 0.2378 (0.9999) | 0.0867 | 0.3645 | 0.2673 |
| interference raw | 2,000 | 24 | 0.4855 (1.0004) | 0.1768 | 0.3643 | 0.2672 |
| interference EMA | 2,000 | 24 | 0.5917 (1.0004) | 0.2155 | 0.3643 | 0.2672 |

Both raw and EMA models beat the direct reverse-scale and scalar Gaussian NMSE baselines at t=24. CP discrepancy also improved at t=24: target EMA 0.1758 versus direct 0.5422, and interference EMA 0.4002 versus direct 0.5382. Communication metrics remain mixed: target EMA t=39 BER was 5,117/537,600 versus direct 5,118/537,600, while t=49 was slightly worse at 27,863 versus 27,847 errors; raw t=39 was also slightly worse, 5,159 versus 5,118. These unfavorable differences are retained. The evidence supports program stability and learning of the epsilon/NMSE objective, but not convergence, consistent BER improvement, source separation, or an FM/RGS advantage.

## Recovery notes

Two bounded pre-stability defects were retained rather than hidden: W&B 0.30 removed `wandb.util.generate_id`, and CUDA `map_location` moved saved CPU RNG ByteTensors to the GPU. Both failed before consuming a new sample. The implementation now lets `wandb.init` create new IDs and explicitly restores every RNG state on its required device; a CUDA regression was added. A later controlled save/restart corrected a status-machine bug that otherwise repeated checkpoint-artifact uploads. The old uploader processes were force-stopped only after atomic checkpoints and `exit_reason=signal` were recorded, then both runs resumed from those exact checkpoints. Failure evidence is preserved under `failed_attempts/` and worker logs.

## Operations

```bash
# Read the completed state.
cat artifacts/prior_pilot/prior-pilot-20260922/status.json
cat artifacts/prior_pilot/prior-pilot-20260922/target/status.json
cat artifacts/prior_pilot/prior-pilot-20260922/interference/status.json

# If a future authorized continuation changes the budget/config explicitly, use a new
# training identity. The following strict command only resumes interrupted in-budget runs;
# the current runs have already exhausted max_updates and will exit immediately.
screen -dmS lte-prior-pilot-20260922 bash -lc \
  'cd /data3/workspace/lab/lte_source_separation && \
   exec .venv-train/bin/python \
   scripts/run_prior_pilot.py orchestrate --resume-only \
   >> artifacts/prior_pilot/orchestrator.log 2>&1'
```

Program stability, learning of the pilot objective, convergence, and separation quality are distinct conclusions. This delivery establishes the first two only. The final full-validation metrics above must be reviewed before any downstream RGS work.
