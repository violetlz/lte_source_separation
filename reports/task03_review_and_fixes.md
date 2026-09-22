# Task 03 review and fixes

Date: 2026-09-23 (Asia/Shanghai)

This report distinguishes engineering acceptance from model quality. The formal jobs are
running and the RGS smoke test is an interface test; neither is evidence that final BER
has improved.

## Audited starting point

- Repository start: `2c9b42321ebb5ec58209d46ad34a710ba182aaa1`.
- Both pilot checkpoints were readable and had `global_step=2000`, cursor/exposures
  `32000`, raw weights, EMA, Adam and RNG state.
- Parent SHA256: target `ec13a8f9...f5826`; interference `82f56b42...aec7`.
- Physical platform hash remained
  `18b2a1bc425f217e36c9e67f9fd64f13701f45c83148ec0ccba42205f8115cb5`.
- The inherited report's full-validation values were not rerun on a training GPU after
  formal launch. The real checkpoint was independently loaded for the RGS smoke test.

## Review findings and changes

| Finding | Implemented correction | Evidence |
|---|---|---|
| Online NumPy generation in every microbatch | Materialized `.npy` shards, read-only mmap Dataset, deterministic recoverable epoch sampler, DataLoader prefetch | `materialized.py`, dataset validation report |
| Tiny microbatches and interference checkpointing | Measured 2/4/8/16; chose 8 for both; checkpointing off | performance report |
| Frequent CUDA/Python synchronization | One loss/gradient finite decision before optimizer commit; no per-step synchronize in formal loop; scalar logging only every 50 updates | `formal.py` |
| VP schedule and timestep frequencies recreated | Registered FP32 schedule buffers and a numerically identical non-persistent frequency buffer | tests |
| 24 skip tensors stacked | Equivalent running skip sum | tests |
| Validation data/raw/EMA repeated | Fixed validation mmap data, batched levels, shared `x0/x_t/epsilon` across raw and EMA | `formal.py` |
| Non-finite values filtered | Formal validation and dataset validation fail on any non-finite value | `formal.py`, validator |
| Single t24/EMA best | Formal full validation selects `best_denoising.pt` using the better predeclared raw/EMA multi-noise mean; RGS pair selection remains dev-only | config and formal trainer |
| Pilot-only config and strict whole-hash resume | Separate formal contract; explicit audited `continue-from`; strict same-formal resume | configs and `_audit_pilot` |
| W&B step collisions | Monotonic `tracking/event_sequence`; optimizer step is a separate metric axis | formal trainer |
| Failure could overwrite `last.pt` | Exception path never serializes possibly damaged live state; save refuses non-finite model parameters | formal trainer |
| Same-GPU assignment and weak launch checks | Physical index and UUID uniqueness, availability check, exactly-one-visible-device assertion, duplicate PID rejection | formal launcher |

## Semantics preserved

The 24-layer DiffWave structure, dilation cycle 12, target/interference widths 64/128,
current timestep-frequency definition, VP50 schedule, `sqrt(2)[Re,Im]`, length 15360,
and source/platform generation rules are unchanged. No per-sample normalization, phase
alignment, physical noise, FM, DDP or multi-GPU model was introduced.

## RGS contract

`load_prior_for_inference` loads raw or EMA only and validates the source, model,
platform/view coordinates and schedule. It does not construct a trainer, optimizer,
dataset, or W&B run. `separate_rgs` implements

`g=(a_t/sigma_t)(eps_s-z_s)-(omega/kappa)(a_u/sigma_u)(eps_b-z_b)`

with `B=(Y-S)/kappa`, default `omega=kappa^2`, timesteps 1..49, cosine learning rate,
frozen priors and no truth argument. Analytic Gaussian tests cover non-unit kappa,
`omega=1`, `omega=kappa^2`, deterministic Monte Carlo and convergence near the known
optimum. The real 2000-step checkpoint smoke covers 12 cases and is deliberately labeled
engineering-only.

Primary references used for the implementation were the
[alpha-RGS paper](https://arxiv.org/html/2306.14411v3), the author's
[`args.py`](https://github.com/tkj516/score_based_source_separation/blob/main/source_separation/args.py)
and [`run_separation.py`](https://github.com/tkj516/score_based_source_separation/blob/main/source_separation/run_separation.py),
plus the [PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
and [profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html).

## Verification

- CPU/unit suite: 32 passed.
- Dataset: four manifests validated, including full-shard finite/shape/hash scans and
  deterministic regeneration across shard boundaries.
- RGS smoke: passed, all parameters frozen, all outputs finite, all estimates moved,
  maximum mixture reconstruction residual `1.91e-6`.
- Formal handoff: both sources exceeded +200 successful updates with new validation,
  known-good formal checkpoint and remotely visible W&B train/validation history.

The exact 300-dimensional OFDM LMMSE and train-PSD Wiener baseline are implemented in
`baselines/structured.py`; the t24 theoretical scale test gives approximately 0.04737.
The running formal handoff's fast validation currently records batched multi-noise
denoising metrics; the full communication-metric evaluator and end-of-training RGS/BER
remain scheduled work and are not claimed complete here.
