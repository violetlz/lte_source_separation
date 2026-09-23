# Formal LTE prior training results

Verified 2026-09-23 (Asia/Shanghai). These are fixed validation results for the
two independently trained source priors. The complete full-validation series is
in [formal_validation_full_20260923.csv](formal_validation_full_20260923.csv):
20 target steps and 30 interference steps, each evaluated on 2,048 windows
(both slots of 1,024 validation parents). Each row contains raw and EMA
noise MSE, reconstruction MSE, and NMSE at diffusion steps
`[0, 1, 4, 9, 24, 39, 49]`. The target run logged its final 200,000-step
full validation twice with identical values; the CSV keeps one row.

## Completion and provenance

| Source | Final cumulative step | Formal / cumulative sample exposures | Final checkpoint SHA-256 | Best-denoising checkpoint SHA-256 | W&B |
|---|---:|---:|---|---|---|
| Target | 200,000 / 200,000 | 3,168,000 / 3,200,000 | `bba67840419fd5951b8e2ac3a70f8be84a2efaa720d637c9e6f925a0418278be` | `2c35b5d7e7cc710b4e60839eaa3e033af48a6d5b173ecff3ba48d08dc0dd4890` | [ewq62hpj](https://wandb.ai/violetftrt-evergarden/lte-source-separation/runs/ewq62hpj) |
| Interference | 300,000 / 300,000 | 4,768,000 / 4,800,000 | `0d2270eaa46886afc9da4571e92fff042af78b348e20d185b1c531c634b13ad0` | `4ed7592871cdcfcbab8f461e74a2ff81d2118a0e5e731e2c4a35f791e2b63bf6` | [pyq671jm](https://wandb.ai/violetftrt-evergarden/lte-source-separation/runs/pyq671jm) |

Both local status files say `completed` with `exit_reason=budget`, and the
final checkpoint hashes above match the files. Their first 2,000 steps and
32,000 exposures came from the respective independent pilots. The target
formal job used one GPU. Interference was gracefully paused at step 95,293,
then continued on two NCCL ranks from that checkpoint with an unchanged
global batch of 16, optimizer, EMA, sampler cursor, training data, and
300,000-step budget. After the switch, each rank's RNG state was checkpointed.
The new noise stream is reproducible from the two-rank checkpoint, though it
is not bitwise identical to a hypothetical single-GPU continuation.

## Final full validation

| Source | Raw mean noise MSE | EMA mean noise MSE | EMA t24 NMSE | EMA t39 NMSE | EMA t49 NMSE |
|---|---:|---:|---:|---:|---:|
| Target, step 200,000 | 0.135499 | **0.135397** | 0.047595 | 0.145347 | 0.259721 |
| Interference, step 300,000 | 0.460265 | **0.460167** | 0.166755 | 0.404509 | 0.582705 |

Both best-denoising checkpoints were selected at their final budget step.
The EMA mean noise MSE changed as follows; lower is better:

| Source | 10k | 50k | 100k | 200k | 300k |
|---|---:|---:|---:|---:|---:|
| Target | 0.151320 | 0.140666 | 0.137099 | **0.135397** | — |
| Interference | 0.490692 | 0.462168 | 0.462455 | 0.460586 | **0.460167** |

From 10k to the final step, the fixed-validation EMA mean noise MSE fell
10.5% for target and 6.2% for interference. The target improved throughout;
the interference curve has small non-monotone segments, including 50k to
100k, which remain in the CSV. Late improvements are small. These results
establish completion and denoising performance on the validation set. They
do not establish final source-separation SIR or BER, a physical AWGN result,
or sealed-test performance. Those require the separate frozen-prior RGS
evaluation protocol.

Regenerate the CSV from the local ignored run artifacts with
`python scripts/export_formal_training_results.py`. The exporter verifies
completion, sample counts, every expected full-validation step, and both
final checkpoint hashes. Model weights, datasets, raw logs, and detailed
run artifacts remain local under ignored `artifacts/`; this repository
contains the code, configs, result summary, and validation series only.
