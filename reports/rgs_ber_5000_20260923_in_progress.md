# LTE waveform α-RGS 5000-step BER: run in progress

This is a progress record, **not a completed four-curve result**. The frozen no-noise
`lte_v1` test uses `y=s+κb`, 11 SIR_RE points `[-24,-21,-18,-15,-12,-9,-6,-3,0,3,6]`,
test parent IDs `2000000..2000255`, and 2,150,400 target bits per SIR/method.
No test labels were used for baseline fitting, RGS initialization, or LR choice.

## Input and method boundary

- Target final EMA checkpoint: 200,000 steps, SHA-256
  `bba67840419fd5951b8e2ac3a70f8be84a2efaa720d637c9e6f925a0418278be`.
- Interference final EMA checkpoint: 300,000 steps, SHA-256
  `0d2270eaa46886afc9da4571e92fff042af78b348e20d185b1c531c634b13ad0`.
- Platform hash: `18b2a1bc425f217e36c9e67f9fd64f13701f45c83148ec0ccba42205f8115cb5`;
  calibration is the non-provisional formal `reports/formal_validation/calibration_summary.json`.
- Direct uses the unchanged target CP/FFT/QPSK receiver.
- `fft_diagonal_wiener_approx` uses train-only source periodogram estimates and a
  diagonal whole-record FFT Wiener transfer. Because CP and aligned target symbols
  make the exact covariance nonstationary, **this is not the exact two-source LMMSE**.
- `single_re_marginal_bit_map_estimated` uses train-only interference RE squared-radius
  density after `target_demodulate`, with 1,024 train parents (`500000..501023`),
  4,300,800 RE samples, 2,048 histogram bins and Gaussian smoothing of eight bins.
  Source file SHA-256: `53ee7ee8ba6ab9457156d2cac70c5876f78d93ea793d94319f6722147bd9760b`.
  Uniform independent global phase makes each individual interference RE circularly
  symmetric. Sharing one radial density across positions remains an approximation.
  This is bit-wise MAP for one RE under that estimated likelihood, **not** full-waveform
  globally minimum-BER detection or symbol MAP. On audit, 512-versus-1,024-parent
  density estimates had L1 distance 0.010; bit-decision disagreement was 2.05% at
  −24 dB, 0.91% at −6 dB, and 0.003% at +6 dB, so low-SIR approximation uncertainty
  must be retained in the conclusion.
- RGS uses both frozen EMA priors on each complete `[B,2,15360]` slot, Direct hard
  QPSK remodulation initialization, `ω=κ²`, VP levels 1–49, 5,000 updates,
  cosine LR to `1e-6`. Independent per `(sir,parent,slot)` seeds make the random
  streams invariant to worker assignment and restart. The formal max LR is `1e-4`,
  selected only on audit; no early stopping or per-SIR step changes.

## Audit-only LR selection

Four audit parents at seven SIR points ran 256-step candidates. At −24 dB,
the mean target NMSE for max LR `0.001`, `0.005`, and `0.01` was respectively
5.58, 50.22, and 139.41. A separate 5,000-step audit trajectory on one
fixed parent showed that `0.001` reached NMSE 114.27 and 4,040/8,400 bit errors
at −24 dB, whereas `0.0001` reached NMSE 7.80 and 3,987/8,400. `0.00003`
reached NMSE 2.99 but 4,001/8,400 at the same point. These are small audit
samples, not formal BER evidence; the stronger-interference tail remains hard.
All 40 audit SIR summaries with integer counts are in
`reports/rgs_ber_5000_audit_20260923.csv`.

At batch one parent/two slots, an actual 5,000-update audit shard took about
58 seconds on one RTX 4090. The paired 11×256×2×5,000 formal workload is
28,160,000 sample-updates, about 23 GPU-pair wall-clock hours before I/O,
startup, and repair. The runner stores an atomic complete shard after each
parent, verifies the frozen config and all file hashes on resume, and retains
failure records.

## Formal status at publication of this progress record

All three non-Oracle baselines have finished 11×256 parents, each with
2,150,400 bits per SIR. Integer counts are in
`reports/rgs_ber_5000_partial/baseline_only_summary.csv` and explicitly labeled
`baseline_complete_rgs_pending`. Clean and oracle-subtraction diagnostics were
zero-error at every point; actual ensemble SIR averaged within 0.003 dB of
nominal. Direct at −12, −6, 0, +6 dB was 0.41205, 0.32713, 0.17680, 0.01622.
Those values are close to—but not reused from—the prior platform run; the fixed
parent range and sample realization differ.

The 11-point RGS curve, paired confidence intervals, modulation diagnostic slices,
PNG/PDF and full run manifest are **pending**. The full aggregator refuses to
draw a curve or mark the run complete until every method/SIR has exactly
256 unique parents, every parent has 8,400 bits, clean/oracle remain zero-error,
all κ values agree, and each RGS result records 5,000 calls to each prior.

The formal dual-GPU RGS workers were launched at 2026-09-23 21:18 CST in
`screen` sessions `lte-rgs5000-gpu0` and `lte-rgs5000-gpu1`, with one parent
(two slots) per atomic shard. The live logs and shards are under the ignored
`artifacts/rgs_ber_5000/formal_frozen_lr_0p0001_20260923/`. This launch is
not a completed result. To check progress: `screen -ls` and
`.venv-train/bin/python scripts/summarize_lte_rgs_ber_5000.py artifacts/rgs_ber_5000/formal_frozen_lr_0p0001_20260923 --output-dir reports/rgs_ber_5000_partial`.

## Paused formal run; paired budget audit first

At the user's request on 2026-09-23, the two formal RGS workers were stopped
with SIGINT, not resumed, and their hourly restart automation was paused.
There were 68 complete, atomic 5,000-step RGS test shards at pause; all are
retained. They are **not** used to select an iteration budget. The baseline
results above remain complete, but there is no complete formal RGS curve.

The budget audit runs only on the fixed audit split, parents `1000000..1000015`,
at SIR_RE `[-24,-12,-6,-3,0,3,6]`. Each parent/SIR generates one observation,
one Direct initialization, and fixed per-slot RGS seeds. The 1,000-, 2,000-,
and 5,000-step *completed budgets* each have their own cosine LR schedule
(`max_lr=1e-4`, `min_lr=1e-6`). The 5,000-step path additionally records states
at steps 1,000 and 2,000, separately labelled as shared-path snapshots, not
completed short-budget runs. All three non-Oracle baselines are evaluated on
the very same observations. The audit reports integer error/bit counts and
parent-paired BER differences with bootstrap intervals; a complete report
requires all 16 parents at every SIR/method. Its running artifacts are under
`artifacts/rgs_ber_5000/audit_paired_budgets_20260923/` and workers are the
`lte-rgs-budget-audit-gpu0` and `lte-rgs-budget-audit-gpu1` screen sessions.

The decision question is the smallest completed budget that shows a credible
BER advantage over the named traditional baselines in the relevant SIR range,
not whether RGS reaches its algorithmic limit. Any reduction of the formal
test grid or budget will be a new, explicitly labelled run; the paused 5,000-step
test will not be relabelled as a shorter-budget result.

The 112-shard audit finished on 2026-09-23. Its full count-level result and
figure are in [the paired budget audit](rgs_budget_audit_20260923.md). Under
the current frozen configuration, no 1k/2k/5k budget demonstrated a credible
BER advantage over the estimated bit-MAP comparator. The formal run and its
restart automation remain paused pending a user decision.

The formal sealed test is not a claim that 5,000 steps converge; the
[source paper's §5.2](https://arxiv.org/html/2306.14411v3) 20,000-step schedule
is a different compute budget. Full-waveform
Bayes-optimal BER and exact mixture LMMSE are not implemented here and must
not be inferred from the approximate baselines.
