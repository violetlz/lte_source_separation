# Progress

## Delivery A

Implementation is present for strict configuration, contracts, stable random streams, QPSK/16QAM/64QAM, common-grid OFDM/CP, continuous target/interference sources, calibration, time-domain mixing, Direct reception, scoring, audits, plots, CLI, and report generation.

Formal validation results are recorded in `reports/platform_validation.md`. Large generated artifacts remain under ignored `artifacts/`.

## Delivery B: independent priors

Implemented the immutable/public configuration boundary, channels-first complex coordinates, deterministic 0.5 ms single-source datasets, target/interference DiffWave priors, VP epsilon objective, EMA, fixed validation, atomic strict-resume checkpoints, online W&B tracking, bounded two-GPU orchestration, and structured status files.

The physical platform hash remains `18b2a1bc425f217e36c9e67f9fd64f13701f45c83148ec0ccba42205f8115cb5`; no source-distribution or calibration rule changed. The suite has 27 CPU-visible passes plus one CUDA-only RNG-device regression pass, and the waveform audit passes 14/14. Both bounded pilots completed 2,000 updates with final full validation and exit code 0. Runtime facts and objective/communications metrics are recorded in `reports/prior_training_handoff.md`; checkpoints and detailed outputs remain ignored under `artifacts/prior_pilot/`.

The pilots were independently reviewed in Task 03. Their unfavorable/favorable diagnostics
remain retained; no early metric was used to claim separation success.

## Delivery C: formal priors and RGS engineering path

Implemented and validated materialized 200,000-parent training datasets per source and
1,024-parent/two-slot validation sets, map-style mmap loading, recoverable epoch sampling,
batched shared raw/EMA validation, formal pilot-to-training migration, two-GPU guarded
launch, frozen-prior inference loading, and waveform alpha-RGS with analytic and real
checkpoint smoke tests.

Both formal jobs started from cumulative step 2000, passed the +200-update stable handoff,
saved known-good formal checkpoints and showed train/validation history on remote W&B.
Both formal jobs have now completed their budgets: target at cumulative step
200,000 and interference at step 300,000. Full fixed-validation results and
checkpoint identities are in `reports/formal_training_results_20260923.md`;
all 50 unique full-validation rows are in
`reports/formal_validation_full_20260923.csv`. The raw artifacts remain under
ignored `artifacts/prior_formal/prior-formal-20260923`. The full RGS-dev
selection, 10k-step sentinel and final sealed functional evaluation remain
pending. FM and physical AWGN remain out of scope.

On 2026-09-23, the interference formal worker was gracefully paused at cumulative
step 95,293 and resumed from its final `last.pt` checkpoint with two NCCL ranks.
The global effective batch remains 16 (eight windows per GPU); the same recoverable
global sampler, optimizer, EMA, source data, validation protocol, and 300,000-step
budget are used. Both rank RNG states are now saved in each continuation checkpoint.
A bounded two-GPU check reached step 95,450 with exact sample accounting
(`1,495,200` formal exposures, `1,527,200` cumulative exposures) and a 100-step
mean of 0.0910 s/update, versus about 0.1729 s/update on one GPU. The long run
resumed from step 95,450 in the same ignored run directory and W&B run. It
completed step 300,000 with 4,768,000 formal exposures, a final full
validation, and both rank RNG states in its checkpoint.

## Delivery D: paired no-noise RGS BER test paused after audit

On 2026-09-23, both final EMA prior checkpoints and the formal calibration were
verified. The strict 11-SIR/256-parent test runner, train-only estimated baselines,
per-sample RGS RNG, atomic two-GPU shards, completeness-gated plotting, and
tests are present. All three non-Oracle baseline sweeps have finished;
clean/oracle are zero-error. The 5,000-step waveform RGS sweep was paused
after 68 complete atomic test shards at the user's request; it has not been
resumed, so no four-method formal BER curve or final performance claim exists.
See `reports/rgs_ber_5000_20260923_in_progress.md` and its explicitly partial
baseline CSV. The exact two-source LMMSE is not implemented; its graph slot is
honestly the FFT-diagonal approximate Wiener baseline.

The independent audit-only paired 1k/2k/5k comparison completed all 112
predeclared `(SIR,parent)` shards. Its count-level report and figure are in
`reports/rgs_budget_audit_20260923.md`: the current frozen model/receiver
did not show a credible BER advantage over the estimated bit-MAP comparator.
No shorter formal budget was selected from the partial sealed test.

## Receiver diagnosis (2026-09-24)

A new audit-only, frozen-weight 5k diagnostic recomputed 16 preselected
trajectories at -24, -6, 0 and +6 dB. Direct/RGS counts and RGS NMSE match
the original atomic audit shards. At -6/0/+6 dB, none of the 100,800 paired
bits changed from the Direct hard-remodulated initialization despite modest
target-grid NMSE reductions. At -24 dB, 90 flips netted 24 corrected bits,
but RGS worsened both waveform and target-grid NMSE relative to that
initialization. The diagnosis and caveats are in
`reports/rgs_receiver_diagnosis_20260924.md`; it is not an algorithm fix or
new formal test. The formal workers and both automations remain paused.
