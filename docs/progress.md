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
They continue in the ignored `artifacts/prior_formal/prior-formal-20260923` run directory.
This is engineering/training progress, not a completed quality or BER result. The full
RGS-dev selection, 10k-step sentinel and final sealed functional evaluation remain pending
the corresponding training milestones. FM and physical AWGN remain out of scope.
