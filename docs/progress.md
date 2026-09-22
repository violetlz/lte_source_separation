# Progress

## Delivery A

Implementation is present for strict configuration, contracts, stable random streams, QPSK/16QAM/64QAM, common-grid OFDM/CP, continuous target/interference sources, calibration, time-domain mixing, Direct reception, scoring, audits, plots, CLI, and report generation.

Formal validation results are recorded in `reports/platform_validation.md`. Large generated artifacts remain under ignored `artifacts/`.

## Delivery B: independent priors

Implemented the immutable/public configuration boundary, channels-first complex coordinates, deterministic 0.5 ms single-source datasets, target/interference DiffWave priors, VP epsilon objective, EMA, fixed validation, atomic strict-resume checkpoints, online W&B tracking, bounded two-GPU orchestration, and structured status files.

The physical platform hash remains `18b2a1bc425f217e36c9e67f9fd64f13701f45c83148ec0ccba42205f8115cb5`; no source-distribution or calibration rule changed. The suite has 27 CPU-visible passes plus one CUDA-only RNG-device regression pass, and the waveform audit passes 14/14. Both bounded pilots completed 2,000 updates with final full validation and exit code 0. Runtime facts and objective/communications metrics are recorded in `reports/prior_training_handoff.md`; checkpoints and detailed outputs remain ignored under `artifacts/prior_pilot/`.

Do not implement RGS/FM until the two pilot priors and their retained unfavorable/favorable diagnostics are reviewed independently.
