# Progress

## Delivery A

Implementation is present for strict configuration, contracts, stable random streams, QPSK/16QAM/64QAM, common-grid OFDM/CP, continuous target/interference sources, calibration, time-domain mixing, Direct reception, scoring, audits, plots, CLI, and report generation.

Formal validation results are recorded in `reports/platform_validation.md`. Large generated artifacts remain under ignored `artifacts/`.

## Next stage

After delivery A remains green, add independent single-source training datasets and two DiffWave priors without changing the platform distribution, receiver, calibration, split, or truth boundary. Do not implement RGS/FM until those priors and denoising diagnostics are independently validated.
