# Project experiment rules

Read `docs/platform_design.md`, `docs/data_contract.md`, and `docs/experiment_protocol.md` before changing the platform.

- `configs/platform/lte_v1.yaml` is the sole source-distribution definition. A physical change requires a new version and revalidation of calibration/models.
- The main path accepts time-domain mixtures and returns time-domain/source estimates or target decisions. Per-RE Oracle code belongs only in explicit diagnostics.
- Train target and interference priors independently; never mix test records or mixture truth into training.
- Keep receiver-visible information and scoring truth in separate typed structures. Formal inference must never access truth.
- Reuse the target receiver, power definition, calibration, sample manifest, and seed scheme across algorithms.
- Never silently discard failures, NaNs, timeouts, samples, or unfavorable SIR points.
- Do not start expensive training until the waveform link and pilot evaluation pass and the compute budget is authorized.
- Unsupported FM/AWGN/training/RGS features must fail clearly; never return a placeholder result.
- Update `docs/progress.md` after each stage with verified facts, open issues, and the next concrete step.
- Large waveforms, datasets, checkpoints, and run logs stay under ignored `artifacts/`; commit only code, config, small fixtures, documentation, and selected result summaries.
