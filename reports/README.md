# Published validation results

`platform_validation.md` is the human-readable delivery-A acceptance report. The
`formal_validation/` directory contains the small, selected machine-readable snapshot
needed to audit its claims:

- `summary.csv`: aggregate Clean, Direct, and Oracle counts for every SIR;
- `validation_checks.json`: hard-gate results and numerical audit errors;
- `calibration_summary.json`: independent 4,096-record calibration statistics;
- `matlab_crosscheck.json`: MATLAB R2025a/LTE Toolbox cross-check;
- `run_manifest.json`: environment and provenance captured when the formal run executed.
- `figures/`: the selected spectrum, CP, and Direct BER diagnostics.

Raw IQ, per-example CSVs, MATLAB `.mat` fixtures, logs, datasets, checkpoints, and
other bulky generated artifacts remain under ignored `artifacts/` and are intentionally
not published in Git.

The no-noise waveform RGS formal BER test is paused. Its complete 112-shard,
audit-split-only 1k/2k/5k comparison is documented in
[`rgs_budget_audit_20260923.md`](rgs_budget_audit_20260923.md), with lightweight
count-level CSVs and a BER figure under `rgs_budget_audit_20260923/`.
The follow-up frozen-model receiver-path diagnosis is in
[`rgs_receiver_diagnosis_20260924.md`](rgs_receiver_diagnosis_20260924.md),
with a small machine-readable audit summary beside it.
