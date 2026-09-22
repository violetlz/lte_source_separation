# lte_source_separation

Reproducible LTE-inspired 5/20 MHz asynchronous OFDM waveform benchmark. Delivery A provides the validated physical waveform platform. Delivery B adds two independent 0.5 ms VP epsilon-prediction priors and their bounded pilot-training machinery; it still does not implement source separation or RGS/FM inference.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
# On the CUDA training host, use the independent environment described below:
uv venv --python /usr/bin/python3.10 .venv-train
uv pip install --python .venv-train/bin/python 'torch==2.9.1' \
  --index https://download.pytorch.org/whl/cu130
uv pip install --python .venv-train/bin/python -e '.[test]' 'wandb==0.30.0'
```

The verified environment is recorded in `requirements.lock` and in each run manifest.

Prior inspection, benchmarking, and training commands are documented in `docs/prior_training.md`.

## Reproduce delivery A

```bash
.venv/bin/python -m pytest
.venv/bin/python -m rfsep audit --config configs/platform/lte_v1.yaml
.venv/bin/python -m rfsep calibrate --config configs/platform/lte_v1.yaml --num-records 4096
.venv/bin/python -m rfsep evaluate --config configs/experiments/smoke.yaml
.venv/bin/python -m rfsep evaluate --config configs/experiments/platform_validation.yaml
.venv/bin/python -m rfsep report --run <run_id>
```

Formal evaluation refuses missing, modified, mismatched, or provisional calibration. Generated data and detailed runs are ignored under `artifacts/`; the selected human-readable acceptance report is `reports/platform_validation.md`.

## Boundaries

This is a synthetic LTE-numerology data waveform, not a complete LTE PHY. It is uncoded and has no pilots, control channels, physical AWGN, multipath, CFO, Doppler, HARQ, MIMO, or 5 MHz front-end filtering. The receiver observes the full 30.72 Msps wideband IQ. See `docs/platform_design.md` and `docs/data_contract.md` before extending it.
