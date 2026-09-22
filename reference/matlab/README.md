# MATLAB reference

`generate_reference.m` uses the installed MATLAB LTE Toolbox to generate deterministic 25-RB and 100-RB normal-CP waveforms at `Nfft=2048`, `Windowing=0`. The output `.mat` file is an ignored local audit artifact, not training/test data. Python compares it only after explicit `[subcarrier,symbol]` to `[symbol,subcarrier]` alignment and one analytically expected fixed scale convention; no per-example best fit is used.
