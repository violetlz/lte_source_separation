# Formal training handoff

Snapshot date: 2026-09-23 Asia/Shanghai. Both workers continue in detached background
processes; values below are the verified stable handoff points, not final training results.

| Source | physical GPU / UUID | PID at launch | stable step | new exposures at step 2200 | throughput | W&B |
|---|---|---:|---:|---:|---:|---|
| target | 0 / `GPU-e4785e2d-2d54-d676-baf3-fab3a0085058` | 1170027 (current) | 2200; reverified after profiler resume | 3,200 | 250.8 samples/s | [ewq62hpj](https://wandb.ai/violetftrt-evergarden/lte-source-separation/runs/ewq62hpj) |
| interference | 1 / `GPU-77cc34c6-695a-dda2-aef2-59145c1f04ce` | 1328003 (current) | 2200; reverified after state-schema resume | 3,200 | 95.8 samples/s | [pyq671jm](https://wandb.ai/violetftrt-evergarden/lte-source-separation/runs/pyq671jm) |

Both formal runs explicitly link pilot W&B IDs `1za8kes6`/`2xif3tr0` and the verified
parent checkpoint SHA256. They inherited raw, EMA, Adam, cumulative step 2000, historical
exposures 32000 and diffusion-noise RNG. The fixed-data sampler starts independently at
epoch 0 offset 0. Resolved configs are frozen beside each ignored run directory.

Each source completed at least 200 new successful optimizer updates, one new batched
multi-noise raw/EMA validation, an atomic formal checkpoint reload boundary and remote
W&B history verification (`train_visible=true`, `validation_visible=true`). The initial
launch rejected the new `validation` status because the pilot state enum was too narrow;
it failed before an optimizer update. The enum was repaired and the same W&B IDs were
resumed. No failed live state overwrote a checkpoint.

After stable handoff, target continued to step 10881, then received a controlled SIGTERM
to save a finite checkpoint while a required short CPU+CUDA pipeline trace ran on GPU0.
It strictly resumed the same formal run/checkpoint and re-established `stable_running`;
interference continued during the profiling step. It was later gracefully checkpointed
and strictly resumed once so its persisted checkpoint also records the corrected formal
`stability_achieved` flag; the model/optimizer/data stream were preserved.

Run root:
`artifacts/prior_formal/prior-formal-20260923/`

Read status without attaching to the processes:

```bash
python -m json.tool artifacts/prior_formal/prior-formal-20260923/target/status.json
python -m json.tool artifacts/prior_formal/prior-formal-20260923/interference/status.json
```

Graceful pause (workers save a known-finite `last.pt`):

```bash
kill -TERM 1170027 1328003
```

Strict resume uses the same command and resolved formal configs:

```bash
.venv-train/bin/python scripts/run_prior_formal.py orchestrate
```

Budgets remain cumulative 200,000 target and 300,000 interference updates. Approximate
compute-only remaining durations at the handoff throughput are 3.5 h and 13.8 h;
validation/checkpoint/W&B overhead makes wall time longer. These are estimates, not a
completion guarantee. Full quality and RGS functional outcomes are not yet available and
must not be inferred from the stable launch.
