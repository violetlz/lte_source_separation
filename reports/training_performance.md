# Formal prior training performance

All measurements use one RTX 4090 per source, BF16, length 15360 and effective batch 16.
GPU-resident candidates used 20 warmup plus 100 measured optimizer updates. No artificial
GPU load was added.

## GPU-resident compute upper bound

| Source | Microbatch | samples/s | step seconds | peak reserved |
|---|---:|---:|---:|---:|
| target | 2 | 95.0 | 0.1684 | 0.48 GiB |
| target | 4 | 177.2 | 0.0903 | 1.00 GiB |
| target | **8** | **263.0** | **0.0608** | **1.72 GiB** |
| target | 16 | 237.1 | 0.0675 | 3.37 GiB |
| interference | 2 | 61.4 | 0.2606 | 1.04 GiB |
| interference | 4 | 64.8 | 0.2468 | 1.88 GiB |
| interference | **8** | **98.1** | **0.1630** | **3.40 GiB** |
| interference | 16 | 86.3 | 0.1855 | 6.54 GiB |

## Real concurrent materialized pipeline

At the formal stability checkpoint (last 100 real updates, both workers concurrent):

| Source | samples/s | step seconds | data wait | peak reserved |
|---|---:|---:|---:|---:|
| target | 250.8 | 0.06380 | about 0.2 ms | 1.69 GiB |
| interference | 95.8 | 0.16696 | about 0.5 ms | 3.33 GiB |

The real pipeline reaches about 95.4%/97.7% of the measured resident-compute upper bound.
The remaining gap is small enough that data synthesis/I/O is no longer the dominant
bottleneck. The pilot report's approximate target/interference throughputs were about
94/46 samples/s, so the observed real-pipeline increases are about 2.67x and 2.08x.

Four workers/source, pinned memory, persistent workers, prefetch factor 2 and nonblocking
H2D were selected. Microbatch 8 wins measured throughput while reserving far below 85% of
48.5 GB device memory; activation checkpointing therefore remains disabled.

The materialized loader alone was measured after 20 warmup and over 100 batches:

| Workers | loader samples/s | mean batch wait |
|---:|---:|---:|
| 2 | 13,097 | 1.22 ms |
| 4 | 13,687 | 1.17 ms |
| 8 | 15,790 | 1.01 ms |

Although eight workers maximized isolated loader throughput, even two are far above model
consumption. Four per source was retained as the balanced concurrent setting; the real
two-process measurements show sub-millisecond observed data wait, so doubling both pools
would add CPU pressure without changing the compute-bound ceiling.

The 100-update handoff interval is a short stability measurement, not the requested final
three-minute performance characterization. The target worker subsequently ran
continuously from the 2200 gate to step 10881 before a controlled profiler pause, providing
well over three minutes of real materialized-pipeline operation at the same approximately
250 samples/s regime while the interference worker ran concurrently. A five-update
CPU+CUDA trace after 20 warmup updates measured DataLoader enumeration at about 0.40 ms
total over five batches and is retained as ignored
`artifacts/runs/task03_target_pipeline_trace.json`. The trace and long-run history used
real data and no artificial load.
