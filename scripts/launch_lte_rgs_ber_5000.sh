#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=${1:?pass formal run directory}
max_lr=${2:?pass audit-frozen max_lr}
cd "$repo_dir"
if screen -ls | rg -q 'lte-rgs5000-(gpu0|gpu1)'; then
  echo 'An LTE RGS worker screen already exists; inspect it before launching another.' >&2
  exit 1
fi
mkdir -p "$run_dir/logs"
for gpu in 0 1; do
  screen -dmS "lte-rgs5000-gpu${gpu}" -L -Logfile "$run_dir/logs/worker_${gpu}.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    .venv-train/bin/python scripts/run_lte_rgs_ber_5000.py \
    --mode rgs --split test --run-dir "$run_dir" \
    --statistics artifacts/rgs_ber_5000/baseline_train_stats_final.json \
    --iterations 5000 --max-lr "$max_lr" --workers 2 --worker "$gpu" \
    --batch-parents 1 --device cuda:0
done
screen -ls
