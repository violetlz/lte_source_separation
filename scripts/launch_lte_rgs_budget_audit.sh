#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
output=artifacts/rgs_ber_5000/audit_paired_budgets_20260923
if screen -ls | rg -q 'lte-rgs-budget-audit-gpu(0|1)'; then
  echo 'Named audit screen already exists; inspect before relaunching.' >&2
  exit 1
fi
mkdir -p "$output/logs"
for gpu in 0 1; do
  screen -dmS "lte-rgs-budget-audit-gpu${gpu}" -L -Logfile "$output/logs/worker_${gpu}.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    .venv-train/bin/python scripts/audit_lte_rgs_budgets.py \
    --output "$output" --worker "$gpu" --workers 2 --device cuda:0
done
screen -ls
