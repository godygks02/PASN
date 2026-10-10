#!/usr/bin/env bash
set -euo pipefail

cd "${PASN_ROOT:-/workspace/pasn_current}"
mkdir -p results

while tmux has-session -t adaptive_full_e10 2>/dev/null; do sleep 30; done

export HF_HOME=/workspace/.hf_fresh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting matched GPT-2-medium PASN_rule full test $(date -u +%FT%TZ)"
/venv/main/bin/python -u experiments/gpt2_wikitext.py \
  --model gpt2-medium --backend mbe_pasn --convert-ops all \
  --pasn-budget rule --n-steps 16 --pasn-target-mse 1e-3 \
  --epochs 10 --fit-device cpu \
  --eval-split test --eval-mode sliding --stride 1024 \
  --benchmark-forwards 3 \
  --json results/rule_full_medium_t16_e10.json \
  --tag rule-full-medium-t16-e10 \
  > results/rule_full_medium_t16_e10.log 2>&1
echo "Finished matched PASN_rule $(date -u +%FT%TZ)"
