#!/usr/bin/env bash
set -euo pipefail

# Bounded GPT-2 Stage-2 pilot on the 8 GB RTX 4060 instance.
# Run under tmux; each arm writes its own JSON/log for recovery.
cd /workspace/pasn
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

arm="${1:?use ann, rule, or fit}"
common=(
  --model gpt2
  --dataset-arrow-dir /workspace/pasn/data/wikitext
  --eval-split validation
  --eval-mode block
  --block 64
  --limit-blocks 8
  --eval-batch-size 1
  --progress-every 2
  --stride 1024
)

case "$arm" in
  ann)
    /venv/main/bin/python -u experiments/gpt2_wikitext.py \
      "${common[@]}" --backend none \
      --json results/pilot_ann.json --tag pilot-ann
    ;;
  rule|fit)
    /venv/main/bin/python -u experiments/gpt2_wikitext.py \
      "${common[@]}" --backend mbe_pasn --convert-ops all \
      --n-steps 8 --epochs 10 --pasn-fit-max-n 2 \
      --pasn-target-mse 1e-3 --pasn-budget "$arm" --fit-device cpu \
      --json "results/pilot_${arm}.json" --tag "pilot-${arm}"
    ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
