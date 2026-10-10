#!/usr/bin/env bash
set -euo pipefail

# Timing/accuracy gate for the large-model offline-fit search. Not a final row.
cd "${PASN_ROOT:-/workspace/pasn}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p results
/venv/main/bin/python -u experiments/gpt2_wikitext.py \
  --model gpt2-medium --backend mbe_pasn --convert-ops all \
  --pasn-budget fit --n-steps 8 --pasn-fit-max-n 2 \
  --pasn-target-mse 1e-3 --epochs 10 --fit-device cpu \
  --eval-split validation --eval-mode sliding --stride 1024 \
  --limit-blocks 32 --benchmark-forwards 3 \
  --json results/adaptive_g1_medium_t8.json --tag adaptive-g1-medium-t8
