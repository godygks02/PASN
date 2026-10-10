#!/usr/bin/env bash
set -euo pipefail

cd "${PASN_ROOT:-/workspace/pasn_current}"
mkdir -p results
export HF_HUB_DISABLE_XET=1

# Match the historical rule rows' 300-epoch fitting budget. Keep the adaptive
# search constraints from the completed 10-epoch run to isolate fit duration.
export HF_HOME=/workspace/.hf_home

if [[ ! -s results/adaptive_vitb_t16_e300.json ]]; then
  echo "Starting ViT-B adaptive 300-epoch full ImageNet $(date -u +%FT%TZ)"
  /venv/main/bin/python -u experiments/vit_imagenet.py \
    --model google/vit-base-patch16-224 \
    --backend mbe_pasn --convert-ops all \
    --pasn-budget fit --n-steps 16 --pasn-fit-max-n 2 \
    --pasn-target-mse 1e-3 --epochs 300 --fit-device cpu \
    --batch-size 32 --calib-batches 4 --ann-acc 80.316 \
    --paper-row ViT-B/16 --benchmark-forwards 3 \
    --progress-json results/adaptive_vitb_t16_e300.progress.json \
    --json results/adaptive_vitb_t16_e300.json \
    --tag adaptive-vitb-t16-e300 \
    > results/adaptive_vitb_t16_e300.log 2>&1
  echo "Finished ViT-B adaptive 300 epochs $(date -u +%FT%TZ)"
fi

if [[ ! -s results/adaptive_vitm_t16_e300.json ]]; then
  echo "Starting ViT-M adaptive 300-epoch full ImageNet $(date -u +%FT%TZ)"
  /venv/main/bin/python -u experiments/vit_imagenet.py \
    --framework timm --model vit_medium_patch16_reg4_gap_256 \
    --backend mbe_pasn --convert-ops all \
    --pasn-budget fit --n-steps 16 --pasn-fit-max-n 2 \
    --pasn-target-mse 1e-3 --epochs 300 --fit-device cpu \
    --batch-size 32 --calib-batches 4 --ann-acc 84.922 \
    --paper-row ViT-M/16 --benchmark-forwards 3 \
    --progress-json results/adaptive_vitm_t16_e300.progress.json \
    --json results/adaptive_vitm_t16_e300.json \
    --tag adaptive-vitm-t16-e300 \
    > results/adaptive_vitm_t16_e300.log 2>&1
  echo "Finished ViT-M adaptive 300 epochs $(date -u +%FT%TZ)"
fi

export HF_HOME=/workspace/.hf_fresh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [[ ! -s results/adaptive_full_medium_t16_e300.json ]]; then
  echo "Starting GPT-2-medium adaptive 300-epoch full test $(date -u +%FT%TZ)"
  /venv/main/bin/python -u experiments/gpt2_wikitext.py \
    --model gpt2-medium --backend mbe_pasn --convert-ops all \
    --pasn-budget fit --n-steps 16 --pasn-fit-max-n 2 \
    --pasn-target-mse 1e-3 --epochs 300 --fit-device cpu \
    --eval-split test --eval-mode sliding --stride 1024 \
    --benchmark-forwards 3 \
    --json results/adaptive_full_medium_t16_e300.json \
    --tag adaptive-full-medium-t16-e300 \
    > results/adaptive_full_medium_t16_e300.log 2>&1
  echo "Finished GPT-2-medium adaptive 300 epochs $(date -u +%FT%TZ)"
fi
