#!/usr/bin/env bash
set -euo pipefail

cd "${PASN_ROOT:-/workspace/pasn_current}"
mkdir -p results

while tmux has-session -t rule_full_e10 2>/dev/null; do sleep 30; done

export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1

echo "Restarting ViT-M full ImageNet validation $(date -u +%FT%TZ)"
/venv/main/bin/python -u experiments/vit_imagenet.py \
  --framework timm --model vit_medium_patch16_reg4_gap_256 \
  --backend mbe_pasn --convert-ops all \
  --pasn-budget fit --n-steps 16 --pasn-fit-max-n 2 \
  --pasn-target-mse 1e-3 --epochs 10 --fit-device cpu \
  --batch-size 32 --calib-batches 4 --ann-acc 84.922 \
  --paper-row ViT-M/16 --benchmark-forwards 3 \
  --progress-json results/adaptive_vitm_t16_e10.progress.json \
  --json results/adaptive_vitm_t16_e10.json \
  --tag adaptive-vitm-t16-e10 \
  > results/adaptive_vitm_t16_e10_retry.log 2>&1
echo "Finished ViT-M $(date -u +%FT%TZ)"
