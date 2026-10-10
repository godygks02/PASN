#!/usr/bin/env bash
set -euo pipefail

cd "${PASN_ROOT:-/workspace/pasn_current}"
mkdir -p results
export HF_HUB_DISABLE_XET=1

if [[ "${WAIT_FOR_RULE:-0}" == "1" ]]; then
  while tmux has-session -t rule_g1 2>/dev/null; do sleep 15; done
fi

# GPT-2-medium: same checkpoint and full WikiText-2 test evaluation as the
# existing rule row. Search N<=2 and T<=16 with the validated 10-epoch budget.
export HF_HOME=/workspace/.hf_fresh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "Starting GPT-2-medium full WikiText test $(date -u +%FT%TZ)"
/venv/main/bin/python -u experiments/gpt2_wikitext.py \
  --model gpt2-medium --backend mbe_pasn --convert-ops all \
  --pasn-budget fit --n-steps 16 --pasn-fit-max-n 2 \
  --pasn-target-mse 1e-3 --epochs 10 --fit-device cpu \
  --eval-split test --eval-mode sliding --stride 1024 \
  --benchmark-forwards 3 \
  --json results/adaptive_full_medium_t16_e10.json \
  --tag adaptive-full-medium-t16-e10 \
  > results/adaptive_full_medium_t16_e10.log 2>&1
echo "Finished GPT-2-medium $(date -u +%FT%TZ)"

# ImageNet validation shards and the gated-data token live in this HF_HOME.
export HF_HOME=/workspace/.hf_home
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE

echo "Starting ViT-B full ImageNet validation $(date -u +%FT%TZ)"
/venv/main/bin/python -u experiments/vit_imagenet.py \
  --model google/vit-base-patch16-224 \
  --backend mbe_pasn --convert-ops all \
  --pasn-budget fit --n-steps 16 --pasn-fit-max-n 2 \
  --pasn-target-mse 1e-3 --epochs 10 --fit-device cpu \
  --batch-size 32 --calib-batches 4 --ann-acc 80.316 \
  --paper-row ViT-B/16 --benchmark-forwards 3 \
  --json results/adaptive_vitb_t16_e10.json \
  --tag adaptive-vitb-t16-e10 \
  > results/adaptive_vitb_t16_e10.log 2>&1
echo "Finished ViT-B $(date -u +%FT%TZ)"

echo "Starting ViT-M full ImageNet validation $(date -u +%FT%TZ)"
/venv/main/bin/python -u experiments/vit_imagenet.py \
  --framework timm --model vit_medium_patch16_reg4_gap_256 \
  --backend mbe_pasn --convert-ops all \
  --pasn-budget fit --n-steps 16 --pasn-fit-max-n 2 \
  --pasn-target-mse 1e-3 --epochs 10 --fit-device cpu \
  --batch-size 32 --calib-batches 4 --ann-acc 84.922 \
  --paper-row ViT-M/16 --benchmark-forwards 3 \
  --json results/adaptive_vitm_t16_e10.json \
  --tag adaptive-vitm-t16-e10 \
  > results/adaptive_vitm_t16_e10.log 2>&1
echo "Finished ViT-M $(date -u +%FT%TZ)"
