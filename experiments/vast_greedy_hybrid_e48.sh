#!/usr/bin/env bash
# E48: PASN-greedy (N=1, mantissa-prefix routing, hybrid target, order-1 folded
# readout, N=1 split identity) on gpt2-medium x WikiText-2 -- the row that joins
# the ANN / MBE / PASN_rule / N-T-adaptive / greedy(absolute) table in
# results/greedy_gpt2_medium_e10_comparison.md.
#
# Same operating point as that table: Stage 2 (--convert-ops all), T <= 16,
# 10 fitting epochs, CPU neuron fitting, sliding eval at context/stride 1024.
# Differences from the earlier greedy row are the E44-E47 decisions:
#   --pasn-target hybrid (r=3e-3, k=0.1)  instead of absolute MSE 1e-3
#   --pasn-readout-order 1                (energy also reported folded)
#   --pasn-greedy-identity split          (N=1 identity, mantissa-deepened prototype)
#   --pasn-id-e-min -12                   (attention P operand, E47)
#
# Run under tmux (memory: a driver started from a bare SSH shell can die silently):
#   tmux new -s e48 'bash experiments/vast_greedy_hybrid_e48.sh 2>&1 | tee results/e48_driver.log'
set -euo pipefail

cd "${PASN_ROOT:-/workspace/pasn_current}"
mkdir -p results
export HF_HOME="${HF_HOME:-/workspace/.hf_fresh}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET=1
PY="${PY:-/venv/main/bin/python}"

echo "commit $(git rev-parse HEAD)  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

COMMON=(--model gpt2-medium --backend mbe_pasn --convert-ops all
        --pasn-budget greedy --pasn-target hybrid --pasn-target-rel 3e-3
        --pasn-floor-k 0.1 --pasn-greedy-identity split --pasn-readout-order 1
        --pasn-id-e-min -12 --n-steps 16 --epochs 10 --fit-device cpu
        --eval-mode sliding --stride 1024 --benchmark-forwards 3)

echo "== G1 gate: validation, 32 windows  $(date -u +%FT%TZ)"
"$PY" -u experiments/gpt2_wikitext.py "${COMMON[@]}" \
  --eval-split validation --limit-blocks 32 \
  --json results/greedy_hybrid_g1_medium_t16_e10.json \
  --tag greedy-hybrid-g1-medium-t16-e10 \
  > results/greedy_hybrid_g1_medium_t16_e10.log 2>&1
echo "== gate done  $(date -u +%FT%TZ)"

echo "== full WikiText-2 test  $(date -u +%FT%TZ)"
"$PY" -u experiments/gpt2_wikitext.py "${COMMON[@]}" \
  --eval-split test \
  --json results/greedy_hybrid_full_medium_t16_e10.json \
  --tag greedy-hybrid-full-medium-t16-e10 \
  > results/greedy_hybrid_full_medium_t16_e10.log 2>&1
echo "== full done  $(date -u +%FT%TZ)"
sha256sum results/greedy_hybrid_*_medium_t16_e10.json
