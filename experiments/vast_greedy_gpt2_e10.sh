#!/usr/bin/env bash
set -euo pipefail

cd "${PASN_ROOT:-/workspace/pasn_current}"
mkdir -p results
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export HF_HUB_DISABLE_XET=1

case "${1:-}" in
gate)
# Gate the new converter on a small validation slice before the full test.
/venv/main/bin/python -u experiments/gpt2_wikitext.py \
  --model gpt2-medium --backend mbe_pasn --convert-ops all \
  --pasn-budget greedy --pasn-m-max 3 --pasn-target-mse 1e-3 \
  --n-steps 16 --epochs 10 --fit-device cpu \
  --eval-split validation --eval-mode sliding --stride 1024 \
  --limit-blocks 32 --progress-every 4 --benchmark-forwards 3 \
  --json results/greedy_g1_medium_t16_e10.json \
  --tag greedy-g1-medium-t16-e10 \
  > results/greedy_g1_medium_t16_e10.log 2>&1
;;
full)
/venv/main/bin/python -u experiments/gpt2_wikitext.py \
  --model gpt2-medium --backend mbe_pasn --convert-ops all \
  --pasn-budget greedy --pasn-m-max 3 --pasn-target-mse 1e-3 \
  --n-steps 16 --epochs 10 --fit-device cpu \
  --eval-split test --eval-mode sliding --stride 1024 \
  --progress-every 10 --benchmark-forwards 3 \
  --json results/greedy_full_medium_t16_e10.json \
  --tag greedy-full-medium-t16-e10 \
  > results/greedy_full_medium_t16_e10.log 2>&1
;;
mbe)
# The existing MBE record is not a matched full WikiText-2 test run.
/venv/main/bin/python -u experiments/gpt2_wikitext.py \
  --model gpt2-medium --backend mbe --convert-ops all \
  --mbe-readout-order 2 --mbe-id-logsample \
  --n-steps 16 --epochs 10 --fit-device cpu \
  --eval-split test --eval-mode sliding --stride 1024 \
  --progress-every 10 --benchmark-forwards 3 \
  --json results/mbe_full_medium_t16_e10.json \
  --tag mbe-full-medium-t16-e10 \
  > results/mbe_full_medium_t16_e10.log 2>&1
;;
*) echo "usage: $0 {gate|full|mbe}" >&2; exit 2;;
esac
