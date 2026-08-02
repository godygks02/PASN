#!/bin/bash
# Remaining Table 2 cells, in order of provenance quality. Each task: ANN then
# the T=16 arm (the one matched to the paper's global timestep).
PY="C:/Users/cm120/miniconda3/envs/SNN/python.exe"
J="results/roberta_nlu.json"
for spec in "mr base" "sst2 large" "subj large" "sst5 large"; do
  set -- $spec; task=$1; size=$2
  echo "##### $task/$size ANN  $(date -Is)"
  "$PY" experiments/roberta_sst2.py --task $task --size $size --backend none \
      --json $J --tag "$task-$size-ann" || { echo "SKIP $task/$size"; continue; }
  echo "##### $task/$size T=16  $(date -Is)"
  "$PY" experiments/roberta_sst2.py --task $task --size $size --backend mbe_pasn \
      --convert-ops all --epochs 300 --pasn-id-target relative \
      --pasn-id-target-rel 1e-2 --pasn-t-fixed 16 \
      --json $J --tag "$task-$size-t16"
done
echo "NLU SWEEP DONE $(date -Is)"
