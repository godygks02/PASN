#!/usr/bin/env bash
set -euo pipefail

cd "${PASN_ROOT:-/workspace/pasn_current}"
mkdir -p results
status=results/greedy_gpt2_e10_pipeline.status
trap 'echo failed > "$status"' ERR
echo waiting_for_gate > "$status"

while tmux has-session -t pasn_greedy_gate 2>/dev/null; do sleep 15; done
/venv/main/bin/python - <<'PY'
import json, math
from pathlib import Path
p = Path('results/greedy_g1_medium_t16_e10.json')
if not p.is_file():
    raise SystemExit('greedy gate ended without a result JSON')
d = json.loads(p.read_text())
if isinstance(d, list):
    d = d[-1]
assert d['model'] == 'gpt2-medium' and d['scope'] == 'all'
assert d['pasn_budget'] == 'greedy' and d['epochs'] == 10 and d['n_steps'] == 16
assert d['eval_split'] == 'validation' and d['stride'] == 1024
assert math.isfinite(d['ppl_snn']) and d['ppl_snn'] < 100, d['ppl_snn']
print('gate PPL', d['ppl_ann'], d['ppl_snn'], flush=True)
PY

echo running_greedy_full > "$status"
bash experiments/vast_greedy_gpt2_e10.sh full
echo running_mbe_full > "$status"
bash experiments/vast_greedy_gpt2_e10.sh mbe
echo complete > "$status"
