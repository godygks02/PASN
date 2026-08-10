# Operator wins, recomputed with identity tying on

From `results/op_pareto.json` + `results/op_pareto_tied.json` via `experiments/op_pareto_merge.py`.
Control: `mbe` arm, 45 builds, reproduced bit-identical.

## Per operator (arm `pasn_rule`)

Iso-accuracy medians; ratios are MBE / PASN (>1 = PASN cheaper).

| operator | tied? | spikes MBE/PASN | bytes MBE/PASN | matched |
|---|---|---|---|---|
| `activation:exp2` | -- |  4.48x |  0.51x | 9/10 |
| `activation:gelu` | -- |  4.95x |  0.35x | 7/7 |
| `activation:inv` | -- |  5.39x |  0.39x | 8/9 |
| `activation:invsqrt` | -- |  2.27x |  0.93x | 6/10 |
| `activation:silu` | -- |  2.63x |  0.30x | 7/7 |
| `activation:tanh` | -- |  3.91x |  0.19x | 8/8 |
| `attention` | **yes** | 1.45x -> **1.70x** | 0.18x -> **0.32x** | 6/6 |
| `fp_multiply` | **yes** | 1.61x -> **1.23x** | 0.06x -> **0.11x** | 2/3 |
| `layernorm` | **yes** | 2.43x -> **1.55x** | 0.13x -> **0.22x** | 6/8 |
| `softmax` | -- |  4.23x |  0.26x | 7/7 |

spike wins: 10/10 (before) -> **10/10 (after)**
winning range: **1.23x - 5.39x**  (median of the winners 3.27x)
byte wins: **0/10**

## Per arm -- a quoted range must name one

| arm | what it is | before | after |
|---|---|---|---|
| `pasn` | routing alone (uniform banks) -- ablation | 10/10 ops, 1.16x-6.41x | **10/10 ops, 1.01x-6.41x** |
| `pasn_rule_T` | + allocate N_j, T pinned to 16 | 8/10 ops, 1.14x-2.35x | **8/10 ops, 1.14x-2.35x** |
| `pasn_rule` | + allocate N_j and T_j -- **the method as it ships** | 10/10 ops, 1.45x-5.39x | **10/10 ops, 1.23x-5.39x** |
