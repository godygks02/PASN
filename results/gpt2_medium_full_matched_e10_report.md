# GPT-2-medium full WikiText-2 test: matched 10-epoch comparison

Both converted arms used the same pretrained `gpt2-medium`, WikiText-2 raw
test split (281 sliding windows, context and stride 1024), Stage-2 conversion,
10 fitting epochs, maximum 16 timesteps, CPU neuron fitting, and the same Vast.ai
RTX 3090 instance. The only allocation-method change was `pasn_budget=rule`
versus `pasn_budget=fit` (`N<=2`, local target MSE `1e-3`). ANN PPL was measured
in each run and agreed exactly.

| Metric | ANN | PASN_rule | PASN_adaptive |
|---|---:|---:|---:|
| Full-test perplexity (lower is better) | 21.7058 | 21.8536 | **21.8513** |
| PPL change vs ANN | baseline | +0.6809% | **+0.6703%** |
| Spikes/input token, first test window | — | 5,903,318 | **3,744,864** |
| Modeled energy, pJ/input element, first window | — | 57,415,432 | **51,380,459** |
| Converted-neuron storage | — | 47,840 B | **46,496 B** |
| Forward median, batch 1 × 1024, 3 runs | — | 28.567 s | **19.657 s** |
| PyTorch peak GPU allocation in benchmark | — | 3.102 GB | 3.102 GB |
| Offline conversion/build time | — | **156 s** | 2,541 s |
| Full SNN evaluation time | — | 7,202 s | **4,945 s** |

Adaptive versus rule changes PPL by **−0.00232** (a very small improvement),
while reducing first-window spikes by 36.6%, modeled energy by 10.5%,
converted-neuron storage by 2.8%, forward time by 31.2%, and full SNN
evaluation time by 31.3%. Offline fitting is 16.3× slower. The two peak GPU
allocation measurements are effectively equal.

Adaptive selected N=1 in 1,036 per-site records and N=2 in 266. Its timestep
choices were T=2: 650, T=3: 48, T=4: 24, T=6: 266, T=8: 48, and T=16: 266.
There were 290 local target misses, which used the lowest-error fallback.
These counts include repeated sites sharing fitted primitives.

Spikes and modeled energy describe only the first test window. The forward
benchmark excludes tokenization and reports PyTorch allocation, not total
device memory or neuromorphic latency. The PPL difference between rule and
adaptive is too small to claim a robust accuracy advantage from this one run.

The earlier 300-epoch rule record had PPL 21.6652, substantially lower than
either 10-epoch arm. It is not fit-budget matched to this experiment; a
300-epoch adaptive run is needed to determine whether the lower spike count
persists at that accuracy level.

Raw records: `adaptive_full_medium_t16_e10.json` and
`rule_full_medium_t16_e10.json`.
