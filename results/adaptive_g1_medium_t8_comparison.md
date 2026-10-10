# GPT-2-medium G1: ANN, PASN_rule, PASN_adaptive

Same pretrained `gpt2-medium`, WikiText-2 raw **validation** first 32 sliding
windows (context 1024, stride 1024), Stage-2 conversion of all supported
operations, 10 fitting epochs, and a maximum of 8 timesteps. Both converted
arms were run on the same Vast.ai RTX 3090 instance. This is a gate experiment,
not the full test-set result.

| Metric | ANN | PASN_rule | PASN_adaptive |
|---|---:|---:|---:|
| Perplexity (lower is better) | 20.3890 | **20.3792** | 20.3939 |
| PPL change vs ANN | baseline | −0.0478% | +0.0243% |
| Spikes/input token, first window | — | 2,512,951 | **1,122,986** |
| Modeled energy, pJ/input element, first window | — | 39,793,179 | **24,268,197** |
| Converted neuron storage, bytes | — | 53,888 | **39,920** |
| Forward median, batch 1 × 1024, 3 runs | — | 15.891 s | **12.890 s** |
| PyTorch peak GPU allocation during benchmark | — | 3.099 GB | **2.929 GB** |
| Offline conversion/build time | — | **243 s** | 1,643 s |
| 32-window SNN evaluation time | — | 535 s | **450 s** |

Adaptive minus rule PPL is +0.01469 (adaptive is worse by 0.0721% relative to
rule). Adaptive uses 55.3% fewer spikes, 39.0% less modeled energy, 25.9% less
converted-neuron storage, 18.9% less measured forward time, and 5.5% less
PyTorch peak allocation. Its offline build is 6.76× slower.

The adaptive budget records select `N=1` in all 1,302 per-site entries. The
selected timestep distribution is T=2: 650, T=3: 48, T=4: 24, T=6: 484, and
T=8: 96. There are 290 entries where no tested candidate met the local target;
the selector used its lowest-error fallback. These entries include repeated
call sites sharing fitted primitives, so they are not independent neuron counts.

Spike and modeled energy numbers are from the first validation window only.
The latency benchmark uses preloaded input and omits tokenization; GPU memory is
PyTorch allocated memory, not device-wide memory use. ANN latency, memory, and
energy were not measured in this gate. The 32-window PPL difference should not
be treated as a full WikiText test-set result or a statistical significance
claim. The full test and ImageNet adaptive evaluations are queued separately.

Raw records: `adaptive_g1_medium_t8.json` and `rule_g1_medium_t8.json`.
