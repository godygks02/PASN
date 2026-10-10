# GPT-2-medium, WikiText-2 test: PASN-greedy network gate

Run completed 2026-10-10 KST. All rows use `gpt2-medium`, WikiText-2 raw test,
Stage-2 conversion (`--convert-ops all`), sliding evaluation with context 1024
and stride 1024, `T <= 16`, and 10 offline fitting epochs. The ANN uses the
same checkpoint and token stream. PASN_rule, N/T-adaptive, and greedy use local
absolute MSE target `1e-3` for untied routed primitives. The identity has a
separate relative target `r=1e-2` and tied prototype. MBE does not use the PASN
MSE target: it has global bases (activation 6, LN/softmax/matmul 8), readout
order 2 where supported, and log-sampled identity calibration.

| Metric | ANN | MBE (new instance) | PASN_rule (old instance) | N/T-adaptive (old instance) | PASN-greedy (new instance) |
|---|---:|---:|---:|---:|---:|
| Test PPL ↓ | 21.7058 | 28,195.3413 | 21.8536 | 21.8513 | **21.8403** |
| ΔPPL vs ANN, relative | — | +129,797.47% | +0.6809% | +0.6703% | **+0.6196%** |
| Spikes per input token, first window ↓ | — | 14.608m | 5.903m | **3.745m** | 5.798m |
| Modeled energy per input token, first window ↓ | — | 251.134m pJ | 57.415m pJ | **51.380m pJ** | 57.229m pJ |
| Converted-neuron storage ↓ | — | 52,812 B | 47,840 B | **46,496 B** | 49,760 B |
| Offline conversion/build ↓ | — | 22.1 min | **2.6 min** | 42.3 min | 48.0 min |
| PyTorch forward median, 1×1024 ↓ | — | 32.18 s | 28.57 s | 19.66 s | **15.16 s** |
| Peak PyTorch GPU allocation | — | 19.38 GB | 3.10 GB | 3.10 GB | 3.47 GB |
| Full SNN test evaluation | — | 150.6 min | 120.0 min | 82.4 min | **76.8 min** |

Greedy vs rule at this operating point: PPL is lower by 0.01332; first-window
spikes fall 1.78%, modeled energy falls 0.33%, converted-neuron storage grows
4.01%, and offline fitting takes 18.4× as long. N/T-adaptive has substantially
lower spikes, modeled energy, and storage than greedy, at a PPL only 0.0110
higher. These results do not reproduce the 3–5× primitive-level spike gains of
E44 in the whole network.

The fitted greedy router selected mantissa depths `m=1` at positive binade
`e=4`, `m=2` at `e=5`, and `m=3` at `e=6`. These are three unique split
decisions in the shared GELU prototype, repeated at 24 activation sites in the
network record. Other primitive kinds had no selected mantissa splits. Tied
identity retains the PASN_rule builder because per-binade split search for its
shared prototype is not yet wired. Attention/LN/matmul identities dominate the
network spike count, so this is a network test of greedy GELU and other untied
primitives, not a full replacement of the identity routing policy.

The in-project MBE conversion diverged badly in accuracy (PPL 28,195). Its
runtime and memory describe the executed implementation, but its cost cannot be
claimed as an accuracy-matched advantage for PASN. The published MBE paper uses
its own checkpoint and evaluation recipe, so this failed local reproduction is
not the paper's reported MBE accuracy.

Forward time and peak allocation for MBE and greedy were measured on the same
new RTX 3090 instance. PASN_rule and N/T-adaptive timings were measured on a
different RTX 3090 instance and are indicative only. Forward medians use three
preloaded runs and are PyTorch GPU wall times, not neuromorphic latency. Peak
allocation is not the memory required to store only the converted neurons.
Spikes and modeled energy are from the first test window, not a whole-test
average. The modeled energy uses the project's 45-nm operation cost table, not
measured electrical power. No multi-seed uncertainty estimate was made.

Raw records: `greedy_g1_medium_t16_e10.json`,
`greedy_full_medium_t16_e10.json`, `mbe_full_medium_t16_e10.json`,
`rule_full_medium_t16_e10.json`, and `adaptive_full_medium_t16_e10.json`.
The new gate/full/MBE JSON and logs were copied to this directory and every
copy matched the remote source by SHA-256 before Vast instance 55020222 was
stopped. The Vast CLI reported `stopping instance 55020222`; subsequent SSH was
refused. The instance was stopped, not destroyed.
