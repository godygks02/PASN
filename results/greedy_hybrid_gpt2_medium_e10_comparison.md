# GPT-2-medium, WikiText-2 test: adding the PASN-greedy (hybrid) row

Run completed 2026-10-10 (E48). All rows: `gpt2-medium`, WikiText-2 raw **test**
(281 sliding windows, context/stride 1024), Stage-2 conversion
(`--convert-ops all`), `T <= 16`, 10 offline fitting epochs, CPU neuron fitting.
ANN PPL was measured inside every run and agreed exactly (21.7058).

| Metric | ANN | MBE (local) | PASN_rule | N/T-adaptive | PASN-greedy (absolute) | **PASN-greedy (hybrid)** |
|---|---:|---:|---:|---:|---:|---:|
| Test PPL ↓ | 21.7058 | 28,195.34 | 21.8536 | 21.8513 | 21.8403 | **21.7076** |
| ΔPPL vs ANN, relative ↓ | — | +129,797% | +0.6809% | +0.6703% | +0.6196% | **+0.0082%** |
| Spikes / input token, first window ↓ | — | 14.608 M | 5.903 M | 3.745 M | 5.798 M | **3.443 M** |
| Modeled energy / input token, first window ↓ | — | 251.13 M pJ | 57.42 M pJ | 51.38 M pJ | 57.23 M pJ | **30.88 M pJ** |
| … with order-1 readouts folded | — | — | — | — | — | **25.94 M pJ** |
| Converted-neuron storage ↓ | — | 52,812 B | 47,840 B | **46,496 B** | 49,760 B | 83,664 B |
| Offline conversion/build ↓ | — | 22.1 min | **2.6 min** | 42.3 min | 48.0 min | 127.2 min |
| PyTorch forward median, 1×1024 ↓ | — | 32.18 s | 28.57 s | 19.66 s | **15.16 s** | 15.19 s |
| Peak PyTorch GPU allocation | — | 19.38 GB | 3.10 GB | 3.10 GB | 3.47 GB | 3.94 GB |
| Full SNN test evaluation | — | 150.6 min | 120.0 min | 82.4 min | 76.8 min | **76.7 min** |

**What changed in the hybrid row** (all at once, so the row is not an ablation):

| Knob | earlier greedy / rule / adaptive | PASN-greedy (hybrid) |
|---|---|---|
| per-bank target (untied) | absolute MSE `1e-3` | hybrid `max((r·rms_j)², (k·r·rms)²)`, r=3e-3, k=0.1 |
| readout order (activation, exp, inv, rsqrt) | 2 | 1 (foldable into the decode kernel) |
| identity (multiply operands) | tied rule build, N≤4 | tied, N=1, prototype deepened by mantissa bits |
| identity router depth `pasn_id_e_min` | −6 | −12 |

Against PASN_rule the hybrid row cuts relative conversion loss from +0.681% to
+0.008% (83× smaller), first-window spikes by 41.7% and modeled energy by 46.2%
(54.8% with readouts folded). Against N/T-adaptive: spikes −8.1%, energy −39.9%.
The price is storage (+74.9% vs rule, +79.9% vs adaptive) and build time
(127 min vs 2.6 / 42 min). E47 measured that the identity depth −6 → −12 alone
cuts attention P·v output error 15–30×, so part of the PPL gain is that change;
this run cannot apportion the gain between the four knobs.

The validation gate (32 windows) gave ΔPPL −0.035% (ANN 20.3890, SNN 20.3819).

Caveats, same as the earlier table: spikes and modeled energy are from the first
test window only, under the project's 45-nm operation model (not measured power;
the folded figure assumes order-1 readouts are pre-multiplied into `d[t]`).
Forward time and peak allocation are PyTorch GPU measurements, three preloaded
runs; this row ran on a new RTX 3090 instance, as did MBE and greedy (absolute),
while rule and adaptive ran on an earlier RTX 3090. One build, no seed spread.
The MBE column is the in-project conversion, which diverged (PPL 28,195); it is
not the paper's published result (Table 3: +1.57% relative, its own recipe).

Raw records: `greedy_hybrid_full_medium_t16_e10.json`,
`greedy_hybrid_g1_medium_t16_e10.json` (+ failed attempt 1:
`greedy_hybrid_g1_medium_t16_e10_attempt1_failed.json`), and for the other rows
the files listed in `greedy_gpt2_medium_e10_comparison.md`. Pulled from vast.ai
host 151.237.25.16 with every JSON and log matching the remote by SHA-256.
Commit of the run: `04862ac`.
