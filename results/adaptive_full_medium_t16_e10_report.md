# GPT-2-medium full WikiText-2 test: interim comparison

**Update:** A matched 10-epoch PASN_rule full-test run has now completed. Use
`gpt2_medium_full_matched_e10_report.md` for the controlled three-way
comparison; the historical 300-epoch rule row below remains background only.

The adaptive run used Stage-2 conversion, offline-fit allocation with `N<=2`
and `T<=16`, 10 fitting epochs, and all 281 WikiText-2 raw test windows
(context 1024, stride 1024). Its ANN and SNN passes used the same checkpoint
and test tokens. The rule row below is the earlier full-test record
`freeze_e1.json` (`freeze-t16-unif`), fitted for **300 epochs** with a fixed
`T=16`; it is an historical reference, not an epoch-matched control.

| Metric | ANN | Historical PASN_rule | PASN_adaptive |
|---|---:|---:|---:|
| Full-test perplexity (lower is better) | 21.7058 | **21.6652** | 21.8513 |
| PPL change vs ANN | baseline | −0.1873% | +0.6703% |
| Fitting epochs | — | 300 | 10 |
| Maximum timestep | — | 16 fixed | 16, selected per bank |
| Spikes/input token, first window | — | 4,690,321 | **3,744,864** |
| Modeled energy, pJ/input element, first window | — | 60,318,398 | **51,380,459** |
| Converted-neuron storage | — | 53,888 B | **46,496 B** |
| Offline conversion/build time | — | 5,262 s | **2,541 s** |
| Full SNN evaluation time | — | 6,732 s | **4,945 s** |

The adaptive run's preloaded batch-1, 1024-token forward benchmark had a
19.657 s median over three runs and 3.102 GB PyTorch peak allocation. The
historical rule and ANN runs did not record matched forward benchmarks, so
these values have no direct hardware-controlled counterpart in this table.

Adaptive versus the historical rule has +0.1861 PPL (worse) while the first
window has 20.2% fewer spikes, 14.8% less modeled energy, and 13.7% less
converted-neuron storage. The 26.5% shorter total SNN evaluation time is
observational because the runs were not benchmarked together on the same host.
Build times are not a method-speed comparison because fitting epochs differ.

The 1,302 per-site adaptive selection records include 1,036 choices of `N=1`
and 266 of `N=2`. T choices were 2: 650, 3: 48, 4: 24, 6: 266, 8: 48,
and 16: 266. In 290 records, no candidate met its local fit target and the
selector used its lowest-error fallback. Selection records include repeated
call sites that share fitted primitives.

The **matched 10-epoch T<=8 validation gate** is separately reported in
`adaptive_g1_medium_t8_comparison.md`; there adaptive had +0.0147 PPL versus
rule on 32 validation windows with 55.3% fewer first-window spikes. That gate
cannot substitute for a matched full-test comparison. To isolate the allocation
method on the full test, run PASN_rule for 10 epochs with the same model,
dataset, machine, and `T<=16`, or run adaptive for 300 epochs.

Raw adaptive result: `adaptive_full_medium_t16_e10.json`.
