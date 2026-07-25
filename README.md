# PASN — Prefix-Adaptive Spiking Neuron (research project)

This project builds an improved training-free ANN→SNN conversion framework
(**PASN**) on top of the **MBE neuron** baseline from *"Training-Free
ANN-to-SNN Conversion for High-Performance Spiking Transformers"* (AAAI-26, paper
in [`paper/`](paper/)). PASN's method is described in
[`PASN_method.md`](PASN_method.md); the overall reproduction plan is in
[`MBE_Implementation_Plan.md`](MBE_Implementation_Plan.md).

## Status

Phases 1–4 are implemented and verified on CPU (16/16 unit tests):

- **MBE neuron core** — [`src/mbe/neuron.py`](src/mbe/neuron.py) (Eq. 4–8),
  surrogate-gradient Heaviside — [`src/mbe/surrogate.py`](src/mbe/surrogate.py).
- **Signed (polarity-split) neuron** — [`src/mbe/signed.py`](src/mbe/signed.py),
  for non-monotonic targets (GELU/SiLU near-zero bend).
- **Function-fitting harness** — [`src/mbe/fit.py`](src/mbe/fit.py) (Adam + LBFGS
  + closed-form readout), targets/calibration — [`src/mbe/functions.py`](src/mbe/functions.py).
- **Table X / VII reproduction** — analysis in [`MBE_RESULTS.md`](MBE_RESULTS.md).
- **Phase 3 — spiking Transformer primitives** — [`src/mbe/spiking_ops.py`](src/mbe/spiking_ops.py)
  (spike-driven FP-mult, Softmax, LayerNorm), verified in
  [`experiments/verify_phase3.py`](experiments/verify_phase3.py); see [`PHASE3_RESULTS.md`](PHASE3_RESULTS.md).
- **Phase 4 — ANN→SNN conversion framework** — [`src/mbe/convert.py`](src/mbe/convert.py)
  (calibrate → build → replace) + toy Transformer [`src/mbe/toy.py`](src/mbe/toy.py),
  verified in [`experiments/verify_phase4.py`](experiments/verify_phase4.py).
- **PASN — Prefix-Adaptive Spiking Neuron** — [`src/mbe/pasn.py`](src/mbe/pasn.py)
  (FP-exponent prefix router + per-binade banks + adaptive budget). Method spec:
  [`PASN_method.md`](PASN_method.md) §9–13. Comparisons vs MBE (function approx,
  assembled ops, Phase-4 conversion) in [`results/pasn_vs_mbe.md`](results/pasn_vs_mbe.md)
  and [`results/pasn_ops_and_phase4.md`](results/pasn_ops_and_phase4.md), figures in
  [`results/`](results/).

Project knowledge base (Obsidian): [`PASN_vault/`](PASN_vault/).

Next: apply the PASN conversion to a real pretrained model (GPT-2 × WikiText-2) on
GPU (vast.ai); ImageNet/NLU/NLG benchmarks are further out (see the plan).

## Environment

Use the `SNN` conda environment (Python 3.10, torch 2.11 CPU):

```bash
conda activate SNN
```

## Quick start

```bash
# unit tests (16)
python tests/test_mbe_neuron.py

# reproduce Table X (MSE vs number of bases N, with/without decay)
python experiments/reproduce_table10.py

# Phase 3: verify spike-driven FP-mult / Softmax / LayerNorm vs exact ops
python experiments/verify_phase3.py

# Phase 4: ANN->SNN conversion of a toy Transformer (forward equivalence)
python experiments/verify_phase4.py

# PASN vs MBE: MSE-vs-spikes Pareto over the nonlinearities (+ figures)
python experiments/compare_pasn_mbe.py --json results/pareto_all.json
python experiments/plot_pasn_comparison.py results/pareto_all.json
# PASN vs MBE on assembled ops and on Phase-4 conversion
python experiments/compare_ops_pasn_mbe.py
python experiments/compare_phase4_pasn_mbe.py
```

## Layout

```
src/mbe/         MBE + PASN package (neuron, signed, pasn, spiking_ops, convert, toy, fit)
experiments/     reproduction + comparison scripts
tests/           unit tests (16)
results/         generated JSON / markdown tables + figures
PASN_vault/      Obsidian knowledge base (overview, implementation, results, planning)
paper/           baseline paper + extracted appendix pages (gitignored; not redistributed)
```
