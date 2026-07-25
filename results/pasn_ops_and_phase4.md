# PASN vs MBE on assembled ops and Phase-4 conversion

CPU, T=16, seed 0. Reproduce:
`python experiments/compare_ops_pasn_mbe.py` and
`python experiments/compare_phase4_pasn_mbe.py`.

## Assembled Transformer ops (Phase-3 primitives swapped MBE ↔ PASN)

| op | backend | output err | spikes/in (driver prim) |
|---|---|---|---|
| **FP multiply** ([0,64]) | MBE_Id | product rel 1.08e-2 | 47.8 (2 operands) |
| | **PASN_Id** | **product rel 1.37e-3** | **22.6** |
| **Softmax** | MBE | 5.64e-4 | 65.2 (2^x) |
| | **PASN** | 6.83e-4 | **17.5** |
| **LayerNorm** | MBE | 3.73e-3 | 75.5 (1/√x) |
| | **PASN** | **3.11e-3** | **6.4** |

- **FP multiply** reconstructs a *wide* operand range — PASN's home turf:
  **~8× lower product error at ~half the spikes**.
- **Softmax / LayerNorm** primitives live on narrow ranges, so the accuracy gap is
  small (softmax essentially tied), but **PASN still reaches it at ~4× (softmax) /
  ~12× (LayerNorm) fewer spikes**.

## Phase-4 conversion: MBE-converted vs PASN-converted toy Transformer

Same toy Transformer (d=16), converted twice; activation + activation×activation
matmuls use MBE vs PASN neurons (softmax/LN held identical). Forward error vs the
full-precision ANN, and spike cost of the dominant nonlinearity (GELU).

| depth | backend | forward rel\|err\| | GELU spikes/in |
|---|---|---|---|
| 1 layer | MBE | 2.49e-2 | 30.8 |
| 1 layer | **PASN** | **3.38e-3** | **11.4** |
| 2 layers | MBE | 6.88e-2 | 31.4 |
| 2 layers | **PASN** | **3.80e-3** | **11.4** |

- The GELU activation was the dominant Phase-4 conversion error (see
  `PASN_vault/10 - Implementation/Phase 4 - Conversion Framework.md`).
- Swapping activation + matmul to PASN **cuts the network forward error and the
  GELU spike cost ~2.7× at once** — better accuracy *and* lower energy.
- **The advantage compounds with depth**: MBE's error grows with layers
  (2.5e-2 → 6.9e-2) while PASN stays flat (3.4e-3 → 3.8e-3), so at 2 layers PASN is
  ~18× more accurate. This is the property that matters for real Transformers.

## Bottom line
Across function approximation, the three assembled ops, and end-to-end Phase-4
conversion, PASN improves the accuracy–spike frontier, with the largest wins where
the operand/activation range is wide (FP-mult, GELU/SiLU) and spike savings even
where accuracy is already tied (softmax/LN). → `results/pasn_vs_mbe.md`.
