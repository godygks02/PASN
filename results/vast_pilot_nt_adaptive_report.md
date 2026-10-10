# Vast.ai N/T-adaptive GPT-2 Stage-2 pilot

## Setup

- Server: Vast.ai RTX 4060, 8,188 MiB VRAM, 16 GB instance disk. PyTorch
  2.11.0+cu128, transformers 5.8.1, datasets 4.8.5.
- Cached pretrained `gpt2` (12 layers), WikiText-2 raw validation, 8 non-overlapping
  blocks of 64 tokens. ANN and both SNN arms use identical evaluation tokens.
- All operations converted (Stage 2), `T<=8`, fit selector `N<=2`, 10 fit
  epochs, local absolute target MSE `1e-3`; tied identity uses its configured
  relative target. Both arms fit the banks on CPU and evaluate the network on GPU.
- The rule uses analytic bank budgets. Fit tests 2 basis counts × 5 timestep
  choices per reachable bank and picks the lowest measured bank spikes among
  feasible candidates, falling back to the lowest-error candidate if none meet
  the target.
- Spike and modeled energy totals are instrumented on the **first** validation
  block only. They are not evaluation-wide measurements. PPL covers all 8 blocks.

## Results

| Measure | Rule | Offline-fit N/T | Fit vs rule |
|---|---:|---:|---:|
| ANN PPL | 57.8123 | 57.8123 | same |
| SNN PPL | **58.5124** | 58.6696 | +0.1572 |
| Relative conversion loss | **+1.2109%** | +1.4828% | +0.2719 pp |
| Spikes per input token, first block | 498,998 | **317,229** | −36.43% |
| Modeled operation energy, pJ/input token, first block | 6,594,557 | **4,599,528** | −30.25% |
| Stored bytes | 23,696 | **20,048** | −15.40% |
| Offline build time | **89.7 s** | 935.5 s | 10.42× |

The fit arm's 654 recorded per-site bank selections chose `N=1` in every case.
Selected T was 2 in 326 entries, 3 in 24, 4 in 12, 6 in 256 and 8 in 36.
These entries include repeated call sites that share fitted primitives; they
are not counts of independent parameter sets. The rule's per-site budget table
includes both `N=1` and `N=2`.

Across all 1,787 per-site bank budget entries, rule used `N=2` in 146 entries.
It used `T=8` in 1,363 entries, whereas fit used `T=8` in only 36 and `T=6`
in 1,327. Thus the spike reduction cannot be attributed to `N=1` alone; the
shorter timestep allocation affects many more bank entries. This is a budget
table observation, not an ablation of the two effects.

**134 of 654 fit selections missed their local MSE target**, often the very
strict near-zero target of the tied identity. The fallback chooses the
lowest-error candidate but cannot call that bank feasible. This is a possible
reason network PPL worsens despite lower first-block cost; the pilot alone does
not establish causation.

## Decision

The implementation runs end to end on an actual pretrained model, and its
offline architecture selection reduces measured first-block spikes and storage.
This point fails the accuracy-preserving improvement criterion: fit has higher
validation PPL, and the cost measure covers only one block. Do not use it as a
paper performance claim. Before a larger model run, revise the selector to
handle local target misses and score candidate budgets against network-level
validation. Then meter costs over the evaluation set and test with full context.

Reproduction: `experiments/vast_small_pilot.sh`.
Records: `results/vast_pilot_ann.json`, `results/vast_pilot_rule.json`,
`results/vast_pilot_fit.json`; corresponding rule/fit logs are also saved.

## Additional global-MBE comparison

A local CPU run used the same cached pretrained `gpt2`, WikiText-2 raw
validation's first 8 × 64 tokens, Stage 2, T=8, 10 fit epochs and the same
calibration prefix. It used the repository's global-MBE defaults: six bases
per GELU side, eight for LayerNorm/Softmax/matmul, order-2 readout where
implemented, and log-sampled identity fitting. ANN PPL was 57.8116 on CPU
versus 57.8123 in the Vast GPU runs.

| Measure | Global MBE | PASN rule | PASN fit |
|---|---:|---:|---:|
| SNN PPL | 3517.0775 | **58.5124** | 58.6696 |
| Spikes/input token, first block | 1,225,724 | 498,998 | **317,229** |
| Modeled energy pJ/input token, first block | 27,754,491 | 6,594,557 | **4,599,528** |
| Stored bytes | 27,132 | 23,696 | **20,048** |

This global-MBE point is substantially worse in both accuracy and measured
first-block cost. It is a **single baseline point, not an MBE Pareto frontier**.
The signed GELU implementation ignores order-2 readout, so the decoder is not
fully matched to PASN. The run also used CPU rather than the Vast GPU for
evaluation; the ANN difference is small but the two measurements were not on
identical hardware. A definitive comparison needs a stronger matched global
MBE frontier and full evaluation-wide costs.

Global-MBE record: `results/mbe_gpt2_stage2_t8_validation.json`.
