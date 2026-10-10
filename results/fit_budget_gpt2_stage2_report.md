# Offline-fitted bank budgets on pretrained GPT-2: CPU pilot

## Setup

- Checkpoint: cached pretrained `gpt2` (12 blocks), identical for both arms.
- Data: cached `Salesforce/wikitext`, `wikitext-2-raw-v1` Arrow files. Direct Arrow
  reading bypassed a stalled `datasets.load_dataset` cache lookup; token joining
  and calibration prefix remain the evaluator's canonical recipe.
- Conversion: Stage 2 (`--convert-ops all`), MBE-PASN, `T <= 4`, `N <= 2`, 10
  fitting epochs, per-bank absolute MSE target `1e-3`; tied identity uses the
  existing relative target `1e-2`. Same checkpoint, calibration and evaluation
  tokens. The only built `ConvertConfig` difference is `pasn_budget`.
- Evaluation: non-overlapping blocks of 64 tokens, first 4 or 16 test blocks.
  This is a small CPU pilot, **not** the paper's full-context, full-corpus result.
- Spike and modeled energy measurements use the same first 64-token test batch.

## Results

| Measure | Rule | Offline fit |
|---|---:|---:|
| Build time | 40.7 s | 274.2 s |
| Stored bytes | 23,696 | 20,048 |
| Spikes per model input element | 219,744 | 213,344 |
| Modeled operation energy, pJ/input element | 4,976,918 | 4,055,507 |
| ANN PPL, 4 blocks | 93.4528 | 93.4528 |
| SNN PPL, 4 blocks | 94.5742 | 91.7327 |
| ANN PPL, 16 blocks | 98.2637 | 98.2637 |
| SNN PPL, 16 blocks | **99.1079** | 99.4714 |

The fit selector saves **2.91% spikes** and **15.40% stored bytes**, at **6.74x**
the offline build time. On 16 blocks its conversion loss is **+1.229%** versus
**+0.859%** for the rule. The first four blocks reverse that accuracy ordering;
on blocks 5–16 alone, inferred PPL is 100.6669 (rule) versus 102.1933 (fit).

## Interpretation

This establishes that offline selection of bank-specific `(N, T)` works inside
the full pretrained GPT-2 Stage 2 conversion and changes the network's cost.
It does **not** establish an accuracy-preserving improvement: the local held-out
bank MSE constraint did not preserve the rule arm's PPL on the longer sample.
The next selector should account for bank occupancy and downstream sensitivity,
and must be validated on a separate network-level calibration slice before a
full-corpus claim.

Machine-readable records: `fit_budget_gpt2_stage2_pilot.json` (4 blocks) and
`fit_budget_gpt2_stage2_16blocks.json` (16 blocks). The 16-block logs are
`fit_budget_gpt2_rule_16.log` and `fit_budget_gpt2_fit_16.log`.
