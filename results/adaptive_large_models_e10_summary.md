# PASN adaptive N/T: large-model 10-epoch results

All adaptive evaluations have finished and their raw JSON and logs are saved
locally. Stage-2 conversion covers activation, LayerNorm, softmax, and the
supported attention matmuls. Adaptive offline fitting searched `N<=2` and
`T<=16` with 10 epochs and local target MSE `1e-3`.

**Run status (2026-10-09):** At the user's request, the subsequent 300-epoch
pipeline was stopped during its first ViT-B primitive fit. It produced no
300-epoch accuracy result. The four completed full-evaluation JSON files were
checked against the Vast.ai copies with matching SHA-256 hashes. No experiment
process remains active on the instance, so it can be removed without losing
these completed records.

## Full-dataset accuracy

| Model / metric | ANN | PASN_rule | PASN_adaptive | Comparison status |
|---|---:|---:|---:|---|
| GPT-2-medium, WikiText-2 test PPL ↓ | 21.7058 | 21.8536 | **21.8513** | Rule and adaptive both 10 epochs; matched full test |
| ViT-B, ImageNet-1k top-1 ↑ | 80.316% | **80.300%** | 80.224% | Historical rule 300 epochs; adaptive 10 |
| ViT-M, ImageNet-1k top-1 ↑ | 84.922% | **84.920%** | 84.812% | Historical rule 300 epochs; adaptive 10 |

The GPT-2 adaptive/rule PPL difference is only −0.00232, with both converted
models about +0.67% above ANN PPL. The ViT-B and ViT-M adaptive conversion
losses versus ANN are −0.092 and −0.110 percentage points, respectively.
These are 0.076 and 0.108 points below the historical 300-epoch rule rows.
The different epoch budgets prevent attributing those ViT gaps to the
allocation method alone.

## Cost and fit tradeoff

| Model | First-sample spikes vs rule | Modeled energy vs rule | Converted-neuron bytes vs rule | Measured adaptive forward median | Adaptive build time |
|---|---:|---:|---:|---:|---:|
| GPT-2-medium | −36.6% | −10.5% | −2.8% | 19.657 s, batch 1 × 1024 | 2,541 s |
| ViT-B | −33.2% | −20.5% | −14.2% | 10.162 s, batch 32 | 1,505 s |
| ViT-M | −28.1% | −18.7% | −13.1% | 8.131 s, batch 32 | 1,297 s |

The matched GPT-2 rule forward median was 28.567 s, so adaptive was 31.2%
faster on the same RTX 3090 and input shape; PyTorch peak GPU allocation was
effectively identical (3.102 GB). GPT-2 rule build time was 156 s, making
adaptive offline fitting 16.3× slower. Historical ViT rule records have no
matched forward or memory benchmark, and their 300-epoch build times are not
directly comparable to the adaptive 10-epoch build times.

Spikes and modeled energy were measured on a single model input, not averaged
over the full test/validation sets. Forward medians exclude data loading and
are three-run wall-clock benchmarks; they are not neuromorphic latency.
Storage is for converted neuron parameters, not entire model weights.

## Observed adaptive choices

| Model | Per-site N=1 | Per-site N=2 | Selected T values | Local target misses |
|---|---:|---:|---|---:|
| GPT-2-medium | 1,036 | 266 | 2, 3, 4, 6, 8, 16 | 290 / 1,302 |
| ViT-B | 532 | 110 | 2, 3, 4, 6, 8, 16 | 134 / 642 |
| ViT-M | 508 | 122 | 2, 3, 4, 6, 8, 16 | 134 / 630 |

The per-site records can repeat shared fitted primitives; they are not counts
of independent parameter sets. The N=1/N=2 and T distributions show actual
allocation of both dimensions, but target misses mean the selected point is
not uniformly feasible under the local error constraint.

## Interpretation

- The matched GPT-2 run supports a cost reduction at essentially unchanged
  PASN-rule PPL, but both 10-epoch conversions are about 0.67% worse than ANN
  PPL. It does not establish accuracy preservation relative to ANN.
- On ViT, the adaptive 10-epoch top-1 losses are small in absolute points,
  but the historical 300-epoch rule rows are more accurate. The fit-budget
  mismatch prevents a causal accuracy comparison.
- Both N and T actually vary in the full adaptive models. The 20–22% local
  target-miss rates and the much longer GPT-2 fitting time remain method
  limitations that should be reported with the cost savings.

For a publication-quality accuracy-cost comparison on ViT, fit either
PASN_rule for 10 epochs on the same instance or PASN_adaptive for 300 epochs.
The latter also tests whether adaptive retains its cost advantage near the
historical rule's higher accuracy. Published MBE results use different
checkpoints and should be compared only through relative conversion loss.

Raw records: `adaptive_full_medium_t16_e10.json`,
`rule_full_medium_t16_e10.json`, `adaptive_vitb_t16_e10.json`,
`adaptive_vitm_t16_e10.json`, `e8_vit_imagenet.json`, and
`e8m_vit_medium.json`.
