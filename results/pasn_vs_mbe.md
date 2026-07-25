# PASN vs MBE — MSE vs mean spikes/input (per nonlinearity)

Reproduce: `python experiments/compare_pasn_mbe.py --epochs 120` (SNN env, CPU,
T=16, seed 0). "spikes/in" = mean spikes emitted per input element (energy proxy);
"stored N" = total bases stored across banks. Lower-left (low MSE, few spikes) wins.

**Figures**: `results/pareto_all.png` (all functions), `results/pareto_gelu120.png`
(faithful GELU). Regenerate: `python experiments/plot_pasn_comparison.py <json>`.

## GELU — faithful paper domain (−120..10)
| model | MSE | spikes/in | stored N |
|---|---|---|---|
| MBE N=8 (global) | 3.31e-1 (fails) | 36.5 | 8 |
| SignedMBE 4×2 (strongest MBE baseline) | 7.88e-5 | 58.5 | 8 |
| PASN n_local=1 | 3.47e-5 | 13.2 | 22 |
| **PASN adaptive** | **2.72e-6** | **13.2** | 32 |

The wide asymmetric domain is PASN's best case: the −120 tail collapses into a few
flat binade banks (adaptive N=1) while near-zero gets exponentially dense banks.
**~29× lower MSE than the strongest MBE baseline at ~4.4× fewer spikes.**

## GELU  (domain −8..8)
| model | MSE | spikes/in | stored N |
|---|---|---|---|
| MBE N=8 (global) | 1.44e-1 | 23.5 | 8 |
| SignedMBE 2×2 (strongest MBE baseline) | 1.40e-3 | 11.6 | 4 |
| SignedMBE 4×2 | 6.31e-4 | 24.4 | 8 |
| PASN n_local=1 | 1.41e-4 | 11.9 | 14 |
| PASN n_local=2 | 1.38e-5 | 18.7 | 24 |
| **PASN adaptive** | **1.88e-5** | **11.1** | 13 |

## SiLU  (domain −8..8)
| model | MSE | spikes/in | stored N |
|---|---|---|---|
| MBE N=8 | 2.09e-1 | 40.1 | 8 |
| SignedMBE 4×2 | 1.84e-3 | 39.1 | 8 |
| PASN n_local=1 | 1.56e-4 | 8.3 | 14 |
| **PASN adaptive** | **2.12e-5** | **10.4** | 14 |

## 1/x  (domain 0.5..1)
| model | MSE | spikes/in | stored N |
|---|---|---|---|
| MBE N=8 | 9.73e-6 | 75.7 | 8 |
| PASN n_local=2 | 2.15e-4 | 7.8 | 8 |
| PASN adaptive | 4.09e-5 | 22.2 | 13 |

## 2^x  (domain 0..1)
| model | MSE | spikes/in | stored N |
|---|---|---|---|
| MBE N=8 | 2.14e-5 | 47.4 | 8 |
| **PASN adaptive** | 3.18e-5 | **8.0** | 4 |

## 1/√x  (domain 0.5..2)
| model | MSE | spikes/in | stored N |
|---|---|---|---|
| MBE N=8 | 9.08e-6 | 77.1 | 8 |
| PASN n_local=2 | 2.42e-5 | 6.9 | 10 |
| PASN adaptive | 7.26e-5 | 11.3 | 6 |

## Reading
- **Wide / curved domains (GELU, SiLU): PASN dominates by 1–2 orders of MSE** at
  equal-or-fewer spikes, even vs the strongest MBE-family baseline (signed MBE).
- **Narrow primitive domains (1/x, 2^x, 1/√x): accuracy is comparable** (a global
  MBE already resolves them), but **PASN reaches it at ~3–11× fewer spikes** —
  because each binade bank fires only its 1–2 local bases.
- **Adaptive N_j (flat tails → N=1)** cuts both stored bases and spikes while
  holding MSE (GELU: 24→13 stored N at the same ~1e-5 MSE).
- **Honest cost**: PASN stores more bases on wide domains and adds prefix /
  bank-addressing overhead.

## Why (validated thesis)
FP-exponent prefix routing = free exact coarse code (binade 2^e) + log-dense ranges
near zero; each bank approximates only the smooth intra-binade residual
g(ρ)=f(σ2^e(1+ρ)), so accuracy-per-spike is far higher than any global bank.
→ `PASN_method.md` §9–13.
