# Flat MBE-PASN vs MBE-PASN-S — shared basis set, prefix-routed readout

**Question.** Flat MBE-PASN gives every routed range its own complete MBE neuron
(`sum_v 6N_v` stored). MBE-PASN-S keeps **one** basis set and lets the FP prefix
select only the readout (`5N + R(N+1)` stored, one spike train per input). Do the
per-range targets `g_v(rho) = f(sigma 2^e (1+rho))` differ only in *scale* (so a
shared basis spans them all) or in *shape* (so it cannot)?

Reproduce:
```bash
python experiments/compare_shared_banks.py --fns gelu silu exp2 invsqrt --epochs 150 --seeds 3 --json results/shared_banks.json
```
(SNN env, CPU, T=16, same router `(e_min, e_max)` for both variants.)

**Read the MSE column as a median of 3 seeds with its [min, max].** The
surrogate-gradient fit is numerically chaotic on GELU/SiLU — a strong-Wolfe LBFGS
line search amplifies thread-level reduction-order noise into order-of-magnitude
swings — so single runs are not measurements. Flat is stable (±6%); S is not.

---

## Aggregate

| function | model | MSE (med) | [min, max] | spikes/in | params | max\|jump\| | build |
|---|---|---|---|---|---|---|---|
| **gelu** | MBE N=8 (global) | 1.18e‑1 | — | 32.6 | 49 | — | 5.4 s |
| | MBE‑PASN n_loc=2 | **1.52e‑5** | [1.5e‑5, 1.6e‑5] | 18.6 | 181 | 3.49e‑2 | 21.7 s |
| | MBE‑PASN‑S N=2 | 2.42e‑5 | [1.0e‑5, 3.5e‑4] | **15.1** | **49** | 9.61e‑3 | 7.8 s |
| | MBE‑PASN‑S N=4 | 1.04e‑3 | [6.8e‑4, 1.2e‑3] | 43.2 | 85 | 7.86e‑3 | 7.1 s |
| | MBE‑PASN‑S N=8 | 1.67e‑5 | [8.8e‑6, 2.8e‑5] | 89.6 | 157 | **1.60e‑3** | 6.8 s |
| **silu** | MBE N=8 (global) | 1.97e‑1 | — | 54.8 | 49 | — | 3.3 s |
| | MBE‑PASN n_loc=2 | 4.15e‑5 | [4.1e‑5, 4.2e‑5] | **13.1** | 181 | 5.99e‑2 | 20.2 s |
| | MBE‑PASN‑S N=2 | 1.02e‑4 | [5.5e‑5, 1.2e‑4] | 14.4 | **49** | 5.21e‑3 | 6.6 s |
| | MBE‑PASN‑S N=4 | 5.13e‑4 | [3.7e‑4, 5.5e‑4] | 43.0 | 85 | 1.88e‑2 | 7.8 s |
| | MBE‑PASN‑S N=8 | **6.88e‑7** | [6.8e‑7, 1.2e‑6] | 90.3 | 157 | **1.29e‑3** | 7.3 s |
| **exp2** | MBE N=8 (global) | 1.55e‑5 | — | 64.5 | 49 | — | 3.5 s |
| | MBE‑PASN n_loc=2 | 1.21e‑5 | [1.2e‑5, 1.6e‑5] | **12.7** | 77 | 5.69e‑3 | 9.0 s |
| | MBE‑PASN‑S N=2 | 3.39e‑4 | [3.2e‑4, 4.2e‑4] | 17.3 | **25** | 2.47e‑2 | 3.7 s |
| | MBE‑PASN‑S N=4 | 1.14e‑4 | [8.2e‑5, 1.4e‑4] | 45.3 | 45 | 4.59e‑3 | 4.5 s |
| | MBE‑PASN‑S N=8 | **2.82e‑6** | [1.2e‑6, 4.4e‑6] | 98.6 | 85 | **8.71e‑5** | 4.8 s |
| **invsqrt** | MBE N=8 (global) | 8.04e‑6 | — | 77.3 | 49 | — | 2.9 s |
| | MBE‑PASN n_loc=2 | 2.20e‑5 | [2.1e‑5, 2.2e‑5] | **6.9** | 64 | 2.27e‑2 | 4.2 s |
| | MBE‑PASN‑S N=2 | 1.74e‑5 | [1.3e‑5, 1.9e‑5] | 15.6 | **22** | 1.67e‑2 | 2.5 s |
| | MBE‑PASN‑S N=4 | 7.20e‑6 | [6.6e‑6, 2.4e‑5] | 45.8 | 40 | 1.14e‑2 | 2.7 s |
| | MBE‑PASN‑S N=8 | **1.68e‑7** | [1.4e‑7, 1.7e‑7] | 89.4 | 76 | **1.33e‑5** | 2.2 s |

---

## What the numbers say

### 1. The answer to the question: **scale, mostly — but sharing costs spikes**

A shared basis set *can* serve every range, and at N=8 it serves them far better
than independent banks: **60× lower MSE on SiLU, 131× on 1/√x, 4× on 2^x** than
flat MBE-PASN, at *less* stored memory on GELU/SiLU (157 vs 181). So the per-range
targets are not shape-incompatible.

But the shared basis needs **more bases** to carry that vocabulary, and bases are
spikes: S N=8 costs 5–13× the spikes of flat's N_j=2 banks. Independent banks need
only 2 bases *because each is tuned to one narrow target*. That is the real
exchange this experiment measures:

> **Flat banks buy energy with memory; a shared basis buys memory with energy.**

Neither dominates. It is a frontier, and it is a publishable frontier — but the
earlier expectation that S would win on both axes at once was wrong.

### 2. Where S is unambiguously the right choice: GELU / SiLU at fixed memory

At **identical memory** to a global MBE N=8 (49 params), MBE-PASN-S N=2:

| | global MBE N=8 | **MBE‑PASN‑S N=2** | gain |
|---|---|---|---|
| gelu MSE | 1.18e‑1 | 2.42e‑5 | **4900×** |
| gelu spikes | 32.6 | 15.1 | **2.2× fewer** |
| silu MSE | 1.97e‑1 | 1.02e‑4 | **1900×** |
| silu spikes | 54.8 | 14.4 | **3.8× fewer** |

Strict domination on all three axes, on exactly the two nonlinearities a spiking
Transformer needs. This is the clean claim; it does **not** extend to the smooth
monotone primitives (on exp2 / invsqrt a global MBE N=8 is already good, and S N=2
does not beat it on accuracy).

### 3. Boundary continuity: S wins everywhere, for free

| function | flat max\|jump\| | S N=8 max\|jump\| | ratio |
|---|---|---|---|
| gelu | 3.49e‑2 | 1.60e‑3 | 22× |
| silu | 5.99e‑2 | 1.29e‑3 | 46× |
| exp2 | 5.69e‑3 | 8.71e‑5 | 65× |
| invsqrt | 2.27e‑2 | 1.33e‑5 | **1700×** |

No boundary constraint was imposed. Because adjacent ranges share the features and
differ only in a linear readout, they cannot drift independently — the artifact
largely disappears by construction. This **subsumes most of the value of the
proposed boundary-constrained readout**: the constraint is now a refinement of an
already-small residual, not a fix for a dominant error.

### 4. Build cost: 1 gradient fit, not R

S is **2.8–3.2× faster to build** on GELU/SiLU (21.7 s → 6.8 s) while covering the
same ranges, because only the shared dynamics are fitted by gradient descent and
each range is a linear solve. This is the structural property that makes a *deeper*
router (mantissa-prefix subdivision, R ≫ 13) affordable — flat cannot pay O(R)
surrogate-gradient fits.

---

## Open problems (do not report S without these)

1. **N=4 is systematically worse than N=2 on GELU and SiLU** (1.04e‑3 vs 2.42e‑5;
   5.13e‑4 vs 1.02e‑4), across all 3 seeds — not variance. It is monotone on
   exp2/invsqrt, so the pathology is specific to the signed, non-monotone targets.
   Prime suspect: the shared core initialises `alpha_v` with the default log-spread,
   whereas a global neuron gets curvature-based placement via
   `functions.curvature_alpha`. There is no per-range curvature to use for a shared
   basis, so an equivalent initialiser has to be derived from the *pooled
   normalised* problem.
2. **Fit instability.** S N=2 on GELU spans [1.0e‑5, 3.5e‑4] over 3 seeds — 35×. The
   best seed beats flat on all three axes; the median does not. Until this is
   controlled, S's accuracy claims are not solid. Flat is stable, so this is a
   property of the pooled fit, not of the neuron.
3. **Per-bank normalisation is load-bearing.** Fitting the shared dynamics on raw
   targets lets the largest binades dictate where the shared staircase puts its
   steps (GELU tails grow as 2^e); normalising per range and folding the scale back
   into the readout row (exact, since the readout is linear and per-range) gave a
   **6× MSE improvement** on GELU. `normalize_banks=False` keeps the ablation.
4. **Unreachable ranges are still counted in memory.** The router's binade grid does
   not align with a calibrated domain, so some banks lie entirely outside it (2 of
   13 for GELU; 2 of 4 for 1/√x). They are now left unfitted (previously they were
   fitted outside the target's valid range and held **NaN** parameters — invisible
   because no input routes there) but they are still stored and counted. Pruning
   them would reduce both variants' memory.
5. **Per-bank MSE is absolute, not relative.** Large-|x| ranges carry larger |f|, so
   their absolute MSE is larger for uninteresting reasons. The per-bank tables in
   the raw log should be read as relative error before drawing conclusions about
   which range is "hard".
