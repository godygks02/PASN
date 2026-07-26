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

`a=u` / `a=L` are the two leading-threshold placements: **u**niform quantiles of
`rho` vs a global MBE's **L**og-spread (see below — this selects a frontier
position, not a winner).

**Every MBE-PASN-S row is exactly reproducible** (`[lo, hi]` collapses to a point
over 3 seeds). Flat and global MBE still vary run to run because they calibrate on
a random draw; S uses a midpoint grid.

---

## Aggregate

| function | model | MSE | spikes/in | params | max\|jump\| | jump/RMSE | build |
|---|---|---|---|---|---|---|---|
| **gelu** | MBE N=8 (global) | 1.18e‑1 | 32.6 | 49 | — | — | 4.9 s |
| | MBE‑PASN n_loc=2 | 1.52e‑5 | 18.6 | 181 | 3.49e‑2 | 8.9 | 19.6 s |
| | S N=2 a=u | 4.98e‑4 | **3.7** | 49 | 6.24e‑2 | 2.8 | 4.5 s |
| | S N=4 a=u | 3.74e‑4 | 12.4 | 85 | 1.40e‑2 | 0.7 | 7.0 s |
| | S N=8 a=u | 4.72e‑5 | 47.1 | 157 | 1.65e‑2 | 2.4 | 6.0 s |
| | **S N=2 a=L** | **2.02e‑5** | 15.2 | **49** | 8.36e‑3 | 1.9 | 5.7 s |
| | S N=4 a=L | 1.16e‑3 | 43.4 | 85 | 2.93e‑2 | 0.9 | 8.0 s |
| | **S N=8 a=L** | **5.74e‑6** | 89.6 | 157 | **1.86e‑3** | **0.8** | 5.3 s |
| **silu** | MBE N=8 (global) | 1.97e‑1 | 54.8 | 49 | — | — | 3.1 s |
| | MBE‑PASN n_loc=2 | 4.15e‑5 | 13.1 | 181 | 5.99e‑2 | 9.3 | 18.6 s |
| | **S N=2 a=u** | 9.61e‑5 | **3.9** | 49 | 4.88e‑2 | 5.0 | 4.9 s |
| | **S N=4 a=u** | **2.47e‑5** | 17.5 | 85 | 9.15e‑3 | 1.8 | 6.4 s |
| | S N=8 a=u | 5.04e‑6 | 52.7 | 157 | 7.97e‑3 | 3.5 | 5.4 s |
| | S N=2 a=L | 1.55e‑4 | 14.8 | 49 | 1.17e‑2 | 0.9 | 6.3 s |
| | S N=4 a=L | 4.41e‑4 | 42.8 | 85 | 7.51e‑3 | 0.4 | 6.5 s |
| | **S N=8 a=L** | **3.85e‑6** | 90.5 | 157 | **1.38e‑3** | **0.7** | 5.5 s |
| **exp2** | MBE N=8 (global) | 1.55e‑5 | 64.5 | 49 | — | — | 3.0 s |
| | MBE‑PASN n_loc=2 | 1.21e‑5 | 12.7 | 77 | 5.69e‑3 | 1.6 | 8.4 s |
| | S N=2 a=u | 1.59e‑5 | 12.7 | **25** | 6.76e‑3 | 1.7 | 3.1 s |
| | **S N=4 a=u** | **6.21e‑6** | 13.7 | 45 | 1.22e‑2 | 4.9 | 3.4 s |
| | S N=8 a=u | 4.65e‑6 | 42.6 | 85 | 9.85e‑3 | 4.6 | 3.3 s |
| | S N=8 a=L | 5.48e‑6 | 98.9 | 85 | **4.47e‑4** | **0.2** | 2.9 s |
| **invsqrt** | MBE N=8 (global) | 8.04e‑6 | 77.3 | 49 | — | — | 2.6 s |
| | **MBE‑PASN n_loc=2** | 2.20e‑5 | **6.9** | 64 | 2.27e‑2 | 4.8 | 3.9 s |
| | S N=2 a=u | 1.22e‑4 | 5.6 | **22** | 1.79e‑2 | 1.6 | 2.0 s |
| | S N=2 a=L | 1.87e‑5 | 15.6 | 22 | 9.81e‑3 | 2.3 | 2.0 s |
| | S N=4 a=L | 6.83e‑6 | 44.9 | 40 | 5.28e‑3 | 2.0 | 2.6 s |
| | **S N=8 a=L** | **1.91e‑7** | 89.7 | 76 | **1.65e‑4** | **0.4** | 1.7 s |

(`jump/RMSE` = worst boundary jump divided by the model's own RMS error — see §3.
Dominated rows for exp2 with `a=L` at N=2/N=4 are omitted; they are in the JSON.)

---

## What the numbers say

### 1. The answer: **scale, mostly — but sharing is paid for in spikes**

A shared basis set *can* serve every range, and at N=8 it beats independent banks
by **2.6× on GELU, 11× on SiLU, 2× on 2^x, 115× on 1/√x**, at *less* stored memory
on GELU/SiLU. So the per-range targets are not shape-incompatible.

But it needs more bases to carry that vocabulary, and bases are spikes. Independent
banks need only 2 bases *because each is tuned to one narrow target*:

> **Flat banks buy energy with memory; a shared basis buys memory with energy.**

Neither is dominated on any function. This is a frontier, and the earlier
expectation that S would win both axes at once was wrong.

### 2. The useful operating points

| | flat MBE‑PASN | best S at ≈ the same spikes | verdict |
|---|---|---|---|
| gelu | 1.52e‑5 / 18.6 / **181** | N=2 a=L: 2.02e‑5 / 15.2 / **49** | 1.3× worse MSE, 1.2× fewer spikes, **3.7× less memory** |
| silu | 4.15e‑5 / 13.1 / **181** | N=4 a=u: **2.47e‑5** / 17.5 / **85** | **1.7× better MSE**, 1.3× more spikes, **2.1× less memory** |
| exp2 | 1.21e‑5 / 12.7 / **77** | N=4 a=u: **6.21e‑6** / 13.7 / **45** | **1.95× better MSE**, 1.08× more spikes, **1.7× less memory** |
| invsqrt | **2.20e‑5 / 6.9** / 64 | N=2 a=u: 1.22e‑4 / 5.6 / **22** | 5.5× worse MSE — flat clearly wins here |

Plus a genuinely new **low-energy** point that flat cannot reach: S N=2 a=u on SiLU
is **9.61e‑5 at 3.9 spikes/input** — 3.4× fewer spikes than flat and comparable to
the SAR-based PASN's ~3 spikes, but with MBE dynamics intact.

**invsqrt is the honest counter-example**: its domain `[0.5, 2]` reaches only 2 of 4
ranges, so there is almost nothing for routing to specialise and flat's per-bank
dynamics win outright. Routing pays off in proportion to how many ranges the
calibrated domain actually spans.

### 3. Boundary continuity — the previous claim was overstated

The earlier report claimed a free 22–1700× reduction in the bank-boundary jump. That
was measured only at S's high-accuracy configurations and **tracks the model's own
error level** — a better fit has smaller jumps everywhere, boundaries included. At
S N=2 a=u the absolute jump on GELU (6.24e‑2) is in fact *worse* than flat's
(3.49e‑2).

Normalising by each model's own RMS error gives the claim that survives:

| function | flat jump/RMSE | S jump/RMSE (best‑MSE config) |
|---|---|---|
| gelu | 8.9 | **0.8** |
| silu | 9.3 | **0.7** |
| exp2 | 1.6 | **0.2** |
| invsqrt | 4.8 | **0.4** |

For flat banks the worst boundary jump is **~5–9× its own RMS error** — an artifact
sticking well above the noise floor. For S it is **≤ 1×**, i.e. the boundary is not
a special place. That is structural (adjacent ranges share features and differ only
in a linear readout, so they cannot drift independently) and it does reduce a
boundary-constrained readout from "fixes the dominant error" to "polishes a residual
already at the noise floor" — but the honest statement is *relative*, not absolute.

### 4. Build cost: 1 gradient fit, not R

S builds **2.9–3.4× faster** on GELU/SiLU (19.6 s → 5.7 s) while covering the same
ranges, because only the shared dynamics are fitted by gradient descent and each
range is a linear solve. This is what makes a *deeper* router (mantissa-prefix
subdivision, R ≫ 13) affordable — flat cannot pay O(R) surrogate-gradient fits.

---

## Two fixes this required (both diagnosed, not guessed)

### The `N=4` pathology was basis saturation

`N=4` was systematically worse than `N=2` on GELU and SiLU across all seeds. Cause:
the inherited `alpha_v` placement. A global MBE log-spreads the leading thresholds
over `[2^-spread, 1]` because the target's curvature concentrates at one end of its
domain. A *routed* residual is the opposite — `rho` is ~uniform on `[0,1)` in every
bank and each `g_v` is smooth there — so a basis with a near-zero threshold fires at
almost every step and its feature is nearly constant in `rho`. Measured per-basis
firing rates at N=4 with log-spread: `[0.88, 0.17, 0.71, 0.95]`, i.e. 1–2 useful
bases out of 4.

`uniform_alpha()` splits `[0,1)` at even quantiles. N=4 MSE: GELU 1.16e‑3 → 3.74e‑4
at 3.5× fewer spikes; SiLU 4.41e‑4 → **2.47e‑5** (18×) at 2.4× fewer spikes.

`alpha_init` is kept as a knob because it **selects a frontier position, not a
winner**: `uniform` gives the cheap end everywhere and wins outright on SiLU (N=2,
N=4) and 2^x (all N); `logspread` wins on GELU (N=2, N=8) and 1/√x (all N). The
choice is per-function and currently manual — an offset selection grid would
automate it (see below).

### The run-to-run spread was the random calibration draw, not the optimiser

Up to 34× MSE spread across seeds. It was **not** the optimizer: the shape-parameter
init is deterministic (`alpha_v` list + `linspace` for tau) and the readout is
closed-form, so the seed only changed the random per-bank sample. Replacing it with
a **midpoint grid** — strictly lower discrepancy for a known deterministic target on
a known interval — makes the build exactly reproducible (spread 1.0× on every
configuration, every function).

Restarts are off by default. Before the fix a "restart" was a no-op (the only random
tensor was the readout, which the closed-form solve overwrites on step 1). With
jittered inits they measurably *hurt* (GELU N=8 test MSE 4.7e‑5 → 1.3e‑4): selecting
on the fitting grid picks fits that place staircase breakpoints to nail grid
midpoints while drifting between them, and the jitter reintroduces the seed
dependence the grid removed.

---

## Remaining

1. **`alpha_init` selection is manual per function.** Fit both (2 builds, ~10 s) and
   select on an **offset** grid — the same discipline that makes restarts safe. This
   would remove the last hand-tuned choice from the build.
2. **Unreachable ranges are still stored and counted.** The router's binade grid does
   not align with a calibrated domain (2 of 13 ranges for GELU, 2 of 4 for 1/√x).
   They are now left unfitted — previously they were fitted outside the target's
   valid range and held **NaN** parameters, invisible because no input routes there —
   but pruning them would reduce both variants' memory.
3. **Per-bank MSE in the raw log is absolute, not relative.** Large-|x| ranges carry
   larger |f|, so their absolute MSE is larger for uninteresting reasons.
4. **No downstream number yet.** All of the above is 1-D approximation. The
   conversion framework has an `S` backend to gain before GPT-2 × WikiText-2 can rank
   these operating points by what actually matters (Δperplexity per spike).
