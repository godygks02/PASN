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

**Every MBE-PASN-S row is exactly reproducible** — `[lo, hi]` collapses to a point
over 3 seeds, because the calibration grids, threshold placement, restart jitter and
readout solve are all deterministic (`seed` has no effect on the result). Flat and
global MBE still vary run to run; they calibrate on a random draw.

---

## Aggregate

| function | model | MSE | spikes/in | params | max\|jump\| | jump/RMSE | build |
|---|---|---|---|---|---|---|---|
| **gelu** | MBE N=8 (global) | 1.18e‑1 | 32.6 | 49 | — | — | 5.3 s |
| | MBE‑PASN n_loc=2 | 1.52e‑5 | **18.6** | 181 | 3.49e‑2 | 8.9 | 19.4 s |
| | S N=4 | 3.74e‑4 | **12.4** | 85 | 1.40e‑2 | 0.7 | 41.0 s |
| | **S N=2** | **2.95e‑6** | 23.3 | **49** | 8.78e‑3 | 5.1 | 34.1 s |
| | **S N=8** | **2.32e‑6** | 65.9 | 157 | **3.58e‑3** | 2.4 | 33.3 s |
| **silu** | MBE N=8 (global) | 1.97e‑1 | 54.8 | 49 | — | — | 3.3 s |
| | MBE‑PASN n_loc=2 | 4.15e‑5 | **13.1** | 181 | 5.99e‑2 | 9.3 | 18.4 s |
| | S N=4 | 2.47e‑5 | 17.5 | 85 | 9.15e‑3 | 1.8 | 40.8 s |
| | **S N=2** | **6.78e‑6** | 22.1 | **49** | 1.11e‑2 | 4.3 | 33.2 s |
| | **S N=8** | **1.94e‑6** | 101.7 | 157 | 6.60e‑3 | 4.7 | 36.2 s |
| **exp2** | MBE N=8 (global) | 1.55e‑5 | 64.5 | 49 | — | — | 3.8 s |
| | MBE‑PASN n_loc=2 | 1.21e‑5 | **12.7** | 77 | 5.69e‑3 | 1.6 | 8.3 s |
| | S N=4 | 6.21e‑6 | 13.7 | 45 | 1.22e‑2 | 4.9 | 21.0 s |
| | S N=2 | 5.74e‑6 | 19.5 | **25** | 1.34e‑3 | 0.6 | 17.6 s |
| | **S N=8** | **1.08e‑6** | 57.9 | 85 | **3.03e‑4** | **0.3** | 21.9 s |
| **invsqrt** | MBE N=8 (global) | 8.04e‑6 | 77.3 | 49 | — | — | 2.9 s |
| | MBE‑PASN n_loc=2 | 2.20e‑5 | **6.9** | 64 | 2.27e‑2 | 4.8 | 3.8 s |
| | S N=2 | 1.53e‑5 | 18.5 | **22** | 1.18e‑2 | 3.0 | 12.4 s |
| | S N=4 | 1.03e‑6 | 43.7 | 40 | 1.24e‑3 | 1.2 | 14.3 s |
| | **S N=8** | **1.91e‑7** | 89.7 | 76 | **1.65e‑4** | **0.4** | 12.4 s |

`jump/RMSE` = worst bank-boundary jump divided by the model's own RMS error.
Build times are with `alpha_init="auto", restarts=3` (6 fits); `restarts=1` builds in
~1/5 the time at 1.2–14× worse MSE.

---

## What the numbers say

### 1. MBE-PASN-S strictly dominates a global MBE on all three axes, on all four targets

| | global MBE N=8 | best S at ≤ its memory | MSE | spikes | params |
|---|---|---|---|---|---|
| gelu | 1.18e‑1 / 32.6 / 49 | S N=2 | **40 000×** | 1.4× fewer | equal |
| silu | 1.97e‑1 / 54.8 / 49 | S N=2 | **29 000×** | 2.5× fewer | equal |
| exp2 | 1.55e‑5 / 64.5 / 49 | S N=2 | **2.7×** | 3.3× fewer | 2.0× fewer |
| invsqrt | 8.04e‑6 / 77.3 / 49 | S N=4 | **7.8×** | 1.8× fewer | 1.2× fewer |

This is the clean claim, and unlike the earlier revision it now holds on the smooth
monotone primitives too, not only on the Transformer activations.

### 2. Against flat banks, the accuracy axis has flipped — but flat keeps the cheapest point

The answer to the opening question is **scale**: one shared basis set spans every
range, and with the threshold placement and candidate selection fixed (below) it is
*more* accurate than independent per-range neurons on **every** function:

| | flat MBE‑PASN | best S | MSE | memory |
|---|---|---|---|---|
| gelu | 1.52e‑5 / 18.6 / 181 | S N=2: 2.95e‑6 / 23.3 / **49** | **5.2× better** | **3.7× less** |
| silu | 4.15e‑5 / 13.1 / 181 | S N=2: 6.78e‑6 / 22.1 / **49** | **6.1× better** | **3.7× less** |
| exp2 | 1.21e‑5 / 12.7 / 77 | S N=4: 6.21e‑6 / 13.7 / **45** | **1.95× better** | **1.7× less** |
| invsqrt | 2.20e‑5 / 6.9 / 64 | S N=4: 1.03e‑6 / 43.7 / **40** | **21× better** | **1.6× less** |

But on every function **flat holds the minimum-spike point** (12.7–18.6, and 6.9 on
1/√x) and no S configuration undercuts it at equal accuracy. So:

> Flat banks own the low-energy end. A shared basis owns accuracy and memory
> everywhere above it.

Both are on the frontier; neither is dominated. That is the honest framing, and it
replaces the earlier "flat buys energy with memory, S buys memory with energy" —
half of which turned out to be an artifact of a bad threshold init.

### 3. Boundary continuity — relative, not absolute

Normalised by each model's own RMS error, the worst boundary jump is:

| | flat | best‑MSE S | improvement |
|---|---|---|---|
| gelu | 8.9 | 2.4 | 3.8× |
| silu | 9.3 | 4.7 | 2.0× |
| exp2 | 1.6 | 0.3 | 5.7× |
| invsqrt | 4.8 | 0.4 | 12.6× |

For flat banks the jump is **2–9× its own RMS error** — an artifact standing above the
noise floor. For S it is **0.3–4.7×**, i.e. 2–13× less pronounced. This is structural
(adjacent ranges share features and differ only in a linear readout, so they cannot
drift independently), but the earlier claim of a *20–1700× absolute* reduction was
overstated: the absolute jump largely tracks the model's overall error.

### 4. Build cost is now a knob, not an advantage

With `restarts=3` and both α candidates, S runs 6 fits and builds **1.7× slower** than
flat's 13 (34 s vs 19 s on GELU) — reversing the earlier claim. What survives is the
*scaling*: S needs `O(1)` surrogate-gradient fits and `O(R)` linear solves, flat needs
`O(R)` gradient fits. At R=13 they are comparable; at R ≫ 13 (mantissa-prefix
subdivision) only S is affordable. `restarts=1` builds in ~6 s, faster than flat.

---

## The two build fixes this required

### `alpha_v` placement: basis saturation

`N=4` was systematically worse than `N=2` on GELU/SiLU across all seeds. Cause: the
inherited placement. A global MBE log-spreads the leading thresholds over
`[2^-spread, 1]` because the target's curvature concentrates at one end of its
domain. A *routed* residual is the opposite — `rho` is ~uniform on `[0,1)` in every
bank and each `g_v` is smooth there — so a basis with a near-zero threshold fires at
almost every step and its feature is nearly constant in `rho`. Measured per-basis
firing rates at N=4 with log-spread: `[0.88, 0.17, 0.71, 0.95]`, i.e. 1–2 useful
bases out of 4.

`uniform_alpha()` splits `[0,1)` at even quantiles. Neither placement wins outright
across targets, so `alpha_init="auto"` fits both and keeps the better — removing the
last hand-tuned choice.

### Candidate selection needs a disjoint grid *and* the right metric

Calibration uses a **midpoint grid** (not a random draw): strictly lower discrepancy
for a known deterministic target on a known interval, and it removed the entire
run-to-run spread (was up to 34×; the seed had changed nothing else, since the init is
deterministic and the readout closed-form).

Candidates are scored on an **offset grid** — the same grid shifted a quarter cell, so
it shares no point with what was fitted. Both properties were necessary:

* **Disjointness.** The same jittered restarts *hurt* when scored on the fitting grid
  (GELU N=8 test MSE 4.7e‑5 → 1.3e‑4: a fit can place staircase breakpoints to nail
  grid midpoints while drifting between them) and *help* when scored on the offset
  grid.
* **Metric.** Scoring on the per-range *relative* loss that the fit minimises picked
  the worse init on 3 of 12 configurations (SiLU N=4 by 18×) — a model can win on
  error averaged across *ranges* while losing on error averaged across *inputs*. The
  offset grid puts equal points in every range regardless of width, so its samples are
  weighted by range width, making the loss an unbiased estimate of the reported
  metric. 11/12 correct after the change; the one residual miss (SiLU N=8, 1.3×) picks
  the cheaper frontier point anyway.
* **Jitter must not be keyed on the seed.** Keyed on `seed + 1000k` restarts became a
  lottery: one lucky seed looked 7× better while the 3-seed median showed no gain at
  all. Keyed on the restart index alone, restarts improve **8 of 12 configurations by
  1.2–14.2× (≈3× geometric mean) with zero regressions**, deterministically.

---

## Spike-aware selection: one build, the whole frontier

Selection scores **(MSE, spikes) under the same width weights**, so a candidate's
accuracy and its energy are measured against each other consistently. `n_shared`
accepts a list, so the candidate pool spans the axis that dominates spike cost, and
`spike_budget` keeps the most accurate candidate that fits (returning the cheapest and
reporting `UNSATISFIABLE` if none do — it never silently overspends).

One build of SiLU with `n_shared=[2,4,8]` (18 candidates, ~3 min) yields the frontier
directly from `model.selection_trace`:

| N | init / restart | MSE | spikes/in | params |
|---|---|---|---|---|
| 2 | uniform / r0 | 1.01e‑4 | **3.90** | **49** |
| 2 | logspread / r2 | 6.29e‑5 | 15.3 | **49** |
| 4 | uniform / r0 | 2.27e‑5 | 17.6 | 85 |
| **2** | **uniform / r2** | **6.32e‑6** | 22.2 | **49** |
| 8 | uniform / r0 | 4.95e‑6 | 52.8 | 157 |
| 8 | uniform / r2 | 2.53e‑6 | 64.7 | 157 |
| 8 | logspread / r2 | **1.97e‑6** | 101.9 | 157 |

Budgets are honoured, and the offset-grid metric tracks an independent uniform test
draw closely (6.32e‑6 → 6.78e‑6; 1.97e‑6 → 1.94e‑6), which is what licenses using it
for selection at all:

| budget | test MSE | measured spikes | params |
|---|---|---|---|
| 4 | 9.61e‑5 | 3.90 | 49 |
| 8 | 9.61e‑5 | 3.90 | 49 |
| 16 | 5.99e‑5 | 15.3 | 49 |
| 32 | 6.78e‑6 | 22.1 | 49 |
| none | 1.94e‑6 | 101.7 | 157 |

Two things this settles:

* **The 3.9-spike operating point is recoverable on request** rather than lost to
  min-MSE selection. It is 3.4× cheaper than flat MBE-PASN's cheapest point (13.1) at
  2.4× worse MSE and 3.7× less memory — a corner of the frontier flat cannot reach at
  any N.
* **Restarts beat extra bases.** `N=2 uniform/r2` reaches 6.32e‑6 at 22.2 spikes with
  **49** parameters, better than every N=4 candidate and within 3× of the best N=8 at
  a third of the memory and a fifth of the spikes. Spending the search budget on
  restarts is cheaper than spending it on bases.

---

## Remaining

1. **The width weights still encode "uniform over the domain".** That is the right
   proxy for the reported 1-D metric, but a spike budget expressed under it is not the
   budget a real network sees — activations concentrate near zero. Calibration already
   measures the per-range visit probabilities; passing them in makes both the fit and
   the budget distribution-aware. This is the same hook for both.
2. **Unreachable ranges are still stored and counted.** The router's binade grid does
   not align with a calibrated domain (2 of 13 ranges for GELU, 2 of 4 for 1/√x). They
   are left unfitted — previously they were fitted outside the target's valid range and
   held **NaN** parameters, invisible because no input routes there — but pruning them
   would reduce both variants' memory.
3. **Per-bank MSE in the raw sweep log is absolute, not relative.** Large-|x| ranges
   carry larger |f|, so their absolute MSE is larger for uninteresting reasons; read
   those columns as relative error before concluding which range is hard.
4. **No downstream number yet.** All of this is 1-D approximation. `convert.py` needs
   an `S` backend before GPT-2 × WikiText-2 can rank these operating points by what
   matters — Δperplexity per spike.
