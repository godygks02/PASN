# Positioning against MBE and NLSpike — draft Related Work + differentiation table

**E10 deliverable.** Written 2026-08-10. Paper-bound text, kept at the repo root so it
is versioned with the numbers it cites (`paper/` is gitignored -- it holds the
copyrighted baseline PDF).

> ⚠️ **Provenance.** The MBE facts are read from the paper in `paper/` (main text
> extracts + appendix images, verified 2026-08-10). The **NLSpike facts are
> second-hand** — they come from the 2026-08-04 reading recorded in
> `PASN_vault/40 - Planning/논문 계획 - 타깃 벤뉴와 실험 로드맵.md` §1, not from a
> re-read here. Everything attributed to NLSpike below is marked ⓝ and **must be
> re-verified against arXiv:2605.20289 before submission.**

---

## 1. The two neighbours

| | **MBE** (our baseline) | **NLSpike** ⓝ | **PASN** (ours) |
|---|---|---|---|
| venue / date | published | arXiv:2605.20289v1, 2026-05-19 | — |
| goal | training-free ANN→SNN for Transformers | same | same |
| primitive decomposition | identity, `exp`, `inv`, `invsqrt` | division, exponential, ℓ₂ norm | identity, `exp2`, `inv`, `invsqrt` |
| how a nonlinearity is computed | multi-basis exponential-decay spike code | LIF population + bit-shift + **lookup table** | multi-basis spike code, **per-binade** |
| where it attaches | the conversion neuron | **post-hoc plugin** behind SpikeLLM / SpikeZIP | the conversion neuron |
| capacity setting | **hand-set**: `N=4` (GELU/Tanh), `N=8` (others), `T=16` | **hand-set**: `H=5`, `K=64` | **solved per range**: `(N_j, T_j)` from a rule |
| identity parameters | **fixed**, shared network-wide | LUT | tied prototype, shared network-wide |
| models | ViT-B/M, RoBERTa, GPT-2-medium | LLaMA-3-8B, LLaMA-2-7B/13B, Mistral-7B, Qwen3-8B, SpikeZIP-BERT | **ViT-B/16**, GPT-2-medium, RoBERTa-base/large |
| modalities | CV + NLU + NLG | NLG + NLU ⓝ | **CV + NLU + NLG** |
| evidence layers | function MSE, firing rates, task | operator error, task accuracy | **function → operator → network** |

**Neither cites the other.** MBE does not appear in NLSpike (a full-text search
for `MBE` / `multi-basis` returned nothing ⓝ), and MBE predates it. Putting both
in one table is itself a contribution.

---

## 2. The claim we can no longer make, and the one we can

*"Training-free spiking nonlinearities"* is no longer a contribution by itself —
NLSpike arrived at the same problem statement with nearly the same primitive
decomposition. The contribution has to be stated more narrowly, and it is:

> **Both neighbours pick their capacity by hand and apply it globally. We solve
> it per input range.** MBE fixes `N=4/8` and `T=16`; NLSpike recommends `H=5,
> K=64` ⓝ. PASN reads the IEEE-754 exponent to route each input to a bank and
> derives that bank's `(N_j, T_j)` from the target's dynamic range over it:
> `b = log₂(Δ_j/√ε)`, `T_j = 2^((b−2)/1.7)`, `N_j` mostly 1.

Three of our own measurements back this specific claim rather than the generic one:

**(a) The allocation is where the value is, and we decomposed it.** Forcing the
paper's global `N=4` costs **2.4–2.55× spikes and 2.55× storage** at unchanged
accuracy; the `T_j` half is only 1.065×. So "solved budget" is not a framing — it
is the measured mechanism.

**(b) The rule is insensitive to how it was calibrated (E12, and then E3).**
Across three disjoint calibration draws the fitted domains move by 3–4% (max
24%), and yet **all 19 recorded per-bank `(N_j, T_j)` decisions are identical**,
storage is byte-identical, and spikes differ by 0.011%. The rule absorbs
calibration noise because a 3–4% change in `Δ_j` is 0.05 of a bit inside `log₂`
and never crosses a threshold.

**E3 then pushed the same test across corpora rather than within one.**
Calibrating on WikiText-103's word-level train — a 49× larger corpus with `<unk>`
substitution, a different token distribution entirely — still produced a
**byte-identical build** (53,888 B / 13,472 params / 339 primitives) and spikes
within **1.001×**. **A solved budget is reproducible in a way a tuned constant is
not**, and that now holds across the calibration corpus, not just the draw.

**(c) Hand-picking a global `N` is a fragile design, and the MBE paper's own
numbers show why (E2).** Fitting error is **not monotone in `N`**: their Table X
has `inv` getting **2.0× worse** from `N=4` to `N=6`, and our own sweep on the
real GPT-2 domains bounces by up to **137×** with more capacity, unfixed by 10×
the epochs. When more capacity can make the fit worse, a single global constant
is a point estimate on a rough surface. Deriving the budget per range does not
make the surface smooth, but it stops one hand-chosen point from being applied
everywhere.

⚠️ **State (c) as an argument, not as a measured comparison.** We have not shown
that our rule lands on better points than a swept global `N` would; P0.3 shows it
beats a *swept* global MBE by 4.92× on a matched toy, and that is the measured
version of the claim.

---

## 3. Where each neighbour is ahead of us — say it first

| axis | who leads | our position |
|---|---|---|
| **Scale** | NLSpike ⓝ (8B-class × 4) | GPT-2-medium (345M) + RoBERTa. **We do not chase this.** E9 mitigates by showing one larger model does not break; it cannot close the gap, and following it hands over the frame. |
| **Data movement** | **contested — E7 priced it 2026-08-10** | Bank switching really does break locality: **48.3%** of consecutive elements switch banks. But a site's whole bank set is **under 1 KB** (median 88 B), so the switch is a register/L0 read, and the cost is **1.8% of compute energy** in the dataflow where we lose outright. In the other bracket routing **wins 4.19×**, because state traffic scales with *active* bases (1.57 vs 7.58). **Neither method touches DRAM** — 52.6 KB and 37.3 KB both sit on-chip, so "zero data movement" ⓝ is a claim about SRAM access counts, not off-chip traffic, and there the metric is active bases. ⚠️ A model, not a measurement; NLSpike's own numbers were not reproduced. |
| **NLU coverage** | NLSpike ⓝ (MR / SST-2 / Subj / SST-5) | SST-2 at both sizes; MR is **undecidable** across checkpoints; SST-5 / Subj have no usable public base fine-tune. Reported as an absence, not filled quietly. |
| **Identity storage** | MBE (fixed, shared network-wide — G.1) | Our tied identity **matches rather than beats it.** The 1.99× tying saving is against our own untied build. |
| **Parameter count** | MBE, on the reading its own appendix supports (5,929 vs our 13,472) | We report all three conventions and lose two. Parameters are not a headline claim. |

Leading with these is the correct move: two of them (data movement, parameters)
are axes we will lose on measurement, and a reviewer who finds them unprompted
will discount the axes we do win.

---

## 4. Where we are ahead

1. **Solved rather than tuned budget** — §2, with the decomposition and the
   calibration-invariance behind it.
2. **Spike code, not a lookup table** ⓝ. NLSpike's nonlinearities are LIF
   populations plus a LUT. *"Is this spiking computation, or quantisation with a
   LUT wearing SNN clothes?"* is a question that will be asked in review, and it
   is not asked of a decayed-basis code that constructs the value from spikes.
   **Do not press this rhetorically** — it is a framing point, and pressing it
   invites the symmetric question about our router's arithmetic (15.9% of our
   modelled energy, and honestly reported as such).
3. **Three evidence layers, matched at each** — function approximation against
   Table X, per-operator iso-accuracy cost (10/10 operators, 1.23–5.39× fewer
   spikes on the arm we ship), and network conversion loss at the paper's own
   `T=16`. NLSpike reports operator error and task accuracy ⓝ; MBE reports
   function MSE, firing rates and task.
4. **A more sensitive metric.** Five-way accuracy has ±0.4pp of noise; perplexity
   lets us argue about 0.18%, and we characterised the recipe sensitivity that
   makes that argument legitimate (band 0.238 pp across three strides, on the
   build we ship).
4b. **Both of Table 3's rows, at the paper's own `T=16`.** Wiki-2 **−0.181%**
   against +1.57% (margin 1.75 pp) and Wiki-103 **−0.196%** against **+3.36%**
   (margin **3.56 pp**, our largest — the paper loses that row by twice as much).
   ⚠️ Absolute perplexity is not comparable in either row; only the relative loss
   against each method's own ANN is.
4c. **All three modalities, which NLSpike does not have.** CV is the axis they
   skip: ViT-B/16 × ImageNet-1k over the full 50k validation set, **−0.020%**
   against the paper's **−0.527%** (margin 0.51 pp). Together with the two NLG
   rows and SST-2 at both sizes, the same neuron, rule and code cover **CV, NLU
   and NLG** — the paper's own coverage, matched. **Both of their ViT rows are
   filled**: ViT-M/16, which is a timm model and needed its own markers
   (`mbe.timm_convert`), comes in at **−0.0024%** against their **−0.745%**.
   ⚠️ Absolute top-1 is not comparable in either row — our pipeline reads
   0.8–1.0 pp below published across two models and two frameworks, so the gap
   is ours, not the checkpoints'. ⚠️ Their CNN rows are still empty.
5. **Low-`T` behaviour, measured on the same axis as the baseline.** At `T=8` —
   the lowest timestep MBE reports — their GPT-2 goes to 41072 from an ANN of
   22.65, and every other model in their Table 4 collapses too (ViT-B 83.44 →
   0.12). We are at **+0.062%**. ⚠️ Their row is WikiText-103 and ours is
   WikiText-2; label the corpus (E3 closes it).

---

## 5. The move that is worth more than any of them: orthogonality

NLSpike attaches **behind** an existing conversion pipeline; PASN **is** the
conversion neuron ⓝ. That makes the natural experiment *"their operators on top
of our routing"* (or the reverse) well-posed rather than rhetorical.

**If orthogonality holds, this stops being a competing paper and becomes a
complementary one** — which is worth more in review than winning a comparison.
It is listed as optional (§4.3(c) of the roadmap) and is **not** in the workshop
scope; it belongs in the ICML cycle if it happens at all.

---

## 6. Draft Related Work paragraph

> Training-free conversion of Transformer nonlinearities has been approached from
> two directions. **MBE** replaces each nonlinear operator with a multi-basis
> spiking neuron whose exponentially decaying bases are fitted per function,
> using a globally fixed capacity (`N=4` for GELU, `N=8` elsewhere, `T=16`), and
> reports near-lossless conversion on CV, NLU and NLG. **NLSpike** ⓝ instead
> decomposes the nonlinearities into division, exponential and ℓ₂-norm and
> implements them with LIF populations, bit-shift scaling and lookup tables,
> attaching as a post-hoc plugin behind existing conversion pipelines at a
> recommended `H=5, K=64`. Both fix their capacity by hand and apply it
> uniformly across the network. We keep MBE's basis dynamics but route each input
> to a bank by its IEEE-754 exponent and **solve** that bank's basis count and
> timestep budget from the target's dynamic range over it. The router consumes no
> parameters and emits no spikes, and with a single bank the neuron is
> bit-identical to MBE, so the method is a strict generalisation of it.

---

## 7. What is left before this can be submitted

- [ ] **Re-verify every ⓝ against arXiv:2605.20289.** This document is written
      from a second-hand reading.
- [x] ~~**E7** — price data movement.~~ Done 2026-08-10: we lose the worst-case
      dataflow by 1.8% of compute, win the other bracket 4.19x, and neither
      method needs DRAM. See the data-movement row above.
- [ ] **E3** — corpus label on the low-`T` comparison (WikiText-103 vs -2).
- [ ] Decide whether the orthogonality experiment (§5) is in the ICML scope.

---

Sources: `PASN_vault/40 - Planning/논문 계획 - 타깃 벤뉴와 실험 로드맵.md` §1 and §4.3
(NLSpike reading, 2026-08-04); `PASN_vault/50 - Reference/MBE 논문 published 설정과
수치.md`; journals E1, E2, E5, E6, E11, E12; `results/RESULTS_2026-08_cycle.md`.
