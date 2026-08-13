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

## 6b. Prior art on the routing mechanism — first pass, 2026-08-13

**This was the open novelty question** (`PAPER_TABLE_COMPARISON.md` §6: *"지수/binade
라우팅 선행연구 조사 안 됨"*). Six searches, first pass, **not exhaustive**. The
finding is negative for the mechanism and it changes how the paper must be framed.

### 🔴 Exponent/binade routing to per-range approximations is established prior art

| source | what it does |
|---|---|
| **BBAL**, arXiv:2504.15721 (2025) | *"a segmented lookup table divides function values into segments based on **different exponents**, with lookups performed according to **mantissas** ... split into sub-tables based on the number of exponent bits"* (5 exponent bits → 2⁵×2 sub-tables). **This is exponent-field routing to per-range tables**, in a block-floating-point LLM accelerator. |
| **US 11163533 / US20210019116A1** (FP exponential unit) | Describes as *existing* practice: *"for each binade/exponent value, the fraction is obtained by a single linear function (A·f+B)"*. Its own contribution is a **fixed** 6-segment scheme, so it is not adaptive-per-binade — but it confirms **per-binade parameters are prior practice.** |
| **Transcendental function evaluation**, US 10725742 / 11099815 / 11733969 | Same family. |

> **We cannot claim "read the IEEE-754 exponent to pick a range" as a contribution.**
> It is a known hardware technique with patents.

### 🔴 Non-uniform, error-driven segment allocation is also established

| source | what it does |
|---|---|
| **NPE**, arXiv:2104.06535 (FPGA NLP overlay, BERT) | **Full text read 2026-08-13.** Non-uniform continuous piecewise-linear approximation, motivated **verbatim as we motivate ours**: *"functions like GELU(x) and √x which are nearly linear except for a very small nonlinear region near zero. Uniform segmentation would require **orders of magnitude more segments** than non-uniform"*. Segment boundaries come from *"one method for finding an **optimal partition** with non-uniform segmentation"* (their ref [3]). |
| **FQA**, arXiv:2606.05627 (2026, VLSI) | **Full text read 2026-08-13.** Piecewise **polynomial** approximation for NAFs (Sigmoid/Tanh/KAN). Its "allocation" is **fractional word lengths of arithmetic units** (slope, intercept, multiplier, adder) — *precision per operator*, not capacity per input range. `spiking`: **0 hits**. `exponent`: only "grows exponentially" + a CORDIC citation. **Further from us than the abstract suggested.** |
| the PPA segmentation literature FQA surveys ([20]–[35], ~16 refs) | **This is the real hit.** *"[25] adopts a sequential approach that starts from the end of each interval and **incrementally searches for the first point satisfying the MAE constraint**"*; *"[26] introduces a bisection method"* to accelerate it. **Determining per-region capacity from an error constraint is the standard practice of this field.** |

> 🔴 **Two things we call contributions are prior art here.** The **GELU near-zero
> diagnosis** is NPE, near-verbatim. And **"solve the capacity from an error
> constraint instead of fixing it globally"** — which `§2` of this document calls
> our central claim — **is what error-bounded non-uniform PPA has always done.**
> Our `b = log₂(Δ_j/√ε)` is a closed form where they run a search, which is a
> difference of method, not of idea.
>
> ⚠️ So *"both neighbours pick capacity by hand; we solve it"* is novel **only
> relative to the two SNN papers**, not relative to the field the technique comes
> from. A reviewer with a hardware-approximation background will say **"this is
> standard error-bounded non-uniform PPA transplanted into a spiking neuron"** —
> and on the mechanism, they will be right.
>
> 📌 Also note NPE's own caveat, which cuts against our emphasis on the rule's
> precision: *"even **sub-optimal segmentation** can result in no accuracy loss for
> BERT inference on the test set."*

### 🟡 MoE routing inside SNNs exists, at a different granularity

**Spiking Transformer with Experts Mixture** (NeurIPS 2024) has a *Spiking Experts
Mixture Mechanism* with **a spiking router allocating computation**;
**SpikingMoE** arXiv:2605.23188 (2026) does dynamic expert fusion. Both route
**blocks/experts**, not **approximation ranges inside one neuron**. The MoE framing
is taken; the granularity is not.

### 🔴🔴 The closest prior art found — Lee et al., TVLSI 2009 (full text read 2026-08-13)

**D.-U. Lee, R. Cheung, W. Luk, J. Villasenor, "Hierarchical Segmentation for
Hardware Function Evaluation", IEEE TVLSI 17(1):103–116, 2009** (earlier: FPT 2003).
Reached via FQA's ref [24]. `doc.ic.ac.uk/~wl/papers/09/tvlsi09dul.pdf`.

This is **the same mechanism as our router**, in fixed-point hardware, seventeen
years earlier:

| our claim | Lee et al. 2009 |
|---|---|
| ranges are **binades** (powers of two) | *"hierarchies involving uniform splines and splines with **size varying by powers of two**"* — and they credit Coleman et al. [14] for a two-level scheme whose *"first segmentation has segments that vary by powers of two"* |
| the range index is a **free bit read** | *"Computation of the segment address ... is based on **detecting the number of leading zeros** for segments beginning with a zero, and ... **leading ones** for segments beginning with a one."* **Leading-zero count is the binade index.** And it is chosen for exactly our reason: *"we are targeting environments in which the **delay introduced by the coefficient address logic must be kept to a minimum**"* |
| the budget is **solved from an error target**, not tuned | *"we enable a designer to **specify an error tolerance** and to **automatically** obtain a segmentation that: 1) meets this tolerance; 2) requires a **small number of segments**; and 3) leads to efficient hardware implementation."* |
| a **two-level** router (near-zero bank + magnitude banks) | hierarchical, multi-level by construction |
| targets `1/x`, `1/√x`, `log`, `exp` | demonstrated on `√x`, `log₂`, `cos`, `ln(x)`, a high-degree rational, `ln(1+x)`, `1/(1+x)` — **the same primitive family** |

They also survey **balanced-error segmentation** — *"the maximum approximation error
in all segments is equal ... since it **minimises the number of segments needed to
meet a given overall approximation error constraint**"* — with Sasao et al. [18]–[20]
giving an algorithm for it. NPE's ref [8] (Frenzen, Sasao & Butler 2010, *"On the
number of segments needed in a piecewise linear approximation"*) is the theory of
that same question.

> 🔴 **Differentiator 2 ("the routing is free because it is an exponent read") is
> dead.** It is documented prior art with the same motivation. Our version reads an
> IEEE-754 exponent field where they run a leading-zero detector on fixed-point —
> cheaper by one small unit, which is an implementation detail, not a contribution.
>
> 🔴 **"Solve the budget from an error target rather than fixing it globally" is also
> theirs**, stated as an explicit design goal and automated.

**What Lee et al. does *not* do:** the per-region capacity is *segment width*; the
polynomial **degree is global to a design** (they report degree-1 and degree-2 as
separate designs, not mixed across regions), and per-region precision is handled
separately by bit-width optimisation (MiniBit). Nothing is spiking; there are no
timesteps.

### ✅ The [20]–[35] sweep — the inversion that is actually ours (2026-08-13)

FQA's survey chain was the most likely place a direct hit was hiding: if one of them
allocates **capacity** per region from an error bound, differentiator 1 collapses to
"spike-domain" alone. It does not happen, and the reason is structural.

| ref | method | what adapts | capacity per region |
|---|---|---|---|
| **[24]** Lee TVLSI'09 | hierarchical splines | segment widths (powers of two) | **fixed** — degree-1 and degree-2 are *separate designs* |
| **[25]** Sun TCAS-I'20 | universal PWL | *"self-adaptive capability to choose the **smallest number of segments** under the constraint of a controllable maximum absolute error"* | **fixed** (linear) |
| **[26]** PLAC TVLSI'20 · **[29]** ML-PLAC'22 | PWL for all unary fns | bisection segment search | **fixed** (linear) |
| **[28]** Lyu TVLSI'21 | FP logarithm | *"automatically segmented into several **maximal subsections** … with restrictions on predefined maximum absolute error"* | **fixed** (linear) |
| **[30]** An, *Electronics*'21 | error-flattened segmenter | *"the segmenter **adaptively selects a minimum number of parabolas**"*; widen each segment until MAE binds | **fixed** (all parabolas) |
| **[31]** QPA TVLSI'23 | quantization-aware PPA | fractional word lengths **per arithmetic unit** | **fixed** degree |

**Full text read: [24], [30]** (plus [30]: `power of two` **0 hits**, `exponent` 2 hits
both incidental — its boundaries are arbitrary, which is why this family needs LUT
cascades ([23] Sasao) or leading-zero detectors ([24]) to find the segment at all).
Read from abstract or from FQA's own description: [25], [26], [28], [29], [31].
Not read: [20]–[22], [23], [27], [32]–[35].

### 📌 The contribution, stated correctly at last — we **invert** the knob

> **Every method in this literature fixes the approximation capacity and adapts the
> region geometry. PASN fixes the region geometry and adapts the capacity.**

They choose *where the boundaries go* at a globally fixed degree; we take the boundaries
as given (binades — free to index off the exponent field) and vary **`(N_j, T_j)` per
region**.

**And the literature explains why nobody did it their way.** In fixed-point hardware,
varying the polynomial degree per segment means variable-latency datapaths and several
multiplier configurations — the reason [24] ships degree-1 and degree-2 as *separate
designs* rather than mixing them. **In a spiking neuron that cost does not exist**: each
bank already carries its own parameter table, so giving bank *j* its own basis count and
timestep budget is free. **The transplant is not arbitrary — the cost structure that made
per-region capacity variation unattractive in hardware is absent in the spike domain,
and that is the whole reason the idea had to wait for this setting.**

⚠️ This is an argument, not a measurement. What *is* measured is E14/E14b: strip the
per-region capacity and the low-`T` collapse returns on both router axes.

### ✅ What survives full-text reading of both

**Two of the three survive. The middle one is gone** (Lee et al., above).

1. ✅ **The allocated resource is spike-domain.** NPE allocates *segment boundaries*;
   FQA allocates *bit widths*; Lee et al. allocates *segment widths* at a globally
   fixed polynomial degree. PASN allocates **`N_j` spiking bases and `T_j`
   timesteps** — the two quantities SNN conversion is actually judged on. Nobody in
   this literature allocates those, because in fixed-point hardware **there is no
   such resource**. This is now the **primary** differentiator.
2. 🔴 ~~**The routing is free.**~~ **Withdrawn** — Lee et al. 2009 computes the
   segment address by leading-zero detection precisely to minimise address-logic
   delay, and BBAL indexes sub-tables by exponent. Prior art, same motivation.
3. ✅ **The failure mode being fixed is SNN-specific.** In fixed-point PPA, too little
   capacity costs accuracy smoothly. In a spiking neuron it causes the **low-`T`
   collapse** of the baseline's Table 4. That failure does not exist in NPE's, FQA's
   or Lee's setting, so **none of them could have reported it.**

> **The residue is one sentence, and it is the whole paper:** *hardware function
> evaluation has solved per-range capacity from an error target since at least 2009,
> using binade segmentation addressed by a leading-zero/exponent read; SNN conversion
> never imported it, and the price of not importing it is that global capacity
> collapses at low timestep budgets.*

### 📌 Reframing: this literature is an asset, not only a threat

Positioning PASN as *"importing mature error-bounded non-uniform approximation from
hardware function evaluation into the spiking conversion neuron, where the allocated
resource becomes spikes and timesteps"* is **stronger and more honest** than claiming
the routing is new. It gives the method a known-good pedigree, and makes the
contribution the thing we can actually defend: **SNN conversion has been fixing
capacity globally by hand while the approximation literature solved this decades ago,
and the cost of that omission is the low-`T` collapse.**

Cite NPE, FQA, BBAL and the PPA survey chain **in the introduction, not buried in
Related Work.** A reviewer who finds them after we omitted them will discount
everything else.

### ✅ What the search did **not** find

- No ANN→SNN conversion work routing by **IEEE-754 exponent inside the conversion
  neuron**.
- No work where the allocated per-range capacity is **spike-domain** — a number of
  spiking bases `N_j` and a timestep budget `T_j` — rather than segments, LUT
  entries or word lengths.
- No work **solving** that spike-domain budget from the target's dynamic range over
  the range (`b = log₂(Δ_j/√ε)`).

### 📌 Consequence for the paper — the mechanism is a transplant, so it cannot be the claim

Every ingredient exists: exponent routing (BBAL, patents), non-uniform error-driven
allocation (NPE, FQA, PLA), MoE routing in SNNs (SEMM, SpikingMoE). What appears
unclaimed is **the combination**: a spiking conversion neuron whose per-range budget
is `(N_j, T_j)` and is solved rather than tuned, with a router that costs **zero
parameters and zero spikes** because it is an exponent read.

That is a **much narrower** claim than *"we introduce exponent routing"*, and an
engineering transplant is not on its own an ICML/NeurIPS contribution. **The paper
therefore has to be carried by the consequence, not the mechanism**:

> the baseline's capacity is fixed globally by hand, **that is what breaks it at low
> timestep budgets** — its own Table 4 collapses all five models at `T=8` and it
> diagnoses the cause as wide identity-mapping ranges — **and per-range solved budgets
> remove that failure mode** at lower spike cost.

⚠️ **This makes the routing ablation (`pasn_e_min` sweep at `T=8`) load-bearing, not
optional.** Without it the paper claims a known mechanism and an unexplained
cross-paper win.

### Limits of this pass

Six searches. ✅ **FQA and NPE full text read 2026-08-13** (13 and 11 pages, `pypdf`).
Still **not** read: BBAL full text, the PPA chain FQA surveys ([20]–[35] — **the most
likely place a direct hit is hiding**, since that is where per-region capacity is
solved from an error bound), and NPE's segmentation reference [3]. Not searched:
pre-2015 DSP/CORDIC, non-English sources, neuromorphic LUT work.

✅ **[20]–[35] swept 2026-08-13** — nobody in that chain varies capacity per region;
they all vary geometry at fixed capacity, so **differentiator 1 survives and got
sharper** (see the inversion section above). Still unread: **[20]–[23], [27],
[32]–[35]**, and BBAL's full text. **Treat this as a first pass that closes the "we
never looked" gap, not as a novelty clearance.**

📌 Baseline provenance confirmed en route: MBE is **AAAI-26**
(`ojs.aaai.org/index.php/AAAI/article/download/37195/41157`, arXiv:2508.07710).

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
