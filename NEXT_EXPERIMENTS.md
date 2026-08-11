# What to run next — cold-start brief

Written 2026-08-03, **updated 2026-08-10**. Self-contained: a session that has
never seen this project should be able to act from this file alone.

> ### ⛔ Read this before quoting any GPT-2 number
>
> **The build is frozen (E0, 2026-08-10) and it is the `pasn_beta={"inv":0.5}`
> build.** Its fingerprint and headline:
>
> | | frozen paper build | ~~old headline~~ |
> |---|---|---|
> | tag / file | `freeze-t16-unif` / `results/freeze_e1.json` | `stride-sens-s1024` |
> | `pasn_beta` | `{"inv": 0.5}` | `{}` (unrecorded at the time) |
> | bytes / params | **53,888 / 13,472** | ~~49,952 / 12,488~~ |
> | **ΔPPL** @ stride 1024 | **−0.18%** (two runs: −0.1873 / −0.1809) | ~~−0.14%~~ |
> | margin over paper's +1.57% | **1.75 pp** | ~~1.71 pp~~ |
> | recipe band (3 strides) | **0.238 pp**, worst \|ΔPPL\| **0.181%** | ~~0.197 pp~~ |
>
> **`−0.14%`, `49,952` and `12,488` now belong to the `--no-pasn-beta` build
> only.** Where a no-beta result must still be cited (the stride sweep, the Stage
> 2 originals), name the build. Every record written since 08-09 carries all 37
> `ConvertConfig` fields in `convert_cfg` — check it against the row above.
>
> **Timestep curve on this build** (E1, 4/4 done 08-10, `results/freeze_e1.json`):
> `T=16` −0.187% · `T=8` +0.062% · `T=4` +2.137% · `T=2` **+60.97%**, at
> 1.00 / 3.25 / 5.44 / 9.52× fewer spikes. **Operating point is `T=8`.**
>
> **Table 4's axis is `T = 8, 10, 12, 16`** — pinned 2026-08-10 from the PDF; it
> was nearly *inferred* as 2/4/8/16, which would have made the whole comparison
> table wrong. **The paper never measures `T=4` or `T=2`.** At `T=8`, its lowest
> point, GPT-2 goes to **41072** (ANN 22.65) and every other model in the table
> collapses too (ViT-B 83.44→0.12, ViT-M 85.95→1.17, RoBERTa 89/91→50). We are at
> **+0.062%** there. Corpus still differs (Wiki-103 vs Wiki-2 → E3).
>
> Write it as *"the paper is destroyed at the lowest `T` it reports and we are
> lossless there"* — **not** "we don't break": we do, at `T=2`, which the paper
> never measures. And never compare across `T` (their `T=16` vs our `T=4` is a
> trade, not a result).

> **The research log is NOT in git.** `PASN_vault/` is gitignored (Obsidian, local
> only). Its index is `PASN_vault/60 - 연구일지/00 - 연구일지 인덱스.md`. If you are
> on a fresh clone the vault is absent and this file is the whole picture.
>
> **Environment**: run everything with the `SNN` conda env, not the default python:
> `C:/Users/cm120/miniconda3/envs/SNN/python.exe` (py3.10, torch 2.11 CPU).

---

## 1. Where the work stands

PASN is a training-free ANN→SNN conversion neuron: an IEEE-754 exponent-bit
router splits the input into binade ranges, and each range gets its own
multi-basis spiking bank with a solved per-range budget `(N_j, T_j)`.

**The baseline is the MBE paper's published tables** — never our own MBE
reimplementation, which cannot reproduce their near-lossless downstream numbers
and is kept only as an internal ablation (decision 2026-08-01).

Measured, all at the paper's own global `T=16`:

| level | task | paper | PASN | status |
|---|---|---|---|---|
| network | GPT-2-medium × WikiText-2 | +1.57% | **−0.19%** | **frozen build**, operator set + T matched |
| network | GPT-2-medium at `T=8` | **41072** — Tab. 4's *lowest* `T`, ANN 22.65 | **+0.062%** | ✅ `T` axis pinned 08-10 (8/10/12/16); corpus differs → E3 |
| network | GPT-2-medium, `T=4` / `T=2` | *paper never measures these* | +2.14% / **+60.97%** | ✅ 4/4 points; **we break at T=2** |
| network | RoBERTa-base × SST-2 | −1.09% | **+0.12%** | robust over 3 checkpoints |
| network | RoBERTa-large × SST-2 | −0.25% | **−0.12%** | hardest cell, still ahead |
| network | RoBERTa-base × MR | −0.44% | −0.73% … −0.10% | **undecidable** (see §3) |
| operator | Table XI firing rates | 7 primitives | **6/7 at 2.2–10.3× fewer spikes** | cross-model caveat |
| operator | whole-op iso-accuracy | — | **10/10 on spikes, 1.23–5.39×** | ✅ recomputed with tying on (E6 §9) |
| **network** | **ViT-B/16 × ImageNet-1k (50k)** | **−0.527%** (Tab. 1) | **−0.020%** | ✅ **CV axis open (E8)** — ⚠️ their ViT-M/16 is timm, unreachable |
| function | Table X MSE vs N | per-function | reproduced and beaten | clean |
| recipe | ΔPPL across stride 1024/512/256 | recipe unstated | **−0.181 … +0.057%** (band 0.238 pp) | ✅ **on the frozen build** (E13) |
| depth | per-layer error, 24 blocks | — | shared fits hold; error saturates | closed |

Budget-rule decomposition on GPT-2 (Stage 2): the value is in `N_j`
(**2.4–2.55×** spikes, **2.55×** storage); `T_j` is only **1.065×** and free in
storage.

**Spike budget by operator — always name the arm** (E0, 2026-08-10). The
often-quoted "attention is 86.5%" is the *per-bank `T_j`* arm. The headline
build runs at global `T=16`, where the split is different:

| arm | matmul | softmax | LN | activation | attention |
|---|---:|---:|---:|---:|---:|
| per-bank `T_j` (`s2fix-rel-1e-2`) | 48.6% | 37.9% | 10.2% | **3.3%** | **86.5%** |
| **global `T=16` = frozen build** | 48.3% | 33.5% | 10.0% | **8.1%** | **81.8%** |
| `T=8` alloc (`e1-t8-alloc`) | 21.7% | 39.2% | 29.6% | 9.6% | 60.9% |

The rule gives activation tails `T≈3`; forcing `T=16` raises exactly that
(`act_spikes_per_input` 1.52 → 3.82). So the ceiling on activation+LayerNorm
work is **1.22×** at the headline operating point, not 1.16×. Direction is
unchanged — attention is still the largest item and fusion is still the target.

---

## 2. Is NLU finished? Yes, effectively.

Done and defensible: the generality claim (decoder *and* encoder, same neuron,
same rule, same code), SST-2 at both sizes, and the checkpoint-variance
characterisation that makes the SST-2 claim safe.

**Do not spend more time on NLU.** Specifically:

* **SST-5 and Subj cannot be completed.** No public base fine-tune exists for
  either; the only `large` candidates are weak (`ghatgetanuj/roberta-large_cls_subj`
  scores **93.35** against the paper's 97.50 — a 4.15pp gap that makes its
  conversion loss uninterpretable). Completing that row means fine-tuning our own
  source ANNs, which is a different project and should be a deliberate decision,
  not a gap quietly filled.
* **MR is undecidable and that is the finding.** Two uploads of the same
  fine-tune family give −0.73% and −0.10%, and the paper's −0.44% sits between
  them. Neither "we lose MR" nor "we win MR" is supportable. Report the range.

---

## 3. Ranked: what is actually left

### ~~1 — GPT-2 evaluation-recipe sensitivity~~ ✅ **RECOVERED 2026-08-11 (E13)**

> **Re-measured on the frozen build** — 3 strides, one build, one box, 13.1 h GPU
> (`results/e13_stride.json`):
>
> | stride | ANN | no-beta | **frozen** | gap |
> |---:|---:|---:|---:|---:|
> | 1024 | 21.7058 | −0.1397% | **−0.1809%** | 0.0413 pp |
> | 512 | 18.4629 | −0.0302% | **−0.0321%** | 0.0019 pp |
> | 256 | 18.0806 | +0.0578% | **+0.0573%** | 0.0006 pp |
>
> Band **0.2382 pp**, worst magnitude **0.181%**, margin over the paper's +1.57%
> **1.75 pp**, band 13.6% of that margin — the same picture the no-beta sweep
> gave (13%). **The claim holds on the build we ship.**
>
> Two things the re-measure changed. **The build gap is not a constant offset**:
> 0.0413 pp at stride 1024 but 0.0019 / 0.0006 pp at 512 / 256, below the
> 0.0064 pp box-to-box noise floor — so the estimate this section used to carry
> ("a roughly uniform shift puts 256 at +0.011%") was wrong, and 256 landed at
> +0.0573% instead. **And the sign still flips at 256**, so write **"measurably
> lossless under every recipe"**, never "−0.18% is an improvement".

### (the sweep, on the no-beta build) ✅ **done 2026-08-04**

The absolute perplexity moves 16.7% across strides (21.706 / 18.463 / 18.081)
while ΔPPL stays inside a **0.197 pp band** (−0.140% / −0.030% / **+0.058%**),
never worse than 0.14% in magnitude. The margin over the paper's +1.57% never
drops below **1.51 pp**, and the whole band is 13% of that margin — **the
conclusion does not depend on the recipe, and the assumption is closed.**

Write it as "lossless under every recipe", **not** "ΔPPL is invariant": the value
drifts monotonically and changes sign at stride 256. `--stride` now takes several
values and evaluates one build at each (`results/stride_sens.json`, 13.3 h).

### ~~3 — Layer-wise error profile~~ ✅ **done 2026-08-04**

`share_fits=True` holds at depth. The shared GELU prototype is *better* at layer
23 than at layer 0 (0.86×) and its worst layer is **L4**, not a deep one;
LayerNorm is flat (1.06–1.14×). In-situ and clean-input error agree to a median
0.65% at the shared sites, so upstream drift contributes essentially nothing.
Cumulative block error **saturates** rather than compounding (1.39× over 23
blocks, and it *falls* at L3). Only 2 of 145 sites ever see an input outside the
pooled fitted range, worst 0.0015%.

The one thing that grows with depth is **`av_matmul`** — 2.85× of L0 at its L21
peak, and the largest per-site error in the network (4–6× every other primitive).
Not a sharing problem (matmuls are fitted per site); it is the FP-multiply
identity path, and it points at the same place the spike budget does.
`experiments/layer_error_profile.py`, 21 min on local CPU.

### 1 — CV: ViT × ImageNet  ★ now the top gap

**Why.** The paper covers CV/NLU/NLG; we have two of three. ViT would complete
the generality argument and is the paper's own strongest cell (ViT-B/16 −0.44%,
ViT-M/16 −0.64% conversion loss, Table 1).

**Wiring is nearly free.** `src/mbe/hf_convert.py` is already model-agnostic, and
ViT was verified to use the same `eager_attention_forward` signature and the same
two-registry dispatch as GPT-2 and RoBERTa. Expect an adapter of the same shape as
`experiments/roberta_sst2.py`, differing only in the eval loop (top-1) and the
data pipeline.

**Blocker.** ImageNet val is 50k images (~7 GB) and needs a license/download.
**Needs vast.ai.** Check `ACT_TARGETS` covers ViT's activation (it maps
`GELUActivation` → exact `gelu`, which is what ViT uses).

### 2 — Method extensions (see `PASN_method.md` §14)

Each carries a measured opening:

| extension | evidence | note |
|---|---|---|
| **softmax→matmul fusion** | attention is 86.5% of spikes; the attention matrix is decoded to FP then re-encoded, a round trip priced at 6.3 spikes/activation (exp 10). **The layer profile now points here too**: `av_matmul` is the largest per-site error in the network and the only one that grows with depth | largest remaining energy target; **changes numerics → full re-eval** |
| ~~**mantissa-prefix router**~~ ✅ **closed 2026-08-06** | ~~`1/x` is the one operator we lose~~ — diagnosed, fixed and defaulted. No new router was needed: `beta=0.5` puts the existing binade ladder on `x − 0.5`. See below. | **do not reopen as an energy item** |
| **budget search where the router degenerates** | rule picks `(2,16)`=17.32 spikes where `(3,8)`=8.76 is *more* accurate — 1.98×, identical across 3 seeds | small change to `rule_budget` |
| ~~**tie the remaining homogeneous primitives**~~ ✅ **closed 2026-08-09 (E6)** | `inv` **cannot** be tied — `beta` shifts the routing key and `1/(t+0.5)` is not homogeneous in `t` (raises, residual 4.2e-3 vs tol 1e-3). `invsqrt` can, but buys 2× storage for 1.19× spikes and 1.11× error on a primitive at **0.0%** of routed elements. Both left off | **do not reopen.** Tying is identity-only by measurement, not oversight |
| **τ sharing across banks** | the four state tensors per basis are **54–61% of stored bytes**, orthogonal to routing, and memory is our weakest axis (level with global MBE at 1.08×) | only route to a memory claim |

**The `1/x` row, closed.** `[0.5, 1)` *is* one binade — the interval `frexp`
normalises a mantissa into — so the exponent router had **1 of 7 banks reachable
and 20000/20000 inputs in it**. PASN there was a global MBE plus router overhead;
it never entered the comparison. `PrefixRouter` already routes on
`(x − beta)/2^gamma`, so `beta=0.5` re-anchors the ladder on the offset and the
banks partition log-densely at `x=0.5`, where `|f''| = 2/x³` is 8× its value at
1.0. `beta` is the IEEE mantissa floor, not a tuned constant.

Now the default (`ConvertConfig.pasn_beta = {"inv": 0.5}`, pinned by
`tests/test_op_cost.py`). At conversion settings: **MSE 4.4e-5 → 1.8e-6 (25×) and
17.0 → 2.9 spikes (5.9×) at once**, for 3.7× stored bytes on that primitive.
`invsqrt` is deliberately left off — `[0.5, 2)` is already two binades, so its
router was never degenerate.

**Why this looked worthless for so long — a measurement trap, not a small
effect.** `1/x` is 0.0–1.3% of softmax's spikes, and both P0.2 / 실험 5 and the
first pass of `inv_router_fix.py` concluded ~1.0× at the op level. Both got there
by **pinning the op's other primitives** (`exp2` and the identity at N=8, T=16)
and varying only the reciprocal. That holds the operator's cost fixed by
construction, so it can only ever report ~1×. It cannot see what happens.

Sweep the whole op budget instead (`op_pareto`, softmax, `pasn` arm):

| | nrmse | spikes |
|---|---|---|
| `beta=0`, b=1 T=8 | 2.60e-2 | 4.65 |
| `beta=0.5`, b=1 T=8 | **6.82e-3** | 4.63 |

**Same spikes, 3.8× the accuracy.** The degenerate reciprocal was never costing
spikes — it was imposing an **accuracy floor on the whole operator**. Lift it and
softmax reaches the MBE front at a much cheaper build, so at matched accuracy:

* isolated `1/x`: **0.85× → 5.39×**
* the softmax operator: **1.88× → 6.41×** (arm `pasn`; on `pasn_rule` it is 4.23×)
* PASN now wins **10/10 operators** on spikes; it was 9/10.

> ⚠️ **Updated 2026-08-10 (E6 §9).** The often-quoted range **1.61–6.41×** picks
> the *best arm per operator*. That is fine as a diagnostic — 실험 12's table
> labels the arm on every row — but it is arm-shopping in a paper, where "the
> method" reads as one configuration. Recomputed on **`pasn_rule` alone** (what
> the conversion path actually builds) and with E6's tying on: **10/10 operators,
> `1.23–5.39×`**, winners' median 3.27×. Bytes remain **0/10**.

So it *is* an energy result; it just arrives through accuracy rather than through
the reciprocal's own spikes. **The general lesson is the one worth keeping: never
judge a primitive-level change by pinning the rest of the operator.** That design
answers "what do this primitive's spikes cost", when the question is "what does
the operator cost at fixed accuracy". `experiments/inv_router_fix.py` (controlled,
~20 min) and `experiments/op_pareto.py --ops activation softmax --acts inv`.

**Tying is on for the identity only — two eligible primitives are not using it.**
`ConvertConfig.pasn_id_tied = True` is the default and **every** GPT-2 record
carries `pasn_id_tied=True` (`gpt2_stage2_fixed`, `stride_sens`, both same-box
decompositions), so the headline (−0.14% then, **−0.19%** on the frozen build)
is a tied number. But the flag reaches
only `_routed_identity` (`convert.py:619`). `invsqrt` (`convert.py:676`) and `inv`
(`convert.py:596`) are built without it, and both are positively homogeneous —
`f(λx) = λ^k f(x)` with `k = −1/2` and `k = −1`, so the router's exponent
factorises out exactly as it does for `k = 1`. Exp 4 §4 verified all three at
`r=1e-3`, **208 params → 13**, accuracy unchanged:

| | free banks | tied |
|---|---|---|
| identity | 1.17e−3 / 208p | 1.08e−3 / **13p** |
| invsqrt | 1.13e−2 / 208p | 1.16e−2 / **13p** |
| inv | 2.75e−2 / 208p | 2.91e−2 / **13p** |

**⚠️ RESOLVED 2026-08-09 (E6) — and the reasoning above was wrong. Do not act on
it.** This section previously argued that `beta=0.5` had *enabled* tying for
`inv` ("the banks began to partition — there are now banks to tie"). That
conflates two different properties. Tying needs `f(from_key(t))` to factorise in
the **routing key**; `beta` shifts that key. `1/(t + 0.5)` is not homogeneous in
`t`, so `_build_tied` **raises** — measured residual `4.21e-3` against a `1e-3`
tolerance. **`beta` and tying are mutually exclusive for `inv`**, and `beta` is
worth far more (MSE 25×, spikes 5.9×) than a storage-only saving. Partitioning is
not homogeneity.

`invsqrt` *does* tie (4 banks → 1 stored basis). But measured at conversion
settings it is **not worth taking**: storage halves (56 → 28 B, 14 → 7 params)
for **1.19× spikes and 1.11× relative error**, on a primitive E11 measured at
**0.0%** of the network's routed elements — roughly 2.5% of stored bytes network
wide, for a numerics change. Wired as `ConvertConfig.pasn_invsqrt_tied`,
**default `False`** (verified bit-identical to the previous build).

So tying stays **identity-only**, and that is now a measured decision rather than
an oversight. It is not additive with the τ row above — `TiedBank` holds a
*reference*, so a tied bank already shares its τ.

**~~Related measurement gap — the op-level table understates PASN.~~ ✅ CLOSED
2026-08-10 (E6 + its §9 follow-up).** Per-primitive gating (`TIED_PRIMS =
{"identity"}`) landed, the three ops were re-swept into
`results/op_pareto_tied.json`, and `experiments/op_pareto_merge.py` folded them
back with the `mbe` arm as a bit-identical control (45 builds × 9 fields).
Bytes improved 1.7–2.7× on those rows (`fp_multiply` 0.06→0.11×, `layernorm`
0.13→0.22×, `attention` 0.18→0.32×) but are **still losses on `pasn_rule`** —
the memory axis stays 0/10 there. Merged artefacts:
`results/op_pareto_merged.json`, `_wins.md`, `op_pareto_merged.md`.
**The original text follows, for provenance.**

**(original) Related measurement gap — the op-level table understates PASN.**
`experiments/op_pareto.py` never passes `tied=True` (`build_mbe_pasn` defaults it
`False`) while the conversion path defaults it on, so the sweep's `pasn_rule` arm
is *not* "the method as it actually is" for the identity-bearing ops, despite the
docstring saying so. Its byte column is measured with tying off on exactly
`fp_multiply` (0.06×), `layernorm` (0.13×) and `attention` (0.18×) — the three ops
where tying applies. **This table is destined for the paper; it currently reports
our own method's memory cost 3–16× too high on those rows.**

The fix needs **per-primitive** gating, not an arm-wide flag: `attention` and
`softmax` also build `exp2`, which is non-homogeneous, and `build_mbe_pasn` raises
on it (`mbe_pasn.py:513`). Gate on `{identity, inv, invsqrt}` in `maker()`.
Cost to re-measure the three ops: the new arm mirrors `pasn_rule` at **6 builds
per op**, whose recorded fit time is 237 + 435 + 417 s ≈ **18 min**, and tied
builds are cheaper still (one prototype fit instead of ~16 per-bank fits). The
existing `mbe` / `pasn` / `pasn_rule_T` / `pasn_rule` records can be reused —
bytes are the deterministic, environment-free axis (§4) — but **re-build one
existing `pasn_rule` record and diff it first**; if it does not reproduce, re-run
all four arms for the three ops (4901 s ≈ 82 min).

### 3 — P0.5 leftovers

Matched toy energy (`waterfall.py` rerun against the matched baseline) and the
`gpt2-medium` re-measure of the Table XI firing rates (that table was taken on
gpt2-small).

---

## 4. Do not repeat these — already settled, and expensive to relearn

* **Do not sweep seeds.** The build is deterministic: `build_mbe_pasn(seed=0)` and
  `seed=7` produce bit-identical state dicts (closed-form readout, fixed init).
  1 h 45 m was spent on two runs that could not have differed. `--seed` is plumbed
  and correct; there is simply nothing stochastic to vary. The real variance
  sources are the **checkpoint** and the **calibration batches**.
* **Do not quote a saving measured at one operating point at another.** The
  `inv_S` broadcast fix recovered 24.3% under `t_fixed=16` on gpt2-small and
  **7.2%** at the headline setting, because its benefit scales with the
  identity's `T`.
* **Do not compare absolute accuracy/perplexity to the paper.** Their ANN is a
  different fine-tune (RoBERTa) or a different eval recipe (GPT-2). Only each
  method's loss against **its own** ANN is comparable. `experiments/p04_report.py`
  enforces this for GPT-2 and refuses to place Stage-1 rows beside Table 3.
* **Do not use our MBE reimplementation as a headline baseline.** Internal
  ablation only.
* **Unit trap in the paper.** Table 3 prints `22.69 (+0.35)` and the prose calls
  it "0.35% conversion loss", but 0.35 is the *absolute* perplexity difference —
  relative it is **+1.57%**. Table 2's "0.24% degradation" is the same conflation.
* **Commit result JSONs as each remote run finishes.** A closed vast.ai box took
  P0.4 Block C and all raw Stage 2 records with it; the numbers survived only
  because they had been transcribed into the vault.
* **Before trusting a knob, check it moves something.** Two knobs were silently
  ignored (`pasn_id_target` under tying, `mbe_readout_order` on the signed
  activation). Build both arms and diff the state dicts.
* **Absolute spike totals are environment-dependent; only ratios travel.** The
  same commit, config and sample on a new box (RTX 5060 Ti, torch 2.12+cu130)
  gave 0.945× the spikes. Per op: matmul 1.001×, activation 0.993×, layernorm
  0.985×, **softmax 0.855×** — bytes identical, ΔPPL identical, so the structure
  is the same and only the firing behaviour moved. Re-measuring the per-bank arm
  on the same box put the `T_j` decomposition at **1.0719×** against the recorded
  1.0649× — ratios survive there because the shift hits both arms alike and
  cancels. **But that cancellation is arm-dependent, not a law.** Re-measuring the
  `n4` arm too put the `N_j` decomposition at **2.5505×** against the recorded
  2.3916% — **6.6% apart**, because matmul moves 1.105× in the `n4` arm while it
  is 1.001× in the per-bank arm, so nothing cancels. Storage is the solid axis:
  bytes came out **identical** (49952 / 127352) on both boxes, so `N_j`'s 2.55×
  memory saving is environment-free. Quote **2.55× memory, 2.4–2.55× spikes,
  1.065× for `T_j`** — and never to four significant figures. Same box is a
  necessary condition for a ratio, not a sufficient one.

---

## 5. Reproducing what exists

```bash
# GPT-2 Stage 2 (the headline), needs GPU for the eval
python experiments/gpt2_wikitext.py --backend mbe_pasn --convert-ops all \
    --model gpt2-medium --epochs 300 --stride 1024 --pasn-t-fixed 16 \
    --pasn-id-target relative --pasn-id-target-rel 1e-2 \
    --json results/gpt2_stage2.json --tag stage2-t16
python experiments/p04_report.py --json results/gpt2_stage2.json

# NLU (all local CPU, ~50 min per base cell)
python experiments/roberta_sst2.py --task sst2 --size base --backend mbe_pasn \
    --convert-ops all --epochs 300 --pasn-t-fixed 16 \
    --pasn-id-target relative --pasn-id-target-rel 1e-2 \
    --json results/roberta_sst2.json --tag sst2-base-t16
python experiments/run_nlu.py --dry-run     # resumable driver, skips finished cells

# cost/energy without paying for an eval
python experiments/gpt2_wikitext.py ... --build-only

# tests (37)
python tests/test_mbe_neuron.py
```

Result files: `results/gpt2_stage2_fixed.json` (post-fix spike totals),
`results/roberta_sst2.json`, `results/roberta_nlu.json`,
`results/sst2_ckpt_var.json`, `results/roberta_mr_ckpt.json`,
`results/firing_rates_gpt2.json`, `results/budget_objective.json`,
`results/stride_sens.json` (recipe sensitivity, 3 records),
`results/tj_decomp_samebox.json` + `results/nj_decomp_samebox.json` (the paired
per-bank and `n4` builds behind the same-box decompositions),
`results/layer_error_profile.json` (per-layer error).

Related in-repo docs: `PASN_method.md` (method spec, §14 = open extensions),
`experiments/NLU_RESUME.md` (NLU cell status), `experiments/P0.4_GPT2_HANDOFF.md`
(GPT-2 gate + trap list).
