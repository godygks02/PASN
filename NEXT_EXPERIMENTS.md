# What to run next — cold-start brief

Written 2026-08-03. Self-contained: a session that has never seen this project
should be able to act from this file alone.

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
| network | GPT-2-medium × WikiText-2 | +1.57% | **−0.14%** | operator set + T matched, verified |
| network | RoBERTa-base × SST-2 | −1.09% | **+0.12%** | robust over 3 checkpoints |
| network | RoBERTa-large × SST-2 | −0.25% | **−0.12%** | hardest cell, still ahead |
| network | RoBERTa-base × MR | −0.44% | −0.73% … −0.10% | **undecidable** (see §3) |
| operator | Table XI firing rates | 7 primitives | **6/7 at 2.2–10.3× fewer spikes** | cross-model caveat |
| function | Table X MSE vs N | per-function | reproduced and beaten | clean |

Budget-rule decomposition on GPT-2 (Stage 2): the value is in `N_j`
(**2.392×** spikes, **2.55×** storage); `T_j` is only **1.065×** and free in
storage. Attention is **86.5%** of the spike budget — activations are 3.3%.

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

### 1 — GPT-2 evaluation-recipe sensitivity  ★ highest value

**Why.** The single open assumption behind the headline. The paper never states
its perplexity recipe (stride/context) — not in the body, not in appendix G.1. We
score at stride 1024 (non-overlapping); shorter strides give every token more
left context and lower perplexity without bound (1024→21.71, 512→18.46,
256→18.08). If ΔPPL is stable across strides, the conclusion does not depend on
knowing their recipe, and the assumption closes.

**Where.** GPU (rent again). gpt2-medium at S=1024 is far too slow on CPU.

```bash
M="--model gpt2-medium --epochs 300 --backend mbe_pasn --convert-ops all \
   --pasn-id-target relative --pasn-id-target-rel 1e-2 --pasn-t-fixed 16 \
   --json results/stride_sens.json"
for s in 1024 512 256; do
  python experiments/gpt2_wikitext.py $M --stride $s --tag stride-$s
done
```

**Cost.** ~2 h per point on an RTX 5060 Ti; shorter strides are *slower* (more
windows). Budget 6–8 h.

**Reads as.** ΔPPL roughly constant → assumption closed, headline safe. ΔPPL
moving a lot → we must say the comparison depends on a recipe we do not know.

### 2 — CV: ViT × ImageNet  ★ third modality

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

### 3 — Layer-wise error profile  ★ cheap, still unmeasured

**Why.** P0.4 §4.4 asked for it and it was never done. `share_fits=True` lets 24
layers reuse one fitted primitive, so per-layer activation distributions could
drift with depth. This is the one "does it actually hold at depth" question we
have not answered directly.

**Where.** Local CPU. Instrument a converted model and compare per-layer hidden
states against the ANN's. No new run needed beyond one build.

### 4 — Method extensions (see `PASN_method.md` §14)

Each carries a measured opening:

| extension | evidence | note |
|---|---|---|
| **softmax→matmul fusion** | attention is 86.5% of spikes; the attention matrix is decoded to FP then re-encoded, a round trip priced at 6.3 spikes/activation (exp 10) | largest remaining energy target; **changes numerics → full re-eval** |
| **mantissa-prefix router** | `1/x` is the one operator we lose (0.26×); its argument is already a mantissa, so exponent routing is a no-op by construction | fixes the story more than the spikes (0.3% of total) |
| **budget search where the router degenerates** | rule picks `(2,16)`=17.32 spikes where `(3,8)`=8.76 is *more* accurate — 1.98×, identical across 3 seeds | small change to `rule_budget` |
| **τ sharing across banks** | the four state tensors per basis are **54–61% of stored bytes**, orthogonal to routing, and memory is our weakest axis (level with global MBE at 1.08×) | only route to a memory claim |

### 5 — P0.5 leftovers

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
`results/firing_rates_gpt2.json`, `results/budget_objective.json`.

Related in-repo docs: `PASN_method.md` (method spec, §14 = open extensions),
`experiments/NLU_RESUME.md` (NLU cell status), `experiments/P0.4_GPT2_HANDOFF.md`
(GPT-2 gate + trap list).
