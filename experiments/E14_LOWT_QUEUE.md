# E14 — the low-`T` headline: queue and gates

Written 2026-08-13. **This is the experiment set that decides whether the paper has a
claim**, after the literature pass (`RELATED_WORK.md` §6b) removed two of the three
differentiators we thought we had.

> ### Why this set exists
>
> Prior art (Lee et al. TVLSI 2009, NPE, FQA, BBAL) already does **binade
> segmentation addressed by a leading-zero/exponent read, with the segment count
> solved from an error tolerance**. So the mechanism is not ours. What survives is:
>
> 1. the allocated resource is **spike-domain** (`N_j` bases, `T_j` timesteps), and
> 2. the failure it removes — the baseline's **low-`T` collapse** — **exists only in
>    the spiking setting**, so no one in that literature could have reported it.
>
> Both surviving claims live at low `T`. **Everything below serves them.**

---

## 0. Gates — do not spend GPU before these pass

| # | gate | where | status |
|---|---|---|---|
| G1 | `--pasn-e-min` actually changes the build (fingerprint differs) | local, `--build-only` | ⏳ running |
| G2 | ANN reproduces (`gpt2-medium` ppl **21.7058** @ stride 1024) | local/GPU, ~1 min | — |
| G3 | frozen fingerprint on the box: **53,888 B / 13,472 p / 339 prims** | any build | — |

**G1 is 함정 9.** `pasn_id_target` and `mbe_readout_order` were both silently
inert once; a knob in the signature is not a knob in the build. If two `e_min`
values produce byte-identical state dicts, the sweep measures nothing.

---

## 1. The router ablation — the one that makes it a mechanism claim

**Question.** Does removing the routing bring the low-`T` collapse back *inside our
own system*? Right now the low-`T` result is entirely cross-paper — there is **no
`mbe`-backend low-`T` run in the repository** — and C9 was deleted (E2), so we cannot
fall back on the global-MBE arm: it diverges at *every* `T`, which makes it useless
as an ablation.

**Design.** Collapse the router from inside the working PASN path by raising
`pasn_e_min`. From `PrefixRouter.__init__`:

* banks = near-zero bank(s) + one magnitude binade per sign for `e ∈ [e_min, e_max)`
* `e_max = ceil(log2(max_mag))` **per site**, and if `e_max ≤ e_min` it is forced to
  `e_min + 1` — **one binade per sign, i.e. the global-neuron limit.**

So raising `e_min` collapses sites progressively, in order of domain width. Measured
domains (E2): LayerNorm identity spans **4775** (`e_max ≈ 12`), GELU input
`(−120, 10)` (`e_max ≈ 7`). Hence:

| `e_min` | what is still routed |
|---:|---|
| **−3** | default / frozen build — everything |
| **4** | GELU near-collapsed, LayerNorm identity still routed |
| **12** | essentially everything collapsed to near-zero + 1 binade |

### ⚠️ It must be a 2-D grid, not a line

**The claim is an interaction, not a main effect.** "Routing is what buys low-`T`
robustness" predicts routing depth barely matters at `T=16` (where everyone is
near-lossless) and matters enormously at `T=8`. A one-dimensional sweep at `T=8`
alone cannot distinguish that from "routing helps everywhere".

| | `T=16` | `T=8` |
|---|---|---|
| `e_min = −3` | ✅ have it (frozen, −0.1809%) | **run** (unif — see §2) |
| `e_min = 4` | run | run |
| `e_min = 12` | run | run |

**5 new points.** ≈ 45–65 min build + 1.9 h eval each → **13–15 h GPU**, one
overnight session.

**Predicted outcome to pre-register** (write it down *before* running):
`T=16` degrades mildly across `e_min`; `T=8` degrades sharply. **If both degrade
equally, the mechanism claim fails** and the paper falls back to a cost claim only.

⚠️ **Risk, stated in advance.** The shallow arms may fail for the E2 reason — fitting
is non-monotone in `N` and a wide undivided domain is exactly where our fit was
measured 10–55× worse. If the collapsed arms fail *at both* `T`, the experiment
cannot separate "no routing" from "bad fit", and it goes in as a negative result.

```bash
# 5 points. Run under tmux. Pull each JSON as it lands (memory: a closed box
# once took Block C and all of Stage 2 with it).
for EM in 4 12; do for T in 16 8; do
python experiments/gpt2_wikitext.py --backend mbe_pasn --convert-ops all \
  --model gpt2-medium --epochs 300 --stride 1024 --pasn-t-fixed $T \
  --pasn-e-min $EM --pasn-id-target relative --pasn-id-target-rel 1e-2 \
  --json results/e14_router_ablation.json --tag e14-emin$EM-t$T
done; done
python experiments/gpt2_wikitext.py --backend mbe_pasn --convert-ops all \
  --model gpt2-medium --epochs 300 --stride 1024 --pasn-t-fixed 8 \
  --pasn-id-target relative --pasn-id-target-rel 1e-2 \
  --json results/e14_router_ablation.json --tag e14-emin-3-t8
```

---

## 2. ⚠️ Arm defect in the existing `T` curve — fix while the box is up

`freeze_e1.json` holds `freeze-t16-**unif**` but `e1-t8-**alloc**` / `t4` / `t2`.
Two different arms:

| flag | effect | arm |
|---|---|---|
| `--n-steps 8` | truncates the rule's `T_GRID` to ≤ 8 (`mbe_pasn.py:662`) | **alloc** |
| `--pasn-t-fixed 8` | forces every bank to `T=8` | **unif** — matches the paper's global `T` |

Reading ΔPPL only, this was safe (the index flagged it). **Reading spikes, it is
not**: the 3.25× at `T=8` cannot be attributed to `T` while the arm also changed.
The `e14-emin-3-t8` run above is the `unif` `T=8` point and repairs this.

📌 `vit_imagenet.py` and `roberta_sst2.py` expose **only** `--pasn-t-fixed`, so
every non-GPT-2 low-`T` point is `unif` automatically. Standardise on `unif`.

---

## 3. ViT-M/16 @ `T=8` — the cell where the comparison is cleanest

The paper's Table 4 has ViT-M/16 going **85.95 → 1.17** at `T=8`, and ViT-M/16 is the
**only** model where we can also reconstruct their spike cost without a cross-model
caveat (Table XI is measured during ViT-M inference — see
`experiments/energy_ratio_vitm.py`). One run puts accuracy and compute on the same
model at the same `T`.

ANN is already measured (**84.922**). ⚠️ **The script re-ran it anyway** —
`acc_ann = None if a.build_only else top1(...)` had no way to skip only the ANN, so
every low-`T` point would have paid ~3 h of GPU to re-measure a number we have.
**`--ann-acc` added 2026-08-13**; it stamps `acc_ann_reused=True` on the record so a
reused value can never be mistaken for a measurement.

```bash
# gate the box first (~10 min) -- E8's rule: if this is not near 84.9, stop
python experiments/vit_imagenet.py --framework timm \
  --model vit_medium_patch16_reg4_gap_256 --backend none --limit 2000 --batch-size 64

python experiments/vit_imagenet.py --framework timm \
  --model vit_medium_patch16_reg4_gap_256 --backend mbe_pasn --convert-ops all \
  --epochs 300 --batch-size 32 --pasn-t-fixed 8 --ann-acc 84.922 \
  --pasn-id-target relative --pasn-id-target-rel 1e-2 \
  --paper-row ViT-M/16 --json results/e8m_vit_t8.json --tag e8m-vitm16-t8
```

⚠️ **Do not predict the result.** GPT-2 degrades gracefully at `T=8` (+0.062%); ViT
may not. E8's own lesson (`RESULTS_CV.md` §6) is that reduced-epoch smokes mispredict
by 48–192×, and we have no ViT low-`T` point at all.

**Cost** ≈ **8–9 h**. The `T=16` run was 11h13m = ~2.5–3 h ANN (**now skipped via
`--ann-acc`**) + 45.8 min build + ~8 h SNN.

⚠️ **Do not assume `T=8` halves the SNN pass.** `RESULTS_CV.md` §7 records that the
ViT eval is **data-bound**, not compute-bound — *"전처리가 단일 스레드라 평가가
데이터에 묶인다(배치당 ≈ 5.9 s 가 디코딩)"*. JPEG decoding does not scale with `T`,
so halving the timesteps removes far less than half the wall clock. Estimate, not
measured: no ViT low-`T` point exists yet.

---

## 4. Local, no GPU needed

| item | command | status |
|---|---|---|
| **RoBERTa MR @ `T=8`** | `roberta_sst2.py --task mr --size base --pasn-t-fixed 8` | ⏳ running |
| Table XI re-measure | `firing_rates.py --model gpt2-medium --epochs 300` | queued |
| `e_min` build gate (G1) | `gpt2_wikitext.py --build-only --pasn-e-min …` | ⏳ running |

**Why MR is worth running even though the `T=16` cell is undecidable.** Our two MR
checkpoints straddle the paper: `textattack/...rotten_tomatoes` gives **−0.738%**,
`textattack/...rotten-tomatoes` (a *different upload*) gives **−0.106%**, and the
paper's is −0.44% — spread 0.63 pp, verdict impossible. **At `T=8` the paper's
RoBERTa-B falls 89.39 → 50.17**, a 43.9% relative collapse — **70× the checkpoint
spread.** The cell that is undecidable at `T=16` becomes decidable at `T=8`, and it
is free (seq 128, local CPU).

---

## 5. Order

1. **G1** (local, running) — if it fails, §1 is dead and must be redesigned.
2. **MR @ `T=8`** (local, running) — cheapest cell in the whole set.
3. **§1 router ablation, 5 points** (~14 h) — this is the one that turns a comparison
   into a mechanism.
4. **§3 ViT-M @ `T=8`** (~6 h) — the cleanest accuracy+compute cell.
5. Table XI re-measure (local).

**§1 before §3.** If the ablation comes back negative, the framing changes and §3's
value changes with it.

---

Related: `RELATED_WORK.md` §6b (why this set exists) · `PAPER_TABLE_COMPARISON.md` §2
(cost axis) and §6 (remaining) · `experiments/energy_ratio_vitm.py` ·
`PASN_vault/60 - 연구일지/E1 - 타임스텝 예산 스윕 (비용 축).md`.
