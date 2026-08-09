# E0 freeze + E1 accuracy axis — resume brief

Written 2026-08-09, after a vast.ai instance was shut down mid-sweep.
Self-contained: a session that has never seen this run should be able to finish
it from this file alone.

Results so far: `results/freeze_e1.json` (2 records), raw stdout
`results/freeze_e1_run.log`. Both committed.

---

## 1. What is done

| # | tag | T | arm | ΔPPL | spikes | bytes | status |
|---|---|---|---|---|---|---|---|
| 1 | `freeze-t16-unif` | 16 | unif (`--pasn-t-fixed 16`) | **−0.1873%** | 2.401e9 | 53,888 | ✅ |
| 2 | `e1-t8-alloc` | 8 | alloc | **+0.0615%** | 7.828e8 | 53,888 | ✅ |
| 3 | `e1-t4-alloc` | 4 | alloc | — | — | — | ❌ lost in flight |
| 4 | `e1-t2-alloc` | 2 | alloc | — | — | — | ❌ never started |
| 5 | `e1-t16-alloc` | 16 | alloc | — | — | — | ⛔ **dropped by decision** |

ANN reference **21.7058** on both, reproducing the recorded value exactly — the
eval recipe, text join and tokenisation are confirmed on a fresh box.

**#5 was deliberately dropped**: it is the T=16 anchor of the alloc curve, and
#1 already occupies that abscissa on the unif arm. Reinstate it only if the
curve's T=16 point being a different arm turns out to matter; it costs ~3.2 h.

---

## 2. Resuming — it is safe to finish on a different box

**Only ΔPPL is needed from #3 and #4**, and ΔPPL reproduces across boxes (the
stride sweep verified this, and #1/#2 above reproduced the ANN exactly). So the
two missing points can be run anywhere.

> ⚠️ **Do not build a spike curve across boxes** (함정 13). Absolute spike totals
> are environment-dependent and the effect is *arm*-dependent, so it does not
> cancel. **The spike axis of E1 is already complete and internally consistent**
> from the cost sweep — `results/timestep_sens_cost.json`, all 8 builds on one
> local CPU, giving 9.52× from T=16→2 (energy only 2.41×). Quote the cost axis
> from there, never by dividing GPU records from two boxes.
>
> Bytes are environment-free (53,888 on both the box and the local build), so
> those may be mixed freely.

### The commands

```bash
# environment (fresh vast.ai box, ~10 min)
tar czf - src experiments tests requirements-vastai.txt | \
  ssh -p PORT root@HOST 'mkdir -p ~/pasn/results && tar xzf - -C ~/pasn'
ssh -p PORT root@HOST '/venv/main/bin/pip install -q -r ~/pasn/requirements-vastai.txt'

# gate: must print 21.7058 before anything else means anything (~45 s)
/venv/main/bin/python experiments/gpt2_wikitext.py --backend none \
    --model gpt2-medium --stride 1024 --json results/sanity.json --tag sanity-ann-full

# the two missing points (~2.4 h total, see §3)
BASE="--backend mbe_pasn --convert-ops all --model gpt2-medium --epochs 300 \
--stride 1024 --pasn-id-target relative --pasn-id-target-rel 1e-2 \
--json results/freeze_e1.json"
for T in 4 2; do
  /venv/main/bin/python experiments/gpt2_wikitext.py $BASE --n-steps $T --tag e1-t$T-alloc
done
```

**Copy `results/freeze_e1.json` onto the box first** — the runner appends to it
and skips tags that are already present, so the finished points will not re-run.

**Launch detached and pull results as each point lands.** A closed box has
already taken P0.4 Block C and Stage 2's raw records once, and it nearly took
this sweep too:

```bash
nohup ./run.sh > ~/pasn/run.log 2>&1 &      # survives the terminal closing
scp -P PORT root@HOST:'~/pasn/results/*.json' results/   # after every point
```

---

## 3. Measured timings — the earlier estimate was wrong

Builds are **~3× slower** than the 30 min I extrapolated from `stride_sens`.
Measured on RTX 5060 Ti / torch 2.11+cu128 / 64 vCPU:

| T | build | SNN eval | total |
|---|---|---|---|
| 16 | 5,262 s (88 m) | 6,732 s (112 m) | **3.3 h** |
| 8 | 2,662 s (44 m) | 4,260 s (71 m) | **1.9 h** |
| 4 | ~1,900 s est. | ~2,400 s est. | ~1.2 h |
| 2 | ~1,500 s est. | ~1,500 s est. | ~0.8 h |

The eval scales close to linearly in `T`; the build does not scale as steeply.
**Remaining work ≈ 2.0–2.4 h** plus ~10 min of setup.

---

## 4. What the two finished points already say

**E0 is answered.** The build the repository actually produces gives ΔPPL
**−0.187%** — the first downstream number for today's defaults, since every
prior headline record predates the `beta=0.5` default of 2026-08-06.

⚠️ **It missed the pre-registered gate.** The completion condition said
`|ΔPPL| ≤ 0.14%`; 0.187% is outside it, in the favourable direction. That band
was measured by the stride sweep **on the no-beta build**, so it was never this
build's band — but the gate was stated in advance and it was not met, so say so
rather than widen the band after the fact. The two builds differ by 0.047 pp at
the same stride, against a recipe-induced spread of 0.197 pp on the other build.
Margin over the paper's +1.57% grows to **1.76 pp**.

⚠️ **New gap**: "lossless under every recipe" was established for the *no-beta*
build. This build's stride sensitivity is unmeasured. Two more stride points
(512, 256) would carry the claim over; until then, state it for stride 1024.

**E1 is half-answered and the half is the interesting one.** At T=8 the loss is
**+0.062%** — inside the band every stride of the sweep produced. The paper's
Table 4 at reduced T diverges to **41072 / 11992** from a 22.65 baseline.

⚠️ **Not yet the same axis**: Table 4 is **WikiText-103**, ours is WikiText-2
(→ E3). Either finish E3 or label the curve with its corpus.

⚠️ **#1 and #2 are different arms** (unif vs alloc). Do not divide their spike
totals and call it a T-scaling ratio; use `timestep_sens_cost.json` for that.

---

Related: `PAPER_PLAN.md` §E0/§E1, `NEXT_EXPERIMENTS.md` §4 (traps),
`PASN_vault/60 - 연구일지/E0 - 빌드 프리즈와 헤드라인 재검증.md`,
`PASN_vault/60 - 연구일지/E1 - 타임스텝 예산 스윕 (비용 축).md`,
`experiments/NLU_RESUME.md` (the same pattern, for the NLU cells).
