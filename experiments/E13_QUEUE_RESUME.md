# E13 → E4 → E3(b) queue — resume brief

> ## ✅ ALL THREE FINISHED 2026-08-11. The box is shut down.
>
> Kept as the record of how the queue was run, and because §3–§5 are the
> reusable parts. **Results and their interpretation live in
> `results/RESULTS_2026-08_cycle.md`.**
>
> | run | result |
> |---|---|
> | E13 | band **0.2382 pp** (−0.1809 / −0.0321 / +0.0573%), worst 0.181%, margin 1.75 pp — **C6 recovered** |
> | E4 | **negative** — S loses 7.73× spikes, 2.49× bytes, no cross-site sharing (1.00× vs 2.23×) |
> | E3(b) | **−0.196%** against the paper's **+3.36%**, margin **3.56 pp** — Table 3's second row filled |
>
> Everything was pulled and committed before shutdown; the box held nothing
> that is not in `results/`.

Written 2026-08-10, while the queue was running. Self-contained: a session that
has never seen this run should be able to finish and interpret it from here.

**Box** (gone): `ssh -p 36003 root@85.10.218.46` — RTX 5060 Ti, torch
2.12.0+cu130, 112 vCPU. Everything lived in `~/pasn`.

**Gate passed before anything else**: ANN `21.7058` at stride 1024, matching every
prior box to four places. The stride 512 / 256 ANN references also reproduced the
recorded sweep exactly (`18.4629` / `18.0806`).

---

## 1. What is queued, and how

Three independent tmux sessions, each waiting on the previous one to **end** —
not to succeed. Each experiment has its own build and does not depend on the
others' results, so a failure upstream must not block what follows.

| session | script | what | ~cost |
|---|---|---|---|
| `e13` | `run_e13.sh` | frozen build, `--stride 1024 512 256` | 14.5 h |
| `e4` | `run_e4.sh` | `--backend mbe_pasn_s`, stride 1024 | ~3 h |
| `e3b` | `run_e3b.sh` | `--dataset wikitext-103-v1`, stride 1024 | ~2.7 h |

```bash
ssh -p 36003 root@85.10.218.46 'tmux ls; tail -3 ~/pasn/run_e13.log'
```

**Records are written per point** (`emit(s)` in `gpt2_wikitext.py`), so an
interrupted run keeps whatever finished. Pull after every point:

```bash
scp -P 36003 root@85.10.218.46:'~/pasn/results/{e13_stride,e4_pasn_s,e3_wiki103}.json' results/
```

---

## 2. Landed so far

| tag | stride | ANN | SNN | ΔPPL | status |
|---|---:|---:|---:|---:|---|
| `e13-frozen-s1024` | 1024 | 21.7058 | 21.6666 | **−0.1809%** | ✅ committed |
| `e13-frozen-s512` | 512 | 18.4629 | — | — | running |
| `e13-frozen-s256` | 256 | 18.0806 | — | — | queued |

Build fingerprint on this box: **53,888 B / 13,472 p / 339 prims, shared 2.23×**
— byte-identical to the frozen build on two previous boxes.

---

## 3. What each result is for

### E13 — carries C6 over to the frozen build

C6 ("the conclusion does not depend on the evaluation recipe") was established on
the **no-beta** build; the freeze moved to the beta build, which had one stride
point. **Completion condition**: ΔPPL at ≥2 strides on the frozen build, and a
statement of that band against the 1.76 pp margin over the paper's +1.57%.

Reference band from the no-beta build: **−0.140 / −0.030 / +0.058%**, width
0.197 pp. If the frozen build's band is comparable, write **"measurably lossless
under every recipe"**; if not, write it for the strides actually measured.

⚠️ **Do not read a negative ΔPPL as an improvement.** On the no-beta build the
sign flipped at stride 256.

### E4 — PASN-S, one point beside flat

Gate already passed locally: the three arms build distinct state dicts and
`pasn_s_n_shared` moves storage, so the knob is live (trap 9). **Completion
condition**: place S beside flat and show **neither dominates** — flat is
expected to hold the minimum-spike point, S to hold accuracy/memory above it.
S's own knobs stay at defaults; porting the `N_j` rule to S is out of scope.

### E3(b) — the Table 3 WikiText-103 row

Uses the **word-level** release deliberately. The `-2` and `-103` **raw** test
splits are byte-identical (measured), so a raw `-103` run would change no
evaluation token — only the calibration corpus. The paper's 22.34 vs 22.65 can
only come from the word-level releases, which differ through `<unk>` (6.31% of
`-2`'s test words against 1.03% of `-103`'s, because the vocabulary is built on
each corpus's own train split, and `-2`'s train is 49× smaller).

⚠️ **Disclose with the number**: `<unk>` is degenerate for a BPE model, which
splits the literal into `<`, `unk`, `>`. It hits the ANN and SNN arms alike, so
the **relative** ΔPPL is legitimate — but the **absolute** ppl is not comparable
to our raw 21.7058, and must not be placed beside it.

Compare our ΔPPL against the paper's Wiki-103 row, **+3.36%** (23.41 against an
ANN of 22.65).

---

## 4. Traps that apply to these runs specifically

* **Only ΔPPL travels between boxes.** Spike totals do not (trap 13). E13's three
  points share one build on one box, so its band is internally clean; but do not
  divide these spike counts against records from other boxes.
* **Measured noise floor: 0.0064 pp.** E13's stride-1024 point re-measured the
  frozen build's headline on an independent box (−0.1873% → −0.1809%). Use this
  when asking whether a ΔPPL difference is real: the beta/no-beta build gap
  (0.0476 pp) is 7.5× it, the recipe band (0.197 pp) is 31×.
* **`--calib-max-rows` defaults to 20000** and is verified token-identical to
  reading every row on the default config. It exists because WikiText-103's train
  split is 540 MB; without it E3(b) would tokenise all of it to keep 32k tokens.
* **The headline is now two measurements**: −0.1873% and −0.1809%. Quote
  **−0.18%**, or the pair. The margin over +1.57% is 1.75–1.76 pp either way.

---

## 5. If the box dies

Everything above the last landed point is safe in the committed JSONs. To
resume, bring up a new box and repeat §1 — but **run the ANN gate first**
(`--backend none --stride 1024`, ~1 min) and refuse to proceed unless it prints
**21.7058**. A closed box has already taken P0.4 Block C and the raw Stage 2
records once, and nearly took the E1 sweep.

Related: `PAPER_PLAN.md` §E13/§E4/§E3, `experiments/E1_RESUME.md` (same pattern),
`results/RESULTS_2026-08_cycle.md`.
