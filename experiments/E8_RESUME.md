# E8 — ViT × ImageNet — resume brief

Written 2026-08-11 while the run is going. Self-contained: a session that has
never seen this run should be able to finish it and write it up from here.

**Box**: `ssh -p 46162 root@85.10.218.46` — RTX 5060 Ti 16 GB, 100 GB disk,
64 vCPU, torch **2.11.0+cu128**. Everything lives in `~/pasn`.

```bash
ssh -p 46162 root@85.10.218.46 'tmux ls; tail -3 ~/pasn/run_e8.log'
scp -P 46162 root@85.10.218.46:'~/pasn/results/e8_vit_imagenet.json' results/
```

---

## 1. What is running

`tmux e8` → `run_e8.sh` → `experiments/vit_imagenet.py`, ViT-B/16 × ImageNet-1k,
**full 50,000-image validation**, Stage 2, `T=16`, epochs 300.
Started 2026-08-11T10:30Z. **~8.2 h** (build ~1 h + eval ~7.2 h at 0.06 batch/s,
1563 batches of 32). One record, written at the end.

⚠️ **Unlike the GPT-2 runs there is no per-point emit** — a single evaluation
means a single record. If the box dies mid-eval, the run is lost and must be
repeated. Pull as soon as it lands.

---

## 2. Already established (committed, `4d648bf`)

| | |
|---|---|
| data | val **50,000 / 1,000 classes / exactly 50 each**, 6.3 GB, read straight from parquet |
| ANN gate | `google/vit-base-patch16-224` → **81.10** = its published top-1 |
| wiring | 12 activations + 12 attention blocks marked → **Stage 2** |
| smoke (epochs 50, 256 imgs) | ANN 81.64 → SNN 80.86, **−0.78 pp / −0.96% relative** |
| storage | 6,500 params / 26,000 B / 171 primitives |

---

## 3. How to read the result

**Compare the RELATIVE column only.** The paper's ViT-B/16 row is
**83.44 → 83.00**; its printed "0.44%" is the *absolute* point difference, and
relative it is **−0.53%**. Our checkpoint is 2.34 pp weaker than theirs, so
absolute top-1 is not comparable — the same discipline the RoBERTa cells use.
`--paper-row ViT-B/16` makes the script print both.

**If we lose, report it.** The smoke was −0.96% against their −0.53%, at 1/6 the
fit epochs. epochs 300 should improve it, but a weaker source ANN can genuinely
convert differently, and RoBERTa MR already showed checkpoint choice moving a
conversion loss by 6.7×. **A loss here is a result, not a failure to hide** — it
would mean writing the CV row as "measured, and we do not lead it", which is
still the third modality the paper has and we did not.

---

## 4. Traps specific to this run

* **timm ViTs convert to nothing.** `make_spikable` and
  `make_attention_spikable` both return **0** on timm models: attention is inline
  in `timm.layers.attention.Attention` and the activation is `torch.nn.GELU`,
  which `ACT_TARGETS` does not list. The "SNN" would be the ANN and the loss
  would print as 0.00%. The adapter hard-fails on a zero marker count. **The
  paper's ViT-M/16 is a timm model, so that cell is out of reach** without new
  wiring.
* **Never use `datasets.load_dataset` for ImageNet here.** It re-encodes into
  arrow (which filled a 100 GB disk once), and `load_dataset(repo,
  split="validation")` downloads the **whole 155 GiB repository** regardless of
  the split. Fetch with `snapshot_download(..., allow_patterns=["data/validation-*"])`
  and read the parquet with pyarrow, which is what `vit_imagenet.py` does.
* **The operator budget is model-dependent.** ViT spends **55.9%** on attention
  against GPT-2's 81.8%, because seq 197 makes the `S²` term 27× cheaper.
  Do not carry "attention is 86.5%" into a ViT sentence → `PAPER_PLAN.md` §4.3.

---

## 5. Where the write-up goes

1. **`results/RESULTS_2026-08_cycle.md`** — add a CV section, and update §1's
   paper-table coverage (Table 1 is the last blank).
2. **`PAPER_PLAN.md`** — close §E8, set C10 (three modalities).
3. **`RELATED_WORK.md`** — §1's "models" row and the evidence-layer claim; the
   three-modality argument becomes available.
4. **`NEXT_EXPERIMENTS.md`** — §1 status table.
5. Journal: `PASN_vault/60 - 연구일지/E8 - ViT x ImageNet.md`.

Related: `PAPER_PLAN.md` §E8, `experiments/E13_QUEUE_RESUME.md` (same pattern),
`results/RESULTS_2026-08_cycle.md`.
