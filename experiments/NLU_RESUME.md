# NLU sweep — where it stopped and how to continue

Interrupted 2026-08-02 ~20:55 KST (machine shut down mid-run). Everything already
finished is committed; nothing needs redoing.

## Done

| cell | ANN | SNN (T=16) | Δ relative | paper |
|---|---|---|---|---|
| **sst2 / base** | 94.04 | 94.15 | **+0.12%** | −1.09% |
| **mr / base** | 89.02 | 88.37 | **−0.74%** | −0.44% |
| sst2 / large | 96.44 | *(interrupted)* | — | −0.25% |

`results/roberta_sst2.json` holds the sst2/base pair (plus its rule-`T_j` arm);
`results/roberta_nlu.json` holds mr/base and the sst2/large ANN.

⚠️ **mr/base is our first cell where the paper does better** (−0.74% vs −0.44%).
Do not bury it. See the caveat below before reading it as a loss: their MR ANN is
89.39 and ours is 89.02, and these are unrelated fine-tunes.

## Remaining

Run from the project root, SNN env. The driver skips nothing and appends, so it
is safe to rerun a whole cell — just give it a fresh `--tag` or accept the
duplicate record.

```bash
bash experiments/run_nlu.sh          # re-runs mr/base too; edit the list to skip
```

Or one cell at a time (this is what the driver does):

```bash
PY="C:/Users/cm120/miniconda3/envs/SNN/python.exe"
J=results/roberta_nlu.json
# sst2/large -- ANN is already recorded, only the T=16 arm is missing
"$PY" experiments/roberta_sst2.py --task sst2 --size large --backend mbe_pasn \
    --convert-ops all --epochs 300 --pasn-id-target relative \
    --pasn-id-target-rel 1e-2 --pasn-t-fixed 16 --json $J --tag sst2-large-t16
# then subj/large, then sst5/large (ANN + T=16 each)
"$PY" experiments/roberta_sst2.py --task subj --size large --backend none \
    --json $J --tag subj-large-ann
"$PY" experiments/roberta_sst2.py --task subj --size large --backend mbe_pasn \
    --convert-ops all --epochs 300 --pasn-id-target relative \
    --pasn-id-target-rel 1e-2 --pasn-t-fixed 16 --json $J --tag subj-large-t16
```

Timing observed: base cell ≈ 50 min (build 15 + eval 35); large cell ≈ 70–80 min
(the SST-2 large arm was showing ETA 63 min for the SNN eval alone). All CPU —
no GPU needed for NLU.

## Checkpoint availability (already probed, don't re-probe)

| cell | checkpoint | provenance |
|---|---|---|
| sst2/base | `textattack/roberta-base-SST-2` | solid |
| mr/base | `textattack/roberta-base-rotten_tomatoes` | solid |
| sst2/large | `philschmid/roberta-large-sst2` | solid |
| sst5/large | `Unso/roberta-large-finetuned-sst5` | weak |
| subj/large | `ghatgetanuj/roberta-large_cls_subj` | weak |
| sst5/base, subj/base | **none found** | — |

The base row cannot be completed from public checkpoints. Fine-tuning them
ourselves would be producing a source ANN, not conversion work — a separate
decision, not a gap to quietly fill.

## The caveat that governs all of these

Every checkpoint is a **third-party fine-tune with an unpublished recipe**, and
so is the paper's. Only each SNN's loss against **its own ANN** is meaningful;
absolute accuracies must never be placed beside the paper's. That cuts both ways
— it is also why mr/base losing to the paper is not yet a settled loss.
