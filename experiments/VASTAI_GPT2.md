# GPT-2 × WikiText-2 on vast.ai — run plan

> **New session? Read [`P0.4_GPT2_HANDOFF.md`](P0.4_GPT2_HANDOFF.md) first.**
> This file is the environment and eval recipe; the handoff carries the eleven
> experiments' conclusions, the three decisions P0.4 has to settle, and the
> traps already stepped in. The research notebook is not in git, so that file
> is the only place those live.

Goal: put our backends next to the **MBE paper's published Table 3**, not next to our
own reimplementation.

| | Param | WikiText-2 ppl | T |
|---|---|---|---|
| ANN GPT-2 (paper's reference) | 346 M | **22.34** | 1 |
| A2S — the paper's method | 346 M | **22.69 (+0.35 %)** | 16 |
| SpikeGPT (directly trained) | 216 M | 18.01 | 1024 |

So the target is: **conversion loss ≤ +0.35 % at T = 16**, on the same model size.

## Setup that must match, or the comparison is meaningless

* **`gpt2-medium` (345 M)**, not `gpt2` (124 M). This is now the script default.
* **T = 16** (`--n-steps` is not exposed; the default `ConvertConfig.n_steps` is 16).
* **Canonical text join.** Blank lines are now *kept* (`"\n\n".join(ds["text"])`), the
  recipe every published GPT-2 WikiText number uses. Our old figure of 34.52 was
  measured with blank lines dropped **and** on `gpt2`, so it is not comparable to
  anything — discard it.
* **Sliding-window perplexity** (default), ctx = 1024, stride = 512.

## What is actually converted (Stage 1)

MLP GELU + every LayerNorm. **Attention stays exact FP** — HF computes QK^T / softmax /
attn·V functionally inside `eager_attention_forward`, so there is no module to hook and
no `Softmax` / `MatMulAA` marker exists in a real GPT-2. Consequences to state in any
write-up:

* the routed-Softmax and spike-driven-matmul work does **not** apply to these numbers;
* this is *not* a fully spiking Transformer yet — it is a spiking-GELU + spiking-LayerNorm
  Transformer.

## Install

```bash
git clone <this-repo> pasn && cd pasn
pip install -r requirements-vastai.txt
```

## Step 1 — sanity + timing (do this first, ~minutes)

Ten windows only. Confirms the download, the eval recipe, and gives a per-window rate to
extrapolate from before committing to a full run.

```bash
python experiments/gpt2_wikitext.py --backend none --model gpt2-medium --limit-blocks 10
```

Expect the ANN perplexity to be in the low 20s. **If it is not, stop and report** — the
absolute number has to be near 22.34 before any conversion result means anything.

Then one converted run, still 10 windows:

```bash
python experiments/gpt2_wikitext.py --backend mbe_pasn_s --model gpt2-medium --limit-blocks 10 --epochs 300
```

Note the printed `blocks/s` for the SNN pass. Spiking LayerNorm runs 49 times per forward
with T=16, so the SNN eval is far slower than the ANN — use the rate to decide whether the
full run is worth it or whether to stay at a few hundred windows.

## Step 2 — the comparison (four runs)

```bash
M="--model gpt2-medium --epochs 300 --eval-batch-size 4"

python experiments/gpt2_wikitext.py --backend none        $M
python experiments/gpt2_wikitext.py --backend mbe         $M
python experiments/gpt2_wikitext.py --backend mbe_pasn    $M
python experiments/gpt2_wikitext.py --backend mbe_pasn_s  $M --pasn-s-n-shared 2 4 --pasn-s-restarts 3
python experiments/gpt2_wikitext.py --backend pasn        $M
```

If the full test set is too slow, add the **same** `--limit-blocks N` to every run — a
partial evaluation is fine for comparing backends, it just is not comparable to the
paper's absolute 22.69.

### Optional: isolate the activation

```bash
python experiments/gpt2_wikitext.py --backend mbe_pasn_s $M --convert-ops activation
```

Much cheaper (LayerNorm stays exact) and isolates the neuron's own contribution. Worth one
run per backend if the full conversion is too slow.

### Optional: a spike budget

`mbe_pasn_s` can be asked for an energy ceiling instead of maximum accuracy:

```bash
python experiments/gpt2_wikitext.py --backend mbe_pasn_s $M --pasn-s-n-shared 2 4 --pasn-s-spike-budget 12
```

The budget is measured under the calibration distribution, so it is the spike count the
model actually spends.

## What to send back

The whole stdout is best, but at minimum these lines from each run:

```
ANN (gpt2-medium) perplexity = ...        [eval=..., ctx=..., stride=...]
[cost <backend>] activations=... activation spikes/input=...
[spikes <backend>] total=... -> ... spikes per input element
    layernorm   ...%
    activation  ...%
SNN (<backend>) perplexity = ...   (delta ...%)
```

Plus the wall time per run, and any `over budget` / `UNSATISFIABLE` lines the
`mbe_pasn_s` builder prints.

## Known caveats to carry into the results

1. Attention is exact FP (above).
2. `[cost ...]` is activations only; `[spikes ...]` is the whole-model total. Quote the
   second one for energy.
3. Our own MBE reimplementation is a *reference point*, not the paper's method — the paper
   reports +0.35 % and our MBE run may not reach that. Report both: our Δ% vs our ANN, and
   the paper's Δ% vs theirs.
4. `n_steps` is fixed at 16 everywhere, matching the paper's T=16 column.
