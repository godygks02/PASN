# Phase 4d — GPT-2 × WikiText-2 on vast.ai

Runs the PASN/MBE conversion on a real pretrained GPT-2 and reports WikiText-2
perplexity vs the ANN. Local CPU can only smoke-test the wiring (no pretrained
download / no GPU); the real numbers come from here.

## What is converted (Stage 1)
MLP GELU + every LayerNorm → spiking (MBE or PASN). Attention (QK^T / softmax /
attn·V) is left exact — it is functional inside `eager_attention_forward` and is
wired in Stage 2. Rationale: GELU is the dominant conversion error (Phase-4 toy
study) and PASN's biggest win, so this isolates the highest-value swap first.

## Setup on the instance
```bash
git clone <this-repo> pasn && cd pasn
pip install -r requirements-vastai.txt        # torch(CUDA) assumed from the image
```

## Run
```bash
# baseline ANN perplexity (sanity)
python experiments/gpt2_wikitext.py --backend none --model gpt2

# Same Stage-1 scope, two independent runs:
#   MBE run  = GELU(MBE)  + LayerNorm primitives(MBE)
#   PASN run = GELU(PASN) + LayerNorm primitives(PASN)
python experiments/gpt2_wikitext.py --backend mbe  --model gpt2 --epochs 300 \
  --fit-device cuda --convert-ops both --eval-batch-size 4
python experiments/gpt2_wikitext.py --backend pasn --model gpt2 --epochs 300 \
  --fit-device cuda --convert-ops both --eval-batch-size 4
```
Conversion fitting defaults to the model device, so on a CUDA instance the PASN
banks and MBE primitives are optimized on the GPU. The script prints each module
and PASN bank as it is fitted. Use `--fit-device cpu` only for debugging; explicit
`--fit-device cuda` is equivalent to the default on a GPU instance.
Repeated GPT-2 blocks share calibration work: one PASN GELU is fitted on the union
of all 12 observed GELU ranges, and LayerNorm primitives use the combined retained
calibration rows and the fitting cache rather than being optimized from scratch
for every block.

Full spiking LayerNorm evaluation is expensive. For the first PASN-vs-MBE
comparison, isolate the operation where the backends differ and batch four
WikiText blocks on the GPU:

```bash
python experiments/gpt2_wikitext.py --backend pasn --model gpt2 --epochs 300 \
  --convert-ops activation --eval-batch-size 4
```

The evaluator prints processed blocks, throughput, and ETA. Use
`--limit-blocks 10` for a quick end-to-end check before a full run. Increase
`--eval-batch-size` until GPU memory is well utilized; reduce it after an OOM.

The script loads WikiText-2 from its canonical Hugging Face repository ID,
`Salesforce/wikitext`. This avoids the invalid unnamespaced `hf://datasets/wikitext`
URI produced by recent `datasets` / `huggingface_hub` combinations when the legacy
`"wikitext"` shorthand is used.

`--limit-blocks N` gives a quick partial pass; omit for the full test set.
`--model gpt2-medium` (345M) for the paper's NLG variant once gpt2 checks out.

## Wiring check (no GPU / no download)
```bash
python experiments/gpt2_wikitext.py --smoke --backend pasn
```
Confirmed on CPU: ANN → SNN perplexity delta ≈ 0% on a tiny random GPT-2 for both
backends (conversion is near-lossless; the tiny random model has near-zero GELU/LN
input ranges — real ranges are wider, which is where the MBE-vs-PASN gap appears).

## Sync back
Copy the printed perplexities (ANN / MBE / PASN + delta%) into
`results/gpt2_wikitext.md` for analysis. Target: PASN delta ≤ MBE delta at equal
or fewer activation spikes.
