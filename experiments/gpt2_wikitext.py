"""GPT-2 -> spiking conversion on WikiText-2 (Phase 4d).

Converts GPT-2's MLP-GELU + LayerNorm to spiking (MBE or PASN) and reports
WikiText-2 perplexity vs the unconverted ANN. Attention is left exact (Stage 1;
see mbe/gpt2_convert.py). Designed to run on vast.ai (GPU); a ``--smoke`` mode
builds a tiny random GPT-2 and skips the download so the wiring can be checked on
CPU.

Usage:
  python experiments/gpt2_wikitext.py --smoke                 # CPU wiring check
  python experiments/gpt2_wikitext.py --backend pasn          # real run (vast.ai)
  python experiments/gpt2_wikitext.py --backend mbe --model gpt2
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import make_spikable, convert_gpt2  # noqa: E402


@torch.no_grad()
def perplexity(model, ids, block, device, limit_blocks=None):
    model.eval()
    nll, ntok, nb = 0.0, 0, 0
    for i in range(0, ids.numel() - 1, block):
        chunk = ids[i:i + block + 1]
        if chunk.numel() < 2:
            break
        inp = chunk[:-1].unsqueeze(0).to(device)
        tgt = chunk[1:].unsqueeze(0).to(device)
        logits = model(inp).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="sum")
        nll += float(loss); ntok += tgt.numel(); nb += 1
        if limit_blocks and nb >= limit_blocks:
            break
    return math.exp(nll / max(ntok, 1))


def load_wikitext_ids(tokenizer, split="test"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return tokenizer(text, return_tensors="pt").input_ids[0]


def build_smoke():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=64, vocab_size=128)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2000,))
    calib = [torch.randint(0, cfg.vocab_size, (2, 32)) for _ in range(2)]
    return model, ids, calib, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["none", "mbe", "pasn"], default="none")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--limit-blocks", type=int, default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        model, ids, calib, block = build_smoke()
        args.block = block
        print("[smoke] tiny random GPT-2 (2 layers, d=32) -- wiring check only")
    else:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
        ids = load_wikitext_ids(tok, "test")
        cids = load_wikitext_ids(tok, "train")[: 64 * args.block]
        calib = [cids[i:i + args.block].unsqueeze(0)
                 for i in range(0, 8 * args.block, args.block)]
        block = args.block

    ppl_ann = perplexity(model, ids, block, device, args.limit_blocks)
    print(f"ANN ({args.model}) perplexity = {ppl_ann:.4f}")

    if args.backend != "none":
        n = make_spikable(model)
        print(f"marked {n} GELU activations; converting (backend={args.backend}) ...")
        cfg = cv.ConvertConfig(epochs=args.epochs, backend=args.backend,
                               spike_mult=True, pasn_n_local=2, pasn_e_min=-3)
        convert_gpt2(model, calib, cfg=cfg, verbose=args.smoke)
        ppl_snn = perplexity(model, ids, block, device, args.limit_blocks)
        drop = 100.0 * (ppl_snn - ppl_ann) / ppl_ann
        print(f"SNN ({args.backend}) perplexity = {ppl_snn:.4f}   "
              f"(delta {drop:+.2f}%)")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
