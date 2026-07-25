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
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import make_spikable, convert_gpt2  # noqa: E402


WIKITEXT_DATASET_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"


@torch.no_grad()
def perplexity(model, ids, block, device, limit_blocks=None, batch_size=1,
               progress_every=10, label="eval"):
    model.eval()
    nll, ntok, nb = 0.0, 0, 0
    chunks = []
    for i in range(0, ids.numel() - 1, block):
        chunk = ids[i:i + block + 1]
        if chunk.numel() == block + 1:
            chunks.append(chunk)
        if limit_blocks and len(chunks) >= limit_blocks:
            break
    total = len(chunks)
    started = time.perf_counter()
    for start in range(0, total, batch_size):
        group = chunks[start:start + batch_size]
        inp = torch.stack([chunk[:-1] for chunk in group]).to(device)
        tgt = torch.stack([chunk[1:] for chunk in group]).to(device)
        logits = model(inp).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="sum")
        nll += float(loss)
        ntok += tgt.numel()
        nb += len(group)
        if (
            progress_every
            and (nb == total or nb % progress_every < len(group))
        ):
            elapsed = time.perf_counter() - started
            rate = nb / max(elapsed, 1e-9)
            eta = (total - nb) / max(rate, 1e-9)
            print(
                f"  [{label}] blocks {nb}/{total}  "
                f"{rate:.2f} blocks/s  ETA {eta / 60:.1f} min",
                flush=True,
            )
    return math.exp(nll / max(ntok, 1))


def load_wikitext_ids(tokenizer, split="test"):
    from datasets import load_dataset
    # Use the canonical namespaced repository ID. The legacy shorthand
    # ``"wikitext"`` is resolved by some datasets releases to an invalid
    # ``hf://datasets/wikitext@...`` URI; recent huggingface_hub parsers require
    # repository IDs in ``namespace/name`` form.
    ds = load_dataset(WIKITEXT_DATASET_ID, WIKITEXT_CONFIG, split=split)
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
    ap.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="WikiText blocks evaluated together (default: 4 on CUDA, 1 on CPU)",
    )
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument(
        "--convert-ops", choices=["both", "activation", "layernorm"],
        default="both",
        help="conversion scope; activation isolates the PASN-vs-MBE contribution",
    )
    ap.add_argument(
        "--fit-device", choices=["auto", "cpu", "cuda"], default="auto",
        help="device for MBE/PASN calibration fitting (default: model device)",
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_batch_size = args.eval_batch_size or (4 if device == "cuda" else 1)

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

    ppl_ann = perplexity(
        model, ids, block, device, args.limit_blocks,
        batch_size=eval_batch_size, progress_every=args.progress_every,
        label="ANN",
    )
    print(f"ANN ({args.model}) perplexity = {ppl_ann:.4f}")

    if args.backend != "none":
        n = make_spikable(model)
        print(f"marked {n} GELU activations; converting (backend={args.backend}) ...")
        cfg = cv.ConvertConfig(epochs=args.epochs, backend=args.backend,
                               spike_mult=True, pasn_n_local=2, pasn_e_min=-3,
                               fit_device=None if args.fit_device == "auto"
                               else args.fit_device,
                               verbose_fits=True)
        only = {
            "activation" if args.convert_ops == "activation" else "layernorm"
        } if args.convert_ops != "both" else None
        convert_gpt2(model, calib, cfg=cfg, only=only, verbose=True)
        ppl_snn = perplexity(
            model, ids, block, device, args.limit_blocks,
            batch_size=eval_batch_size, progress_every=args.progress_every,
            label=f"SNN-{args.backend}",
        )
        drop = 100.0 * (ppl_snn - ppl_ann) / ppl_ann
        print(f"SNN ({args.backend}) perplexity = {ppl_snn:.4f}   "
              f"(delta {drop:+.2f}%)")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
