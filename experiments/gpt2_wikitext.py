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


@torch.no_grad()
def perplexity_sliding(model, ids, device, max_length, stride,
                       limit_windows=None, progress_every=10, label="eval"):
    """Canonical sliding-window perplexity (the HF/paper recipe).

    Each token is scored once with up to ``max_length`` tokens of left context;
    only the last ``stride`` tokens of each window contribute (the rest are
    context, masked with -100). Non-overlapping blocks over-estimate perplexity,
    which is why our earlier block eval gave 38 where the paper reports ~22.3 for
    GPT-2-medium.
    """
    model.eval()
    n = ids.numel()
    specs, prev_end = [], 0
    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        specs.append((begin, end, end - prev_end))
        prev_end = end
        if end == n:
            break
    if limit_windows:
        specs = specs[:limit_windows]
    total = len(specs)
    nll_sum, ntok, done = 0.0, 0, 0
    started = time.perf_counter()
    for begin, end, trg in specs:
        input_ids = ids[begin:end].unsqueeze(0).to(device)
        target = input_ids.clone()
        target[:, :-trg] = -100                    # score only the new tokens
        loss = model(input_ids, labels=target).loss
        nll_sum += float(loss) * trg
        ntok += trg
        done += 1
        if progress_every and (done == total or done % progress_every == 0):
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            print(f"  [{label}] win {done}/{total}  {rate:.2f} win/s  "
                  f"ETA {(total - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    return math.exp(nll_sum / max(ntok, 1))


def load_wikitext_ids(tokenizer, split="test", drop_blank: bool = False):
    """Tokenised WikiText split.

    Defaults to the canonical recipe, ``"\\n\\n".join(ds["text"])`` with blank lines
    **kept** -- the same text every published GPT-2 WikiText perplexity is measured on.
    Dropping blank lines (the previous behaviour, kept behind ``drop_blank``) changes the
    token stream and inflates perplexity, which is why our earlier GPT-2 baseline read
    34.52 where the literature reports the high 20s.
    """
    from datasets import load_dataset
    # Use the canonical namespaced repository ID. The legacy shorthand
    # ``"wikitext"`` is resolved by some datasets releases to an invalid
    # ``hf://datasets/wikitext@...`` URI; recent huggingface_hub parsers require
    # repository IDs in ``namespace/name`` form.
    ds = load_dataset(WIKITEXT_DATASET_ID, WIKITEXT_CONFIG, split=split)
    texts = (t for t in ds["text"] if t.strip()) if drop_blank else ds["text"]
    return tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]


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
    ap.add_argument("--backend",
                    choices=["none", "mbe", "mbe_pasn", "mbe_pasn_s", "pasn"],
                    default="none")
    # The paper's NLG table (Tab. 3) is GPT-2 346M = gpt2-medium at T=16:
    # ANN 22.34 ppl on WikiText-2, their conversion 22.69 (+0.35%). Compare there.
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--drop-blank-lines", action="store_true",
                    help="pre-fix behaviour: drop blank lines before joining the "
                         "WikiText split (changes the token stream and inflates ppl)")
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
    # Budget knobs for iso-storage / iso-spike fairness sweeps. MBE signed GELU
    # stores 2*n_basis_act bases; mbe_pasn stores ~(#binades)*pasn_n_local.
    ap.add_argument("--n-basis-act", type=int, default=6,
                    help="MBE bases per activation (signed GELU uses 2x this)")
    ap.add_argument("--pasn-n-local", type=int, default=2,
                    help="mbe_pasn: MBE bases per binade bank")
    ap.add_argument("--pasn-e-min", type=int, default=-3,
                    help="smallest binade exponent (near-zero resolution); "
                         "shared by mbe_pasn and pasn so only the encoder differs")
    ap.add_argument("--pasn-T", type=int, default=6,
                    help="pasn: SAR bits per bank (spike budget)")
    ap.add_argument("--pasn-order", type=int, default=2,
                    help="pasn: per-bank readout polynomial order")
    ap.add_argument("--pasn-s-n-shared", nargs="+", type=int, default=[4],
                    help="mbe_pasn_s: candidate shared-basis counts")
    ap.add_argument("--pasn-s-spike-budget", type=float, default=None,
                    help="mbe_pasn_s: cap on mean spikes/input, measured under the "
                         "calibration distribution")
    ap.add_argument("--pasn-s-restarts", type=int, default=3,
                    help="mbe_pasn_s: fits per candidate init")
    ap.add_argument("--eval-mode", choices=["sliding", "block"], default="sliding",
                    help="sliding = canonical HF window ppl (matches the paper); "
                         "block = fast non-overlapping (over-estimates ppl)")
    ap.add_argument("--max-length", type=int, default=0,
                    help="sliding-window context length (0 = model n_positions)")
    ap.add_argument("--stride", type=int, default=512,
                    help="sliding-window stride (tokens scored per window)")
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
        ids = load_wikitext_ids(tok, "test", args.drop_blank_lines)
        cids = load_wikitext_ids(tok, "train", args.drop_blank_lines)[: 64 * args.block]
        calib = [cids[i:i + args.block].unsqueeze(0)
                 for i in range(0, 8 * args.block, args.block)]
        block = args.block

    max_length = args.max_length or getattr(model.config, "n_positions", 1024)

    def eval_ppl(label):
        if args.eval_mode == "block":
            return perplexity(model, ids, block, device, args.limit_blocks,
                              batch_size=eval_batch_size,
                              progress_every=args.progress_every, label=label)
        return perplexity_sliding(model, ids, device, max_length, args.stride,
                                  limit_windows=args.limit_blocks,
                                  progress_every=args.progress_every, label=label)

    ppl_ann = eval_ppl("ANN")
    which = "smoke: tiny random GPT-2" if args.smoke else args.model
    print(f"ANN ({which}) perplexity = {ppl_ann:.4f}  "
          f"[eval={args.eval_mode}, ctx={max_length}, stride={args.stride}]")

    if args.backend != "none":
        n = make_spikable(model)
        print(f"marked {n} GELU activations; converting (backend={args.backend}) ...")
        cfg = cv.ConvertConfig(epochs=args.epochs, backend=args.backend,
                               spike_mult=True, n_basis_act=args.n_basis_act,
                               pasn_n_local=args.pasn_n_local,
                               pasn_e_min=args.pasn_e_min,
                               pasn_T=args.pasn_T,
                               pasn_order=args.pasn_order,
                               pasn_s_n_shared=(args.pasn_s_n_shared[0]
                                                if len(args.pasn_s_n_shared) == 1
                                                else args.pasn_s_n_shared),
                               pasn_s_spike_budget=args.pasn_s_spike_budget,
                               pasn_s_restarts=args.pasn_s_restarts,
                               fit_device=None if args.fit_device == "auto"
                               else args.fit_device,
                               verbose_fits=True)
        only = {
            "activation" if args.convert_ops == "activation" else "layernorm"
        } if args.convert_ops != "both" else None
        convert_gpt2(model, calib, cfg=cfg, only=only, verbose=True)
        # Fair-comparison cost metrics (MBE and PASN store different #bases, so
        # accuracy alone is not comparable): stored/active bases + spikes per input.
        sample_batch = ids[:block].unsqueeze(0)
        costs = cv.activation_cost_report(model, sample_batch)
        print(cv.format_cost_report(costs, label=args.backend), flush=True)
        # Whole-model spike cost. The line above is activations only; the FP-multiply
        # identities inside LayerNorm (and attention, once it is converted) are not
        # in it, so it is not an energy total.
        print(cv.format_spiking_cost_report(
            cv.spiking_cost_report(model, sample_batch), label=args.backend),
            flush=True)
        ppl_snn = eval_ppl(f"SNN-{args.backend}")
        drop = 100.0 * (ppl_snn - ppl_ann) / ppl_ann
        print(f"SNN ({args.backend}) perplexity = {ppl_snn:.4f}   "
              f"(delta {drop:+.2f}%)")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
