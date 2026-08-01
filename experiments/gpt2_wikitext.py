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
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import (make_spikable, make_attention_spikable,  # noqa: E402
                              convert_gpt2)
from mbe.metrics import neuron_params, storage_breakdown  # noqa: E402


WIKITEXT_DATASET_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"

# ``ConvertConfig`` is the authority on every method decision (audit_defaults.py
# checks it, not this file). Pull the CLI defaults from it so a flag left unset
# means "whatever the audited default is" rather than a second copy that can
# drift away from it.
_DEF = cv.ConvertConfig()


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
    context, masked with -100).

    ``stride == max_length`` (the default) is the non-overlapping recipe the
    published GPT-2 numbers use, and it is what reproduces them: 21.71 for
    gpt2-medium against Table 3's 22.76. Overlap is *not* a correction to apply on
    top -- shortening the stride hands every token more left context and lowers the
    number without bound (512 -> 18.46, 256 -> 18.08), so it measures an easier
    quantity than the tables do. What over-estimated our earlier 38 was the ``block``
    evaluator at ``block=512``, which caps context at 512 *and* scores the first
    tokens of every block with almost none.
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


def storage_report(model) -> dict:
    """Stored cost of the converted primitives, on two axes.

    ``params`` is ``neuron_params`` summed over the *unique* spiking primitives --
    the same quantity the toy waterfall tables report, so a GPT-2 row is directly
    comparable to them. ``bytes`` is the real stored footprint (parameters **and**
    buffers): ``learn_tau=False`` moves the taus into buffers, which makes the
    learnable count fall without anything actually being freed, so any memory
    claim has to be made on this number instead.

    Both axes de-duplicate by object -- see :func:`~mbe.metrics.storage_bytes`.
    The de-duplication is taken over the whole primitive list at once, not per
    neuron: ``share_fits=True`` lets 24 layers reuse one fitted primitive, and the
    model stores that once. ``bytes_naive`` is the pre-P0.5 sum, which charged
    every shared tensor once per bank referencing it and so inflated the routed
    backend alone; it is kept only to keep the P0.4 rows readable.
    """
    prims = cv._spiking_primitives(model)
    neurons = [n for _, _, n in prims]
    br = storage_breakdown(neurons)
    return dict(params=sum(neuron_params(n) for n in neurons),
                bytes=br["bytes"], bytes_naive=br["bytes_naive"],
                shared_factor=br["shared_factor"], by_role=br["by_role"],
                primitives=len(prims))


def write_record(path: str, rec: dict):
    """Append one run to a JSON array, creating it on first use.

    Sequential runs only -- the sweep is a shell loop over GPU evals, so a
    read-modify-write is enough. If you ever run two of these at once, give each
    its own ``--json`` path and merge afterwards.
    """
    rows = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                rows = [rows]
        except (json.JSONDecodeError, OSError) as e:
            backup = f"{path}.bad"
            print(f"[json] {path} is unreadable ({e}); moving it to {backup}")
            os.replace(path, backup)
            rows = []
    rows.append(rec)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"[json] wrote record {len(rows)} to {path}")


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
    ap.add_argument("--build-only", action="store_true",
                    help="build + report storage and spikes, skip both perplexity "
                         "evals. P0.5 re-measures bytes on rows whose ppl is "
                         "already known; the eval is the expensive half and the "
                         "conversion is deterministic, so it need not be repeated")
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
        "--convert-ops",
        choices=["all", "both", "activation", "layernorm", "attention"],
        default="both",
        help="conversion scope. 'all' = Stage 2, the whole network including "
             "attention -- the ONLY scope comparable to the paper's Table 3. "
             "'both' = Stage 1 (GELU + LayerNorm, attention left exact FP), kept "
             "as the default so the recorded P0.4 rows keep their meaning. The "
             "single-op scopes isolate one contribution",
    )
    ap.add_argument(
        "--fit-device", choices=["auto", "cpu", "cuda"], default="auto",
        help="device for MBE/PASN calibration fitting (default: model device)",
    )
    # Budget knobs for iso-storage / iso-spike fairness sweeps. MBE signed GELU
    # stores 2*n_basis_act bases; mbe_pasn stores ~(#binades)*pasn_n_local.
    ap.add_argument("--n-basis-act", type=int, default=_DEF.n_basis_act,
                    help="MBE bases per activation (signed GELU uses 2x this)")
    ap.add_argument("--n-basis-ln", type=int, default=_DEF.n_basis_ln,
                    help="MBE bases per LayerNorm primitive; sweep together with "
                         "--n-basis-act to give the mbe baseline its own frontier")
    ap.add_argument("--pasn-n-local", type=int, default=_DEF.pasn_n_local,
                    help="mbe_pasn: MBE bases per binade bank")
    ap.add_argument("--pasn-e-min", type=int, default=_DEF.pasn_e_min,
                    help="smallest binade exponent (near-zero resolution); "
                         "shared by mbe_pasn and pasn so only the encoder differs")

    # --- P0.4 decision knobs (§3 of P0.4_GPT2_HANDOFF.md) -------------------
    # These exist in ConvertConfig but were not reachable from the command line,
    # so the sweep that settles them could not be run.
    ap.add_argument("--pasn-id-target", choices=["relative", "absolute"],
                    default=_DEF.pasn_id_target,
                    help="budget for the routed identity inside the spike-driven FP "
                         "multiply. THE P0.4 decision: exp 10 measured 15-49%% "
                         "relative error on a real product under an absolute budget, "
                         "while P0.3 measured relative costing 0.81x spikes at "
                         "iso-forward-error. Only dPPL settles it -- run both")
    ap.add_argument("--pasn-id-target-rel", type=float,
                    default=_DEF.pasn_id_target_rel,
                    help="operating point r for --pasn-id-target=relative (ignored "
                         "when absolute -- sweeping it there gives identical configs)")
    ap.add_argument("--pasn-id-tied", action=argparse.BooleanOptionalAction,
                    default=_DEF.pasn_id_tied,
                    help="tie the identity's magnitude banks to one prototype (exp 4). "
                         "Required off to compare --pasn-id-target: a tied prototype is "
                         "fitted on the unit residual, so its budget is relative by "
                         "construction and 'absolute' is not expressible")
    ap.add_argument("--pasn-id-e-min", type=int, default=_DEF.pasn_id_e_min,
                    help="router depth for the identity primitives only; their "
                         "operands span many decades. -6 on the toy, unverified on "
                         "GPT-2's activation distribution")
    ap.add_argument("--pasn-readout-order", type=int,
                    default=_DEF.pasn_readout_order,
                    help="mbe_pasn activation decoder order (2 = exp 6; free "
                         "accuracy at zero extra spikes)")

    # --- matched-baseline knobs --------------------------------------------
    # An unmatched mbe baseline is what produced the dead 14.2x. Give it the same
    # advantages: readout order 2 and a log-uniform identity calibration draw.
    ap.add_argument("--mbe-readout-order", type=int,
                    default=_DEF.mbe_readout_order,
                    help="global-MBE decoder order. WARNING: reaches LayerNorm only. "
                         "GELU takes the polarity-split SignedMBENeuron, which has no "
                         "order-2 readout, so the activation stays order-1 however this "
                         "is set -- a matched baseline is matched only on LayerNorm")
    ap.add_argument("--mbe-id-logsample", action="store_true",
                    default=_DEF.mbe_id_logsample,
                    help="draw the global identity's calibration log-uniformly, the "
                         "fairest global analogue of the routed relative budget")
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
    ap.add_argument("--stride", type=int, default=1024,
                    help="sliding-window stride (tokens scored per window). 1024 = "
                         "non-overlapping, the published recipe: gpt2-medium 21.71 and "
                         "gpt2 29.94 vs Table 3's 22.76 / 29.41. stride 512 was the old "
                         "default and reads 18.46, 17%% low -- more left context per "
                         "token is an easier quantity, not the paper's")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="append this run to a JSON array (results/gpt2_p04.json "
                         "is the P0.4 deliverable). Sequential runs only")
    ap.add_argument("--tag", default=None,
                    help="free-form label stored with the JSON record")
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

    which = "smoke: tiny random GPT-2" if args.smoke else args.model
    if args.build_only:
        # Storage and spike counts come off the built model; only perplexity needs
        # the sweep over windows. P0.5 re-measures bytes for rows whose ppl is
        # already recorded, and paying 1-2 h of eval to reprint it would be waste.
        ppl_ann, eval_ann_s = None, None
        print(f"[build-only] skipping perplexity for {which}")
    else:
        t0 = time.perf_counter()
        ppl_ann = eval_ppl("ANN")
        eval_ann_s = time.perf_counter() - t0
        print(f"ANN ({which}) perplexity = {ppl_ann:.4f}  "
              f"[eval={args.eval_mode}, ctx={max_length}, stride={args.stride}]")

    # Record only the knobs that reach this backend. ``_build_one`` branches on
    # ``backend in _ROUTED_BACKENDS``, so the pasn_* family is dead for a global mbe
    # run and the mbe_*/n_basis_* family is dead for a routed one. Storing all of
    # them anyway puts a column in the deliverable table that varies without
    # anything changing -- the shape of trap 6, one level up in the analysis.
    routed = args.backend in cv._ROUTED_BACKENDS
    converting = args.backend != "none"
    # The identity's operating point likewise only exists under a relative budget.
    rel = (args.pasn_id_target_rel
           if routed and args.pasn_id_target == "relative" else None)

    def rt(v):                       # routed-only knob
        return v if routed else None

    def gl(v):                       # global-MBE-only knob
        return v if converting and not routed else None

    rec = dict(
        tag=args.tag, backend=args.backend, scope=args.convert_ops,
        # 1 = GELU + LayerNorm only (attention exact FP); 2 = whole network.
        # Only stage 2 may be placed next to the paper's Table 3.
        stage=(None if not converting
               else 2 if args.convert_ops in ("all", "attention") else 1),
        model=which, smoke=args.smoke, device=device,
        id_target=rt(args.pasn_id_target), r=rel,
        pasn_id_tied=rt(args.pasn_id_tied),
        pasn_e_min=rt(args.pasn_e_min), pasn_id_e_min=rt(args.pasn_id_e_min),
        pasn_readout_order=rt(args.pasn_readout_order),
        pasn_n_local=rt(args.pasn_n_local),
        mbe_readout_order=gl(args.mbe_readout_order),
        mbe_id_logsample=gl(args.mbe_id_logsample),
        n_basis_act=gl(args.n_basis_act), n_basis_ln=gl(args.n_basis_ln),
        epochs=args.epochs if converting else None, n_steps=_DEF.n_steps,
        eval_mode=args.eval_mode, ctx=max_length, stride=args.stride,
        limit_blocks=args.limit_blocks, eval_batch_size=eval_batch_size,
        ppl_ann=ppl_ann, ppl_snn=None, delta_pct=None,
        spikes_per_token=None, total_spikes=None, by_kind=None,
        stored_params=None, stored_bytes=None, n_primitives=None,
        n_activations=None, act_spikes_per_input=None,
        build_s=None, eval_ann_s=eval_ann_s, eval_snn_s=None,
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    if args.backend != "none":
        n = make_spikable(model)
        print(f"marked {n} GELU activations; converting (backend={args.backend}) ...")
        cfg = cv.ConvertConfig(epochs=args.epochs, backend=args.backend,
                               spike_mult=True, n_basis_act=args.n_basis_act,
                               n_basis_ln=args.n_basis_ln,
                               pasn_n_local=args.pasn_n_local,
                               pasn_e_min=args.pasn_e_min,
                               pasn_id_e_min=args.pasn_id_e_min,
                               pasn_id_target=args.pasn_id_target,
                               pasn_id_target_rel=args.pasn_id_target_rel,
                               pasn_id_tied=args.pasn_id_tied,
                               pasn_readout_order=args.pasn_readout_order,
                               mbe_readout_order=args.mbe_readout_order,
                               mbe_id_logsample=args.mbe_id_logsample,
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
        # Scope -> primitive kinds. "both" stays Stage 1 (GELU + LayerNorm) so the
        # recorded P0.4 rows keep their meaning; "all" is Stage 2, the whole
        # network, and is the only scope comparable to the paper's Table 3.
        _SCOPE_KINDS = {
            "all": None,                                  # every kind
            "both": {"activation", "layernorm"},
            "activation": {"activation"},
            "layernorm": {"layernorm"},
            "attention": {"matmul", "softmax"},
        }
        only = _SCOPE_KINDS[args.convert_ops]
        if args.convert_ops in ("all", "attention"):
            n_attn = make_attention_spikable(model)
            print(f"marked {n_attn} attention blocks "
                  f"(QK^T / softmax / attn*V); Stage 2")
        t0 = time.perf_counter()
        convert_gpt2(model, calib, cfg=cfg, only=only, verbose=True)
        rec["build_s"] = time.perf_counter() - t0
        print(f"[build] conversion took {rec['build_s'] / 60:.1f} min")
        # Fair-comparison cost metrics (MBE and PASN store different #bases, so
        # accuracy alone is not comparable): stored/active bases + spikes per input.
        sample_batch = ids[:block].unsqueeze(0)
        costs = cv.activation_cost_report(model, sample_batch)
        print(cv.format_cost_report(costs, label=args.backend), flush=True)
        # Whole-model spike cost. The line above is activations only; the FP-multiply
        # identities inside LayerNorm (and attention, once it is converted) are not
        # in it, so it is not an energy total.
        spikes = cv.spiking_cost_report(model, sample_batch)
        print(cv.format_spiking_cost_report(spikes, label=args.backend), flush=True)
        store = storage_report(model)
        print(f"[store {args.backend}] params={store['params']}  "
              f"bytes={store['bytes']}  primitives={store['primitives']}  "
              f"(naive={store['bytes_naive']}, "
              f"shared {store['shared_factor']:.2f}x)  "
              f"roles={ {k: v for k, v in sorted(store['by_role'].items())} }",
              flush=True)
        rec.update(spikes_per_token=spikes["spikes_per_input"],
                   total_spikes=spikes["total_spikes"],
                   by_kind=spikes["by_kind"],
                   stored_params=store["params"], stored_bytes=store["bytes"],
                   stored_bytes_naive=store["bytes_naive"],
                   stored_shared_factor=store["shared_factor"],
                   stored_by_role=store["by_role"],
                   n_primitives=store["primitives"], n_activations=len(costs),
                   act_spikes_per_input=(sum(c["spikes"] for c in costs.values())
                                         / len(costs) if costs else None))

        if not args.build_only:
            t0 = time.perf_counter()
            ppl_snn = eval_ppl(f"SNN-{args.backend}")
            rec["eval_snn_s"] = time.perf_counter() - t0
            drop = 100.0 * (ppl_snn - ppl_ann) / ppl_ann
            rec.update(ppl_snn=ppl_snn, delta_pct=drop)
            print(f"SNN ({args.backend}) perplexity = {ppl_snn:.4f}   "
                  f"(delta {drop:+.2f}%)")

    if args.json:
        write_record(args.json, rec)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
