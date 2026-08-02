"""RoBERTa -> spiking conversion on SST-2 (the second task).

GPT-2 x WikiText-2 alone cannot support a claim about a *general* training-free
conversion neuron, and the paper reports CV, NLU and NLG. This is the cheapest
second task: 872 validation examples, public fine-tuned checkpoints, no license.

**Paper baseline (Table 2, SST-2).** RoBERTa-base 125M: ANN 94.49 -> 93.46 at
T=16. RoBERTa-large 355M: ANN 96.22 -> 95.98. As in Table 3 the prose calls the
gap a percentage ("only 0.24% degradation") when it is the *absolute* point
difference; on a 96%-accuracy task the two nearly coincide, but the relative
figures are what our numbers belong beside: **-1.09%** (base) and **-0.25%**
(large).

**The baseline mismatch is the real risk here, worse than it was for GPT-2.**
The paper follows SpikeZIP-TF's setup and does not publish its fine-tuning
recipe, so our ANN is a *different* fine-tune of the same architecture. With
perplexity a differing baseline was mostly a level shift; with accuracy, a weaker
or stronger starting model can genuinely change how much conversion costs. Report
our own ANN alongside every number and never quote the absolute accuracy against
theirs.

Attention is included (Stage 2) -- the same operator set the paper converts.

    python experiments/roberta_sst2.py --smoke                  # CPU wiring check
    python experiments/roberta_sst2.py --model textattack/roberta-base-SST-2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.hf_convert import convert_hf, make_attention_spikable, make_spikable  # noqa: E402
from mbe.metrics import neuron_params, storage_breakdown  # noqa: E402

_DEF = cv.ConvertConfig()

#: Table 2, SST-2 column. ``(ANN, theirs, T)`` -- accuracy in percent.
PAPER_SST2 = {"base": (94.49, 93.46, 16), "large": (96.22, 95.98, 16)}


@torch.no_grad()
def accuracy(model, batches, device, label="eval", progress_every=10) -> float:
    model.eval()
    right = total = 0
    started = time.perf_counter()
    for i, b in enumerate(batches, 1):
        b = {k: v.to(device) for k, v in b.items()}
        y = b.pop("labels")
        pred = model(**b).logits.argmax(-1)
        right += int((pred == y).sum())
        total += y.numel()
        if progress_every and (i == len(batches) or i % progress_every == 0):
            rate = i / max(time.perf_counter() - started, 1e-9)
            print(f"  [{label}] batch {i}/{len(batches)}  {rate:.2f} b/s  "
                  f"ETA {(len(batches) - i) / max(rate, 1e-9) / 60:.1f} min "
                  f"acc {100.0 * right / max(total, 1):.2f}", flush=True)
    return 100.0 * right / max(total, 1)


def load_sst2(tok, split="validation", batch_size=16, max_length=128, limit=None):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for i in range(0, len(ds), batch_size):
        chunk = ds[i:i + batch_size]
        enc = tok(chunk["sentence"], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(chunk["label"])
        out.append(enc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="textattack/roberta-base-SST-2")
    ap.add_argument("--backend", default="mbe_pasn",
                    choices=["none", "mbe_pasn", "mbe", "pasn", "mbe_pasn_s"])
    ap.add_argument("--convert-ops",
                    choices=["all", "both", "activation", "layernorm", "attention"],
                    default="all",
                    help="'all' = Stage 2, the operator set the paper converts")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random RoBERTa, no download -- wiring check only")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--calib-batches", type=int, default=4)
    ap.add_argument("--pasn-id-target", choices=["relative", "absolute"],
                    default=_DEF.pasn_id_target)
    ap.add_argument("--pasn-id-target-rel", type=float,
                    default=_DEF.pasn_id_target_rel)
    ap.add_argument("--pasn-t-fixed", type=int, default=None,
                    help="16 matches the paper's global T")
    ap.add_argument("--pasn-n-fixed", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        from transformers import RobertaConfig, RobertaForSequenceClassification
        cfg = RobertaConfig(num_hidden_layers=2, num_attention_heads=2,
                            hidden_size=32, intermediate_size=64,
                            max_position_embeddings=64, vocab_size=128,
                            num_labels=2)
        model = RobertaForSequenceClassification(cfg).eval()
        batches = [dict(input_ids=torch.randint(0, 128, (4, 16)),
                        attention_mask=torch.ones(4, 16, dtype=torch.long),
                        labels=torch.randint(0, 2, (4,))) for _ in range(3)]
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model, attn_implementation="eager").eval()
        batches = load_sst2(tok, "validation", a.batch_size, a.max_length, a.limit)
    model.config._attn_implementation = "eager"
    model.to(device)
    calib = [{k: v for k, v in b.items() if k != "labels"}
             for b in batches[:a.calib_batches]]

    rec = dict(tag=a.tag, backend=a.backend, scope=a.convert_ops, model=a.model,
               device=device, smoke=a.smoke, epochs=a.epochs,
               stage=(None if a.backend == "none"
                      else 2 if a.convert_ops in ("all", "attention") else 1),
               n_eval=sum(b["labels"].numel() for b in batches),
               started=time.strftime("%Y-%m-%dT%H:%M:%S"))

    acc_ann = None if a.build_only else accuracy(model, batches, device, "ANN")
    rec["acc_ann"] = acc_ann
    if acc_ann is not None:
        print(f"ANN ({a.model}) accuracy = {acc_ann:.2f}")

    if a.backend != "none":
        print(f"marked {make_spikable(model)} activations", flush=True)
        _KINDS = {"all": None, "both": {"activation", "layernorm"},
                  "activation": {"activation"}, "layernorm": {"layernorm"},
                  "attention": {"matmul", "softmax"}}
        if a.convert_ops in ("all", "attention"):
            print(f"marked {make_attention_spikable(model)} attention blocks; "
                  f"Stage 2", flush=True)
        cfg = cv.ConvertConfig(epochs=a.epochs, backend=a.backend, spike_mult=True,
                               pasn_id_target=a.pasn_id_target,
                               pasn_id_target_rel=a.pasn_id_target_rel,
                               pasn_t_fixed=a.pasn_t_fixed,
                               pasn_n_fixed=a.pasn_n_fixed,
                               verbose_fits=True)
        t0 = time.perf_counter()
        convert_hf(model, calib, cfg=cfg, only=_KINDS[a.convert_ops], verbose=True)
        rec["build_s"] = time.perf_counter() - t0
        print(f"[build] conversion took {rec['build_s'] / 60:.1f} min")

        spikes = cv.spiking_cost_report(model, calib[0])
        print(cv.format_spiking_cost_report(spikes, label=a.backend), flush=True)
        prims = cv._spiking_primitives(model)
        store = storage_breakdown([n for _, _, n in prims])
        print(f"[store {a.backend}] params="
              f"{sum(neuron_params(n) for _, _, n in prims)}  "
              f"bytes={store['bytes']}  primitives={len(prims)}", flush=True)
        rec.update(spikes_per_input=spikes["spikes_per_input"],
                   by_kind=spikes["by_kind"],
                   ops_per_input=spikes.get("ops_per_input"),
                   energy_pj_per_input=spikes.get("energy_pj_per_input"),
                   stored_params=sum(neuron_params(n) for _, _, n in prims),
                   stored_bytes=store["bytes"], n_primitives=len(prims))

        if not a.build_only:
            acc_snn = accuracy(model, batches, device, f"SNN-{a.backend}")
            drop_pp = acc_snn - acc_ann
            rec.update(acc_snn=acc_snn, delta_pp=drop_pp,
                       delta_pct=100.0 * drop_pp / acc_ann)
            print(f"SNN ({a.backend}) accuracy = {acc_snn:.2f}   "
                  f"({drop_pp:+.2f} pp, {rec['delta_pct']:+.2f}% relative)")
            size = "large" if "large" in a.model else "base"
            p_ann, p_snn, p_T = PAPER_SST2[size]
            print(f"  paper (RoBERTa-{size}): {p_ann} -> {p_snn} at T={p_T}  "
                  f"= {100.0 * (p_snn - p_ann) / p_ann:+.2f}% relative")
            print("  compare the RELATIVE columns only -- our ANN is a different "
                  "fine-tune of the same architecture")

    if a.json:
        prev = []
        if os.path.exists(a.json):
            with open(a.json, encoding="utf-8") as fh:
                prev = json.load(fh)
        prev.append(rec)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(prev, fh, indent=1)
        print(f"[json] wrote record {len(prev)} to {a.json}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
