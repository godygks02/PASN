"""ViT -> spiking conversion on ImageNet-1k (the CV axis, E8).

The paper covers CV, NLU and NLG; we had the latter two. This is the third.

**Paper baseline (Table 1, all at T=16).** ViT-B/16 86M: ANN 83.44 -> 83.00.
ViT-M/16 64M: ANN 85.95 -> 85.31. As in Tables 2 and 3 the prose calls the gap a
percentage ("conversion losses of 0.44% and 0.64%") when it is the *absolute*
point difference; relative they are **-0.53%** and **-0.74%**, and those are the
figures ours belong beside.

⚠️ **We cannot use the paper's checkpoints.** It names architectures, not weights,
and ViT-M/16 is "ViT-Medium-Patch16-Reg4-Gap-256" -- a **timm** model. timm's ViT
computes attention inline in ``timm.layers.attention.Attention`` and uses
``torch.nn.GELU``, so ``make_spikable`` and ``make_attention_spikable`` both mark
**zero** modules on it: the conversion would silently be a no-op and the "SNN"
would be the ANN. Verified 2026-08-11. This script therefore uses **HF
``transformers`` ViT checkpoints**, whose two-registry attention dispatch and
``GELUActivation`` our converter does handle, and whose absolute top-1 differs
from the paper's. **Only the relative conversion loss is comparable** -- the same
discipline the RoBERTa adapter states for its third-party fine-tunes.

**Data.** The ImageNet-1k validation parquet shards are read directly with
pyarrow and decoded per batch. ``datasets.load_dataset`` is deliberately not used:
it re-encodes into arrow, which for 50k images filled a 100 GB disk once already,
and ``load_dataset(repo, split="validation")`` fetches the whole repository (155
GiB) regardless of the split argument.

    python experiments/vit_imagenet.py --smoke                    # CPU wiring check
    python experiments/vit_imagenet.py --backend none --limit 2000   # ANN gate
    python experiments/vit_imagenet.py --model google/vit-base-patch16-224
"""
from __future__ import annotations

import argparse
import glob
import io
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

#: Paper Table 1, at T=16: ``label -> (ANN top-1, theirs)``. The relative loss is
#: derived, never the printed "0.44%" -- that is the absolute point difference.
PAPER_TABLE1 = {"ViT-B/16": (83.44, 83.00), "ViT-M/16": (85.95, 85.31)}

#: Where the validation shards live once fetched with
#: ``snapshot_download("imagenet-1k", repo_type="dataset",
#:                     allow_patterns=["data/validation-*"])``.
VAL_GLOB = ("/workspace/.hf_home/hub/datasets--imagenet-1k/snapshots/*/"
            "data/validation-*.parquet")


class ImageNetVal:
    """Lazy batches of preprocessed validation images.

    Materialising 50,000 preprocessed tensors would be ~30 GB, so batches are
    produced on demand and only the current one is held. ``__len__`` is the batch
    count, so the progress printer still knows the total.
    """

    def __init__(self, files, processor, batch_size=32, limit=None):
        import pyarrow.parquet as pq
        self.pq = pq
        self.files = files
        self.processor = processor
        self.batch_size = batch_size
        counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
        self.n = sum(counts) if limit is None else min(limit, sum(counts))

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def _rows(self):
        seen = 0
        for f in self.files:
            pf = self.pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["image", "label"])
                for rec in tbl.to_pylist():
                    if seen >= self.n:
                        return
                    seen += 1
                    yield rec

    def __iter__(self):
        from PIL import Image
        batch = []
        for rec in self._rows():
            im = Image.open(io.BytesIO(rec["image"]["bytes"]))
            # A handful of ImageNet validation files are greyscale or CMYK.
            batch.append((im.convert("RGB"), rec["label"]))
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    def _collate(self, batch):
        images = [im for im, _ in batch]
        enc = self.processor(images=images, return_tensors="pt")
        enc["labels"] = torch.tensor([lab for _, lab in batch])
        return dict(enc)


@torch.no_grad()
def top1(model, batches, device, label="eval", progress_every=20) -> float:
    model.eval()
    right = total = 0
    started = time.perf_counter()
    n_batches = len(batches)
    for i, b in enumerate(batches, 1):
        b = {k: v.to(device) for k, v in b.items()}
        y = b.pop("labels")
        pred = model(**b).logits.argmax(-1)
        right += int((pred == y).sum())
        total += y.numel()
        if progress_every and (i == n_batches or i % progress_every == 0):
            rate = i / max(time.perf_counter() - started, 1e-9)
            print(f"  [{label}] batch {i}/{n_batches}  {rate:.2f} b/s  "
                  f"ETA {(n_batches - i) / max(rate, 1e-9) / 60:.1f} min  "
                  f"top-1 {100.0 * right / max(total, 1):.2f}", flush=True)
    return 100.0 * right / max(total, 1)


def build_smoke():
    """Tiny random ViT and random images -- wiring only, seconds on CPU."""
    from transformers import ViTConfig, ViTForImageClassification
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, image_size=32, patch_size=16,
                    num_labels=10)
    model = ViTForImageClassification(cfg).eval()
    batches = [dict(pixel_values=torch.randn(4, 3, 32, 32),
                    labels=torch.randint(0, 10, (4,))) for _ in range(3)]
    return model, batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/vit-base-patch16-224",
                    help="HF ViT checkpoint. ⚠ timm models are NOT usable: their "
                         "attention is inline and their activation is nn.GELU, so "
                         "both markers mark zero modules and the conversion is a "
                         "silent no-op")
    ap.add_argument("--backend", default="mbe_pasn",
                    choices=["none", "mbe", "mbe_pasn", "mbe_pasn_s", "pasn"])
    ap.add_argument("--convert-ops", default="all",
                    choices=["all", "both", "activation", "layernorm", "attention"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random ViT, no download")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N images (smoke / gate runs)")
    ap.add_argument("--calib-batches", type=int, default=4)
    ap.add_argument("--val-glob", default=VAL_GLOB)
    ap.add_argument("--paper-row", default=None, choices=sorted(PAPER_TABLE1),
                    help="print the paper's row beside ours (relative only)")
    ap.add_argument("--pasn-id-target", choices=["relative", "absolute"],
                    default=_DEF.pasn_id_target)
    ap.add_argument("--pasn-id-target-rel", type=float,
                    default=_DEF.pasn_id_target_rel)
    ap.add_argument("--pasn-t-fixed", type=int, default=None,
                    help="force the paper's global T=16")
    ap.add_argument("--pasn-n-fixed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=_DEF.seed)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        model, batches = build_smoke()
        n_eval = sum(b["labels"].numel() for b in batches)
    else:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        files = sorted(glob.glob(a.val_glob))
        if not files:
            raise SystemExit(
                f"no validation shards at {a.val_glob}. Fetch them with "
                f"snapshot_download('imagenet-1k', repo_type='dataset', "
                f"allow_patterns=['data/validation-*'])  -- and NOT with "
                f"load_dataset, which pulls the whole 155 GiB repository.")
        processor = AutoImageProcessor.from_pretrained(a.model)
        model = AutoModelForImageClassification.from_pretrained(
            a.model, attn_implementation="eager").eval()
        batches = ImageNetVal(files, processor, a.batch_size, a.limit)
        n_eval = batches.n
        print(f"[data] {len(files)} shards, evaluating {n_eval} images "
              f"in {len(batches)} batches of {a.batch_size}", flush=True)
    model.config._attn_implementation = "eager"
    model.to(device)

    calib = []
    for i, b in enumerate(batches):
        if i >= a.calib_batches:
            break
        calib.append({k: v for k, v in b.items() if k != "labels"})

    rec = dict(tag=a.tag, task="imagenet-1k", model=a.model, seed=a.seed,
               backend=a.backend, scope=a.convert_ops, device=device,
               smoke=a.smoke, epochs=a.epochs, batch_size=a.batch_size,
               limit=a.limit, n_eval=n_eval, calib_batches=a.calib_batches,
               stage=(None if a.backend == "none"
                      else 2 if a.convert_ops in ("all", "attention") else 1),
               started=time.strftime("%Y-%m-%dT%H:%M:%S"))

    acc_ann = None if a.build_only else top1(model, batches, device, "ANN")
    rec["acc_ann"] = acc_ann
    if acc_ann is not None:
        print(f"ANN ({a.model}) top-1 = {acc_ann:.2f}")

    if a.backend != "none":
        n_act = make_spikable(model)
        print(f"marked {n_act} activations", flush=True)
        if n_act == 0:
            raise SystemExit(
                "make_spikable marked 0 modules -- this is a silent no-op and the "
                "'SNN' would just be the ANN. timm ViTs hit exactly this: their "
                "activation is torch.nn.GELU, which ACT_TARGETS does not cover.")
        _KINDS = {"all": None, "both": {"activation", "layernorm"},
                  "activation": {"activation"}, "layernorm": {"layernorm"},
                  "attention": {"matmul", "softmax"}}
        if a.convert_ops in ("all", "attention"):
            n_attn = make_attention_spikable(model)
            print(f"marked {n_attn} attention blocks; Stage 2", flush=True)
            if n_attn == 0:
                raise SystemExit("make_attention_spikable marked 0 blocks -- "
                                 "attention would stay exact FP; see above.")
        cfg = cv.ConvertConfig(epochs=a.epochs, backend=a.backend, spike_mult=True,
                               seed=a.seed,
                               pasn_id_target=a.pasn_id_target,
                               pasn_id_target_rel=a.pasn_id_target_rel,
                               pasn_t_fixed=a.pasn_t_fixed,
                               pasn_n_fixed=a.pasn_n_fixed,
                               verbose_fits=True)
        # E0's lesson: record the whole built config, not a hand-kept subset.
        rec["convert_cfg"] = {f.name: _jsonable(getattr(cfg, f.name))
                              for f in __import__("dataclasses").fields(cfg)}
        t0 = time.perf_counter()
        convert_hf(model, calib, cfg=cfg, only=_KINDS[a.convert_ops], verbose=True)
        rec["build_s"] = time.perf_counter() - t0
        print(f"[build] conversion took {rec['build_s'] / 60:.1f} min")

        spikes = cv.spiking_cost_report(model, calib[0])
        print(cv.format_spiking_cost_report(spikes, label=a.backend), flush=True)
        prims = cv._spiking_primitives(model)
        store = storage_breakdown([n for _, _, n in prims])
        params = sum(neuron_params(n) for _, _, n in prims)
        print(f"[store {a.backend}] params={params}  bytes={store['bytes']}  "
              f"primitives={len(prims)}", flush=True)
        rec.update(spikes_per_input=spikes["spikes_per_input"],
                   by_kind=spikes["by_kind"],
                   ops_per_input=spikes.get("ops_per_input"),
                   energy_pj_per_input=spikes.get("energy_pj_per_input"),
                   stored_params=params, stored_bytes=store["bytes"],
                   n_primitives=len(prims))

        if not a.build_only:
            acc_snn = top1(model, batches, device, f"SNN-{a.backend}")
            drop_pp = acc_snn - acc_ann
            rec.update(acc_snn=acc_snn, delta_pp=drop_pp,
                       delta_pct=100.0 * drop_pp / acc_ann)
            print(f"SNN ({a.backend}) top-1 = {acc_snn:.2f}   "
                  f"({drop_pp:+.2f} pp, {rec['delta_pct']:+.2f}% relative)")
            if a.paper_row:
                p_ann, p_snn = PAPER_TABLE1[a.paper_row]
                rec["paper_row"] = a.paper_row
                rec["paper_rel_pct"] = 100.0 * (p_snn - p_ann) / p_ann
                print(f"  paper ({a.paper_row}): {p_ann} -> {p_snn} at T=16 "
                      f"= {rec['paper_rel_pct']:+.2f}% relative")
                print("  compare the RELATIVE columns only -- our checkpoint is a "
                      "different model from the paper's, which we cannot run "
                      "(theirs is timm; our converter does not reach it)")

    if a.json:
        prev = []
        if os.path.exists(a.json):
            with open(a.json, encoding="utf-8") as fh:
                prev = json.load(fh)
        prev.append(rec)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(prev, fh, indent=1)
        print(f"[json] wrote record {len(prev)} to {a.json}")


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
