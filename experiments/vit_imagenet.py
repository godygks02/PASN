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
from mbe import hf_convert as _hf  # noqa: E402
from mbe import timm_convert as _tm  # noqa: E402
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
        labels = torch.tensor([lab for _, lab in batch])
        if callable(getattr(self.processor, "__call__", None)) and \
                not hasattr(self.processor, "image_mean"):
            # timm transform: PIL -> CHW tensor, one image at a time
            px = torch.stack([self.processor(im) for im in images])
            return dict(pixel_values=px, labels=labels)
        enc = self.processor(images=images, return_tensors="pt")
        enc["labels"] = labels
        return dict(enc)


def _hf_style(model) -> bool:
    """HF models take keyword inputs and return an output object; timm does not."""
    return hasattr(model, "config")


@torch.no_grad()
def top1(model, batches, device, label="eval", progress_every=20) -> float:
    model.eval()
    right = total = 0
    started = time.perf_counter()
    n_batches = len(batches)
    for i, b in enumerate(batches, 1):
        b = {k: v.to(device) for k, v in b.items()}
        y = b.pop("labels")
        # HF returns an output object; timm returns the logit tensor directly.
        out = model(**b) if _hf_style(model) else model(b["pixel_values"])
        pred = (out.logits if hasattr(out, "logits") else out).argmax(-1)
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
                    help="HF checkpoint, or a timm model name with --framework timm")
    ap.add_argument("--framework", choices=["hf", "timm"], default="hf",
                    help="timm needs its own markers (mbe.timm_convert): its "
                         "attention is inline and its activation is plain "
                         "nn.GELU, so the HF markers mark zero modules and the "
                         "conversion would be a silent no-op. The paper's "
                         "ViT-M/16 is a timm model")
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
    ap.add_argument("--ann-acc", type=float, default=None, metavar="TOP1",
                    help="reuse a known ANN top-1 instead of re-running the ANN "
                         "pass (~3 h). Only valid when the checkpoint, eval set "
                         "and preprocessing are unchanged -- the record is marked "
                         "with acc_ann_reused=True so it can never be mistaken "
                         "for a measurement.")
    ap.add_argument("--pasn-t-fixed", type=int, default=None,
                    help="force the paper's global T=16")
    ap.add_argument("--pasn-n-fixed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=_DEF.seed)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        if a.framework == "timm":
            import timm
            torch.manual_seed(a.seed)
            a.model = "vit_medium_patch16_reg4_gap_256 (random, smoke)"
            model = timm.create_model("vit_medium_patch16_reg4_gap_256",
                                      pretrained=False, num_classes=10).eval()
            batches = [dict(pixel_values=torch.randn(2, 3, 256, 256),
                            labels=torch.randint(0, 10, (2,))) for _ in range(3)]
        else:
            model, batches = build_smoke()
        n_eval = sum(b["labels"].numel() for b in batches)
    else:
        files = sorted(glob.glob(a.val_glob))
        if not files:
            raise SystemExit(
                f"no validation shards at {a.val_glob}. Fetch them with "
                f"snapshot_download('imagenet-1k', repo_type='dataset', "
                f"allow_patterns=['data/validation-*'])  -- and NOT with "
                f"load_dataset, which pulls the whole 155 GiB repository.")
        if a.framework == "timm":
            import timm
            model = timm.create_model(a.model, pretrained=True).eval()
            dcfg = timm.data.resolve_model_data_config(model)
            processor = timm.data.create_transform(**dcfg, is_training=False)
            print(f"[timm] {a.model}  data_config={dcfg}", flush=True)
        else:
            from transformers import (AutoImageProcessor,
                                      AutoModelForImageClassification)
            processor = AutoImageProcessor.from_pretrained(a.model)
            model = AutoModelForImageClassification.from_pretrained(
                a.model, attn_implementation="eager").eval()
        batches = ImageNetVal(files, processor, a.batch_size, a.limit)
        n_eval = batches.n
        print(f"[data] {len(files)} shards, evaluating {n_eval} images "
              f"in {len(batches)} batches of {a.batch_size}", flush=True)
    if a.framework == "hf":
        model.config._attn_implementation = "eager"
    model.to(device)

    # HF forwards take keyword dicts; timm's takes the tensor positionally, so a
    # dict would raise "unexpected keyword argument 'pixel_values'" during
    # calibration -- after the markers are already in, which makes it look like a
    # conversion failure rather than a plumbing one.
    calib = []
    for i, b in enumerate(batches):
        if i >= a.calib_batches:
            break
        calib.append({k: v for k, v in b.items() if k != "labels"}
                     if _hf_style(model) else b["pixel_values"])

    rec = dict(tag=a.tag, task="imagenet-1k", model=a.model, seed=a.seed,
               framework=a.framework,
               backend=a.backend, scope=a.convert_ops, device=device,
               smoke=a.smoke, epochs=a.epochs, batch_size=a.batch_size,
               limit=a.limit, n_eval=n_eval, calib_batches=a.calib_batches,
               stage=(None if a.backend == "none"
                      else 2 if a.convert_ops in ("all", "attention") else 1),
               started=time.strftime("%Y-%m-%dT%H:%M:%S"))

    if a.build_only:
        acc_ann = None
    elif a.ann_acc is not None:
        # Reuse a previously measured ANN pass. The ANN is ~3 h of the ~11 h run and
        # is identical across our low-`T` points -- only the SNN arm changes -- so
        # re-measuring it per point buys nothing. Gate the box with
        # ``--backend none --limit 2000`` instead of paying for the full sweep.
        acc_ann = float(a.ann_acc)
        rec["acc_ann_reused"] = True
        print(f"ANN ({a.model}) top-1 = {acc_ann:.3f}  [REUSED via --ann-acc, "
              f"not measured in this run]")
    else:
        acc_ann = top1(model, batches, device, "ANN")
    rec["acc_ann"] = acc_ann
    if acc_ann is not None and not a.ann_acc:
        print(f"ANN ({a.model}) top-1 = {acc_ann:.2f}")

    if a.backend != "none":
        adapter = _tm if a.framework == "timm" else _hf
        n_act = adapter.make_spikable(model)
        print(f"marked {n_act} activations", flush=True)
        if n_act == 0:
            raise SystemExit(
                "make_spikable marked 0 modules -- this is a silent no-op and the "
                "'SNN' would just be the ANN. Use --framework timm for timm "
                "models: their activation is torch.nn.GELU, which the HF "
                "ACT_TARGETS does not cover.")
        _KINDS = {"all": None, "both": {"activation", "layernorm"},
                  "activation": {"activation"}, "layernorm": {"layernorm"},
                  "attention": {"matmul", "softmax"}}
        if a.convert_ops in ("all", "attention"):
            n_attn = adapter.make_attention_spikable(model)
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
        conv = (_tm.convert_timm if a.framework == "timm" else _hf.convert_hf)
        conv(model, calib, cfg=cfg, only=_KINDS[a.convert_ops], verbose=True)
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
