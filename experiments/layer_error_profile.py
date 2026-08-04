"""Per-layer conversion error at depth (P0.4 §4.4, the last open item).

``share_fits=True`` fits **one** activation prototype and **one** LayerNorm
prototype and deep-copies them into all 24 blocks. Nothing has ever checked that
the shared fit is equally good at every depth: per-layer activation distributions
could drift, and a whole-network perplexity cannot tell a good fit everywhere
from a good fit early and a bad one late.

Three measurements, deliberately separated -- a single number would confound
"this site's own fit is worse" with "the input reaching it has already drifted".

1. **in-situ** -- every converted site's spiking output against the exact op it
   replaced, on the input that site sees *inside the converted model*. This is
   the error the deployed network actually pays, upstream drift included.
2. **clean-input** -- the same comparison with the *ANN's* input at that site
   replayed into the spiking module. Upstream drift removed, so what is left is
   the fit's own quality at that depth. Restricted to activation and LayerNorm,
   which are exactly the two kinds ``_shared_fit_slots`` pools (softmax and
   matmul are fitted per site, so depth cannot surprise them the same way).
3. **domain** -- what fraction of the in-situ inputs fall outside the pooled
   range the shared prototype was fitted on. This is the mechanism drift would
   have to act through, and it separates the two readings above: inputs inside
   the fitted range with a rising error means the fit is mismatched at depth;
   inputs escaping it means accumulation is the cause.

Plus **cumulative** block-output error, the quantity that hides all of this.

Local CPU. No new evaluation run -- one build and a few forwards.

Usage:
  python experiments/layer_error_profile.py --model gpt2-medium --seq-len 256
  python experiments/layer_error_profile.py --smoke        # wiring check, seconds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import make_spikable, make_attention_spikable  # noqa: E402

from gpt2_wikitext import load_wikitext_ids  # noqa: E402


# --------------------------------------------------------------------------
# The exact op behind each conversion point
# --------------------------------------------------------------------------

def exact_specs(model: nn.Module, kinds: dict) -> dict:
    """``name -> spec`` describing the ANN op each marked site will replace.

    Captured **before** conversion: the spiking wrappers do not all keep what is
    needed to reproduce their reference (``_SpikingActModule`` drops the target
    name, and reading it back off the fitted neuron would be circular anyway).
    """
    specs = {}
    for name, kind in kinds.items():
        mod = model.get_submodule(name)
        if kind == "activation":
            specs[name] = dict(kind=kind, fn=cv.Activation(mod.kind), target=mod.kind)
        elif kind == "layernorm":
            specs[name] = dict(kind=kind, shape=tuple(mod.normalized_shape),
                               eps=float(mod.eps), weight=mod.weight, bias=mod.bias)
        elif kind == "softmax":
            specs[name] = dict(kind=kind, dim=mod.dim)
        elif kind == "matmul":
            specs[name] = dict(kind=kind)
    return specs


def exact_forward(spec: dict, args):
    """Run the ANN op for ``spec`` on the arguments the spiking module received."""
    kind = spec["kind"]
    if kind == "activation":
        return spec["fn"](args[0])
    if kind == "layernorm":
        return F.layer_norm(args[0], spec["shape"], spec["weight"], spec["bias"],
                            spec["eps"])
    if kind == "softmax":
        return torch.softmax(args[0], dim=spec["dim"])
    if kind == "matmul":
        return args[0] @ args[1]
    raise ValueError(kind)


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative Frobenius error, the metric exp 10 reports for the FP multiply."""
    denom = ref.norm()
    if denom == 0:
        return float(got.norm())
    return float((got - ref).norm() / denom)


def layer_of(name: str) -> int | None:
    """Block index in ``transformer.h.<i>.…`` / ``encoder.layer.<i>.…``, else None."""
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part in ("h", "layer", "layers") and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def site_role(name: str) -> str:
    """Short label for what the site is, independent of its depth."""
    tail = name.split(".")[-1]
    return {"act": "mlp.act", "attn_softmax": "softmax", "qk_matmul": "qk_matmul",
            "av_matmul": "av_matmul"}.get(tail, tail)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_smoke():
    """Tiny random GPT-2 -- checks the wiring of this script in seconds."""
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(n_layer=4, n_head=2, n_embd=32, n_positions=64, vocab_size=256)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, 256, (8 * 64,))
    return model, ids, 64


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random GPT-2, no download -- wiring check only")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="tokens per profiled sequence. Depth is what is under "
                         "test, not context length, and the spiking softmax is "
                         "quadratic in this -- 256 keeps a CPU build tractable")
    ap.add_argument("--n-seq", type=int, default=2,
                    help="sequences profiled (errors are averaged over them)")
    ap.add_argument("--epochs", type=int, default=300,
                    help="fit epochs, as in the headline build")
    ap.add_argument("--backend", default="mbe_pasn")
    ap.add_argument("--convert-ops", default="all",
                    choices=["all", "both", "activation", "layernorm", "attention"])
    ap.add_argument("--pasn-t-fixed", type=int, default=16,
                    help="headline arm forces the paper's global T=16")
    ap.add_argument("--pasn-id-target", default="relative")
    ap.add_argument("--pasn-id-target-rel", type=float, default=1e-2)
    ap.add_argument("--no-share-fits", action="store_true",
                    help="fit every site separately. The control arm: if the "
                         "depth profile flattens under this, sharing is the cause")
    ap.add_argument("--json", default="results/layer_error_profile.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    t_start = time.perf_counter()

    if args.smoke:
        model, ids, block = build_smoke()
        seq_len = min(args.seq_len, 64)
    else:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
        ids = load_wikitext_ids(tok, "test")
        block = 1024
        seq_len = args.seq_len

    # Calibration follows the headline build: 8 batches of `block` train tokens.
    if args.smoke:
        calib = [ids[i:i + block].unsqueeze(0) for i in range(0, 4 * block, block)]
    else:
        cids = load_wikitext_ids(tok, "train")[: 64 * block]
        calib = [cids[i:i + block].unsqueeze(0) for i in range(0, 8 * block, block)]

    probes = [ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0)
              for i in range(args.n_seq)]

    n_act = make_spikable(model)
    n_attn = 0
    _SCOPE = {"all": None, "both": {"activation", "layernorm"},
              "activation": {"activation"}, "layernorm": {"layernorm"},
              "attention": {"matmul", "softmax"}}
    only = _SCOPE[args.convert_ops]
    if args.convert_ops in ("all", "attention"):
        n_attn = make_attention_spikable(model)
    print(f"marked {n_act} activations, {n_attn} attention blocks "
          f"(scope={args.convert_ops})", flush=True)

    blocks = model.transformer.h
    n_layers = len(blocks)

    # ---------------------------------------------------------------- pass 1
    # ANN reference. The markers are numerically inert (pinned by
    # test_stage2_attention_markers_are_numerically_inert), so this *is* the ANN.
    ann_block_out: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    ann_site_in: dict[str, list[torch.Tensor]] = {}
    shared_kinds = {"activation", "layernorm"}
    handles = []

    for i, blk in enumerate(blocks):
        def blk_hook(mod, inp, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            ann_block_out[i].append(h.detach().clone())
        handles.append(blk.register_forward_hook(blk_hook))

    # Only the shared-fit kinds get their inputs stored: those are the sites the
    # clean-input replay needs, and they are the small tensors. Keeping the
    # attention operands too would mean S x S per head per layer.
    for name, mod in model.named_modules():
        kind = cv.classify(mod)
        if kind in shared_kinds:
            ann_site_in.setdefault(name, [])

            def in_hook(mod, inp, name=name):
                ann_site_in[name].append(inp[0].detach().clone())
            handles.append(mod.register_forward_pre_hook(in_hook))

    with torch.no_grad():
        for probe in probes:
            model(probe)
    for h in handles:
        h.remove()
    print(f"[pass 1] ANN reference captured "
          f"({len(ann_site_in)} shared-fit sites)", flush=True)

    # ---------------------------------------------------------------- build
    cfg = cv.ConvertConfig(epochs=args.epochs, backend=args.backend,
                           spike_mult=True,
                           pasn_t_fixed=args.pasn_t_fixed,
                           pasn_id_target=args.pasn_id_target,
                           pasn_id_target_rel=args.pasn_id_target_rel,
                           share_fits=not args.no_share_fits,
                           verbose_fits=False)
    t0 = time.perf_counter()
    rec = cv.calibrate(model, calib)
    specs = exact_specs(model, dict(rec.kinds))
    # The range each prototype was actually fitted on, per site. Under sharing
    # this is the pooled union; without it, the site's own.
    shared_slots = cv._shared_fit_slots(model, rec) if cfg.share_fits else {}
    fit_range = {}
    for name in rec.kinds:
        slots = shared_slots.get(name) or rec.ranges.get(name)
        if slots:
            fit_range[name] = (float(slots[0].lo), float(slots[0].hi))
    cv.convert(model, rec, cfg=cfg, only=only, verbose=False)
    build_s = time.perf_counter() - t0
    print(f"[build] {build_s / 60:.1f} min "
          f"(share_fits={cfg.share_fits})", flush=True)

    # ---------------------------------------------------------------- pass 2
    # In-situ error, and the domain check, at every converted site.
    insitu: dict[str, list[float]] = {}
    outside: dict[str, list[float]] = {}
    seen_range: dict[str, tuple[float, float]] = {}
    snn_block_out: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    handles = []

    converted = {}
    for name, mod in model.named_modules():
        if isinstance(mod, cv._SPIKING_TYPES) and name in specs:
            converted[name] = mod

    for name, mod in converted.items():
        spec = specs[name]

        def site_hook(mod, inp, out, name=name, spec=spec):
            with torch.no_grad():
                ref = exact_forward(spec, inp)
                insitu.setdefault(name, []).append(rel_err(out, ref))
                x = inp[0]
                lo, hi = fit_range.get(name, (float("-inf"), float("inf")))
                frac = float(((x < lo) | (x > hi)).float().mean())
                outside.setdefault(name, []).append(frac)
                seen = seen_range.get(name)
                cur = (float(x.min()), float(x.max()))
                seen_range[name] = cur if seen is None else (
                    min(seen[0], cur[0]), max(seen[1], cur[1]))
        handles.append(mod.register_forward_hook(site_hook))

    for i, blk in enumerate(blocks):
        def blk_hook(mod, inp, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            snn_block_out[i].append(h.detach().clone())
        handles.append(blk.register_forward_hook(blk_hook))

    with torch.no_grad():
        for k, probe in enumerate(probes):
            t = time.perf_counter()
            model(probe)
            print(f"  [pass 2] sequence {k + 1}/{len(probes)} "
                  f"({time.perf_counter() - t:.1f}s)", flush=True)
    for h in handles:
        h.remove()

    # ---------------------------------------------------------------- pass 3
    # Clean-input replay: the ANN's input at each shared-fit site, so the number
    # carries no upstream drift.
    clean: dict[str, list[float]] = {}
    with torch.no_grad():
        for name, xs in ann_site_in.items():
            mod = converted.get(name)
            if mod is None:
                continue                      # site outside the converted scope
            spec = specs[name]
            for x in xs:
                clean.setdefault(name, []).append(
                    rel_err(mod(x), exact_forward(spec, (x,))))
    print(f"[pass 3] clean-input replay over {len(clean)} sites", flush=True)

    # ---------------------------------------------------------------- report
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    rows = []
    for name in sorted(converted, key=lambda n: (layer_of(n) if layer_of(n)
                                                 is not None else -1, n)):
        lo, hi = fit_range.get(name, (float("nan"),) * 2)
        seen = seen_range.get(name, (float("nan"),) * 2)
        rows.append(dict(
            site=name, layer=layer_of(name), role=site_role(name),
            kind=specs[name]["kind"],
            insitu=mean(insitu.get(name, [])),
            clean=mean(clean[name]) if name in clean else None,
            outside_frac=mean(outside.get(name, [])),
            fit_lo=lo, fit_hi=hi, seen_lo=seen[0], seen_hi=seen[1],
        ))

    cumulative = []
    for i in range(n_layers):
        errs = [rel_err(s, a) for s, a in zip(snn_block_out[i], ann_block_out[i])]
        cumulative.append(dict(layer=i, rel_err=mean(errs)))

    roles = sorted({r["role"] for r in rows})
    width = max(len(r) for r in roles) if roles else 8

    print("\n=== per-site relative error (in-situ | clean-input) ===")
    header = "layer  " + "  ".join(f"{r:>{max(width, 17)}}" for r in roles)
    print(header + "   cumulative")
    for i in range(n_layers):
        cells = []
        for role in roles:
            hit = [r for r in rows if r["layer"] == i and r["role"] == role]
            if not hit:
                cells.append(f"{'-':>{max(width, 17)}}")
                continue
            r = hit[0]
            c = f"{r['clean']:.1e}" if r["clean"] is not None else "  --   "
            cells.append(f"{r['insitu']:.1e}|{c}".rjust(max(width, 17)))
        print(f"{i:>5}  " + "  ".join(cells) +
              f"   {cumulative[i]['rel_err']:.3e}")

    print("\n=== depth trend (first block vs last, in-situ) ===")
    for role in roles:
        series = [r for r in rows if r["role"] == role and r["layer"] is not None]
        series.sort(key=lambda r: r["layer"])
        if len(series) < 2:
            continue
        first, last = series[0], series[-1]
        ratio = last["insitu"] / first["insitu"] if first["insitu"] else float("nan")
        line = (f"  {role:<12} in-situ {first['insitu']:.2e} -> "
                f"{last['insitu']:.2e}  ({ratio:.2f}x)")
        if first["clean"] is not None and last["clean"] is not None:
            cr = last["clean"] / first["clean"] if first["clean"] else float("nan")
            line += (f" | clean {first['clean']:.2e} -> {last['clean']:.2e} "
                     f"({cr:.2f}x)")
        print(line)

    esc = [r for r in rows if r["outside_frac"] > 1e-6]
    print(f"\n=== inputs outside the fitted range: {len(esc)}/{len(rows)} sites ===")
    for r in sorted(esc, key=lambda r: -r["outside_frac"])[:12]:
        print(f"  L{r['layer']:<3} {r['role']:<12} {r['outside_frac'] * 100:6.3f}%  "
              f"seen [{r['seen_lo']:.3g},{r['seen_hi']:.3g}] "
              f"fitted [{r['fit_lo']:.3g},{r['fit_hi']:.3g}]")
    if not esc:
        print("  none -- every site stayed inside the range it was fitted on")

    out = dict(
        model=args.model if not args.smoke else "smoke",
        backend=args.backend, scope=args.convert_ops,
        share_fits=cfg.share_fits, epochs=args.epochs,
        pasn_t_fixed=args.pasn_t_fixed, id_target=args.pasn_id_target,
        r=args.pasn_id_target_rel,
        seq_len=seq_len, n_seq=len(probes), n_layers=n_layers,
        build_s=build_s, total_s=time.perf_counter() - t_start,
        sites=rows, cumulative=cumulative,
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        prev = []
        if os.path.exists(args.json):
            with open(args.json) as fh:
                prev = json.load(fh)
        prev.append(out)
        with open(args.json, "w") as fh:
            json.dump(prev, fh, indent=1)
        print(f"\nwrote {args.json} ({len(prev)} records)")


if __name__ == "__main__":
    main()
