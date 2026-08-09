"""E11b -- does a deeper identity router recover what ``av_matmul`` loses?

E11 found the attention identity collapsed at all 24 sites: 97-99% of the
attention-probability mass falls in the near-zero bank, because the operand is a
softmax output whose mean ``1/S`` sits below the near-zero floor ``2^-6``. That
is the *opposite* mechanism to ``1/x`` (whose domain was too narrow to split);
here the domain is 17 binades wide and the floor is simply in the wrong place.
The knob is ``pasn_id_e_min``, not ``beta``.

**Why this cannot be measured on the synthetic operator.** ``op_pareto``'s
``attention`` op runs at ``seq=32``, where the mean attention probability is
``1/32 = 0.031`` -- *above* the ``2^-6 = 0.0156`` floor. The pathology does not
exist there, so sweeping the depth knob on it would report "no benefit" for the
same structural reason the two wrong ``1/x`` closures did (함정 15): the
measurement is taken where the problem is absent. This script therefore runs the
operator on **real GPT-2 attention operands**, captured at the depth where E11
measured the worst collapse.

**Why the whole operator, not the primitive.** 함정 15 again: pinning the rest of
the op fixes its cost by construction and can only report ~1x. The question is
not "what do this identity's spikes cost" but "at equal accuracy, what does
attention cost". So this builds QK^T -> softmax -> AV end to end and compares
iso-accuracy, reusing ``op_pareto``'s metered cell (``SpikeMeter``, which already
handles the reconstruct-delegates-to-forward double charge, 함정 14).

Two arms of depth, deliberately separated:

* ``id_a`` only -- the ``[0,1]`` identities (the attention matrix consumed by the
  softmax and by AV). Attribution: is there anything at the site of the
  pathology?
* all identities -- what the shipped ``pasn_id_e_min`` knob actually does, since
  it also deepens ``id_qk`` / ``id_v``, whose domains are wide and *not*
  concentrated, so they pay memory for nothing.

Local CPU. Capture is one forward of gpt2-medium; the sweep is builds only.

Usage:
  python experiments/id_router_depth.py --capture          # ~2 min, writes cache
  python experiments/id_router_depth.py --smoke            # wiring, minutes
  python experiments/id_router_depth.py                    # the sweep
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import spiking_ops as so  # noqa: E402

from op_pareto import errors, maker, run_cell  # noqa: E402


DEFAULT_CACHE = ".tmp/attn_operands_gpt2-medium.pt"


# --------------------------------------------------------------------------
# Capture -- real q/k/v, at the layer E11 measured as worst
# --------------------------------------------------------------------------

def capture(model_name: str, layers: list[int], seq: int, heads: int,
            out_path: str) -> dict:
    """Save ``(q, k, v)`` per requested layer from one real forward.

    Heads are subsampled: the concentration that E11 found scales with the
    sequence length (mean probability ``1/S``), not with how many heads are
    averaged, so dropping heads cuts the sweep cost without touching the
    quantity under test. ``seq`` is the one dimension that must stay realistic.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from gpt2_wikitext import load_wikitext_ids

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    ids = load_wikitext_ids(tok, "test")[:seq].unsqueeze(0)

    n_head = model.config.n_head
    n_embd = model.config.n_embd
    dh = n_embd // n_head
    grabbed: dict[int, dict] = {}
    handles = []

    for li in layers:
        attn = model.transformer.h[li].attn

        def hook(mod, inp, out, li=li):
            qkv = out if torch.is_tensor(out) else out[0]
            q, k, v = qkv.split(n_embd, dim=2)

            def split(t):
                # (1, S, E) -> (heads, S, dh), then keep the first `heads`
                return (t.reshape(t.shape[1], n_head, dh)
                         .permute(1, 0, 2)[:heads].detach().clone())
            grabbed[li] = dict(q=split(q), k=split(k), v=split(v))
        handles.append(attn.c_attn.register_forward_hook(hook))

    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    payload = dict(model=model_name, seq=seq, heads=heads, dh=dh,
                   n_head=n_head, layers=layers,
                   data={str(k): v for k, v in grabbed.items()})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(payload, out_path)
    return payload


# --------------------------------------------------------------------------
# The operator, on real operands, with per-primitive router depth
# --------------------------------------------------------------------------

def make_attention_real(qkv: dict, id_depth: int, deep_all: bool):
    """``op_pareto``-shaped op dict: QK^T -> softmax -> AV on captured operands.

    Mirrors ``op_pareto.make_attention`` exactly except that ``data`` returns the
    real tensors and the ``[0,1]`` identities get their own router depth. Keeping
    the structure identical is what makes the numbers comparable to the existing
    ``attention`` row.

    ``id_depth`` is a **binade span**: the router resolves that many binades below
    the operand's largest magnitude and everything smaller lands in the near-zero
    bank. For an operand in ``(0, 1]`` that is exactly ``e_min = -id_depth``.
    """
    q, k, v = qkv["q"], qkv["k"], qkv["v"]
    dh = q.shape[-1]
    scale = 1.0 / math.sqrt(dh)

    def data(cfg):
        ref = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v
        return dict(q=q, k=k, v=v, scale=scale, ref=ref, n_out=ref.numel())

    def build(backend, b, t, cfg, d):
        qk_max = float(max(d["q"].abs().max(), d["k"].abs().max())) * 1.1
        v_max = float(d["v"].abs().max()) * 1.1
        if backend == "mbe":
            sm = so.build_softmax(d["q"] @ d["k"].transpose(-1, -2) * d["scale"],
                                  n_basis=b, n_steps=t, epochs=cfg.epochs,
                                  seed=cfg.seed, spike_mult=True)
            com = dict(epochs=cfg.epochs, seed=cfg.seed)
            return dict(
                exp=sm.exp, inv=sm.inv, idn=sm.idn,
                id_qk=so.calibrate_identity(0.0, qk_max, n_basis=b, n_steps=t,
                                            **com),
                id_a=so.calibrate_identity(0.0, 1.0, n_basis=b, n_steps=t, **com),
                id_v=so.calibrate_identity(0.0, v_max, n_basis=b, n_steps=t,
                                           **com))
        deep = copy.copy(cfg)
        deep.binades = id_depth
        mk = maker(backend, b, t, deep if deep_all else cfg)
        mk_a = maker(backend, b, t, deep)       # the [0,1] identities
        return dict(exp=mk("exp2", (0.0, 1.0)), inv=mk("inv", (0.5, 1.0)),
                    idn=mk_a("identity", (0.0, 1.0)),
                    id_qk=mk("identity", (-qk_max, qk_max)),
                    id_a=mk_a("identity", (0.0, 1.0)),
                    id_v=mk("identity", (-v_max, v_max)))

    def run(prims, d):
        scores = so.spiking_matmul(prims["id_qk"], d["q"],
                                   d["k"].transpose(-1, -2),
                                   signed=None) * d["scale"]
        sm = so.SpikingSoftmax(prims["exp"], prims["inv"], prims["idn"],
                               spike_mult=True)
        attn = sm(scores, dim=-1)
        return so.spiking_matmul(prims["id_a"], attn, d["v"], signed=None,
                                 idn2=prims["id_v"])

    return dict(name="attention_real", fn="identity", data=data, build=build,
                run=run, primary="nrmse")


def occupancy(prims: dict, qkv: dict) -> dict:
    """Where the real attention matrix lands in ``id_a``'s banks.

    The necessary condition for the fix: if deepening the router does not move
    the mass off bank 0, nothing downstream can improve and the arm is dead.
    """
    idn = prims.get("id_a")
    router = getattr(idn, "router", None)
    if router is None:
        return {}
    q, k, v = qkv["q"], qkv["k"], qkv["v"]
    with torch.no_grad():
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]),
                          dim=-1)
        counts = idn.bank_usage(a)
    tot = sum(counts.values()) or 1
    p = torch.tensor([counts[i] / tot for i in sorted(counts)],
                     dtype=torch.float64)
    nz = p[p > 0]
    return dict(n_banks=router.n_banks, e_min=int(router.e_min),
                top1=float(p.max()),
                n_eff=float(torch.exp(-(nz * nz.log()).sum())),
                reached=int((p > 0).sum()),
                occupancy=[float(x) for x in p])


# --------------------------------------------------------------------------
# Iso-accuracy comparison
# --------------------------------------------------------------------------

def iso_table(records: list, thresholds: list) -> list:
    """Cheapest build (by spikes) reaching each nrmse threshold, per depth.

    Comparing single operating points is what lets accuracy be bought with
    spikes; the frontier at a fixed accuracy is the only fair axis (§4.2).
    """
    out = []
    depths = sorted({r["id_depth"] for r in records})
    for thr in thresholds:
        row = dict(nrmse_le=thr, by_depth={})
        for d in depths:
            ok = [r for r in records if r["id_depth"] == d and r["nrmse"] <= thr]
            if ok:
                best = min(ok, key=lambda r: r["spikes"])
                row["by_depth"][d] = dict(spikes=best["spikes"],
                                          bytes=best["bytes"],
                                          nrmse=best["nrmse"],
                                          budget=best["budget"])
        out.append(row)
    return out


def report(records: list, iso: list, occ: dict) -> str:
    L = []
    depths = sorted({r["id_depth"] for r in records})

    L.append("## 1. Does the mass actually split? (necessary condition)\n")
    L.append("| id e_min | banks | reached | top-1 | n_eff |")
    L.append("|---:|---:|---:|---:|---:|")
    for d in depths:
        o = occ.get(d)
        if o:
            L.append(f"| -{d} | {o['n_banks']} | {o['reached']} | "
                     f"{o['top1']:.4f} | {o['n_eff']:.2f} |")
    L.append("")

    L.append("\n## 2. Iso-accuracy cost of the whole operator\n")
    L.append("Cheapest build reaching each nrmse, by router depth. "
             "Spikes per output element; bytes stored.\n")
    hdr = "| nrmse <= | " + " | ".join(f"-{d}: spikes / bytes" for d in depths) + " |"
    L.append(hdr)
    L.append("|---|" + "---|" * len(depths))
    for row in iso:
        cells = []
        for d in depths:
            e = row["by_depth"].get(d)
            cells.append(f"{e['spikes']:.2f} / {e['bytes']}" if e else "-")
        L.append(f"| {row['nrmse_le']:.0e} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("\n## 3. Raw frontier (every cell)\n")
    L.append("The iso-accuracy table above depends on where the thresholds fall; "
             "this does not.\n")
    L.append("| target | " + " | ".join(f"-{d}: nrmse / spikes" for d in depths) + " |")
    L.append("|---|" + "---|" * len(depths))
    for b in sorted({r["budget"] for r in records}, reverse=True):
        cells = []
        for d in depths:
            m = [r for r in records if r["id_depth"] == d and r["budget"] == b]
            cells.append(f"{m[0]['nrmse']:.2e} / {m[0]['spikes']:.1f}"
                         if m else "-")
        L.append(f"| {b:.0e} | " + " | ".join(cells) + " |")
    L.append("")

    base = depths[0]
    L.append(f"\n## 4. Verdict vs the shipped default (e_min = -{base})\n")
    L.append("| nrmse <= | best depth | spikes ratio | bytes ratio |")
    L.append("|---|---:|---:|---:|")
    for row in iso:
        b = row["by_depth"].get(base)
        if not b:
            continue
        cands = [(d, e) for d, e in row["by_depth"].items() if d != base]
        if not cands:
            continue
        d, e = min(cands, key=lambda de: de[1]["spikes"])
        L.append(f"| {row['nrmse_le']:.0e} | -{d} | "
                 f"{b['spikes'] / e['spikes']:.2f}x | "
                 f"{b['bytes'] / e['bytes']:.2f}x |")
    L.append("\n⚠️ A ratio below 1.00x means the deeper router **costs** more. "
             "Report it either way -- a negative result closes the item and "
             "pins the freeze at the shipped default.\n")
    return "\n".join(L)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", action="store_true",
                    help="capture real q/k/v from GPT-2 into the cache and exit")
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--layers", nargs="+", type=int, default=[21],
                    help="21 is where E11 measured the worst collapse")
    ap.add_argument("--seq", type=int, default=256,
                    help="the dimension the pathology scales with (mean prob 1/S)")
    ap.add_argument("--heads", type=int, default=4,
                    help="head subsample; does not affect the concentration")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--depths", nargs="+", type=int, default=[6, 8, 10, 12],
                    help="binade spans for the [0,1] identities; 6 is shipped")
    ap.add_argument("--deep-all", action="store_true",
                    help="deepen every identity, i.e. what pasn_id_e_min does, "
                         "instead of only the [0,1] ones")
    ap.add_argument("--rule-targets", nargs="+", type=float,
                    default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3])
    ap.add_argument("--backend", default="pasn_rule")
    ap.add_argument("--t", type=int, default=None,
                    help="pin every bank's T after the rule has run. The "
                         "headline build ships --pasn-t-fixed 16, so `--backend "
                         "pasn_rule_T --t 16` is the deployed operating point; "
                         "the default (dynamic T_j) is the method's own front")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", default="results/id_router_depth.json")
    ap.add_argument("--md", default="results/id_router_depth.md")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    torch.set_num_threads(os.cpu_count() or 4)

    if args.capture:
        p = capture(args.model, args.layers, args.seq, args.heads, args.cache)
        for li, d in p["data"].items():
            print(f"layer {li}: q{tuple(d['q'].shape)} "
                  f"k{tuple(d['k'].shape)} v{tuple(d['v'].shape)}")
        print(f"[capture] -> {args.cache}")
        return

    if args.smoke:
        args.depths, args.rule_targets = [6, 10], [1e-1, 1e-2]
        args.epochs, args.seq, args.heads = 40, 64, 2

    if not os.path.exists(args.cache):
        raise SystemExit(f"no operand cache at {args.cache}; "
                         f"run with --capture first")
    cache = torch.load(args.cache, weights_only=False)
    layer = str(args.layers[0])
    if layer not in cache["data"]:
        raise SystemExit(f"layer {layer} not in cache (has {list(cache['data'])})")
    qkv = cache["data"][layer]
    if args.smoke:
        qkv = {k: v[: args.heads, : args.seq] for k, v in qkv.items()}
    print(f"[operands] layer {layer}, q{tuple(qkv['q'].shape)} "
          f"from {cache['model']} seq={cache['seq']}", flush=True)

    cfg = types.SimpleNamespace(epochs=args.epochs, seed=args.seed, binades=6,
                                m_eval=4000)
    records, occ = [], {}
    t_start = time.time()

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)

    def save():
        """Write after **every cell**: this sweep is hours and a lost run is a
        rerun rather than a gap (the same reason ``op_pareto`` does it, and the
        reason a closed vast.ai box once took Stage 2's raw records with it)."""
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(model=cache["model"], layer=int(layer),
                           seq=cache["seq"], heads=int(qkv["q"].shape[0]),
                           depths=args.depths, deep_all=args.deep_all,
                           backend=args.backend, epochs=args.epochs,
                           rule_targets=args.rule_targets,
                           total_s=time.time() - t_start,
                           occupancy=occ, records=records), f, indent=1)

    for depth in args.depths:
        op = make_attention_real(qkv, depth, args.deep_all)
        d = op["data"](cfg)
        for b in args.rule_targets:
            t0 = time.time()
            rec = run_cell(op, args.backend, b, args.t, cfg, d)
            rec.update(id_depth=depth, layer=int(layer), deep_all=args.deep_all,
                       seconds=time.time() - t0)
            records.append(rec)
            print(f"  e_min=-{depth:<3d} target={b:<8.0e} "
                  f"nrmse={rec['nrmse']:.3e} spikes={rec['spikes']:7.2f} "
                  f"bytes={rec['bytes']:6d}  ({rec['seconds']:.0f}s)", flush=True)
            if depth not in occ:
                occ[depth] = occupancy(op["build"](args.backend, b, args.t, cfg, d),
                                       qkv)
            save()

    thresholds = sorted({round(r["nrmse"], 12) for r in records})
    # A small ladder of round thresholds reads better than every observed value.
    ladder = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
    ladder = [t for t in ladder if any(r["nrmse"] <= t for r in records)]
    iso = iso_table(records, ladder or thresholds[:4])
    md = report(records, iso, occ)

    save()
    with open(args.json, "r+", encoding="utf-8") as f:
        payload = json.load(f)
        payload["iso"] = iso
        f.seek(0), f.truncate()
        json.dump(payload, f, indent=1)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(f"# E11b -- identity router depth ({cache['model']}, "
                f"layer {layer}, seq {cache['seq']})\n\n" + md + "\n")

    print("\n" + md)
    print(f"\n[done] {len(records)} cells, "
          f"{(time.time() - t_start) / 60:.1f} min -> {args.json}")


if __name__ == "__main__":
    main()
