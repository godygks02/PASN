"""E11 -- bank utilisation audit: does the router actually partition anything?

**Why this exists.** The `1/x` router degeneracy was found *by accident*, while
chasing the one row PASN lost in Table XI. `1/S` lives on `[0.5, 1)`, which is
exactly one binade -- the interval `frexp` normalises a mantissa into -- so 1 of
7 banks was reachable and 20000/20000 inputs landed in it. PASN there was a
global MBE neuron plus router overhead. Before that was understood the question
was closed **twice** as "~1.0x, not worth it" (P0.2, and the first pass of
`inv_router_fix.py`), because both pinned the rest of the operator (함정 15).

Nothing has ever checked the other sites. GPT-2-medium has 339 primitives, and
the argument that protects them is *reasoning*, not measurement: `invsqrt` is
assumed fine because `[0.5, 2)` spans two binades. Two is one away from the
failure mode, and no one has looked at what the calibrated domain actually is.

**Why it also earns its place in the paper.** `PAPER_PLAN.md` §0 presents PASN as
an MoE contract -- store `R` banks, fire one -- and the field's standard question
about that contract is expert load: are the experts used evenly, are any dead?
This script answers it directly, and the same pass produces the *network-level*
total/active basis table that §0 currently fills with isolated-operator numbers.

Three outputs:

1. **degeneracy** -- sites where routing buys nothing: one bank reached, or one
   bank holding ~all the mass. For each, the observed domain and its width in
   binades, which is what a `beta` fix keys off (`beta = 0.5` for `1/x` was the
   domain's lower bound, not a tuned constant).
2. **load distribution** -- per-site occupancy, and the effective bank count
   `exp(H)` of the occupancy distribution. The MoE-convention load figure.
3. **MoE contract, network level** -- stored bases vs expected bases per input,
   `sum_b p_b * N_b`, against the paper's G.1 (N=4 activation / 8 elsewhere,
   always all active).

Build-only: one conversion and a few forwards, no perplexity evaluation.
Local CPU. Runs before the E0 freeze, because a degenerate site found *after*
the freeze costs the GPU session over again.

Usage:
  python experiments/bank_usage_audit.py --smoke          # wiring check, seconds
  python experiments/bank_usage_audit.py --model gpt2-medium --seq-len 256
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import make_spikable, make_attention_spikable  # noqa: E402

from gpt2_wikitext import load_wikitext_ids  # noqa: E402
from layer_error_profile import layer_of, site_role  # noqa: E402


#: A bank holding at least this share of the inputs makes the others decorative.
TOP_HEAVY = 0.99
#: Effective bank count below this, with this many banks built, is a bad split.
LOW_NEFF, MIN_BANKS = 2.0, 4


# --------------------------------------------------------------------------
# Occupancy probe
# --------------------------------------------------------------------------

def install_usage_probe(neuron, entry, saved):
    """Tally the routed bank of every element the neuron is handed.

    Wraps ``forward`` **and** ``reconstruct`` with a depth counter, exactly as
    :func:`convert._install_cost_probe` does and for the same reason (함정 14):
    ``MBEPASNNeuron.reconstruct`` is ``return self.forward(x)``, so a naive wrap
    tallies a routed reconstruct twice while a plain ``MBENeuron`` -- whose
    ``reconstruct`` goes elsewhere -- is tallied once. Shares would survive that
    (the bias is uniform within a site), but the element counts would not, and
    those are what weight the network-level aggregate across sites.

    Two *sequential* calls still count twice: the depth returns to 0 between them.
    """
    router = neuron.router
    state = {"depth": 0}

    def wrap(fn):
        def inner(x, *a, **k):
            outermost = state["depth"] == 0
            state["depth"] += 1
            try:
                if outermost and torch.is_tensor(x) and x.numel():
                    with torch.no_grad():
                        flat = x.reshape(-1)
                        idx = router.route(flat)
                        entry["counts"] += torch.bincount(
                            idx, minlength=router.n_banks).to(torch.float64)
                        entry["elements"] += int(flat.numel())
                        entry["calls"] += 1
                        # Observed domain, in x-space and in router-key space.
                        # The key is what the binade ladder actually sees, so its
                        # span is the quantity that decides degeneracy.
                        t = router.to_key(flat).abs()
                        t = t[t > 0]
                        lo, hi = float(flat.min()), float(flat.max())
                        entry["x_lo"] = min(entry["x_lo"], lo)
                        entry["x_hi"] = max(entry["x_hi"], hi)
                        if t.numel():
                            entry["t_lo"] = min(entry["t_lo"], float(t.min()))
                            entry["t_hi"] = max(entry["t_hi"], float(t.max()))
                return fn(x, *a, **k)
            finally:
                state["depth"] -= 1
        return inner

    for attr in ("forward", "reconstruct"):
        orig = getattr(neuron, attr, None)
        if orig is None:
            continue
        saved.append((neuron, attr, orig))
        setattr(neuron, attr, wrap(orig))


def remove_probes(saved):
    for neuron, attr, orig in saved:
        try:
            delattr(neuron, attr)
        except AttributeError:
            setattr(neuron, attr, orig)


def new_entry(n_banks):
    return dict(counts=torch.zeros(n_banks, dtype=torch.float64),
                elements=0, calls=0,
                x_lo=float("inf"), x_hi=float("-inf"),
                t_lo=float("inf"), t_hi=float("-inf"))


# --------------------------------------------------------------------------
# Per-site statistics
# --------------------------------------------------------------------------

def site_stats(name, kind, neuron, entry) -> dict:
    """Occupancy, load balance, degeneracy verdict and MoE currency for one site."""
    router = neuron.router
    counts = entry["counts"]
    total = float(counts.sum())
    p = (counts / total) if total > 0 else counts

    reached = int((counts > 0).sum())
    top1 = float(p.max()) if total > 0 else 0.0
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum()) if nz.numel() else 0.0
    n_eff = math.exp(entropy) if total > 0 else 0.0

    # Bases: stored counts each shared/tied prototype once; active is what one
    # input actually fires, which is the MoE "active parameters" column.
    n_per_bank = [int(b.cfg.n_basis) for b in neuron.bank_mods]
    t_per_bank = [int(b.cfg.n_steps) for b in neuron.bank_mods]
    active_bases = float(sum(float(p[i]) * n_per_bank[i] for i in range(len(p))))
    active_slots = float(sum(float(p[i]) * n_per_bank[i] * t_per_bank[i]
                             for i in range(len(p))))

    # Key-space width in binades: how many the ladder can possibly split into.
    span = (math.log2(entry["t_hi"] / entry["t_lo"])
            if entry["t_lo"] > 0 and entry["t_hi"] > entry["t_lo"] else 0.0)

    dead = reached <= 1 and router.n_banks > 1
    top_heavy = top1 >= TOP_HEAVY and router.n_banks > 1
    imbalanced = (n_eff < LOW_NEFF and router.n_banks >= MIN_BANKS
                  and not (dead or top_heavy))

    return dict(
        site=name, op=kind, role=site_role(name), layer=layer_of(name),
        n_banks=int(router.n_banks), n_reached=reached,
        top1_share=top1, n_eff=n_eff, entropy=entropy,
        elements=int(entry["elements"]), calls=int(entry["calls"]),
        x_lo=entry["x_lo"] if entry["elements"] else None,
        x_hi=entry["x_hi"] if entry["elements"] else None,
        key_lo=entry["t_lo"] if entry["t_lo"] < float("inf") else None,
        key_hi=entry["t_hi"] if entry["t_hi"] > float("-inf") else None,
        key_binades=span,
        beta=float(router.beta), e_min=int(router.e_min), e_max=int(router.e_max),
        stored_bases=int(neuron.stored_bases()),
        active_bases=active_bases, active_slots=active_slots,
        n_per_bank=n_per_bank, t_per_bank=t_per_bank,
        occupancy=[float(v) for v in p],
        degenerate=bool(dead or top_heavy),
        flags=[f for f, on in (("dead", dead), ("top_heavy", top_heavy),
                               ("imbalanced", imbalanced)) if on],
    )


#: The paper's G.1 basis counts, by our op kind. Every one of them is always
#: active -- a global multi-basis neuron fires all N bases for every input.
PAPER_N = {"activation": 4, "layernorm": 8, "softmax": 8, "matmul": 8}


def aggregate(rows: list) -> dict:
    """Network-level MoE contract, per op kind and overall.

    ``active`` is element-weighted: a site that runs on S x S attention matrices
    must not count the same as one that runs on a hidden vector.
    """
    out = {}
    for kind in sorted({r["op"] for r in rows} | {"__all__"}):
        sel = rows if kind == "__all__" else [r for r in rows if r["op"] == kind]
        el = sum(r["elements"] for r in sel) or 1
        out[kind] = dict(
            sites=len(sel),
            elements=sum(r["elements"] for r in sel),
            stored_bases=sum(r["stored_bases"] for r in sel),
            active_bases=sum(r["active_bases"] * r["elements"] for r in sel) / el,
            active_slots=sum(r["active_slots"] * r["elements"] for r in sel) / el,
            paper_n=PAPER_N.get(kind),
            degenerate=sum(1 for r in sel if r["degenerate"]),
            mean_n_eff=sum(r["n_eff"] * r["elements"] for r in sel) / el,
        )
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def report(rows: list, agg: dict) -> str:
    L = []
    bad = [r for r in rows if r["flags"]]

    L.append("## 1. Degeneracy\n")
    if not bad:
        L.append(f"**None.** All {len(rows)} routed sites split their input: every "
                 f"one reaches >1 bank with top-1 share < {TOP_HEAVY:.2f} and "
                 f"effective bank count >= {LOW_NEFF}.\n")
    else:
        L.append(f"**{len(bad)} of {len(rows)} sites flagged.**\n")
        L.append("| site | op | banks | reached | top-1 | n_eff | key range | binades | beta | flags |")
        L.append("|---|---|---:|---:|---:|---:|---|---:|---:|---|")
        for r in sorted(bad, key=lambda r: (-len(r["flags"]), -r["top1_share"])):
            kr = (f"[{r['key_lo']:.3g}, {r['key_hi']:.3g}]"
                  if r["key_lo"] is not None else "-")
            L.append(f"| `{r['site']}` | {r['op']} | {r['n_banks']} | "
                     f"{r['n_reached']} | {r['top1_share']:.4f} | {r['n_eff']:.2f} | "
                     f"{kr} | {r['key_binades']:.2f} | {r['beta']:.3g} | "
                     f"{', '.join(r['flags'])} |")
        L.append("")
        L.append("**Reading a flagged row.** `binades` is the width of the observed "
                 "routing key in powers of two -- the number of ranges the ladder "
                 "*can* split into. Below ~1 the router cannot partition at all and "
                 "the bank is a global neuron; the fix is to re-anchor `beta` on the "
                 "domain's hard end, as `beta=0.5` did for `1/x`.\n")

    L.append("\n## 2. Load distribution\n")
    L.append("| op | sites | banks (mean) | reached (mean) | n_eff (elem-weighted) |")
    L.append("|---|---:|---:|---:|---:|")
    for kind in sorted(k for k in agg if k != "__all__"):
        sel = [r for r in rows if r["op"] == kind]
        L.append(f"| {kind} | {len(sel)} | "
                 f"{sum(r['n_banks'] for r in sel)/len(sel):.1f} | "
                 f"{sum(r['n_reached'] for r in sel)/len(sel):.1f} | "
                 f"{agg[kind]['mean_n_eff']:.2f} |")

    L.append("\n## 3. MoE contract -- network level\n")
    L.append("Stored = bases held (shared/tied prototypes counted once). "
             "Active = `sum_b p_b * N_b`, the bases one input fires, "
             "element-weighted across sites.\n")
    L.append("| op | sites | stored bases | active bases / input | paper N (all active) | ratio |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for kind in sorted(k for k in agg if k != "__all__"):
        a = agg[kind]
        ratio = (a["paper_n"] / a["active_bases"]) if a["active_bases"] else float("nan")
        L.append(f"| {kind} | {a['sites']} | {a['stored_bases']} | "
                 f"{a['active_bases']:.2f} | {a['paper_n']} | {ratio:.2f}x |")
    a = agg["__all__"]
    L.append(f"| **all** | {a['sites']} | **{a['stored_bases']}** | "
             f"**{a['active_bases']:.2f}** | - | - |")
    L.append("")
    L.append("⚠️ `active bases` is not an energy figure on its own -- energy is "
             "`T*eta*N`, and `T_j` varies per bank. The `active_slots` field in the "
             "JSON carries the `N_j*T_j` version. Quote ratios, not absolutes "
             "(함정 13).\n")
    return "\n".join(L)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random GPT-2, no download -- wiring check only")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="tokens per probed sequence; the spiking softmax is "
                         "quadratic in this")
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300,
                    help="fit epochs, as in the headline build")
    ap.add_argument("--backend", default="mbe_pasn")
    ap.add_argument("--convert-ops", default="all",
                    choices=["all", "both", "activation", "layernorm", "attention"])
    ap.add_argument("--pasn-t-fixed", type=int, default=16)
    ap.add_argument("--pasn-id-target", default="relative")
    ap.add_argument("--pasn-id-target-rel", type=float, default=1e-2)
    ap.add_argument("--json", default="results/bank_usage_audit.json")
    ap.add_argument("--md", default="results/bank_usage_audit.md")
    args = ap.parse_args()

    # The report carries 함정 references and a warning glyph; the default Windows
    # console codec (cp949) cannot encode either, and the files are already on
    # disk by the time we print, so a crash here would lose only the echo.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    device = torch.device("cpu")
    t_start = time.perf_counter()

    if args.smoke:
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg_m = GPT2Config(n_layer=4, n_head=2, n_embd=32, n_positions=64,
                           vocab_size=256)
        torch.manual_seed(0)
        model = GPT2LMHeadModel(cfg_m).eval()
        ids = torch.randint(0, 256, (8 * 64,))
        block, seq_len = 64, min(args.seq_len, 64)
        calib = [ids[i:i + block].unsqueeze(0) for i in range(0, 4 * block, block)]
    else:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
        ids = load_wikitext_ids(tok, "test")
        block, seq_len = 1024, args.seq_len
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

    cfg = cv.ConvertConfig(epochs=args.epochs, backend=args.backend,
                           spike_mult=True,
                           pasn_t_fixed=args.pasn_t_fixed,
                           pasn_id_target=args.pasn_id_target,
                           pasn_id_target_rel=args.pasn_id_target_rel,
                           verbose_fits=False)
    t0 = time.perf_counter()
    rec = cv.calibrate(model, calib)
    cv.convert(model, rec, cfg=cfg, only=only, verbose=False)
    build_s = time.perf_counter() - t0
    print(f"[build] {build_s / 60:.1f} min", flush=True)

    # Routed primitives only: a plain MBENeuron has no router to audit, and
    # counting it would dilute the very statistic under test.
    prims = cv._spiking_primitives(model)
    routed = [(n, k, ne) for n, k, ne in prims if hasattr(ne, "router")]
    print(f"[sites] {len(routed)} routed of {len(prims)} spiking primitives",
          flush=True)
    if not routed:
        raise SystemExit("no routed primitives -- is --backend a PASN backend?")

    entries = {n: new_entry(ne.router.n_banks) for n, _, ne in routed}
    saved = []
    for n, _, ne in routed:
        install_usage_probe(ne, entries[n], saved)

    with torch.no_grad():
        for i, probe in enumerate(probes):
            model(probe)
            print(f"[probe] sequence {i + 1}/{len(probes)}", flush=True)
    remove_probes(saved)

    rows = [site_stats(n, k, ne, entries[n]) for n, k, ne in routed]
    rows = [r for r in rows if r["elements"] > 0]
    agg = aggregate(rows)
    md = report(rows, agg)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    payload = dict(
        model=args.model, smoke=args.smoke, backend=args.backend,
        convert_ops=args.convert_ops, seq_len=seq_len, n_seq=args.n_seq,
        epochs=args.epochs, pasn_t_fixed=args.pasn_t_fixed,
        pasn_id_target=args.pasn_id_target,
        pasn_id_target_rel=args.pasn_id_target_rel,
        pasn_beta=dict(cfg.pasn_beta or {}), pasn_id_tied=bool(cfg.pasn_id_tied),
        pasn_e_min=int(cfg.pasn_e_min),
        n_sites=len(rows), n_degenerate=sum(1 for r in rows if r["degenerate"]),
        build_s=build_s, total_s=time.perf_counter() - t_start,
        aggregate=agg, sites=rows,
    )
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(f"# E11 -- bank utilisation audit ({args.model})\n\n" + md + "\n")

    print("\n" + md)
    print(f"\n[done] {len(rows)} sites, "
          f"{payload['n_degenerate']} degenerate, "
          f"{(time.perf_counter() - t_start) / 60:.1f} min -> {args.json}")


if __name__ == "__main__":
    main()
