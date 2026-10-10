"""E27b -- redo the storage floor with tying at its *shipped* setting.

E27 measured a floor of exactly the bank count (`gelu` 11.00x, `inv`/`exp2`/
`identity` 7.00x, `invsqrt` 2.00x against MBE's cheapest) and concluded that no
allocation can beat it, because every bank must store at least one basis so
`sum_j N_j >= B`.

**That conclusion does not hold for the identity, and the identity is the case
that matters.** E27 called `_fit_bank` per bank and never went through the
`tied=True` path, but `ConvertConfig.pasn_id_tied` is **on by default** and
`op_pareto.TIED_PRIMS == {"identity"}`. For a positively homogeneous target the
magnitude banks share **one** fitted prototype and differ only by a readout scale
the router already computed, so the floor is not `B` bases but

    1 (the shared prototype)  +  the near-zero banks, which are not homogeneous

and the identity is the largest single consumer of spikes in a converted
Transformer (45.1% on GPT-2), with the worst measured overprovisioning
(`matmul.av.idn`: 8 banks, `n_eff` 1.09). So E27 got the most important cell
wrong, in the direction that flatters nobody: it made PASN look *worse* on
memory than the shipped build actually is.

This script re-measures with the tying policy the conversion path uses, and runs
an explicit `tied=False` arm beside it so the delta is attributable rather than
inferred. Both arms go through `op_pareto.pasn`, which is the builder the sweep
and the conversion path already share -- E27's hand-rolled bank loop is exactly
where the two wiring bugs (near-zero polarity, sign-less fit cache) came from.

⚠️ Operating point: **function level**, `span=6` binades, uniform draw,
`budget="fixed"` so `N` is swept rather than solved. Not the network. The tied
saving scales with the number of magnitude binades, and the network's
`matmul.av.idn` spans 16.9-19.9 of them against this bench's 6 -- so these rows
are a **lower bound** on what tying is worth there, not an estimate of it.

Usage::
    python experiments/e27b_tied_floor.py --smoke
    python experiments/e27b_tied_floor.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics as st
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mbe import functions                                            # noqa: E402
from mbe.metrics import storage_bytes                                # noqa: E402
from mbe.mbe_pasn import _fit_bank                                   # noqa: E402
import op_pareto as OP                                               # noqa: E402
import e26_bank_merge as E26                                         # noqa: E402

FNS = ["gelu", "invsqrt", "inv", "exp2", "identity"]
N_GRID = [1, 2, 3, 4]
T_GRID = [4, 8, 16]
MBE_N_GRID = [1, 2, 4, 6, 8]
SEEDS = [0, 1, 2]
SPAN = 6
EPOCHS = 250
M_EVAL = 4000
DIST = "unif"


def measure(neuron, xe, ref):
    with torch.no_grad():
        err = OP.errors(neuron(xe), ref)
        spk = float(neuron.mean_spikes(xe))
    return dict(nrmse=err["nrmse"], mse=err["mse"], spikes=spk,
                bytes=int(storage_bytes(neuron)),
                stored_bases=int(neuron.stored_bases()))


def mbe_points(name, seed, epochs, xe, ref, cache):
    """Global MBE neuron over its own single `N` -- the baseline's frontier."""
    fn, domain = functions.REGISTRY[name]
    out = []
    for N in MBE_N_GRID:
        for T in T_GRID:
            ck = ("mbe", name, N, T, seed, epochs)
            if ck not in cache:
                cache[ck] = _fit_bank(lambda v: fn(v), domain[0], domain[1], N, T,
                                      epochs, seed, "cpu", alpha_init="auto",
                                      readout_order=1, learn_tau=True)[0]
            bank = cache[ck]
            with torch.no_grad():
                err = OP.errors(bank(xe), ref)
                fr = float(bank.firing_rate(xe))
            out.append(dict(arm="mbe", N=N, T=T, nrmse=err["nrmse"], mse=err["mse"],
                            spikes=fr * N * T, bytes=int(storage_bytes(bank)),
                            stored_bases=N))
    return out


def front(pts):
    out, best = [], float("inf")
    for p in sorted(pts, key=lambda r: r["spikes"]):
        if p["nrmse"] < best:
            out.append(p)
            best = p["nrmse"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e27b_tied_floor.json"))
    a = ap.parse_args()

    fns = ["identity"] if a.smoke else FNS
    seeds = SEEDS[:1] if a.smoke else SEEDS
    ngrid = N_GRID[:2] if a.smoke else N_GRID

    rows, t0 = [], time.time()
    for name in fns:
        shipped_tied = name in OP.TIED_PRIMS
        _, domain = functions.REGISTRY[name]
        for seed in seeds:
            xe = E26.draw(name, M_EVAL, seed + 101, DIST)
            ref = functions.REGISTRY[name][0](xe)
            cache = {}
            rows += [dict(fn=name, seed=seed, **m)
                     for m in mbe_points(name, seed, a.epochs, xe, ref, cache)]
            for tied in (False, True):
                if tied and not shipped_tied:
                    continue          # only homogeneous targets can tie at all
                t1 = time.time()
                for N in ngrid:
                    for T in T_GRID:
                        OP._PASN_CACHE.clear()   # keep memory flat across the grid
                        n = OP.pasn(name, domain, N, T, a.epochs, seed, SPAN,
                                    tied=tied)
                        rows.append(dict(fn=name, seed=seed,
                                         arm="pasn_tied" if tied else "pasn_untied",
                                         N=N, T=T, tied=tied,
                                         n_banks=n.router.n_banks,
                                         **measure(n, xe, ref)))
                print(f"  {name:<9} s{seed} tied={str(tied):<5} "
                      f"({time.time() - t1:.0f}s)", flush=True)

    # ------------------------------------------------------------- the floor
    floor = {}
    for name in fns:
        r = {}
        for arm in ("pasn_untied", "pasn_tied", "mbe"):
            sel = [x for x in rows if x["fn"] == name and x["arm"] == arm]
            if sel:
                r[arm] = dict(min_bytes=min(x["bytes"] for x in sel),
                              min_bases=min(x["stored_bases"] for x in sel),
                              n_banks=sel[0].get("n_banks"))
        if "mbe" in r:
            for arm in ("pasn_untied", "pasn_tied"):
                if arm in r:
                    r[arm]["floor_ratio"] = r[arm]["min_bytes"] / r["mbe"]["min_bytes"]
        floor[name] = r

    print("\n" + "=" * 78)
    print("THE FLOOR -- cheapest storage any build on the grid can reach")
    print("=" * 78)
    print(f"{'fn':<10} {'banks':>6} {'untied B':>9} {'tied B':>8} {'MBE B':>7} "
          f"{'untied':>8} {'tied':>8}")
    print("-" * 78)
    for name in fns:
        r = floor[name]
        u, t, m = r.get("pasn_untied"), r.get("pasn_tied"), r.get("mbe")
        tied_bytes = str(t["min_bytes"]) if t else "--"
        tied_ratio = f"{t['floor_ratio']:.2f}x" if t else "--"
        print(f"{name:<10} {u['n_banks']:>6} {u['min_bytes']:>9} {tied_bytes:>8} "
              f"{m['min_bytes']:>7} {u['floor_ratio']:>7.2f}x {tied_ratio:>8}")

    # ------------------------------------------------- iso-accuracy, 3 rules
    verdict = {}
    for name in fns:
        for arm in ("pasn_untied", "pasn_tied"):
            cells = []
            for seed in seeds:
                mbe = [x for x in rows if x["fn"] == name and x["seed"] == seed
                       and x["arm"] == "mbe"]
                cand = [x for x in rows if x["fn"] == name and x["seed"] == seed
                        and x["arm"] == arm]
                if not (mbe and cand):
                    continue
                for m in front(mbe):
                    ok = [c for c in cand if c["nrmse"] <= m["nrmse"]]
                    if not ok:
                        continue
                    pick = min(ok, key=lambda c: c["spikes"])
                    under = [c for c in ok if c["bytes"] <= m["bytes"]]
                    iso = min(under, key=lambda c: c["spikes"]) if under else None
                    dual = [c for c in ok if m["spikes"] / c["spikes"] >= 1.0
                            and m["bytes"] / c["bytes"] >= 1.0]
                    cells.append(dict(
                        minspike_spike=m["spikes"] / pick["spikes"],
                        minspike_byte=m["bytes"] / pick["bytes"],
                        isobyte_spike=(m["spikes"] / iso["spikes"]) if iso else None,
                        dual_win=bool(dual)))
            if cells:
                iso_ok = [c for c in cells if c["isobyte_spike"] is not None]
                verdict[f"{name}|{arm}"] = dict(
                    n_matched=len(cells),
                    n_dual_win=sum(c["dual_win"] for c in cells),
                    n_iso_feasible=len(iso_ok),
                    minspike_spike=st.median(c["minspike_spike"] for c in cells),
                    minspike_byte=st.median(c["minspike_byte"] for c in cells),
                    isobyte_spike=(st.median(c["isobyte_spike"] for c in iso_ok)
                                   if iso_ok else None))

    print("\n" + "=" * 78)
    print("ISO-ACCURACY vs MBE -- three stated rules, no best-of")
    print("=" * 78)
    print(f"{'fn|arm':<26} {'matched':>7} {'dual':>5} {'spk/byt':>16} "
          f"{'iso-byte spk':>13} {'feas':>5}")
    print("-" * 78)
    for k, v in verdict.items():
        iso = f"{v['isobyte_spike']:.2f}x" if v["isobyte_spike"] is not None else "--"
        print(f"{k:<26} {v['n_matched']:>7} {v['n_dual_win']:>5} "
              f"{v['minspike_spike']:>7.2f}x/{v['minspike_byte']:>6.2f}x "
              f"{iso:>13} {v['n_iso_feasible']:>5}")

    rec = [dict(
        tag="e27b-tied-floor" + ("-smoke" if a.smoke else ""),
        experiment="E27b", idea="research/ideas/idea-004.md",
        backend="mbe_pasn", model="function-level",
        convert_cfg=None, pasn_beta=OP.BETA,
        stored_bytes=None, stored_params=None, n_primitives=None,
        smoke=bool(a.smoke), status="ok",
        corrects=dict(
            record="results/e27_joint_budget.json",
            reason=("E27 never took the tied=True path, but pasn_id_tied is on by "
                    "default and TIED_PRIMS == {'identity'}. Its identity floor "
                    "(7.00x) is therefore an untied variant, not the shipped "
                    "build, and the identity is 45.1% of GPT-2's spikes. E27's "
                    "untied rows remain valid *as an untied ablation*."),
            raised_by="user, 2026-08-18"),
        started=_dt.datetime.now().isoformat(timespec="seconds"),
        total_s=round(time.time() - t0, 1),
        operating_point=dict(level="function", span_binades=SPAN, dist=DIST,
                             budget="fixed", epochs=a.epochs, m_eval=M_EVAL,
                             readout_order=1, N_grid=ngrid, T_grid=T_GRID,
                             mbe_N_grid=MBE_N_GRID,
                             tied_prims=sorted(OP.TIED_PRIMS),
                             builder="op_pareto.pasn (the shipped path)"),
        caveat=("the tied saving grows with the number of magnitude binades; this "
                "bench has 6 while the network's matmul.av.idn spans 16.9-19.9, so "
                "these are a lower bound on the network, not an estimate of it."),
        fns=fns, seeds=seeds, rows=rows, floor=floor, verdict=verdict,
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {a.json}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
