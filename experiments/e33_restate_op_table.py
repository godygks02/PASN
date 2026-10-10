"""E33 -- restate the operator table against a baseline we did not weaken.

`results/op_pareto_merged_wins.md` says PASN wins **10/10 operators on spikes,
1.23-5.39x**, and E30 says strict 3-axis domination is MBE 6 : PASN 3. Every one
of those numbers was measured against an `mbe` arm that fitted each primitive
**once**, while the routed arms have always tried two alpha placements and kept
the better (`mbe_pasn._fit_bank`). That is a handicap on the baseline with nothing
to do with routing, and E32 measured it as large: adding the `tau = dt/ln2`
initialisation to the global path cut the network profile's per-site error median
**16.6x**.

So the operator table has to be restated. This script does the restatement and
nothing else:

1. Takes the 240 routed records from `op_pareto_merged.json` **unchanged** --
   `FIT_TAU_SEARCH` lives in `spiking_ops` and the routed path never calls it, so
   those builds are bit-identical with the flag on or off. Only the baseline moved.
2. Swaps in the 150 re-fitted `mbe` records from `e33_op_pareto_mbe_tau.json`.
3. Recomputes the iso-accuracy medians and E30's strict 3-axis domination, and
   prints **before vs after** so the cost of honesty is visible rather than
   quietly absorbed.

⚠️ Expect the margins to shrink. That is the point: a win over a baseline we fit
worse than our own method is not a win. The restated numbers are the ones that
can go in a paper.

⚠️ Small regressions in individual baseline cells are expected and are not a bug.
`_fit_best` selects on a held-out draw and `op_pareto` reports on a third draw, so
keep-best can lose slightly on the reporting set. Measured range 0.86-0.99x on
5 of 150 cells.

Usage::  python experiments/e33_restate_op_table.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics as st
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import e30_pareto_3axis as E30                                       # noqa: E402

MBE = "mbe"
SHIPS = "pasn_rule"
AXES = ("bytes", "spikes", "err")


def front(pts, err):
    out, best = [], float("inf")
    for p in sorted(pts, key=lambda r: r["spikes"]):
        if p[err] < best:
            out.append(p)
            best = p[err]
    return out


def iso(recs, op, arm):
    """Iso-accuracy medians: for each MBE front point, the cheapest matching arm build."""
    pts = [r for r in recs if r["op"] == op]
    if not pts:
        return None
    err = pts[0]["primary"]
    ref = front([p for p in pts if p["backend"] == MBE], err)
    cand = [p for p in pts if p["backend"] == arm]
    sr, br = [], []
    for r in ref:
        ok = [c for c in cand if c[err] <= r[err]]
        if not ok:
            continue
        c = min(ok, key=lambda p: p["spikes"])
        sr.append(r["spikes"] / c["spikes"])
        br.append(r["bytes"] / c["bytes"])
    if not sr:
        return dict(matched=0, n_ref=len(ref), spikes=None, bytes=None)
    return dict(matched=len(sr), n_ref=len(ref),
                spikes=st.median(sr), bytes=st.median(br))


def domination(recs, op, arm):
    """E30's strict 3-axis counts, recomputed on whichever baseline is supplied."""
    pts = [E30.coords(r) for r in recs if r["op"] == op]
    mbe = [p for p in pts if p["backend"] == MBE]
    cand = [p for p in pts if p["backend"] == arm]
    if not (mbe and cand):
        return None
    fm, fc = E30.front(mbe), E30.front(cand)
    return dict(
        cand_beats_mbe=sum(1 for m in fm
                           if any(E30.dominates(p, m, strict_all=True) for p in cand)),
        of_mbe=len(fm),
        mbe_beats_cand=sum(1 for p in fc
                           if any(E30.dominates(m, p, strict_all=True) for m in mbe)),
        of_cand=len(fc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=os.path.join(ROOT, "results", "op_pareto_merged.json"))
    ap.add_argument("--new-mbe", default=os.path.join(ROOT, "results", "e33_op_pareto_mbe_tau.json"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e33_op_table_restated.json"))
    a = ap.parse_args()

    merged = json.load(open(a.merged, encoding="utf-8"))
    old = merged["records"]
    nm = json.load(open(a.new_mbe, encoding="utf-8"))
    new_mbe = nm["records"] if isinstance(nm, dict) else nm

    routed = [r for r in old if r["backend"] != MBE]
    old_mbe = [r for r in old if r["backend"] == MBE]
    k = lambda r: (r["op"], r["budget"], r["T"])
    assert {k(r) for r in new_mbe} == {k(r) for r in old_mbe}, "baseline grid moved"
    restated = routed + new_mbe

    ops = sorted({r["op"] for r in old})
    rows = {}
    print("=" * 96)
    print("E33 -- operator table restated against the strengthened baseline")
    print("=" * 96)
    print(f"{'operator':<22}{'spikes old':>12}{'spikes new':>12}{'byte old':>11}"
          f"{'byte new':>11}{'matched':>10}")
    print("-" * 96)
    win_old = win_new = 0
    sp_new = []
    for op in ops:
        o, n = iso(old, op, SHIPS), iso(restated, op, SHIPS)
        rows[op] = dict(old=o, new=n)
        if not (o and n and o["spikes"] and n["spikes"]):
            continue
        win_old += o["spikes"] > 1.0
        win_new += n["spikes"] > 1.0
        sp_new.append(n["spikes"])
        print(f"{op:<22}{o['spikes']:>11.2f}x{n['spikes']:>11.2f}x"
              f"{o['bytes']:>10.2f}x{n['bytes']:>10.2f}x"
              f"{str(n['matched']) + '/' + str(n['n_ref']):>10}")
    print("-" * 96)
    print(f"  spike wins: {win_old}/{len(ops)}  ->  **{win_new}/{len(ops)}**")
    if sp_new:
        s = sorted(sp_new)
        w = [x for x in s if x > 1.0]
        print(f"  winning range: {s[0]:.2f}x - {s[-1]:.2f}x"
              + (f"   winners' median {st.median(w):.2f}x" if w else ""))

    print("\n## strict 3-axis domination (E30), recomputed")
    print(f"{'operator':<22}{'PASN>MBE old':>14}{'PASN>MBE new':>14}"
          f"{'MBE>PASN old':>14}{'MBE>PASN new':>14}")
    print("-" * 96)
    dom = {}
    to, tn, mo, mn = 0, 0, 0, 0
    for op in ops:
        d_o, d_n = domination(old, op, SHIPS), domination(restated, op, SHIPS)
        dom[op] = dict(old=d_o, new=d_n)
        if not (d_o and d_n):
            continue
        to += d_o["cand_beats_mbe"]; tn += d_n["cand_beats_mbe"]
        mo += d_o["mbe_beats_cand"]; mn += d_n["mbe_beats_cand"]
        print(f"{op:<22}{d_o['cand_beats_mbe']:>14}{d_n['cand_beats_mbe']:>14}"
              f"{d_o['mbe_beats_cand']:>14}{d_n['mbe_beats_cand']:>14}")
    print("-" * 96)
    print(f"{'TOTAL':<22}{to:>14}{tn:>14}{mo:>14}{mn:>14}")

    rec = [dict(
        tag="e33-op-table-restated", experiment="E33",
        backend="analysis", model="op-level",
        convert_cfg=None, pasn_beta=None,
        stored_bytes=None, stored_params=None, n_primitives=None,
        status="ok", started=_dt.datetime.now().isoformat(timespec="seconds"),
        sources=dict(routed_from=os.path.relpath(a.merged, ROOT),
                     baseline_from=os.path.relpath(a.new_mbe, ROOT),
                     n_routed=len(routed), n_mbe=len(new_mbe)),
        what_changed=("only the `mbe` arm. FIT_TAU_SEARCH lives in spiking_ops and "
                      "the routed path never calls it, so the 240 routed records "
                      "are bit-identical and were reused unchanged."),
        iso_accuracy=rows, domination=dom,
        spike_wins=dict(old=win_old, new=win_new, n_ops=len(ops)),
        strict_domination_totals=dict(pasn_beats_mbe=dict(old=to, new=tn),
                                      mbe_beats_pasn=dict(old=mo, new=mn)),
        caveat=("small per-cell baseline regressions (0.86-0.99x on 5/150) are "
                "selection noise: _fit_best selects on a held-out draw and "
                "op_pareto reports on a third draw."),
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
