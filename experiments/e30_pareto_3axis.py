"""E30 -- compare both methods at their own best, without picking a point for either.

The iso-accuracy table (`op_pareto_merged_wins.md`) pins one axis and reads the
other two off. That is the rigorous comparison, but it answers a narrow question:
*given MBE's accuracy, what does PASN cost?* It cannot answer *what is each method
capable of when tuned for itself*, and it hides what happens when one method
simply cannot reach the other's operating range.

The obvious alternative -- "take each method's best build" -- is not well posed.
"Best" over (bytes, spikes, error) needs a scalarisation, and choosing one after
seeing the data is arm shopping: whichever weighting flatters each side gets
picked, and the reader cannot tell. This project has already ruled that out twice
(idea-001 §5, idea-003 "이 런이 주장하지 않는 것").

So this script does the assumption-free version:

1. **Per-method 3-axis Pareto front.** A build survives if no other build of the
   *same method* is at least as cheap in bytes, at least as cheap in spikes, and
   at least as accurate, with one strict. No weighting is chosen, so no weighting
   can be gamed.
2. **Strict cross-method domination.** How many of MBE's front points are
   dominated by some PASN build on **all three axes at once**, and vice versa.
   This is the only comparison that needs no scalarisation and no matching -- if
   PASN strictly dominates a point, there is no weighting under which MBE's point
   is preferable.
3. **Each method's own extremes**, printed with all three coordinates so the
   trade is visible: cheapest storage, fewest spikes, best accuracy.

Both arms are swept over **their own knobs**, which is the point of the request:
MBE over `N in {1,2,4,6,8}` x `T in {4,8,16}`, PASN over its budget targets, its
`N`, and the same `T` grid. `T` is *not* pinned to 16 anywhere here.

⚠️ Two arms are reported for PASN and they answer different questions.
`pasn_rule` is **the method as it ships** (the conversion path builds this).
`pasn_any` pools every routed arm, which is the method's full configuration
space -- legitimate for "what is PASN capable of", but **it is not one
configuration**, so a number taken from it must say so. They are never merged.

⚠️ Operator level only. Network-level totals cannot be added here: our MBE
reimplementation diverges at every network setting we have run (E2: ppl
611-589,287 against ANN 21.71), so its network accuracy axis does not exist to
be optimised over. That gap is E15's, and it is a property of the
reimplementation, not of this analysis.

Usage::  python experiments/e30_pareto_3axis.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

AXES = ("bytes", "spikes", "err")
MBE = "mbe"
SHIPS = "pasn_rule"


def coords(r):
    return dict(bytes=float(r["bytes"]), spikes=float(r["spikes"]),
                err=float(r[r["primary"]]),
                label=f"{r['backend']} N={r['budget']} T={r['T']}",
                backend=r["backend"], budget=r["budget"], T=r["T"])


def dominates(a, b, strict_all=False):
    """`a` dominates `b`: no worse on every axis, strictly better on at least one.

    With ``strict_all`` the win must be strict on **all three** -- that is the
    form used for the cross-method claim, because a tie on an axis leaves room to
    argue the point is not really beaten.
    """
    if strict_all:
        return all(a[k] < b[k] for k in AXES)
    return all(a[k] <= b[k] for k in AXES) and any(a[k] < b[k] for k in AXES)


def front(points):
    return [p for p in points
            if not any(dominates(q, p) for q in points if q is not p)]


def extremes(points):
    out = {}
    for k in AXES:
        p = min(points, key=lambda r: (r[k], r["bytes"], r["spikes"], r["err"]))
        out[f"min_{k}"] = {a: p[a] for a in AXES} | dict(build=p["label"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", default=os.path.join(ROOT, "results", "op_pareto_merged.json"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e30_pareto_3axis.json"))
    a = ap.parse_args()

    d = json.load(open(a.pareto, encoding="utf-8"))
    recs = d["records"]
    ops = sorted({r["op"] for r in recs})

    out, summary = {}, []
    for op in ops:
        pts = [coords(r) for r in recs if r["op"] == op]
        arms = {
            MBE: [p for p in pts if p["backend"] == MBE],
            SHIPS: [p for p in pts if p["backend"] == SHIPS],
            "pasn_any": [p for p in pts if p["backend"].startswith("pasn")],
        }
        fronts = {k: front(v) for k, v in arms.items() if v}
        row = dict(n_builds={k: len(v) for k, v in arms.items()},
                   front_size={k: len(v) for k, v in fronts.items()},
                   extremes={k: extremes(v) for k, v in arms.items() if v})

        # strict 3-axis domination, both directions, against MBE's front
        for name in (SHIPS, "pasn_any"):
            if name not in fronts:
                continue
            mbe_dom = [m for m in fronts[MBE]
                       if any(dominates(p, m, strict_all=True) for p in arms[name])]
            pasn_dom = [p for p in fronts[name]
                        if any(dominates(m, p, strict_all=True) for m in arms[MBE])]
            row[f"{name}_strictly_dominates_mbe"] = dict(
                count=len(mbe_dom), of=len(fronts[MBE]),
                points=[{k: m[k] for k in AXES} | dict(build=m["build"]
                        if "build" in m else m["label"]) for m in mbe_dom[:4]])
            row[f"mbe_strictly_dominates_{name}"] = dict(
                count=len(pasn_dom), of=len(fronts[name]))
        out[op] = row
        summary.append((op, row))

    # ------------------------------------------------------------------ print
    print("=" * 92)
    print("E30 -- each method on its OWN front. T is swept, not pinned. No scalarisation.")
    print("=" * 92)
    print("\n## Each method's own extremes  (bytes / spikes / err at that build)")
    for op, row in summary:
        print(f"\n  {op}")
        for arm in (MBE, SHIPS):
            e = row["extremes"].get(arm)
            if not e:
                continue
            print(f"    {arm:<11}"
                  f" min-bytes {e['min_bytes']['bytes']:>6.0f}B "
                  f"{e['min_bytes']['spikes']:>7.2f}sp {e['min_bytes']['err']:>9.2e}"
                  f" | min-spikes {e['min_spikes']['bytes']:>6.0f}B "
                  f"{e['min_spikes']['spikes']:>7.2f}sp {e['min_spikes']['err']:>9.2e}"
                  f" | best-err {e['min_err']['bytes']:>6.0f}B "
                  f"{e['min_err']['spikes']:>7.2f}sp {e['min_err']['err']:>9.2e}")

    print("\n\n## Strict 3-axis domination -- cheaper storage AND fewer spikes AND more accurate")
    print(f"{'operator':<22}{'PASN(ships) beats MBE':>24}{'PASN(any) beats MBE':>22}"
          f"{'MBE beats PASN(ships)':>24}")
    print("-" * 92)
    tot_s = tot_a = tot_m = 0
    for op, row in summary:
        s = row.get(f"{SHIPS}_strictly_dominates_mbe", {})
        an = row.get("pasn_any_strictly_dominates_mbe", {})
        m = row.get(f"mbe_strictly_dominates_{SHIPS}", {})
        tot_s += s.get("count", 0)
        tot_a += an.get("count", 0)
        tot_m += m.get("count", 0)
        print(f"{op:<22}{s.get('count',0):>13}/{s.get('of',0):<10}"
              f"{an.get('count',0):>12}/{an.get('of',0):<9}"
              f"{m.get('count',0):>14}/{m.get('of',0):<9}")
    print("-" * 92)
    print(f"{'TOTAL front points':<22}{tot_s:>13}{'':<10}{tot_a:>12}{'':<9}{tot_m:>14}")
    print("\n  A strictly dominated point is beaten on all three axes at once, so no")
    print("  weighting of the three makes it preferable. Ties do not count.")

    rec = [dict(
        tag="e30-pareto-3axis", experiment="E30",
        backend="analysis", model="op-level",
        convert_cfg=None, pasn_beta=None,
        stored_bytes=None, stored_params=None, n_primitives=None,
        status="ok", started=_dt.datetime.now().isoformat(timespec="seconds"),
        source=dict(path=os.path.relpath(a.pareto, ROOT), n_records=len(recs),
                    grid=d["meta"]),
        method=("per-method 3-axis Pareto front over (bytes, spikes, error), each "
                "method swept over its own knobs. T is NOT pinned. Cross-method "
                "claim uses STRICT domination on all three axes, which needs no "
                "scalarisation and no accuracy matching."),
        arms=dict(mbe="N in {1,2,4,6,8} x T in {4,8,16}",
                  pasn_rule="the method as it ships (conversion path builds this)",
                  pasn_any=("every routed arm pooled -- the method's configuration "
                            "space, NOT one configuration. Never merge with "
                            "pasn_rule in a quoted number.")),
        caveat=("operator level only. Network totals cannot be added: our MBE "
                "reimplementation diverges at every network setting (E2: ppl "
                "611-589,287 vs ANN 21.71), so its network accuracy axis does not "
                "exist to optimise over. That is E15's gap."),
        per_op=out,
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
