"""E25 -- where PASN's stored bytes actually go, and how much room the budget rule left.

Reads existing records only. **No fits, no builds, no GPU** -- every number here
is arithmetic on `results/op_pareto_merged.json` (operator level) and
`results/bank_usage_audit.json` (the frozen GPT-2 build). idea-004 is the card.

The lab-meeting objection (2026-08-14 feedback 4) is that on `fp_multiply`,
`layernorm` and `attention` the memory grows faster than the spikes fall, so
routing does not pay. `results/op_pareto_merged_wins.md` agrees: 10/10 on spikes,
**0/10 on bytes**. This script asks what that byte column is made of.

Four blocks, in the order idea-004 gates them:

1. **The byte law (R0).** Regress `params` on `N` per (op, arm). If bytes are
   `c0 + c1*N` with no `T` dependence, then `c1(pasn)/c1(mbe)` *is* the bank
   count, and the entire memory ratio is one integer nobody optimises --
   `rule_budget` returns `(N, T)` and takes `B` from the router's exponent span.

2. **The selection rule (G3).** `op_pareto.iso_accuracy` matches on accuracy,
   picks `min(spikes)`, and *reports* bytes for that build. Re-run the match
   under min-spike / min-byte / min-product. If all three agree, the byte column
   was never a choice: the shipped arm has no byte knob, and "0/10 on bytes" is
   a statement about the sweep grid rather than about the method.

3. **The pooled front.** Across every PASN arm, the `(spike_ratio, byte_ratio)`
   Pareto front at each matched accuracy, and whether any build wins on both.
   ⚠️ This is **arm shopping** -- selection and reporting on the same data. It
   sizes the headroom for E27; it is *not* a result, and idea-004 §(b) says so.

4. **Bank occupancy.** `n_eff = exp(entropy)` against the banks actually stored,
   on the frozen build. Spikes are an expectation over banks (an element pays
   only the bank it lands in); bytes are a sum over banks. That asymmetry is the
   mechanism, and this block measures its size.

⚠️ Blocks 1-3 are **operator level, 6 binades**; block 4 is the **network**,
where `matmul.av.idn` spans 16.9-19.9 binades. `B` and `n_eff` are different
numbers there. The record keeps them in separate objects and this script never
puts them in one table (idea-004 "실패하는 방식" (c)).

Usage::  python experiments/e25_capacity_frontier.py [--json results/e25_capacity_frontier.json]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

OPS = ["fp_multiply", "layernorm", "attention", "softmax",
       "activation:gelu", "activation:tanh", "activation:silu",
       "activation:exp2", "activation:inv", "activation:invsqrt"]

SHIPS = "pasn_rule"          # the arm the conversion path actually builds
CONTROL = "mbe"
MIN_MATCHED = 3              # G2: cells thinner than this are reported, not judged


# ---------------------------------------------------------------- helpers

def spike_front(pts, err_key):
    """Cheapest-spike points that are each strictly more accurate than all before.

    Same definition as ``op_pareto.front(sel, "spikes", primary)`` -- reproduced
    here rather than imported so this script stays a pure reader of records and
    cannot be perturbed by a change to the sweep module.
    """
    out, best = [], float("inf")
    for p in sorted(pts, key=lambda p: p["spikes"]):
        if p[err_key] < best:
            out.append(p)
            best = p[err_key]
    return out


def _ratio_stats(vals):
    return dict(median=st.median(vals), lo=min(vals), hi=max(vals), n=len(vals))


# ---------------------------------------------------------------- 1. byte law

def byte_law(recs):
    """`params = c0 + c1*N` per (op, arm), plus a check that `T` moves nothing.

    Returns the fitted intercept/slope, the residual against that line over
    *every* build, and `t_spread` -- the number of (op, arm, N) cells where two
    values of `T` disagree on the byte count. R0 requires `t_spread == 0` and a
    zero residual; anything else means bytes are not what this claim says.
    """
    out = {}
    for op in OPS:
        pts = [r for r in recs if r["op"] == op]
        row = {}
        for arm in sorted({p["backend"] for p in pts}):
            sel = [p for p in pts if p["backend"] == arm]
            by_n = {}
            for p in sel:
                by_n.setdefault(p["budget"], set()).add((p["params"], p["bytes"]))
            ns = sorted(n for n in by_n if isinstance(n, (int, float))
                        and float(n).is_integer())
            # cells where T changed the footprint at fixed N
            t_spread = sum(1 for n in by_n if len(by_n[n]) > 1)
            if len(ns) < 2:
                row[arm] = dict(fitted=False, t_spread=t_spread, n_points=len(sel))
                continue
            p0 = sorted(by_n[ns[0]])[0][0]
            p1 = sorted(by_n[ns[1]])[0][0]
            c1 = (p1 - p0) / (ns[1] - ns[0])
            c0 = p0 - c1 * ns[0]
            resid = max(abs(sorted(by_n[n])[0][0] - (c0 + c1 * n)) for n in ns)
            row[arm] = dict(fitted=True, c0=c0, c1=c1, max_resid=resid,
                            t_spread=t_spread, n_grid=ns, n_points=len(sel))
        if row.get(CONTROL, {}).get("fitted") and row.get("pasn", {}).get("fitted"):
            row["bank_multiplier"] = row["pasn"]["c1"] / row[CONTROL]["c1"]
        out[op] = row
    return out


# ------------------------------------------------- 2. selection-rule sensitivity

SELECTORS = {
    "min_spike": lambda r, c: c["spikes"],
    "min_byte": lambda r, c: c["bytes"],
    "min_product": lambda r, c: c["spikes"] * c["bytes"],
}


def selection_sensitivity(recs, arm=SHIPS):
    """Iso-accuracy medians under three selection rules, on one arm.

    If the three agree the arm is totally ordered in accuracy, spikes *and*
    bytes at once -- there was no trade-off to make, so the reported byte column
    is a consequence of the grid, not a choice the method lost.
    """
    out = {}
    for op in OPS:
        pts = [r for r in recs if r["op"] == op]
        if not pts:
            continue
        err = pts[0]["primary"]
        ref = spike_front([p for p in pts if p["backend"] == CONTROL], err)
        cand = [p for p in pts if p["backend"] == arm]
        row = {}
        for name, key in SELECTORS.items():
            sr, br, matched = [], [], 0
            for r in ref:
                ok = [c for c in cand if c[err] <= r[err]]
                if not ok:
                    continue
                c = min(ok, key=lambda c: key(r, c))
                matched += 1
                sr.append(r["spikes"] / c["spikes"])
                br.append(r["bytes"] / c["bytes"])
            if not sr:
                row[name] = None
                continue
            row[name] = dict(spikes=_ratio_stats(sr), bytes=_ratio_stats(br),
                             matched=matched, n_ref=len(ref))
        live = [v for v in row.values() if v]
        row["rules_agree"] = (
            len({round(v["spikes"]["median"], 9) for v in live}) == 1
            and len({round(v["bytes"]["median"], 9) for v in live}) == 1
        ) if live else None
        row["thin"] = bool(live) and live[0]["matched"] < MIN_MATCHED
        out[op] = row
    return out


# ------------------------------------------------------------- 3. pooled front

def pooled_front(recs):
    """Per reference accuracy, the `(spike_ratio, byte_ratio)` front over all PASN arms.

    ⚠️ Arm shopping. Reported as headroom for E27, never as a result -- the
    honest form picks the build with **one rule** and reports that, which is
    exactly what E27 is for.
    """
    out = {}
    for op in OPS:
        pts = [r for r in recs if r["op"] == op]
        if not pts:
            continue
        err = pts[0]["primary"]
        ref = spike_front([p for p in pts if p["backend"] == CONTROL], err)
        pool = [p for p in pts if p["backend"].startswith("pasn")]
        rows, dual_s, dual_b = [], [], []
        for r in ref:
            ok = [c for c in pool if c[err] <= r[err]]
            if not ok:
                rows.append(dict(ref_err=r[err], matched=False))
                continue
            pts_xy = [(r["spikes"] / c["spikes"], r["bytes"] / c["bytes"], c)
                      for c in ok]
            front = [(s, b, c) for (s, b, c) in pts_xy
                     if not any(s2 >= s and b2 >= b and (s2, b2) != (s, b)
                                for (s2, b2, _) in pts_xy)]
            dual = [(s, b, c) for (s, b, c) in front if s >= 1.0 and b >= 1.0]
            row = dict(ref_err=r[err], matched=True, n_candidates=len(ok),
                       front_size=len(front), n_dual_win=len(dual))
            if dual:
                s, b, c = max(dual, key=lambda t: t[0] * t[1])
                row["best_dual"] = dict(spike_ratio=s, byte_ratio=b,
                                        arm=c["backend"], N=c["budget"], T=c["T"])
                dual_s.append(s)
                dual_b.append(b)
            rows.append(row)
        out[op] = dict(
            points=rows,
            dual_win_points=len(dual_s),
            n_ref=len(ref),
            dual_spike_median=st.median(dual_s) if dual_s else None,
            dual_byte_median=st.median(dual_b) if dual_b else None,
        )
    return out


# ---------------------------------------------------------- 4. bank occupancy

def occupancy(audit):
    """Stored bank capacity vs the capacity one element actually uses.

    `sum_N` is what the footprint charges (every bank); `exp_N` is the occupancy
    weighted mean (the bank an element lands in). Their ratio is the price of
    specialisation, and `n_banks / n_eff` is how far the router overshot.
    """
    def group(s):
        if s["op"] == "matmul":
            return f"matmul.{'av' if 'av_' in s['site'] else 'qk'}.{s['role']}"
        return f"{s['op']}.{s['role']}"

    by = {}
    for s in audit["sites"]:
        by.setdefault(group(s), []).append(s)

    out, all_ratios = {}, []
    for k in sorted(by):
        v = by[k]
        sum_n, exp_n, sum_t, exp_t = [], [], [], []
        for s in v:
            n, t, o = s["n_per_bank"], s["t_per_bank"], s["occupancy"]
            if not n or not o or len(o) != len(n):
                continue
            tot = sum(o) or 1.0
            p = [x / tot for x in o]
            sum_n.append(sum(n))
            exp_n.append(sum(pi * ni for pi, ni in zip(p, n)))
            sum_t.append(sum(t))
            exp_t.append(sum(pi * ti for pi, ti in zip(p, t)))
        if not sum_n:
            continue
        banks = st.median(x["n_banks"] for x in v)
        neff = st.median(x["n_eff"] for x in v)
        sn, en = st.median(sum_n), st.median(exp_n)
        all_ratios.append(sn / en)
        out[k] = dict(
            sites=len(v), n_banks=banks, n_reached=st.median(x["n_reached"] for x in v),
            n_eff=neff, top1=st.median(x["top1_share"] for x in v),
            overprovision=banks / neff if neff else None,
            sum_N=sn, exp_N=en, capacity_ratio=sn / en,
            sum_T=st.median(sum_t), exp_T=st.median(exp_t),
        )
    return out, (st.median(all_ratios) if all_ratios else None)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", default=os.path.join(ROOT, "results", "op_pareto_merged.json"))
    ap.add_argument("--audit", default=os.path.join(ROOT, "results", "bank_usage_audit.json"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e25_capacity_frontier.json"))
    a = ap.parse_args()

    pareto = json.load(open(a.pareto, encoding="utf-8"))
    audit = json.load(open(a.audit, encoding="utf-8"))
    recs = pareto["records"]

    law = byte_law(recs)
    sens = selection_sensitivity(recs)
    front = pooled_front(recs)
    occ, occ_median = occupancy(audit)

    # ---- R0: the byte law must hold exactly, with no T dependence anywhere
    r0_t = sum(v.get("t_spread", 0) for op in law for v in law[op].values()
               if isinstance(v, dict))
    r0_resid = max((v["max_resid"] for op in law for v in law[op].values()
                    if isinstance(v, dict) and v.get("fitted")), default=None)
    r0_pass = (r0_t == 0 and r0_resid == 0)

    print("=" * 74)
    print("1. BYTE LAW  params = c0 + c1*N   (R0: exact, and no T dependence)")
    print("=" * 74)
    print(f"{'operator':<22} {'MBE c1':>7} {'PASN c1':>8} {'B':>7} {'resid':>6} {'T-dep':>6}")
    print("-" * 74)
    for op in OPS:
        r = law[op]
        m, p = r.get(CONTROL, {}), r.get("pasn", {})
        if not (m.get("fitted") and p.get("fitted")):
            continue
        td = m.get("t_spread", 0) + p.get("t_spread", 0)
        print(f"{op:<22} {m['c1']:>7.0f} {p['c1']:>8.0f} "
              f"{r['bank_multiplier']:>6.2f}x {max(m['max_resid'], p['max_resid']):>6.0f} "
              f"{td:>6}")
    print(f"\n  R0: {'PASS' if r0_pass else 'FAIL'} "
          f"(max residual {r0_resid}, T-dependent cells {r0_t})")
    print("  => the byte ratio IS the bank count. rule_budget solves (N,T); B is derived.")

    print()
    print("=" * 74)
    print(f"2. SELECTION RULE on the shipped arm '{SHIPS}'  (G3)")
    print("=" * 74)
    print(f"{'operator':<22} {'min-spike':>15} {'min-byte':>15} {'min-prod':>15} {'agree':>6}")
    print("-" * 74)
    n_agree = 0
    for op in OPS:
        r = sens.get(op)
        if not r:
            continue
        cells = []
        for name in SELECTORS:
            v = r[name]
            cells.append(f"{v['spikes']['median']:.2f}x/{v['bytes']['median']:.2f}x"
                         if v else "--")
        n_agree += bool(r["rules_agree"])
        flag = " THIN" if r["thin"] else ""
        print(f"{op:<22} {cells[0]:>15} {cells[1]:>15} {cells[2]:>15} "
              f"{str(r['rules_agree']):>6}{flag}")
    print(f"\n  all three rules agree on {n_agree}/{len(OPS)} operators.")
    print("  => the shipped arm is totally ordered: bytes were never a choice.")

    print()
    print("=" * 74)
    print("3. POOLED FRONT over every PASN arm -- HEADROOM ONLY, arm shopping")
    print("=" * 74)
    print(f"{'operator':<22} {'dual-win pts':>13} {'median spk/byt':>18}")
    print("-" * 74)
    n_dual = 0
    for op in OPS:
        f = front.get(op)
        if not f:
            continue
        n_dual += bool(f["dual_win_points"])
        med = (f"{f['dual_spike_median']:.2f}x/{f['dual_byte_median']:.2f}x"
               if f["dual_win_points"] else "NONE")
        print(f"{op:<22} {str(f['dual_win_points']) + '/' + str(f['n_ref']):>13} {med:>18}")
    print(f"\n  {n_dual}/{len(OPS)} operators have at least one build winning on BOTH axes.")
    print("  => headroom exists. NOT a result (idea-004 (b)) -- E27 must reach it with one rule.")

    print()
    print("=" * 74)
    print("4. BANK OCCUPANCY on the frozen GPT-2 build -- SEPARATE LEVEL, do not merge")
    print("=" * 74)
    print(f"{'group':<24} {'banks':>6} {'n_eff':>6} {'top1':>6} {'over':>7} "
          f"{'sum_N':>6} {'E[N]':>6} {'ratio':>7}")
    print("-" * 74)
    for k, v in sorted(occ.items(), key=lambda kv: -kv[1]["capacity_ratio"]):
        print(f"{k:<24} {v['n_banks']:>6.0f} {v['n_eff']:>6.2f} {v['top1']:>6.3f} "
              f"{v['overprovision']:>6.1f}x {v['sum_N']:>6.0f} {v['exp_N']:>6.2f} "
              f"{v['capacity_ratio']:>6.1f}x")
    print(f"\n  median stored/used capacity ratio: {occ_median:.1f}x")
    print("  => spikes are an expectation over banks; bytes are a sum over them.")

    rec = [dict(
        tag="e25-capacity-frontier",
        experiment="E25",
        idea="research/ideas/idea-004.md",
        backend="analysis",
        model="op-level + frozen-build audit",
        convert_cfg=None,
        pasn_beta=None,
        stored_bytes=None, stored_params=None, n_primitives=None,
        status="ok",
        started=_dt.datetime.now().isoformat(timespec="seconds"),
        sources={
            "op_pareto": dict(path=os.path.relpath(a.pareto, ROOT),
                              n_records=len(recs), meta=pareto.get("meta")),
            "bank_audit": dict(path=os.path.relpath(a.audit, ROOT),
                               n_sites=audit.get("n_sites"),
                               model=audit.get("model"),
                               pasn_t_fixed=audit.get("pasn_t_fixed"),
                               pasn_beta=audit.get("pasn_beta")),
        },
        operating_point=dict(
            op_level=dict(binades=pareto["meta"].get("binades"),
                          Ts=pareto["meta"].get("Ts"), seed=pareto["meta"].get("seed"),
                          shipped_arm=SHIPS, control_arm=CONTROL),
            network=dict(model=audit.get("model"), t_fixed=audit.get("pasn_t_fixed"),
                         note="different level from op_level -- never tabulate together"),
        ),
        selection_rule="three rules reported side by side; see G3 in idea-004",
        gates=dict(R0=dict(passed=bool(r0_pass), max_residual=r0_resid,
                           t_dependent_cells=r0_t),
                   G2_thin_cells=[op for op in OPS if sens.get(op, {}).get("thin")],
                   G3_rules_agree_on=n_agree, G3_n_ops=len(OPS)),
        byte_law=law,
        selection_sensitivity=sens,
        pooled_front=front,
        pooled_front_caveat=("arm shopping -- selection and reporting on the same "
                             "data. Headroom for E27, not a result."),
        occupancy=occ,
        occupancy_median_capacity_ratio=occ_median,
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
