"""E18 Stage 1 -- per-operator `net` energy, bracketed over residency and access class.

idea-005 is the card. The question is the lab meeting's feedback 3/4 in the only
form that can be answered: **does routing pay for itself, operator by operator,
once the router and the memory it touches are charged?**

    net = (spike energy saved) - (router energy + PASN access - MBE access)

all per **routed element**, all through `OP_ENERGY_PJ` (45 nm, ac 0.9 / mac 4.6 /
cmp 0.9 pJ) and Horowitz 45 nm access tiers.

**Why this is not just E7 re-run per op.** E7 charges PASN a full parameter read
for *every* element (`memory_cost_model.py:101`), computes the bank-switch
probability right above it, and then never uses it. That is a machine with no
cache at all. The same script measures the per-site bank-table footprint and
reports that a site's whole bank set is register/L0 class -- so the assumption
contradicts the measurement. Residency is therefore an **axis**, not a constant:

    R0  nothing resident          reads = active_bases * params_per_basis   (E7)
    R1  one active bank cached    reads = p_switch * (same)                 p_switch = 1 - sum p^2
    R2  site bank set resident    reads ~ 0                                 (what MBE already gets)

MBE's parameter reads are 0 under all three because a global neuron's bases are
loop-invariant and hoist. That is an *assumption*, and G3 checks it: its
footprint has to fit the same register file PASN's R2 would need.

**Provenance is per term, and the terms do not come from one level.** Written on
every row rather than buried:

* `cmp`, `mac`, `poly`, `router`, `bitop`, and all access terms -- **network
  level**, the frozen GPT-2 build's own occupancy (`bank_usage_audit`, 339 sites).
  These need no firing rate, so both methods are computable there: MBE's compare
  count is `N*T` per element with `N` from G.1.
* the **spike** term -- **operator level, iso-accuracy** (`op_pareto_merged`).
  It cannot come from the network: our MBE reimplementation diverges at every
  network setting (E2: ppl 611-589,287 vs ANN 21.71), so a network spike ratio
  divides a working system by a broken one (MEMORY_OVERHEAD §6).

The two levels differ (6 binades vs 16.9-19.9 for `matmul.av.idn`), so a row
whose sign depends on the spike term is **weaker** than one whose sign is already
fixed by the network-level terms alone. Both are printed, and `sign_source` says
which.

⚠️ This does not pick a dataflow or a residency. Choosing the one that flatters
us is the same error as the 5.27x headline (idea-005 §a).

Usage::  python experiments/e18_op_net_energy.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from mbe.metrics import OP_ENERGY_PJ                                  # noqa: E402

ACCESS_PJ = {"register": 0.1, "sram_8kb": 10.0, "sram_64kb": 20.0}
RESIDENCY = ("R0_none", "R1_one_bank_cached", "R2_bank_set_resident")

#: G.1: N=4 for GELU/Tanh, N=8 for exp / inv / invsqrt / identity. Same table E7
#: uses; this is the **paper's** MBE, not our reimplementation (2026-08-01).
PAPER_N = {"mlp.act": 4, "rsqrt": 8, "id_dev": 8, "id_istd": 8,
           "idn": 8, "idn2": 8, "exp": 8, "inv": 8}
MBE_T = 16

#: network op kind -> the operator-level bench that measures its iso-accuracy
#: spike ratio. GPT-2 uses GELU, and `matmul` is the activation x activation
#: product that `fp_multiply` benches.
OP_BENCH = {"activation": "activation:gelu", "layernorm": "layernorm",
            "softmax": "softmax", "matmul": "fp_multiply"}
MIN_MATCHED = 3          # G1: thinner cells are reported, not judged


def energy(ops):
    e = OP_ENERGY_PJ
    return (ops["ac"] * e["ac"] + (ops["mac"] + ops["poly"]) * e["mac"]
            + (ops["cmp"] + ops["router"]) * e["cmp"] + ops["bitop"] * e["bitop"])


# ---------------------------------------------------------------- spike term

def iso_spike_saving(recs, bench):
    """(MBE - PASN) spikes per element at the operator's iso-accuracy match.

    Median over the MBE spike-front points PASN can match, so one lucky pairing
    cannot carry it. Returns the spread too -- idea-005 §e refuses to read a sign
    off a net smaller than its own noise.
    """
    pts = [r for r in recs if r["op"] == bench]
    if not pts:
        return None
    err = pts[0]["primary"]
    mbe = [p for p in pts if p["backend"] == "mbe"]
    pasn = [p for p in pts if p["backend"] == "pasn_rule"]
    front, best = [], float("inf")
    for p in sorted(mbe, key=lambda r: r["spikes"]):
        if p[err] < best:
            front.append(p)
            best = p[err]
    diffs = []
    for m in front:
        ok = [c for c in pasn if c[err] <= m[err]]
        if ok:
            diffs.append(m["spikes"] - min(ok, key=lambda c: c["spikes"])["spikes"])
    if not diffs:
        return None
    diffs.sort()
    return dict(median=diffs[len(diffs) // 2], lo=diffs[0], hi=diffs[-1],
                matched=len(diffs), n_ref=len(front),
                thin=len(diffs) < MIN_MATCHED)


# --------------------------------------------------- network-level per-op terms

def op_terms(sites, params_per_basis, readout_order=1):
    """Per-op, per-routed-element op counts for both methods, from the audit.

    PASN's are occupancy-weighted over its own banks; MBE's follow G.1 (`N` from
    the role, `T=16`, linear readout). No firing rate is needed for any of these
    classes, which is why they are computable at network level for both sides --
    the spike class is not, and is handled separately.
    """
    by_op = {}
    for s in sites:
        by_op.setdefault(s["op"], []).append(s)

    out = {}
    for op, grp in by_op.items():
        el_tot = sum(s["elements"] for s in grp)
        acc = dict(p_cmp=0.0, p_mac=0.0, p_poly=0.0, p_bases=0.0, p_switch=0.0,
                   m_cmp=0.0, m_mac=0.0, m_bases=0.0, foot=[])
        for s in grp:
            n, t, o = s["n_per_bank"], s["t_per_bank"], s["occupancy"]
            el = s["elements"]
            if not n or not o or len(o) != len(n):
                continue
            tot = sum(o) or 1.0
            p = [x / tot for x in o]
            en = sum(pi * ni for pi, ni in zip(p, n))        # E[N] for this site
            et = sum(pi * ni * ti for pi, ni, ti in zip(p, n, t))   # E[N*T]
            acc["p_cmp"] += el * et
            acc["p_mac"] += el * en * readout_order
            acc["p_poly"] += el * en * (readout_order - 1)
            acc["p_bases"] += el * en
            acc["p_switch"] += el * (1.0 - sum(x * x for x in p))
            nm = PAPER_N[s["role"]]
            acc["m_cmp"] += el * nm * MBE_T
            acc["m_mac"] += el * nm * readout_order
            acc["m_bases"] += el * nm
            acc["foot"].append(s["stored_bases"] * params_per_basis * 4)
        if el_tot == 0:
            continue
        f = sorted(acc["foot"])
        out[op] = dict(
            elements=el_tot, sites=len(grp),
            pasn=dict(cmp=acc["p_cmp"] / el_tot, mac=acc["p_mac"] / el_tot,
                      poly=acc["p_poly"] / el_tot, router=6.0, bitop=1.0),
            mbe=dict(cmp=acc["m_cmp"] / el_tot, mac=acc["m_mac"] / el_tot,
                     poly=0.0, router=0.0, bitop=0.0),
            pasn_active_bases=acc["p_bases"] / el_tot,
            mbe_bases=acc["m_bases"] / el_tot,
            p_switch=acc["p_switch"] / el_tot,
            site_footprint_B=dict(median=f[len(f) // 2], max=f[-1]),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(ROOT, "results", "bank_usage_audit.json"))
    ap.add_argument("--pareto", default=os.path.join(ROOT, "results", "op_pareto_merged.json"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e18_op_net_energy.json"))
    a = ap.parse_args()

    audit = json.load(open(a.audit, encoding="utf-8"))
    ppb = 7.353711790393013            # params per basis, E7's measured constant
    terms = op_terms(audit["sites"], ppb)
    recs = json.load(open(a.pareto, encoding="utf-8"))["records"]

    # G3 -- MBE's zero reads are an assumption. Check its bases fit the same
    # register file R2 would need for PASN, and say so either way.
    g3 = {}
    for op, t in terms.items():
        mbe_B = t["mbe_bases"] * ppb * 4
        g3[op] = dict(mbe_resident_B=mbe_B,
                      pasn_R2_resident_B=t["site_footprint_B"]["median"],
                      mbe_fits_if_pasn_does=mbe_B <= t["site_footprint_B"]["median"])

    print("=" * 100)
    print("E18 Stage 1 -- per-op net energy, pJ per routed element. 45 nm, one price table.")
    print("=" * 100)
    print(f"{'op':<12}{'sites':>6}{'E[N] PASN':>10}{'N MBE':>7}{'p_switch':>10}"
          f"{'compute PASN':>14}{'compute MBE':>13}{'router':>8}")
    print("-" * 100)
    rows = {}
    for op in sorted(terms):
        t = terms[op]
        # compute-side energy WITHOUT the spike class (no firing rate needed)
        p_c = energy(dict(t["pasn"], ac=0.0))
        m_c = energy(dict(t["mbe"], ac=0.0))
        rows[op] = dict(pasn_compute_no_ac_pj=p_c, mbe_compute_no_ac_pj=m_c)
        print(f"{op:<12}{t['sites']:>6}{t['pasn_active_bases']:>10.2f}"
              f"{t['mbe_bases']:>7.1f}{t['p_switch']:>10.3f}"
              f"{p_c:>13.2f}p{m_c:>12.2f}p{6.0*OP_ENERGY_PJ['cmp']:>7.2f}p")

    print("\n\n## net per routed element, 3 residency x 3 access")
    print("   net = (spike saving, op-level iso-accuracy) + (MBE compute - PASN compute)")
    print("         - (PASN access - MBE access).   MBE access = 0 (loop-invariant, see G3)")
    print()
    hdr = f"{'op':<12}{'spike save':>11}{'cmp delta':>11}"
    for r in RESIDENCY:
        hdr += f"{r.split('_')[0]:>26}"
    print(hdr)
    print(f"{'':<12}{'pJ/elem':>11}{'pJ/elem':>11}" +
          "".join(f"{'reg / 8kb / 64kb':>26}" for _ in RESIDENCY))
    print("-" * 100)

    verdict = {}
    for op in sorted(terms):
        t = terms[op]
        bench = OP_BENCH.get(op)
        sp = iso_spike_saving(recs, bench) if bench else None
        # spikes -> ac ops: each spike costs 2 accumulate ops (metrics.py)
        spike_pj = (2.0 * sp["median"] * OP_ENERGY_PJ["ac"]) if sp else None
        cmp_delta = rows[op]["mbe_compute_no_ac_pj"] - rows[op]["pasn_compute_no_ac_pj"]

        reads = {
            "R0_none": t["pasn_active_bases"] * ppb,
            "R1_one_bank_cached": t["p_switch"] * t["pasn_active_bases"] * ppb,
            "R2_bank_set_resident": 0.0,
        }
        cells, line = {}, f"{op:<12}"
        line += (f"{spike_pj:>11.2f}" if spike_pj is not None else f"{'N/A':>11}")
        line += f"{cmp_delta:>11.2f}"
        for r in RESIDENCY:
            trio = []
            for acc, e in ACCESS_PJ.items():
                access = reads[r] * e
                net_hard = cmp_delta - access               # no spike term
                net_full = (net_hard + spike_pj) if spike_pj is not None else None
                cells[f"{r}|{acc}"] = dict(
                    access_pj=access, net_without_spike_term=net_hard,
                    net=net_full,
                    sign_source=("network-only" if net_hard > 0 else
                                 ("spike-term" if net_full and net_full > 0 else "negative")))
                trio.append("+" if (net_full or net_hard) > 0 else "-")
            line += f"{'  '.join(trio):>26}"
        print(line)
        verdict[op] = dict(spike_saving=sp, spike_saving_pj=spike_pj,
                           compute_delta_pj=cmp_delta, reads_per_elem=reads,
                           cells=cells,
                           all_positive=all(v["net"] is not None and v["net"] > 0
                                            for v in cells.values()),
                           any_negative=any(v["net"] is not None and v["net"] <= 0
                                            for v in cells.values()),
                           sign_stable=len({(v["net"] or 0) > 0
                                            for v in cells.values()}) == 1,
                           thin=bool(sp and sp["thin"]))

    n_all_pos = sum(v["all_positive"] for v in verdict.values())
    n_unstable = sum(not v["sign_stable"] for v in verdict.values())
    gate = ("R1_refuted_all_positive" if n_all_pos == len(verdict) else
            "R3_undecided_sign_flips" if n_unstable > len(verdict) / 2 else
            "R2_refuted_all_negative" if n_all_pos == 0 and n_unstable == 0 else
            "S_split_and_stable")

    print("\n" + "=" * 100)
    print(f"GATE: {gate}")
    print(f"  ops net>0 in every bracket : {n_all_pos}/{len(verdict)}")
    print(f"  ops whose sign flips       : {n_unstable}/{len(verdict)}")
    print("\n## G3 -- is MBE's zero-read assumption safe?")
    for op, g in g3.items():
        print(f"  {op:<12} MBE resident {g['mbe_resident_B']:>7.0f} B   "
              f"PASN R2 needs {g['pasn_R2_resident_B']:>7.0f} B   "
              f"{'ok' if g['mbe_fits_if_pasn_does'] else 'MBE needs MORE than PASN R2'}")

    rec = [dict(
        tag="e18-op-net-energy", experiment="E18-stage1",
        idea="research/ideas/idea-005.md",
        backend="analysis", model="gpt2-medium (frozen build audit) + op-level bench",
        convert_cfg=None, pasn_beta=audit.get("pasn_beta"),
        stored_bytes=None, stored_params=None, n_primitives=None,
        status="ok", started=_dt.datetime.now().isoformat(timespec="seconds"),
        sources=dict(audit=os.path.relpath(a.audit, ROOT),
                     pareto=os.path.relpath(a.pareto, ROOT),
                     params_per_basis=ppb),
        provenance=dict(
            network_level=["cmp", "mac", "poly", "router", "bitop", "access"],
            op_level_iso_accuracy=["spike saving"],
            why=("the spike term cannot come from the network: our MBE diverges "
                 "there (ppl 611-589,287 vs ANN 21.71), so a network spike ratio "
                 "divides a working system by a broken one. The other classes need "
                 "no firing rate and are computable at network level for both.")),
        price_table=OP_ENERGY_PJ, access_pj=ACCESS_PJ, residency=list(RESIDENCY),
        mbe_convention=dict(N=PAPER_N, T=MBE_T,
                            note="paper G.1, NOT our reimplementation (2026-08-01)"),
        per_op=verdict, terms=terms, gates=dict(G3_mbe_zero_reads=g3,
                                                G1_thin=[o for o, v in verdict.items()
                                                         if v["thin"]]),
        gate=gate,
        caveat=("residency and access class are hardware assumptions. This script "
                "brackets them and does not choose. Picking the flattering one is "
                "the 5.27x error (idea-005 §a)."),
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
