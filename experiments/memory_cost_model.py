"""E7 -- price memory access, the axis our own framing invites and a rival claims.

Two things put this on the critical path. Adopting the MoE total/active contract
(PAPER_PLAN section 0) invites "what does a bank switch cost?", and NLSpike claims
zero data movement from in-core execution. Bank switching has worse locality than
a single global neuron, so **this is an axis we can lose**, and the completion
condition says to report the number either way.

The model has two access classes, because they point in opposite directions:

1. **Parameter reads.** A global MBE reads the *same* N bases for every element,
   so a compiler hoists them into registers once per tensor and the marginal cost
   per element is ~0. PASN reads whichever bank the router picked, which changes
   per element and **cannot be hoisted**. This is where we lose.

2. **State traffic.** Membrane `u` and accumulator `o` are per element per basis
   per timestep, so this scales with *active* bases -- 1.57 for PASN against the
   paper's 4/8, always all of them. This is where we win, and it is the larger
   term.

Both parameter sets are small enough to sit on-chip (PASN 52.6 KB, global MBE
37.3 KB), so the honest range for an access is register-to-SRAM, **not DRAM**.
Quoting a DRAM number here would be modelling a machine neither method needs.

Two dataflows bracket the answer, and which one a chip uses is not ours to
choose, so both are reported:

* **element-stationary** -- one element's state lives in registers for its whole
  T-step evolution. State traffic ~0; only parameter reads count. *Worst case for
  PASN.*
* **tensor-stationary** -- state is held in SRAM across the tensor. State traffic
  dominates. *Best case for PASN.*

Usage::  python experiments/memory_cost_model.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Appendix G.1: N=4 for GELU/Tanh, N=8 for exp / inv / invsqrt / identity.
PAPER_N = {"mlp.act": 4, "rsqrt": 8, "id_dev": 8, "id_istd": 8,
           "idn": 8, "idn2": 8, "exp": 8, "inv": 8}

# Horowitz, ISSCC 2014, 45 nm, 32-bit accesses. The same technology node the
# paper's G.4 energy constants come from (E_AC 0.9 pJ, E_MAC 4.6 pJ), so the two
# halves of the total are at least commensurable.
ACCESS_PJ = {"register": 0.1, "sram_8kb": 10.0, "sram_64kb": 20.0}
DRAM_PJ = 640.0          # reported for scale only -- neither method needs it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/bank_usage_audit.json")
    ap.add_argument("--pasn-bytes", type=int, default=53888)
    ap.add_argument("--pasn-params", type=int, default=13472)
    ap.add_argument("--mbe-bytes", type=int, default=38148,
                    help="global MBE storage, de-duplicated (P0.5)")
    # Frozen build (E0): energy_pj_per_input, and the audit's 2 sequences x 256.
    ap.add_argument("--compute-pj-per-token", type=float, default=6.0783e7)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--json", default="results/e7_memory_cost.json")
    args = ap.parse_args()

    audit = json.load(open(args.audit, encoding="utf-8"))
    sites = audit["sites"]
    params_per_basis = args.pasn_params / sum(s["stored_bases"] for s in sites)

    print(f"=== inputs ===")
    print(f"  sites {len(sites)}   routed elements "
          f"{sum(s['elements'] for s in sites):,}")
    print(f"  stored bases {sum(s['stored_bases'] for s in sites)}  "
          f"-> {params_per_basis:.2f} params/basis "
          f"({args.pasn_params} params total)")
    print(f"  on-chip? PASN {args.pasn_bytes/1024:.1f} KB, "
          f"global MBE {args.mbe_bytes/1024:.1f} KB "
          f"-- both fit in on-chip SRAM, so DRAM ({DRAM_PJ:g} pJ/access) "
          f"is not on either path")

    tot_el = 0
    p_par = p_state = 0.0          # PASN: parameter reads, state accesses
    m_par = m_state = 0.0          # global MBE, same
    switch_weighted = 0.0
    for s in sites:
        el, T = s["elements"], max(s["t_per_bank"])
        n_pasn = s["active_bases"]
        n_mbe = PAPER_N[s["role"]]
        occ = s["occupancy"]
        # Probability two independent consecutive elements land in different
        # banks. This is what "bank switching breaks locality" means, quantified.
        p_switch = 1.0 - sum(p * p for p in occ)

        tot_el += el
        switch_weighted += p_switch * el
        # Parameter reads. PASN pays per element (the bank changes); the global
        # neuron's are loop-invariant, so it pays them once per tensor -- rounded
        # to zero per element, which is the assumption most favourable to it.
        p_par += el * n_pasn * params_per_basis
        m_par += 0.0
        # State: read u, write u, write o, per basis per timestep.
        p_state += el * n_pasn * T * 3
        m_state += el * n_mbe * T * 3

    print(f"\n  mean bank-switch probability (element-weighted): "
          f"{switch_weighted / tot_el:.3f}")
    print(f"  active bases/element: PASN "
          f"{p_state / (tot_el * 3 * 16):.2f}  vs  global MBE "
          f"{m_state / (tot_el * 3 * 16):.2f}")

    # Which access class the router's table actually lands in is the question the
    # element-stationary bracket turns on, so measure the footprint rather than
    # assume it. A site's whole bank set is what would have to be resident.
    foot = sorted(s["stored_bases"] * params_per_basis * 4 for s in sites)
    print(f"\n  per-site bank-table footprint: median {foot[len(foot)//2]:.0f} B, "
          f"max {foot[-1]:.0f} B  (whole model {args.pasn_bytes/1024:.1f} KB)")
    print(f"    -> a site's bank set is register/L0 class; even all 339 fit in "
          f"one 64 KB SRAM")

    # Compute energy per routed element, so memory reads as a fraction of a
    # quantity we already report rather than as a bare pJ.
    compute_pj_per_elem = args.compute_pj_per_token / (tot_el / args.tokens)
    print(f"  compute energy: {args.compute_pj_per_token:.3e} pJ/token "
          f"/ {tot_el / args.tokens:,.0f} routed elem per token "
          f"= {compute_pj_per_elem:.2f} pJ per routed element")

    rows = []
    print(f"\n=== memory energy per routed element, and vs compute ===")
    print(f"{'dataflow':20s} {'access':10s} {'PASN pJ':>10s} {'MBE pJ':>10s} "
          f"{'MBE/PASN':>9s} {'PASN mem/compute':>17s}")
    for flow, use_state in (("element-stationary", False),
                            ("tensor-stationary", True)):
        for lab, e in ACCESS_PJ.items():
            p = (p_par + (p_state if use_state else 0.0)) * e / tot_el
            m = (m_par + (m_state if use_state else 0.0)) * e / tot_el
            ratio = (m / p) if p else float("nan")
            overhead = p / compute_pj_per_elem
            rows.append(dict(dataflow=flow, access=lab, access_pj=e,
                             pasn_pj=p, mbe_pj=m, ratio=ratio,
                             pasn_overhead_vs_compute=overhead))
            print(f"{flow:20s} {lab:10s} {p:>10.2f} {m:>10.2f} "
                  f"{ratio:>8.2f}x {overhead:>16.1%}")

    print("\n=== reading it ===")
    print("  * ELEMENT-STATIONARY is the bracket where we lose, and the loss is")
    print("    structural: a global neuron's parameters are loop-invariant, so")
    print("    they hoist and its marginal memory cost is ZERO. Any PASN cost")
    print("    loses against zero. The ratio column is meaningless there -- read")
    print("    the last column instead, which is the honest question: how much")
    print("    does routing add on top of compute we already price?")
    print("  * At register/L0 access -- which the footprint above supports, since")
    print("    a site's whole bank set is under a kilobyte -- that overhead is")
    print(f"    {rows[0]['pasn_overhead_vs_compute']:.1%}. If a bank read is")
    print("    instead a full SRAM access, it is the dominant cost and the")
    print("    method's energy story does not survive that dataflow.")
    print("  * TENSOR-STATIONARY: state traffic scales with ACTIVE bases (1.57 vs")
    print("    7.58), so routing wins by the same mechanism that saves spikes.")
    print("  * Neither method touches DRAM for parameters: both fit on-chip. The")
    print("    'zero data movement' axis a rival claims is therefore about access")
    print("    counts inside SRAM, not off-chip traffic -- and on that axis the")
    print("    comparison is the two rows above, not a claim either side wins")
    print("    outright.")

    out = dict(sites=len(sites), total_elements=tot_el,
               params_per_basis=params_per_basis,
               mean_switch_prob=switch_weighted / tot_el,
               pasn_param_reads=p_par, pasn_state_accesses=p_state,
               mbe_param_reads=m_par, mbe_state_accesses=m_state,
               pasn_bytes=args.pasn_bytes, mbe_bytes=args.mbe_bytes,
               access_pj=ACCESS_PJ, dram_pj=DRAM_PJ, rows=rows,
               paper_n=PAPER_N,
               source="results/bank_usage_audit.json (E11, 339 sites)")
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
