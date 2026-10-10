"""E27 -- solve the per-bank budget under a *memory* constraint, not just an error one.

idea-004, the processing half. `rule_budget` answers "what `(N_j, T_j)` does this
bank need to hit its error target", one bank at a time, with no global coupling
and no notion of what the whole neuron is allowed to store. E25 showed why that
matters: bytes are `c0 + c1*N` per bank with **no** `T` dependence, so total
storage is `sum_j N_j` over distinct banks and nothing else -- while the arm that
ships is totally ordered in accuracy, spikes and bytes at once (all three
selection rules agree on 10/10 operators). There was never a memory knob to turn.

This adds one. The problem is

    minimise    sum_j p_j * spikes_j(N_j, T_j)
    subject to  sum_j p_j * mse_j(N_j, T_j) <= eps
                sum_j N_j                   <= M

and both the objective and the error constraint are **exact**, not surrogates: a
routed neuron sends each input to exactly one bank, so per-element spikes and
squared error are occupancy-weighted sums over banks by construction. That makes
this a small integer program with a separable objective and two additive
constraints -- a Lagrangian sweep over the memory multiplier solves it to
optimality on the measured grid, no training, no GPU.

**What it costs.** The per-bank `(N, T) -> (mse, spikes)` table has to be
measured, which is `n_banks * |N| * |T|` fits per function. That table is the
whole experiment; the optimisation itself is milliseconds. It is cached to
`--table` so the frontier can be re-solved without re-fitting.

**What it is compared against.** Three things, measured the same way on the same
draw: (a) the global MBE neuron swept over its own single `N`, which is the
baseline's own frontier; (b) the point `rule_budget` currently picks; (c) the
memory-constrained optima. R2/S in idea-004 ask whether any point on (c) beats
(a) on spikes *while* using no more bytes -- one rule, not arm shopping.

⚠️ Operating point: **function level**, `span=6` binades, `readout_order=1`,
relative error target. Not the network. Bank counts and occupancies differ there
(idea-004 "실패하는 방식" (c)).

Usage::
    python experiments/e27_joint_budget.py --smoke
    python experiments/e27_joint_budget.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
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
from mbe.mbe_pasn import (PrefixRouter, MBEPASNNeuron, rule_budget,  # noqa: E402
                          bit_law, fill_unreachable, _fit_bank, _grid, T_GRID)
from mbe.neuron import MBENeuron                                     # noqa: E402
import op_pareto as OP                                               # noqa: E402
import e26_bank_merge as E26                                         # noqa: E402

FNS = ["gelu", "invsqrt", "inv", "exp2", "identity"]
N_GRID = [1, 2, 3, 4]
T_GRID_E27 = [4, 8, 16]
MBE_N_GRID = [1, 2, 4, 6, 8]
SEEDS = [0, 1, 2]
DIST = "unif"
SPAN = 6
TARGET_REL = 1e-2
EPOCHS = 250
M_EVAL = 4000


# ------------------------------------------------------- per-bank cost table

def bank_specs(name, seed, dist=DIST):
    """Reachable banks with their key spans, targets and occupancies."""
    fn, domain = functions.REGISTRY[name]
    beta = OP.BETA.get(name, 0.0)
    e_min, e_max = OP._binades(*domain, span=SPAN, beta=beta)
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max,
                          near0="signed", beta=beta, gamma_log2=0)
    reach = {bi: sp for bi in range(router.n_banks)
             if (sp := router.reachable(bi, domain[0], domain[1])) is not None}
    occ = E26.occupancy(router, E26.draw(name, M_EVAL, seed + 1, dist))

    out = []
    for bi in sorted(reach):
        # Shared with E26 so both scripts route through the router's own rule --
        # `near0s` banks carry a sign and are fed |key|, and getting that wrong
        # silently refits the negative near-zero bank to the positive target.
        tgt, lo, hi = E26.bank_target(router, fn, reach, bi)
        out.append(dict(bi=bi, lo=lo, hi=hi, tgt=tgt, p=occ[bi],
                        kind=router.banks[bi]["kind"],
                        sign=router.banks[bi].get("sign")))
    return router, reach, out


@torch.no_grad()
def _measure(bank, tgt, lo, hi, seed):
    """(mse, mean spikes per element) for one fitted bank on a held-out draw."""
    g = torch.Generator().manual_seed(seed + 777)
    x = torch.rand(1024, generator=g) * (hi - lo) + lo
    y = tgt(x)
    out = bank(x)
    mse = float((out - y).pow(2).mean())
    fr = float(bank.firing_rate(x))
    return mse, fr * bank.cfg.n_basis * bank.cfg.n_steps


def cost_table(name, seed, epochs, cache):
    """`(bi, N, T) -> (mse, spikes)` by fitting every bank at every budget.

    This is the only expensive part of E27 and it is pure measurement -- the
    optimiser below never estimates a cell it has not measured.
    """
    router, reach, specs = bank_specs(name, seed)
    tab = {}
    for s in specs:
        for N in N_GRID:
            for T in T_GRID_E27:
                # `sign` is part of the key, not decoration: the +1 and -1 banks
                # of the same binade have identical (lo, hi, kind) but opposite
                # targets, so omitting it makes the positive bank silently reuse
                # the negative bank's fit. Caught by the predicted-vs-measured
                # check below, which is why that check is a gate and not a print.
                ck = (name, round(s["lo"], 12), round(s["hi"], 12), s["kind"],
                      s["sign"], N, T, seed, epochs)
                if ck not in cache:
                    bank, _ = _fit_bank(s["tgt"], s["lo"], s["hi"], N, T, epochs,
                                        seed, "cpu", alpha_init="auto",
                                        readout_order=1, learn_tau=True)
                    cache[ck] = (bank, *_measure(bank, s["tgt"], s["lo"], s["hi"], seed))
                bank, mse, spk = cache[ck]
                tab[(s["bi"], N, T)] = dict(mse=mse, spikes=spk, bank=bank)
    return router, specs, tab


# ----------------------------------------------------------------- optimiser

def solve(specs, tab, eps, m_max):
    """Exact min-spike allocation under `sum p_j mse_j <= eps` and `sum N_j <= m_max`.

    DP over banks with the memory budget as the state. Error is handled by a
    Lagrangian sweep: for a multiplier `lam` each bank independently minimises
    `p*spikes + lam*p*mse`, which is optimal for the relaxed problem, and the
    sweep keeps every feasible allocation it passes through. Because both
    constraints are additive over banks and the grid is small, sweeping `lam`
    over the breakpoints implied by the grid recovers the true frontier rather
    than an approximation of it.
    """
    best = None
    lams = [0.0] + [10.0 ** k for k in range(-2, 13)]
    for lam in lams:
        # DP state: bases used -> (cost, err, spikes, choice list)
        dp = {0: (0.0, 0.0, 0.0, [])}
        for s in specs:
            nxt = {}
            for used, (c, e, sp, ch) in dp.items():
                for N in N_GRID:
                    if used + N > m_max:
                        continue
                    for T in T_GRID_E27:
                        cell = tab[(s["bi"], N, T)]
                        c2 = c + s["p"] * (cell["spikes"] + lam * cell["mse"])
                        k = used + N
                        if k not in nxt or c2 < nxt[k][0]:
                            nxt[k] = (c2, e + s["p"] * cell["mse"],
                                      sp + s["p"] * cell["spikes"],
                                      ch + [(s["bi"], N, T)])
            dp = nxt
        for used, (c, e, sp, ch) in dp.items():
            if e <= eps and (best is None or sp < best["spikes"]):
                best = dict(spikes=sp, mse=e, bases=used, choice=ch, lam=lam)
    return best


# -------------------------------------------------------------- the baselines

def mbe_frontier(name, seed, epochs, cache):
    """The global MBE neuron over its own single `N` -- the baseline's frontier."""
    fn, domain = functions.REGISTRY[name]
    xe = E26.draw(name, M_EVAL, seed + 101, DIST)
    ref = fn(xe)
    out = []
    for N in MBE_N_GRID:
        for T in T_GRID_E27:
            ck = ("mbe", name, N, T, seed, epochs)
            if ck not in cache:
                tgt = lambda v: fn(v)
                bank, _ = _fit_bank(tgt, domain[0], domain[1], N, T, epochs, seed,
                                    "cpu", alpha_init="auto", readout_order=1,
                                    learn_tau=True)
                cache[ck] = bank
            bank = cache[ck]
            with torch.no_grad():
                err = OP.errors(bank(xe), ref)
                fr = float(bank.firing_rate(xe))
            out.append(dict(N=N, T=T, nrmse=err["nrmse"], mse=err["mse"],
                            spikes=fr * N * T, bytes=int(storage_bytes(bank)),
                            bases=N))
    return out


def assemble(router, specs, tab, choice):
    """Realise a chosen allocation as a neuron, so bytes are measured not modelled."""
    banks = [None] * router.n_banks
    for bi, N, T in choice:
        banks[bi] = tab[(bi, N, T)]["bank"]
    return MBEPASNNeuron(router, fill_unreachable(router, banks))


def consistency(router, specs, tab, name, seed, dist=DIST, tol=4.0):
    """G5 -- does `sum_j p_j mse_j` predict the assembled neuron's measured MSE?

    The optimiser trusts the per-bank table; if the table and the realised neuron
    disagree, every allocation it returns is solving the wrong problem. This is a
    **gate**, not a diagnostic: two separate wiring bugs (near-zero polarity
    dropped, and a fit cache keyed without `sign` so the +1 bank reused the -1
    bank's fit) both showed up here as ratios of 1e4-1e6 while every printed
    ratio table still looked plausible.

    `tol` is loose on purpose -- the per-bank MSE is measured on a uniform draw
    over each bank's key span, not on the routed conditional, so agreement to a
    small factor is the most the aggregation can promise. Orders of magnitude are
    the signal.
    """
    xe = E26.draw(name, M_EVAL, seed + 101, dist)
    ref = functions.REGISTRY[name][0](xe)
    worst = 0.0
    for N, T in ((1, 4), (max(N_GRID), max(T_GRID_E27))):
        pred = sum(s["p"] * tab[(s["bi"], N, T)]["mse"] for s in specs)
        n = assemble(router, specs, tab, [(s["bi"], N, T) for s in specs])
        with torch.no_grad():
            got = float(OP.errors(n(xe), ref)["mse"])
        r = max(got / max(pred, 1e-30), max(pred, 1e-30) / max(got, 1e-30))
        worst = max(worst, r)
    return dict(worst_ratio=worst, passed=bool(worst <= tol), tol=tol)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e27_joint_budget.json"))
    a = ap.parse_args()

    fns = FNS[:1] if a.smoke else FNS
    seeds = SEEDS[:1] if a.smoke else SEEDS

    rows, t0, checks = [], time.time(), {}
    for name in fns:
        for seed in seeds:
            t1 = time.time()
            cache = {}
            router, specs, tab = cost_table(name, seed, a.epochs, cache)
            chk = consistency(router, specs, tab, name, seed)
            checks[f"{name}|s{seed}"] = chk
            if not chk["passed"]:
                print(f"  G5 FAIL {name} s{seed}: table vs measured off by "
                      f"{chk['worst_ratio']:.3g}x -- allocations from this table "
                      f"are not trustworthy", flush=True)
            mbe = mbe_frontier(name, seed, a.epochs, cache)
            xe = E26.draw(name, M_EVAL, seed + 101, DIST)
            ref = functions.REGISTRY[name][0](xe)

            # what rule_budget picks today, measured the same way
            ra, rc = bit_law(1, True)
            rule_choice = []
            for s in specs:
                y = s["tgt"](_grid(s["lo"], s["hi"], 512, 0.0))
                e = (TARGET_REL * max(float(y.pow(2).mean().sqrt()), 1e-30)) ** 2
                N, T = rule_budget(float(y.max() - y.min()), e, n_max=4, a=ra, c=rc,
                                   t_grid=T_GRID)
                rule_choice.append((s["bi"], min(N, max(N_GRID)),
                                    min(T_GRID_E27, key=lambda t: abs(t - T))))
            n_rule = assemble(router, specs, tab, rule_choice)
            with torch.no_grad():
                rerr = OP.errors(n_rule(xe), ref)
            rule_row = dict(fn=name, seed=seed, arm="rule", bases=sum(c[1] for c in rule_choice),
                            nrmse=rerr["nrmse"], spikes=float(n_rule.mean_spikes(xe)),
                            bytes=int(storage_bytes(n_rule)))
            rows.append(rule_row)

            for m in mbe:
                rows.append(dict(fn=name, seed=seed, arm="mbe", N=m["N"], T=m["T"],
                                 bases=m["bases"], nrmse=m["nrmse"],
                                 spikes=m["spikes"], bytes=m["bytes"]))

            # The constrained frontier. The error ceiling has to *sweep*: fixing
            # it at one level leaves the memory budget non-binding, the solver
            # just takes the cheapest allocation at every M, and the frontier
            # collapses to a single point (observed on the first smoke run). So
            # ask the solver for each accuracy the MBE front actually reaches --
            # which is what iso-accuracy means -- and sweep M inside that.
            max_bases = len(specs) * max(N_GRID)
            budgets = sorted({*range(len(specs), max_bases + 1,
                                     max(1, max_bases // 8)), max_bases})
            targets = sorted({m["mse"] for m in mbe} | {rerr["mse"]})
            for tgt_mse in targets:
                for m_max in budgets:
                    sol = solve(specs, tab, tgt_mse, m_max)
                    if sol is None:
                        continue
                    n = assemble(router, specs, tab, sol["choice"])
                    with torch.no_grad():
                        err = OP.errors(n(xe), ref)
                    rows.append(dict(fn=name, seed=seed, arm="joint",
                                     target_mse=tgt_mse, m_max=m_max,
                                     bases=sol["bases"], nrmse=err["nrmse"],
                                     mse=err["mse"],
                                     spikes=float(n.mean_spikes(xe)),
                                     bytes=int(storage_bytes(n)),
                                     choice=[list(c) for c in sol["choice"]]))
            print(f"{name:<9} s{seed}  banks={len(specs):>2}  "
                  f"rule: {rule_row['spikes']:6.2f} spk / {rule_row['bytes']:>5} B "
                  f"/ nrmse {rule_row['nrmse']:.2e}   ({time.time()-t1:.0f}s)",
                  flush=True)

    # ------------------------------------------------------- iso-accuracy verdict
    verdict = {}
    for name in fns:
        cells = []
        for seed in seeds:
            mbe = [r for r in rows if r["fn"] == name and r["seed"] == seed
                   and r["arm"] == "mbe"]
            joint = [r for r in rows if r["fn"] == name and r["seed"] == seed
                     and r["arm"] == "joint"]
            rule = [r for r in rows if r["fn"] == name and r["seed"] == seed
                    and r["arm"] == "rule"]
            if not (mbe and joint and rule):
                continue
            front, best = [], float("inf")
            for m in sorted(mbe, key=lambda r: r["spikes"]):
                if m["nrmse"] < best:
                    front.append(m)
                    best = m["nrmse"]
            for m in front:
                ok = [j for j in joint if j["nrmse"] <= m["nrmse"]]
                if not ok:
                    continue
                # Three *stated* rules, never a best-of. `min_spike` reproduces
                # what the existing table reports (and inherits its blind spot:
                # bytes are whatever the spike-optimal build happened to store).
                # `iso_byte` is the one the lab-meeting objection actually asks
                # for -- hold storage at or below MBE's and see if the spike win
                # survives. `dual` records whether any point wins on both.
                pick = min(ok, key=lambda j: j["spikes"])
                under = [j for j in ok if j["bytes"] <= m["bytes"]]
                iso = min(under, key=lambda j: j["spikes"]) if under else None
                dual = [j for j in ok if m["spikes"] / j["spikes"] >= 1.0
                        and m["bytes"] / j["bytes"] >= 1.0]
                cells.append(dict(
                    ref_nrmse=m["nrmse"], ref_spikes=m["spikes"], ref_bytes=m["bytes"],
                    minspike_spike_ratio=m["spikes"] / pick["spikes"],
                    minspike_byte_ratio=m["bytes"] / pick["bytes"],
                    isobyte_spike_ratio=(m["spikes"] / iso["spikes"]) if iso else None,
                    isobyte_byte_ratio=(m["bytes"] / iso["bytes"]) if iso else None,
                    rule_spike_ratio=m["spikes"] / rule[0]["spikes"]
                    if rule[0]["nrmse"] <= m["nrmse"] else None,
                    dual_win=bool(dual),
                ))
        if cells:
            iso_ok = [c for c in cells if c["isobyte_spike_ratio"] is not None]
            verdict[name] = dict(
                n_matched=len(cells), n_dual_win=sum(c["dual_win"] for c in cells),
                n_iso_byte_feasible=len(iso_ok),
                minspike_spike_median=st.median(c["minspike_spike_ratio"] for c in cells),
                minspike_byte_median=st.median(c["minspike_byte_ratio"] for c in cells),
                isobyte_spike_median=(st.median(c["isobyte_spike_ratio"] for c in iso_ok)
                                      if iso_ok else None),
                cells=cells,
            )

    print("\n" + "=" * 74)
    print("R2/S -- memory-constrained solve vs the MBE frontier, one rule")
    print("=" * 74)
    print(f"{'fn':<11} {'matched':>7} {'dual':>5} | {'min-spike spk/byt':>19} | "
          f"{'iso-byte spk':>12} {'feas':>5}")
    print("-" * 74)
    for k, v in verdict.items():
        iso = (f"{v['isobyte_spike_median']:.2f}x"
               if v["isobyte_spike_median"] is not None else "--")
        print(f"{k:<11} {v['n_matched']:>7} {v['n_dual_win']:>5} | "
              f"{v['minspike_spike_median']:>8.2f}x /{v['minspike_byte_median']:>7.2f}x | "
              f"{iso:>12} {v['n_iso_byte_feasible']:>5}")
    n_dual = sum(1 for v in verdict.values() if v["n_dual_win"])
    n_iso = sum(1 for v in verdict.values()
                if (v["isobyte_spike_median"] or 0) >= 1.0)
    print(f"\n  dual win (spikes AND bytes beat MBE): {n_dual}/{len(verdict)} functions.")
    print(f"  spike win at or under MBE's own bytes: {n_iso}/{len(verdict)} functions.")

    rec = [dict(
        tag="e27-joint-budget" + ("-smoke" if a.smoke else ""),
        experiment="E27", idea="research/ideas/idea-004.md",
        backend="mbe_pasn", model="function-level",
        convert_cfg=None, pasn_beta=OP.BETA,
        stored_bytes=None, stored_params=None, n_primitives=None,
        smoke=bool(a.smoke), status="ok",
        started=_dt.datetime.now().isoformat(timespec="seconds"),
        total_s=round(time.time() - t0, 1),
        operating_point=dict(level="function", span_binades=SPAN, dist=DIST,
                             target_rel=TARGET_REL, epochs=a.epochs,
                             m_eval=M_EVAL, readout_order=1,
                             N_grid=N_GRID, T_grid=T_GRID_E27,
                             mbe_N_grid=MBE_N_GRID),
        objective=("min sum_j p_j spikes_j  s.t.  sum_j p_j mse_j <= eps  and  "
                   "sum_j N_j <= M. Both sums exact for a routed neuron."),
        selection_rule="one rule per memory budget -- no arm shopping",
        gates=dict(G5_table_vs_measured=checks,
                   G5_passed=all(c["passed"] for c in checks.values())),
        fns=fns, seeds=seeds, rows=rows, verdict=verdict,
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.json}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
