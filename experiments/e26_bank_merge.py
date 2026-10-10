"""E26 -- coarsen the low-occupancy tail of the router and see what it costs.

idea-004 is the card; E25 established the premise it rests on. Bytes are
`c0 + c1*N` per bank with **no** `T` dependence, so the whole memory ratio
against MBE is the bank count `B` -- and `B` is the one capacity knob
`rule_budget` never sees (it returns `(N, T)`; `B` falls out of the router's
exponent span). The frozen build stores a median **7.3x** more basis capacity
than any single element uses, worst on `matmul.av.idn`: 8 banks, `n_eff` 1.09,
top-1 occupancy 0.988.

This script asks the obvious next question and tries hard to get a *no*.

**What is varied.** Banks whose calibration occupancy is at or below `theta` are
"weak". Contiguous runs of weak banks -- within one (kind, sign) run, so no group
straddles zero or a polarity split -- collapse into a single bank fitted over the
union of their key spans, with `rule_budget` re-solved on that wider range.
Strong banks are left exactly as they were. The router is **untouched**: same
`e_min`, same `e_max`, same index space, same routing arithmetic. Only how many
distinct neurons those indices point at changes (idea-004 "실패하는 방식" (f) --
merging and router-width clamping must not move at once or neither is
attributable).

**Why it can fail, stated before running.** `n_eff` is the effective number of
banks by *mass*, not by *error contribution*. Tail banks hold the small operands,
and the identity's budget is **relative**, so coarsening the tail attacks exactly
the budget that protects them. `MEMORY_OVERHEAD.md` §4 E23 predicted failure is
more likely than success; this script is written to let that happen.

**Two input distributions, on purpose.** Occupancy is a property of the input
draw, not of the neuron. `functions.sample` is uniform on the domain, which puts
~half the mass in the top binade by construction -- merging would look good for a
reason that has nothing to do with the method. The `logu` draw is log-uniform in
magnitude, spreading mass evenly across binades, which is the *hostile* case. A
result that only survives under `unif` is a sampler artefact and the record says
so.

⚠️ Operating point: **function level**, `span=6` binades (op_pareto's own),
`budget="rule"`, `target="relative"`. Not the network. The frozen build's
`matmul.av.idn` spans 16.9-19.9 binades, so `B` and `n_eff` are different numbers
there and these rows must not be quoted against it.

Usage::
    python experiments/e26_bank_merge.py --smoke        # 1 fn, 1 seed -- timing
    python experiments/e26_bank_merge.py                # the gate
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
import op_pareto as OP                                               # noqa: E402

FNS = ["gelu", "invsqrt", "inv", "exp2", "identity"]
# Measured occupancies on this grid run 1.7e-2 .. 0.49 (probe, 2026-08-18), so a
# threshold below 1e-2 marks nothing weak and reproduces the baseline exactly --
# useful as a null check but wasteful as a sweep point. These four bite.
THETAS = [0.0, 0.02, 0.05, 0.10, 0.20]    # 0.0 == baseline, no bank is weak
SEEDS = [0, 1, 2]
DISTS = ["unif", "logu"]
SPAN = 6                                   # op_pareto's binade count
TARGET_REL = 1e-2
EPOCHS = 250                               # op_pareto's, so fits stay comparable
M_EVAL = 4000


# ------------------------------------------------------------------ sampling

def draw(name, m, seed, dist):
    """Held-out input draw. ``unif`` is op_pareto's own; ``logu`` is the hostile one.

    ``logu`` samples the *magnitude* log-uniformly over the domain's binades, so
    every binade gets comparable mass and the router's tail is genuinely used.
    Under ``unif`` the top binade holds ~half the domain by construction, which
    would hand this experiment a favourable occupancy for free.
    """
    lo, hi = functions.REGISTRY[name][1]
    if dist == "unif":
        x, _, _ = functions.sample(name, m=m, seed=seed, domain=(lo, hi))
        return x
    g = torch.Generator().manual_seed(seed)
    mag_hi = max(abs(lo), abs(hi))
    mag_lo = max(min(abs(lo), abs(hi)), mag_hi * 2.0 ** -SPAN)
    if lo < 0 < hi:                       # both polarities present -> split evenly
        s = torch.where(torch.rand(m, generator=g) < 0.5, -1.0, 1.0)
    else:
        s = torch.full((m,), 1.0 if hi > 0 else -1.0)
    u = torch.rand(m, generator=g)
    mag = torch.exp(u * (math.log(mag_hi) - math.log(mag_lo)) + math.log(mag_lo))
    x = (s * mag).clamp(min=lo, max=hi)
    return torch.sort(x).values


# ------------------------------------------------------------------ grouping

def occupancy(router, x):
    idx = router.route(x.reshape(-1))
    n = x.numel()
    return [float((idx == bi).sum()) / n for bi in range(router.n_banks)]


def bank_target(router, fn, reach, bi):
    """``(tgt, lo, hi)`` in key space -- mirrors ``build_mbe_pasn``'s own closure.

    The test is ``kind == "near0"`` **exactly**, not ``startswith("near0")``.
    With ``near0="signed"`` the router emits ``near0s`` banks that carry a
    ``sign`` and are fed ``|key|`` just like magnitude banks (``raw`` is False);
    only the unsplit ``near0`` bank takes the raw signed key. Treating ``near0s``
    as raw drops the polarity and fits the negative near-zero bank to the
    *positive* target -- which is exactly the error floor ``near0="signed"``
    exists to remove, so the bug is invisible in aggregate and fatal in detail.
    """
    a, b = reach[bi]
    ta, tb = router.to_key(a), router.to_key(b)
    spec = router.banks[bi]
    if spec["kind"] == "near0":
        return (lambda v: fn(router.from_key(v))), min(ta, tb), max(ta, tb)
    s = spec["sign"]
    return ((lambda v, s=s: fn(router.from_key(s * v))),
            min(abs(ta), abs(tb)), max(abs(ta), abs(tb)))


def groups_for(router, occ, theta, reach):
    """Bank index -> group id. Weak runs collapse; strong banks stay singletons.

    A group never spans two ``(kind, sign)`` runs: near-zero banks are fed the raw
    signed key while magnitude banks are fed ``|key|``, and opposite-sign magnitude
    banks map through different targets, so a shared fit across either boundary
    would be fitting two different functions at once.
    """
    gid, out, run_key, open_weak = 0, {}, None, None
    for bi in range(router.n_banks):
        if bi not in reach:
            continue
        spec = router.banks[bi]
        key = (spec["kind"], spec.get("sign"))
        if key != run_key:                       # new run -> close any open group
            run_key, open_weak = key, None
        if occ[bi] <= theta:
            if open_weak is None:
                open_weak = gid
                gid += 1
            out[bi] = open_weak
        else:
            out[bi] = gid
            gid += 1
            open_weak = None
    return out


# ------------------------------------------------------------------- building

def build(name, dist, theta, seed, epochs, device="cpu", cache=None):
    """Routed neuron with weak-run banks sharing one fitted object.

    Returns ``(neuron, info)``. Fits are memoised on the group's *span* so the
    theta sweep re-fits only the groups it actually changed -- the strong banks
    are bit-identical across theta by construction, and re-fitting them would
    burn time and introduce noise where the design says there is none.
    """
    fn, domain = functions.REGISTRY[name]
    beta = OP.BETA.get(name, 0.0)
    e_min, e_max = OP._binades(*domain, span=SPAN, beta=beta)
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max,
                          near0="signed", beta=beta, gamma_log2=0)
    reach = {bi: sp for bi in range(router.n_banks)
             if (sp := router.reachable(bi, domain[0], domain[1])) is not None}

    xs = draw(name, M_EVAL, seed + 1, dist)
    occ = occupancy(router, xs)
    gmap = groups_for(router, occ, theta, reach)

    by_group = {}
    for bi, g in gmap.items():
        by_group.setdefault(g, []).append(bi)

    banks = [None] * router.n_banks
    n_fits = 0
    ra, rc = bit_law(1, True)
    for g, members in by_group.items():
        spec = router.banks[members[0]]
        # Every member of a group shares (kind, sign) by construction, so the
        # group's target map is any member's and its span is their union.
        tgt, _, _ = bank_target(router, fn, reach, members[0])
        spans = [bank_target(router, fn, reach, bi)[1:] for bi in members]
        lo = min(s[0] for s in spans)
        hi = max(s[1] for s in spans)
        ck = (name, beta, spec["kind"], spec.get("sign"), round(lo, 12),
              round(hi, 12), seed, epochs)
        if cache is not None and ck in cache:
            bank = cache[ck]
        else:
            y = tgt(_grid(lo, hi, 512, 0.0))
            eps = (TARGET_REL * max(float(y.pow(2).mean().sqrt()), 1e-30)) ** 2
            cap, steps = rule_budget(float(y.max() - y.min()), eps, n_max=4,
                                     a=ra, c=rc, t_grid=T_GRID)
            bank, _ = _fit_bank(tgt, lo, hi, cap, steps, epochs, seed, device,
                                alpha_init="auto", readout_order=1, learn_tau=True)
            n_fits += 1
            if cache is not None:
                cache[ck] = bank
        for bi in members:
            banks[bi] = bank

    neuron = MBEPASNNeuron(router, fill_unreachable(router, banks))
    distinct = len({id(b) for b in banks if b is not None})
    return neuron, dict(n_banks=router.n_banks, n_reached=len(reach),
                        n_weak=sum(1 for bi in reach if occ[bi] <= theta),
                        n_groups=len(by_group), n_distinct=distinct,
                        n_fits=n_fits, occupancy=occ,
                        n_eff=float(math.exp(-sum(p * math.log(p)
                                                  for p in occ if p > 0))))


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e26_bank_merge.json"))
    a = ap.parse_args()

    fns = FNS[:1] if a.smoke else FNS
    seeds = SEEDS[:1] if a.smoke else SEEDS
    dists = DISTS[:1] if a.smoke else DISTS
    thetas = THETAS[:2] if a.smoke else THETAS

    rows, t0 = [], time.time()
    for name in fns:
        for dist in dists:
            for seed in seeds:
                cache = {}
                xe = draw(name, M_EVAL, seed + 101, dist)   # scoring draw != occupancy draw
                ref = functions.REGISTRY[name][0](xe)
                for th in thetas:
                    t1 = time.time()
                    neuron, info = build(name, dist, th, seed, a.epochs, cache=cache)
                    with torch.no_grad():
                        out = neuron(xe)
                    err = OP.errors(out, ref)
                    row = dict(fn=name, dist=dist, seed=seed, theta=th,
                               nrmse=err["nrmse"], mse=err["mse"],
                               rel=err.get("rel"),
                               spikes=float(neuron.mean_spikes(xe)),
                               bytes=int(storage_bytes(neuron)),
                               stored_bases=int(neuron.stored_bases()),
                               fit_s=round(time.time() - t1, 1), **info)
                    row.pop("occupancy")
                    row["occ_top1"] = max(info["occupancy"])
                    rows.append(row)
                    print(f"{name:<9} {dist} s{seed} th={th:<6} "
                          f"B={info['n_distinct']:>2}/{info['n_reached']:<2} "
                          f"neff={info['n_eff']:.2f} nrmse={err['nrmse']:.3e} "
                          f"spk={row['spikes']:7.2f} byt={row['bytes']:>5} "
                          f"({row['fit_s']}s, {info['n_fits']} fits)", flush=True)

    # ---- gates
    cells, verdict = {}, {}
    for name in fns:
        for dist in dists:
            base = [r for r in rows if r["fn"] == name and r["dist"] == dist
                    and r["theta"] == 0.0]
            if len(base) < 2:
                continue
            spread = max(r["nrmse"] for r in base) / max(min(r["nrmse"] for r in base), 1e-30)
            b_err = st.median(r["nrmse"] for r in base)
            b_spk = st.median(r["spikes"] for r in base)
            b_byt = st.median(r["bytes"] for r in base)
            for th in thetas:
                if th == 0.0:
                    continue
                sel = [r for r in rows if r["fn"] == name and r["dist"] == dist
                       and r["theta"] == th]
                if not sel:
                    continue
                e = st.median(r["nrmse"] for r in sel)
                cells[f"{name}|{dist}|{th}"] = dict(
                    base_spread=spread, err_ratio=e / max(b_err, 1e-30),
                    spike_ratio=st.median(r["spikes"] for r in sel) / max(b_spk, 1e-30),
                    byte_ratio=b_byt / max(st.median(r["bytes"] for r in sel), 1e-30),
                    B_ratio=st.median(r["n_distinct"] for r in sel)
                    / max(st.median(r["n_distinct"] for r in base), 1e-30),
                    within_base_spread=bool(e / max(b_err, 1e-30) <= spread),
                )

    print("\n" + "=" * 78)
    print("GATE -- error vs the 3-seed baseline spread (R3), and what B bought")
    print("=" * 78)
    print(f"{'cell':<30} {'spread':>7} {'err':>8} {'spikes':>8} {'bytes':>7} "
          f"{'B':>6} {'R3':>5}")
    print("-" * 78)
    for k, v in cells.items():
        print(f"{k:<30} {v['base_spread']:>6.2f}x {v['err_ratio']:>7.2f}x "
              f"{v['spike_ratio']:>7.2f}x {v['byte_ratio']:>6.2f}x "
              f"{v['B_ratio']:>5.2f}x {'ok' if v['within_base_spread'] else 'NO':>5}")
    n_ok = sum(v["within_base_spread"] for v in cells.values())
    verdict = dict(cells_within_base_spread=n_ok, n_cells=len(cells),
                   note=("R3 asks whether coarsening the tail keeps the error inside "
                         "the seed noise it already had. R1 is decided at op level "
                         "against MBE and is NOT decided here."))
    print(f"\n  R3: {n_ok}/{len(cells)} cells inside the baseline seed spread.")

    rec = [dict(
        tag="e26-bank-merge" + ("-smoke" if a.smoke else ""),
        experiment="E26", idea="research/ideas/idea-004.md",
        backend="mbe_pasn", model="function-level",
        convert_cfg=None, pasn_beta=OP.BETA,
        stored_bytes=None, stored_params=None, n_primitives=None,
        smoke=bool(a.smoke), status="ok",
        supersedes=dict(
            record="results/e26_bank_merge_v1_invalid.json",
            reason=("the first run tested `kind.startswith('near0')` and treated "
                    "`near0s` banks as taking the raw signed key. They do not -- "
                    "they carry a `sign` and are fed |key| like magnitude banks "
                    "(`raw` is False), so the negative near-zero bank was fitted "
                    "to the *positive* target. That is exactly the error floor "
                    "`near0='signed'` exists to remove, so every row of that "
                    "record is invalid. Kept, not deleted."),
            found_by="E27's predicted-vs-measured consistency check (G5)"),
        started=_dt.datetime.now().isoformat(timespec="seconds"),
        total_s=round(time.time() - t0, 1),
        operating_point=dict(level="function", span_binades=SPAN,
                             budget="rule", target="relative",
                             target_rel=TARGET_REL, epochs=a.epochs,
                             m_eval=M_EVAL, readout_order=1,
                             router="unchanged -- merge only, no e_min/e_max clamp"),
        fns=fns, dists=dists, seeds=seeds, thetas=thetas,
        caveat=("occupancy is a property of the input draw. 'unif' is uniform on "
                "the domain and puts ~half the mass in the top binade by "
                "construction; 'logu' spreads mass across binades and is the "
                "hostile case. A result that holds only under 'unif' is a sampler "
                "artefact."),
        rows=rows, gate_cells=cells, verdict=verdict,
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.json}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
