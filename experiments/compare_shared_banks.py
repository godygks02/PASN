"""Flat MBE-PASN vs MBE-PASN-S: does one shared basis set span every range?

MBE-PASN-S keeps a single MBE basis set and lets the FP prefix pick only the
readout, so routing costs zero spikes and memory drops from ``sum_v 6 N_v`` to
``5N + R(N+1)``. The bet is that the per-range targets

    g_v(rho) = f(sigma 2^e (1+rho)),   rho in [0,1)

differ mostly in *scale*, so ``N`` shared bases span them all. If they differ in
*shape*, the shared basis cannot and flat banks win on accuracy.

This decides it: same router, matched budgets, and a **per-bank** MSE table
alongside the aggregate accuracy / spikes / memory and the worst bank-boundary
jump. All CPU.

Usage:  python experiments/compare_shared_banks.py --fns gelu
        python experiments/compare_shared_banks.py --json results/shared_banks.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import (MBENeuron, functions, fit_model, build_mbe_pasn,  # noqa: E402
                 build_mbe_pasn_s)
from mbe.metrics import neuron_cost, neuron_params  # noqa: E402

FN_SETUP = {
    "gelu":    dict(domain=(-8.0, 8.0), e_min=-2, e_max=4),
    "silu":    dict(domain=(-8.0, 8.0), e_min=-2, e_max=4),
    "inv":     dict(domain=(0.5, 1.0),  e_min=-2, e_max=0),
    "exp2":    dict(domain=(0.0, 1.0),  e_min=-4, e_max=0),
    "invsqrt": dict(domain=(0.5, 2.0),  e_min=-2, e_max=1),
}


def bank_label(spec) -> str:
    if spec["kind"] == "near0":
        return "near0"
    return f"{'neg' if spec['sign'] < 0 else 'pos'} e={spec['e']:+d}"


def bank_samples(router, domain, m: int, seed: int = 11):
    """Uniform test samples per routed range, clamped to the calibrated domain.

    Ranges the domain never reaches are skipped -- they are unfitted by
    construction, so scoring them would report noise, not error.
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for bi, spec in enumerate(router.banks):
        span = router.reachable(bi, domain[0], domain[1])
        if span is None:
            continue
        a, b = span
        out.append((bank_label(spec),
                    torch.rand(m, generator=g) * (b - a) + a))
    return out


@torch.no_grad()
def max_boundary_jump(model, e_min, e_max, domain, signed, d=1e-4) -> float:
    """Largest |f_hat(b-) - f_hat(b+)| over the internal range boundaries."""
    bnds = []
    for e in range(e_min, e_max):
        bnds.append(2.0 ** e)
        if signed:
            bnds.append(-(2.0 ** (e + 1)))
    bnds.append(2.0 ** e_min)
    if signed:
        bnds.append(-(2.0 ** e_min))
    worst = 0.0
    for b in sorted(set(b for b in bnds if domain[0] < b < domain[1])):
        yl = float(model(torch.tensor([b - d])))
        yr = float(model(torch.tensor([b + d])))
        worst = max(worst, abs(yr - yl))
    return worst


def _median_build(builder, seeds, xte, yte):
    """Build ``builder(seed)`` for every seed and return the median-MSE run.

    The surrogate-gradient fit is numerically chaotic on GELU/SiLU -- a
    strong-Wolfe LBFGS line search amplifies thread-level reduction-order noise
    into order-of-magnitude MSE swings -- so a single run is not a measurement.
    Carries the spread and the wall time of the median run.
    """
    runs = []
    for sd in seeds:
        t0 = time.perf_counter()
        model = builder(sd)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            mse = (model(xte) - yte).pow(2).mean().item()
        runs.append((mse, secs, model))
    runs.sort(key=lambda r: r[0])
    mid = runs[len(runs) // 2]
    return dict(model=mid[2], mse=mid[0], secs=mid[1],
                mse_lo=runs[0][0], mse_hi=runs[-1][0])


def run_fn(name, epochs, n_shared_list, seeds, rows, per_bank):
    s = FN_SETUP[name]
    dom, e_min, e_max = s["domain"], s["e_min"], s["e_max"]
    signed = dom[0] < 0.0
    fn, _ = functions.REGISTRY[name]
    xte, yte, _ = functions.sample(name, m=4000, seed=7, domain=dom)
    print(f"\n### {name}  domain={dom}  router e_min={e_min} e_max={e_max}")

    xtr, ytr, _ = functions.sample(name, m=4000, seed=0, domain=dom)

    def build_global(sd):
        m = MBENeuron(functions.make_config(name, n_basis=8, n_steps=16,
                                            domain=dom))
        fit_model(m, xtr, ytr, seed=sd, epochs=epochs)
        return m

    specs = {
        "MBE N=8 (global)": build_global,
        "MBE-PASN n_loc=2": lambda sd: build_mbe_pasn(
            name, dom, e_min=e_min, e_max=e_max, n_local=2, n_near0=4,
            epochs=epochs, seed=sd),
    }
    # The method as built: alpha placement and restart both chosen on the offset
    # grid. The a=u / a=L ablation lives in the report's history section.
    for N in n_shared_list:
        specs[f"MBE-PASN-S N={N}"] = lambda sd, N=N: build_mbe_pasn_s(
            name, dom, e_min=e_min, e_max=e_max, n_shared=N, n_steps=16,
            epochs=epochs, seed=sd)

    results = {tag: _median_build(b, seeds, xte, yte) for tag, b in specs.items()}
    banks = bank_samples(results["MBE-PASN n_loc=2"]["model"].router, dom, m=1500)

    print(f"\n{'model':22s} {'MSE(med)':>10s} {'[lo, hi]':>19s} {'spikes/in':>10s} "
          f"{'params':>7s} {'max|jump|':>10s} {'build s':>8s}")
    print("-" * 90)
    for tag, r in results.items():
        model = r["model"]
        c = neuron_cost(model, xte)
        p = neuron_params(model)
        jump = (float("nan") if tag.startswith("MBE N=")
                else max_boundary_jump(model, e_min, e_max, dom, signed))
        rows.append(dict(fn=name, model=tag, mse=r["mse"], mse_lo=r["mse_lo"],
                         mse_hi=r["mse_hi"], spikes=c["spikes"], params=p,
                         max_jump=jump, build_s=r["secs"]))
        span = f"[{r['mse_lo']:.1e}, {r['mse_hi']:.1e}]"
        print(f"{tag:22s} {r['mse']:10.2e} {span:>19s} {c['spikes']:10.2f} "
              f"{p:7d} {jump:10.2e} {r['secs']:8.1f}")

        with torch.no_grad():
            for label, xs in banks:
                pb = (model(xs) - fn(xs)).pow(2).mean().item()
                per_bank.setdefault((name, label), {})[tag] = pb

    # -- the decisive table -------------------------------------------------
    tags = [t for t in specs if not t.startswith("MBE N=")]
    print(f"\nper-bank MSE  ({name})")
    print(f"{'bank':12s} " + " ".join(f"{t:>22s}" for t in tags))
    print("-" * (13 + 23 * len(tags)))
    for label, _ in banks:
        vals = per_bank[(name, label)]
        cells = " ".join(f"{vals[t]:22.2e}" for t in tags)
        star = "  *" if (name in ("gelu", "silu") and label == "neg e=-1") else ""
        print(f"{label:12s} {cells}{star}")
    if name in ("gelu", "silu"):
        print("  * holds the non-monotone extremum of the target")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fns", nargs="+", default=["gelu", "silu", "exp2", "invsqrt"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-shared", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--seeds", type=int, default=3,
                    help="fits per configuration; the median-MSE run is reported")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    print("flat MBE-PASN vs MBE-PASN-S (shared basis, routed readout)")
    rows, per_bank = [], {}
    for fn in args.fns:
        run_fn(fn, args.epochs, args.n_shared, range(args.seeds), rows, per_bank)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dict(aggregate=rows,
                           per_bank=[dict(fn=k[0], bank=k[1], **v)
                                     for k, v in per_bank.items()]), f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
