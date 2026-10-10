"""E39b: on the banks that wanted N > 1, does a mantissa split do it at N = 1?

E38 found ``N > 1`` chosen only on wide-output top binades pinned at ``T = 8``
(gelu x in [1, 10], silu x in [1, 12]). E39 compares whole neurons at a few
targets; this zooms in on exactly those binades -- plus ``1/x`` on [0.5, 1)
*without* beta, the case where the exponent router has a single binade and is
useless -- and builds the full error / spikes / bytes frontier per binade:

  N-arm   one bank over the binade, N in 1..4,                   T in grid
  m-arm   2^m banks (top-m mantissa bits), N = 1 each, m = 1..3, T per sub-bank

For an error budget ``eps`` the selector rule is the deployed one: every bank must
reach ``eps`` on its own held-out grid; among feasible choices take the fewest
spikes. Binade spikes / MSE are width-weighted over sub-banks (uniform input).
Function level only.

  python experiments/e39b_bank_frontier.py --functions gelu \
      --json results/e39b_bank_frontier_gelu.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.mbe_pasn import _fit_bank, _grid  # noqa: E402
from mbe.metrics import storage_bytes  # noqa: E402

# Positive binades only (sign +, beta 0): the bank is fed x itself.
BINADES = {
    "gelu": [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 10.0)],   # E38 banks 15-18
    "silu": [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0)],   # E38 banks 12-15
    "inv": [(0.5, 1.0)],                                          # no beta: 1 binade
}
EPS = (1e-3, 1e-4, 1e-5, 1e-6)


def sub_ranges(lo, hi, m):
    """Nominal mantissa sub-ranges of the binade containing [lo, hi), clipped."""
    import math
    e = math.floor(math.log2(lo))
    w = 2.0 ** e / 2 ** m
    out = []
    for j in range(2 ** m):
        a, b = 2.0 ** e + j * w, 2.0 ** e + (j + 1) * w
        a, b = max(a, lo), min(b, hi)
        if b - a > 1e-12:
            out.append((a, b))
    return out


def fit_cell(fn, a, b, N, T, args):
    t0 = time.perf_counter()
    bank, _ = _fit_bank(fn, a, b, N, T, args.epochs, args.seed, "cpu",
                        alpha_init="auto", readout_order=args.readout_order)
    fit_s = time.perf_counter() - t0
    x = _grid(a, b, 1024, 0.75)
    with torch.no_grad():
        mse = float((bank(x) - fn(x)).pow(2).mean())
        spikes = float(bank.firing_rate(x)) * N * T
    return dict(lo=a, hi=b, N=N, T=T, mse=mse, spikes=spikes,
                bytes=int(storage_bytes(bank)), fit_s=round(fit_s, 3))


def best_for(cells_per_sub, widths, eps):
    """Fewest-spike choice with every sub-bank at or under eps (None if none)."""
    total = sum(widths)
    pick = []
    for cells in cells_per_sub:
        feas = [c for c in cells if c["mse"] <= eps]
        if not feas:
            return None
        pick.append(min(feas, key=lambda c: (c["spikes"], c["N"], c["T"])))
    w = [x / total for x in widths]
    return dict(
        spikes=sum(wi * c["spikes"] for wi, c in zip(w, pick)),
        mse=sum(wi * c["mse"] for wi, c in zip(w, pick)),
        bytes=sum(c["bytes"] for c in pick),
        T=[c["T"] for c in pick], N=[c["N"] for c in pick])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--functions", nargs="+", default=list(BINADES),
                   choices=list(BINADES))
    p.add_argument("--max-n", type=int, default=4)
    p.add_argument("--max-m", type=int, default=3)
    p.add_argument("--t-grid", nargs="+", type=int, default=[2, 3, 4, 6, 8])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--readout-order", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json", required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.json)
    rec = dict(experiment="e39b_bank_frontier", status="running",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:],
               eps=list(EPS), binades=[])
    for name in args.functions:
        fn = functions.REGISTRY[name][0]
        for lo, hi in BINADES[name]:
            entry = dict(function=name, lo=lo, hi=hi, arms={})
            arms = [("N%d" % n, n, 0) for n in range(1, args.max_n + 1)]
            arms += [("m%d" % m, 1, m) for m in range(1, args.max_m + 1)]
            for label, N, m in arms:
                subs = sub_ranges(lo, hi, m) if m else [(lo, hi)]
                cells = [[fit_cell(fn, a, b, N, T, args) for T in args.t_grid]
                         for a, b in subs]
                widths = [b - a for a, b in subs]
                entry["arms"][label] = dict(
                    N=N, m=m, n_banks=len(subs), cells=cells,
                    fit_s=round(sum(c["fit_s"] for cs in cells for c in cs), 2),
                    at_eps={f"{e:.0e}": best_for(cells, widths, e) for e in EPS})
                summ = " ".join(
                    f"{e:.0e}:" + ("none" if (r := entry["arms"][label]["at_eps"][f"{e:.0e}"]) is None
                                   else f"{r['spikes']:.2f}sp/{r['bytes']}B")
                    for e in EPS)
                print(f"{name:5s} [{lo:g},{hi:g}) {label:3s} banks={len(subs)} {summ}",
                      flush=True)
            rec["binades"].append(entry)
            out.write_text(json.dumps(rec, indent=1))
    rec["status"] = "ok"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
