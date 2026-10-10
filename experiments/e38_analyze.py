"""Read E38 records and answer: does N>1 ever earn its place?

Per bank and fit budget it reports
  * nesting violations: best-over-T MSE at N=k worse than at N=1. A converged
    fit cannot do this (N=k contains N=1), so the rate measures under-fitting;
  * gain: best MSE at N=1 / best MSE at max N (>1 means extra bases help);
  * the N the G1 selector would pick (spikes, then N, then T, then MSE) at
    several targets, re-run offline on the stored candidates. Banks whose
    target is relative (tied identity) keep their own stored target.

  python experiments/e38_analyze.py results/e38_n_basis_value_ro2.json ...
"""
from __future__ import annotations

import collections
import json
import statistics
import sys


def select(cands, eps):
    feas = [c for c in cands if c["mse"] <= eps]
    if feas:
        return min(feas, key=lambda c: (c["spikes"], c["N"], c["T"], c["mse"])), True
    return min(cands, key=lambda c: (c["mse"], c["spikes"], c["N"], c["T"])), False


def best_by_n(cands):
    out = {}
    for c in cands:
        out[c["N"]] = min(out.get(c["N"], float("inf")), c["mse"])
    return out


def main(paths, targets=(1e-3, 1e-4, 1e-5)):
    for path in paths:
        rec = json.load(open(path))
        print(f"\n=== {path}  status={rec['status']}")
        for row in rec["rows"]:
            if row.get("status") == "failed":
                print(f"{row['function']:9s} ep={row['epochs']} FAILED")
                continue
            sels = row["selection"]
            viol = gain_n = 0
            gains = []
            for s in sels:
                b = best_by_n(s["candidates"])
                nmax = max(b)
                if any(b[k] > b[1] for k in b if k > 1):
                    viol += 1
                gains.append(b[1] / max(b[nmax], 1e-30))
            picks = {}
            for eps in targets:
                ns, miss = collections.Counter(), 0
                for s in sels:
                    rel = s.get("bank") == "tied_magnitude" or row["function"] == "identity"
                    c, ok = select(s["candidates"], s["target_mse"] if rel else eps)
                    ns[c["N"]] += 1
                    miss += not ok
                picks[eps] = (dict(sorted(ns.items())), miss)
            g = sorted(gains)
            print(f"{row['function']:9s} ep={row['epochs']:4d} ro={row['readout_order']} "
                  f"banks={len(sels):2d} viol={viol:2d} "
                  f"gain(N1/Nmax) med={statistics.median(g):6.2f} max={g[-1]:8.2f} "
                  f"eval_mse={row['eval_mse']:.2e}")
            for eps, (ns, miss) in picks.items():
                print(f"{'':14s}target {eps:.0e}: N picks {ns}  infeasible={miss}")


if __name__ == "__main__":
    main(sys.argv[1:])
