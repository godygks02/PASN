"""E38: does a bank's second basis buy anything, or was G1's all-N=1 an under-fit?

The GPT-2-medium G1 adaptive gate (``results/adaptive_g1_medium_t8.json``) picked
``N=1`` in all 1,302 per-site entries. But it searched only ``N in {1, 2}`` at
10 epochs, and in 1,254 of those entries the best ``N=2`` fit had *higher* held-out
MSE than the best ``N=1`` fit -- impossible for a converged fit, since ``N=2``
nests ``N=1`` (zero the second basis). So the selection may be reading the
optimiser, not the function.

This sweeps the full ``(N, T)`` grid per bank with ``budget="fit"`` (the same
selector G1 used) at the G1 routing and readout, for several fit-epoch budgets,
and keeps every candidate so the selection can be re-run offline at any target.
Function level only: no network, energy or latency claim.

  python experiments/e38_n_basis_value.py --epochs 10 100 300 \
      --json results/e38_n_basis_value.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.convert import _erange_beta  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402

# G1 routing: pasn_e_min=-3, pasn_id_e_min=-6, tied relative identity at 1e-2,
# beta=0.5 on inv with 6 binades, signed near-zero split. e_max from the domain.
ROUTES = {
    "gelu": dict(e_min=-3, e_max=7),
    "silu": dict(e_min=-3, e_max=4),
    "exp2": dict(e_min=-3, e_max=0),
    "invsqrt": dict(e_min=-3, e_max=1),
    "identity": dict(e_min=-6, e_max=0, tied=True, target="relative",
                     target_rel=1e-2),
    "inv": dict(beta=0.5),
}


def route_for(name):
    r = dict(ROUTES[name])
    if "beta" in r:
        lo, hi = functions.REGISTRY[name][1]
        r["e_min"], r["e_max"] = _erange_beta(lo, hi, r["beta"], 6)
    return r


def run(name, epochs, args):
    fn, domain = functions.REGISTRY[name]
    x, y, _ = functions.sample(name, m=args.eval_samples,
                                seed=args.seed + 913, domain=domain)
    t0 = time.time()
    neuron = build_mbe_pasn(
        name, domain, **route_for(name), near0="signed", budget="fit",
        n_steps=args.max_t, n_max=args.max_n, n_local=2, n_near0=2,
        epochs=epochs, seed=args.seed, target_mse=args.target_mse,
        readout_order=args.readout_order, alpha_init="auto",
    )
    with torch.no_grad():
        mse = float((neuron(x) - y).pow(2).mean())
        spikes = float(neuron.mean_spikes(x))
    row = dict(function=name, epochs=epochs, seed=args.seed,
               readout_order=args.readout_order, max_n=args.max_n,
               max_t=args.max_t, target_mse=args.target_mse,
               route={k: v for k, v in route_for(name).items()},
               eval_mse=mse, eval_spikes=spikes,
               selected_N=[d["selected"]["N"] for d in neuron.fit_budget_diagnostics],
               selection=neuron.fit_budget_diagnostics,
               build_s=round(time.time() - t0, 1))
    print(f"{name:9s} ep={epochs:4d} ro={args.readout_order} "
          f"mse={mse:.3e} spikes={spikes:.3f} N={row['selected_N']} "
          f"({row['build_s']}s)", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--functions", nargs="+", default=list(ROUTES),
                   choices=list(ROUTES))
    p.add_argument("--epochs", nargs="+", type=int, default=[10, 100, 300])
    p.add_argument("--max-n", type=int, default=4)
    p.add_argument("--max-t", type=int, default=8)
    p.add_argument("--target-mse", type=float, default=1e-3)
    p.add_argument("--readout-order", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=20000)
    p.add_argument("--json", required=True)
    args = p.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    out = Path(args.json)
    rec = dict(experiment="e38_n_basis_value", status="running",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"),
               argv=sys.argv[1:], rows=[])
    for ep in args.epochs:
        for name in args.functions:
            try:
                rec["rows"].append(run(name, ep, args))
            except Exception as e:  # keep the failure in the record
                rec["rows"].append(dict(function=name, epochs=ep,
                                        status="failed", error=repr(e)))
                print(f"{name} ep={ep} FAILED: {e!r}", flush=True)
            out.write_text(json.dumps(rec, indent=1))   # partial, every row
    rec["status"] = ("ok" if all(r.get("status") != "failed" for r in rec["rows"])
                     else "partial")
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
