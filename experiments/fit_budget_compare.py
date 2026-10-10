"""Compare analytic and offline-fitted per-bank (N, T) allocations.

Examples:
  python experiments/fit_budget_compare.py --quick --functions gelu
  python experiments/fit_budget_compare.py --functions gelu identity \
      --epochs 100 --json results/fit_budget_compare.json

This is a function-level experiment. It does not claim network energy or latency.
The fitted selector uses a separate offset grid; this script evaluates on an
independent random draw and records both feasibility and storage cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402
from mbe.metrics import storage_bytes  # noqa: E402


ROUTES = {
    "gelu": dict(e_min=-2, e_max=4, near0="signed"),
    "silu": dict(e_min=-2, e_max=4, near0="signed"),
    "exp2": dict(e_min=-4, e_max=0, near0="signed"),
    "invsqrt": dict(e_min=-2, e_max=1, near0="signed"),
    "identity": dict(e_min=-4, e_max=1, near0="signed", tied=True,
                     target="relative"),
    "inv": dict(e_min=-4, e_max=0, near0="signed", beta=0.5),
}


def run_one(name: str, args) -> list[dict]:
    fn, domain = functions.REGISTRY[name]
    route = dict(ROUTES[name])
    if route.get("tied"):
        route["target_rel"] = args.target_rel
    x, y, _ = functions.sample(name, m=args.eval_samples,
                                seed=args.seed + 913, domain=domain)
    rows = []
    for budget in ("rule", "fit"):
        neuron = build_mbe_pasn(
            name, domain, **route, budget=budget, n_steps=args.max_t,
            n_max=args.max_n, n_local=args.max_n, n_near0=args.max_n,
            epochs=args.epochs, seed=args.seed,
            target_mse=args.target_mse, alpha_init="uniform",
        )
        with torch.no_grad():
            mse = float((neuron(x) - y).pow(2).mean())
            spikes = neuron.mean_spikes(x)
        row = dict(function=name, budget=budget, mse=mse, spikes=spikes,
                   bytes=int(storage_bytes(neuron)),
                   stored_bases=neuron.stored_bases(),
                   bank_budget=neuron.budget_table())
        if budget == "fit":
            row["selection"] = neuron.fit_budget_diagnostics
            row["infeasible_banks"] = sum(
                not d["selected"]["feasible"]
                for d in neuron.fit_budget_diagnostics)
        rows.append(row)
        print(f"{name:9s} {budget:4s}  mse={mse:.4e}  "
              f"spikes={spikes:.3f}  bytes={row['bytes']}  "
              f"bases={row['stored_bases']}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--functions", nargs="+", choices=sorted(ROUTES),
                   default=["gelu", "identity"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-n", type=int, default=4)
    p.add_argument("--max-t", type=int, default=16)
    p.add_argument("--target-mse", type=float, default=1e-4)
    p.add_argument("--target-rel", type=float, default=1e-2,
                   help="relative RMS target for tied identity banks")
    p.add_argument("--eval-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="small wiring check; results are not research evidence")
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    if args.quick:
        args.epochs, args.max_n, args.max_t = 2, 2, 4
        args.eval_samples = 256
    if args.max_n < 1 or args.max_t < 2 or args.target_mse <= 0 or args.target_rel <= 0:
        p.error("max-n >= 1, max-t >= 2 and positive error targets are required")
    rows = []
    for name in args.functions:
        rows.extend(run_one(name, args))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(
            experiment="offline_fit_budget", quick=args.quick,
            config=dict(functions=args.functions, epochs=args.epochs,
                        max_n=args.max_n, max_t=args.max_t,
                        target_mse=args.target_mse, target_rel=args.target_rel,
                        eval_samples=args.eval_samples, seed=args.seed),
            rows=rows), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
