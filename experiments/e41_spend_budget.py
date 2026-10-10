"""E41: spend a byte budget on spikes, fitting only the path the greedy takes.

E40's greedy stops at the least memory that meets the target; its frontier shows
more bytes keep buying spikes, but the frontier fits every depth of every binade.
``build_mbe_pasn_split(byte_budget=B, sample=...)`` adds a second phase: deepen
the binade with the most sample-weighted spikes per added byte, keep the split
only if the target still holds and spikes drop, else close the binade.

For each function, target and budget factor f, B = f x (E40 greedy bytes). The
spike weights come from a calibration draw (seed 0); evaluation uses the E39/E40
draw (seed 913). Compared against the E40 frontier at the same B. Threads are
pinned to 4, the setting E40 ran at -- fits are bit-identical only then.

  python experiments/e41_spend_budget.py --target-mse 1e-4 \
      --json results/e41_spend_t1e-4.json --save-dir logs/e41_neurons
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn_split  # noqa: E402
from e38_n_basis_value import route_for  # noqa: E402
from e40_adaptive_mantissa import FUNCS, measure  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--functions", nargs="+", default=FUNCS, choices=FUNCS)
    p.add_argument("--target-mse", type=float, default=1e-4)
    p.add_argument("--factors", nargs="+", type=float, default=[1.5, 2.0, 3.0, 6.0])
    p.add_argument("--m-max", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-t", type=int, default=8)
    p.add_argument("--readout-order", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json", required=True)
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tag = f"{args.target_mse:.0e}".replace("e-0", "e-")
    base = {r["function"]: r for r in
            json.loads(Path(f"results/e40_greedy_t{tag}.json").read_text())["rows"]}
    frontier = {(c["function"], c["target_mse"]): c["points"] for c in
                json.loads(Path("results/e40_frontier.json").read_text())["cells"]}

    out = Path(args.json)
    rec = dict(experiment="e41_spend_budget", status="running",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:],
               threads=args.threads, rows=[])
    for name in args.functions:
        _, domain = functions.REGISTRY[name]
        cal, _, _ = functions.sample(name, m=20000, seed=0, domain=domain)
        b0 = base[name]["storage_bytes"]
        for f in args.factors:
            budget = int(f * b0)
            try:
                t0 = time.perf_counter()
                n = build_mbe_pasn_split(
                    name, domain, **route_for(name), m_max=args.m_max,
                    n_steps=args.max_t, epochs=args.epochs, seed=args.seed,
                    target_mse=args.target_mse, readout_order=args.readout_order,
                    alpha_init="auto", byte_budget=budget, sample=cal)
                build_s = time.perf_counter() - t0
                pts = [q for q in frontier[(name, args.target_mse)]
                       if q["storage_bytes"] <= budget]
                fb = min(pts, key=lambda q: q["spikes_per_input"]) if pts else None
                row = dict(function=name, factor=f, byte_budget=budget,
                           base_bytes=b0, base_spikes=base[name]["spikes_per_input"],
                           target_mse=args.target_mse, build_s=round(build_s, 2),
                           banks_fitted=len(n.fit_budget_diagnostics),
                           m_map={f"{s:+.0f},{e}": m for (s, e), m in n.m_map.items()},
                           trajectory=[dict(t, binade=f"{t['binade'][0]:+.0f},{t['binade'][1]}")
                                       for t in n.spend_trajectory],
                           **measure(n, name),
                           frontier_spikes=fb["spikes_per_input"] if fb else None,
                           frontier_bytes=fb["storage_bytes"] if fb else None)
                if args.save_dir:
                    pth = Path(args.save_dir) / f"{name}__x{f:g}__{args.target_mse:.0e}.pt"
                    torch.save(n, pth)
                    row["saved"] = str(pth).replace("\\", "/")
                gap = (f"{row['spikes_per_input'] / fb['spikes_per_input']:.3f}x of frontier"
                       if fb else "no frontier pt")
                print(f"{name:8s} x{f:<4g} B<={budget:5d}: spikes {row['base_spikes']:.3f} -> "
                      f"{row['spikes_per_input']:.3f} bytes {row['storage_bytes']} "
                      f"fitted={row['banks_fitted']} ({gap}) build={build_s:.0f}s", flush=True)
            except Exception as e:
                row = dict(function=name, factor=f, status="failed", error=repr(e))
                print(f"{name} x{f} FAILED {e!r}", flush=True)
            rec["rows"].append(row)
            out.write_text(json.dumps(rec, indent=1))
    rec["status"] = ("ok" if all(r.get("status") != "failed" for r in rec["rows"])
                     else "partial")
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
