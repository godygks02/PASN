"""E44: a relative target with an absolute floor, then the T <= 16 check.

E43's relative per-bank target fixed E42's outlier waste but turned unreachable
where the output goes to zero: GELU's negative tail asked for MSE 1e-13 and was
split to m=3 for nothing. ``target="hybrid"`` floors it:

    eps_j = max((r * rms_j)^2, eps_abs),   eps_abs = (k * r * rms_dist)^2

where ``rms_dist`` is the output RMS over the real calibration half -- "no bank
needs to be more exact than r of the signal the network actually carries". k=1
and k=0.1 are run. ``--max-t 16`` repeats a point with T up to 16 to see whether
r=1e-3's outlier splits were the T=8 bit ceiling (E43 §2.2).

Same samples, split, routing, arms and settings as E42/E43.

  python experiments/e44_hybrid_target.py --target hybrid --floor-k 1 \
      --target-rel 1e-3 --json results/e44_hyb_k1_r1e-3.json
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
from mbe.mbe_pasn import build_mbe_pasn, build_mbe_pasn_split  # noqa: E402
from e42_gpt2_dist import FUNCS, TARGET_FN, measure  # noqa: E402

ARMS = ["rule", "n_adapt", "greedy"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", default="logs/e42_gpt2_samples.pt")
    p.add_argument("--functions", nargs="+", default=FUNCS, choices=FUNCS)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--target", choices=["relative", "hybrid"], default="hybrid")
    p.add_argument("--target-rel", type=float, default=1e-3)
    p.add_argument("--floor-k", type=float, default=1.0)
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
    data = torch.load(args.samples, weights_only=False)

    out = Path(args.json)
    rec = dict(experiment="e44_hybrid_target", status="running",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:],
               samples=args.samples, threads=args.threads, rows=[])
    for name in args.functions:
        d = data["prims"][name]
        domain, route = tuple(d["domain"]), dict(d["route"])
        s = d["sample"]
        s = s[(s >= domain[0]) & (s <= domain[1])]
        perm = torch.randperm(s.numel(), generator=torch.Generator().manual_seed(1))
        cal, ev = s[perm[: s.numel() // 2]], s[perm[s.numel() // 2:]]
        reg = TARGET_FN.get(name, name)
        fn = functions.REGISTRY[reg][0]
        rms_dist = float(fn(cal).pow(2).mean().sqrt())
        eps_abs = (args.floor_k * args.target_rel * rms_dist) ** 2
        common = dict(n_steps=args.max_t, epochs=args.epochs, seed=args.seed,
                      target=args.target, target_rel=args.target_rel,
                      target_mse=eps_abs, readout_order=args.readout_order,
                      alpha_init="auto")
        makers = {
            "rule": lambda: build_mbe_pasn(reg, domain, **route, near0="signed",
                                           budget="rule", n_max=4, n_local=2,
                                           n_near0=4, **common),
            "n_adapt": lambda: build_mbe_pasn(reg, domain, **route, near0="signed",
                                              budget="fit", n_max=4, n_local=2,
                                              n_near0=4, **common),
            "greedy": lambda: build_mbe_pasn_split(reg, domain, **route,
                                                   m_max=args.m_max, **common),
        }
        for arm in args.arms:
            try:
                t0 = time.perf_counter()
                n = makers[arm]()
                build_s = time.perf_counter() - t0
                row = dict(function=name, arm=arm, target=args.target,
                           target_rel=args.target_rel, floor_k=args.floor_k,
                           eps_abs=eps_abs, rms_dist=rms_dist, max_t=args.max_t,
                           domain=domain, route=route, build_s=round(build_s, 2),
                           **measure(n, ev, fn))
                diag = getattr(n, "fit_budget_diagnostics", None) or []
                row["N_used"] = sorted({b.cfg.n_basis for b in n.bank_mods})
                row["T_used"] = sorted({b.cfg.n_steps for b in n.bank_mods})
                final = [dd for dd in diag
                         if not isinstance(dd["bank"], tuple) or dd["bank"][0] != "mag"
                         or n.router.m_of.get((dd["bank"][1], dd["bank"][2]), 0) == dd["bank"][3]]
                row["infeasible_banks"] = sum(not dd["selected"]["feasible"] for dd in final)
                if hasattr(n, "m_map"):
                    idx = n.router.route(ev)
                    mass = {}
                    for bi, sp in enumerate(n.router.banks):
                        if sp["kind"] == "mag":
                            k = (sp["sign"], sp["e"])
                            mass[k] = mass.get(k, 0.0) + float((idx == bi).float().mean())
                    row["m_map"] = {f"{k[0]:+.0f},{k[1]}": dict(m=m, mass=round(mass.get(k, 0.0), 6))
                                    for k, m in n.m_map.items()}
                    extra = sum(2 ** m - 1 for m in n.m_map.values())
                    low = sum(2 ** m - 1 for k, m in n.m_map.items() if mass.get(k, 0.0) < 1e-3)
                    row["split_banks_added"] = extra       # nominal; clipped binades add fewer
                    row["split_banks_low_mass"] = low
                    row["banks_fitted"] = len(diag)
                if args.save_dir:
                    pth = Path(args.save_dir) / (f"{name}__{arm}__{args.target}_k{args.floor_k:g}"
                                                 f"_r{args.target_rel:.0e}_T{args.max_t}.pt")
                    torch.save(n, pth)
                    row["saved"] = str(pth).replace("\\", "/")
                print(f"{name:8s} {arm:8s} nrmse={row['eval_nrmse']:.3e} "
                      f"spikes={row['spikes_per_input']:.3f} bytes={row['storage_bytes']:5d} "
                      f"banks={row['n_banks']:3d} N={row['N_used']} T={row['T_used']} "
                      f"infeas={row['infeasible_banks']} "
                      f"split={row.get('split_banks_added', '-')}/low={row.get('split_banks_low_mass', '-')} "
                      f"build={build_s:.0f}s", flush=True)
            except Exception as e:
                row = dict(function=name, arm=arm, status="failed", error=repr(e))
                print(f"{name} {arm} FAILED {e!r}", flush=True)
            rec["rows"].append(row)
            out.write_text(json.dumps(rec, indent=1))
    rec["status"] = ("ok" if all(r.get("status") != "failed" for r in rec["rows"])
                     else "partial")
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
