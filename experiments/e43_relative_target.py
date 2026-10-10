"""E43: the greedy split with a *relative* per-bank target, on the real GPT-2 mix.

E42 found that an absolute per-bank MSE target sends the greedy split (and
n_adapt) after GELU's outlier binades -- x in [8, 82], output ~ x, ~0 mass -- which
no T <= 8 bank can hold to the same absolute error as a bank whose output is 0.1.
A relative target, eps_j = (r * rms(f on bank j))^2, scales with the output, so
those binades should stop asking for splits. The cost to watch: where the output
is ~0 (GELU's negative tail), a relative target becomes very strict instead.

Arms per target r (all with target="relative"):
  rule / n_adapt / greedy / spend_x2 (B = 2 x greedy bytes, real-mass weighted)
Same samples, split, routing and settings as E42; scored on the real mix, so
the comparison with E42's absolute-target arms is at measured nrmse.

  python experiments/e43_relative_target.py --target-rel 3e-3 \
      --json results/e43_rel_t3e-3.json --save-dir logs/e43_neurons
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", default="logs/e42_gpt2_samples.pt")
    p.add_argument("--functions", nargs="+", default=FUNCS, choices=FUNCS)
    p.add_argument("--target-rel", type=float, default=3e-3)
    p.add_argument("--factors", nargs="+", type=float, default=[2.0])
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
    rec = dict(experiment="e43_relative_target", status="running",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:],
               samples=args.samples, sample_meta=data["meta"], threads=args.threads,
               rows=[])
    common = dict(n_steps=args.max_t, epochs=args.epochs, seed=args.seed,
                  target="relative", target_rel=args.target_rel,
                  readout_order=args.readout_order, alpha_init="auto")
    for name in args.functions:
        d = data["prims"][name]
        domain, route = tuple(d["domain"]), dict(d["route"])
        s = d["sample"]
        s = s[(s >= domain[0]) & (s <= domain[1])]
        perm = torch.randperm(s.numel(), generator=torch.Generator().manual_seed(1))
        cal, ev = s[perm[: s.numel() // 2]], s[perm[s.numel() // 2:]]
        reg = TARGET_FN.get(name, name)
        fn = functions.REGISTRY[reg][0]
        arms = [("rule", lambda: build_mbe_pasn(reg, domain, **route, near0="signed",
                                                budget="rule", n_max=4, n_local=2,
                                                n_near0=4, **common)),
                ("n_adapt", lambda: build_mbe_pasn(reg, domain, **route, near0="signed",
                                                   budget="fit", n_max=4, n_local=2,
                                                   n_near0=4, **common)),
                ("greedy", lambda: build_mbe_pasn_split(reg, domain, **route,
                                                        m_max=args.m_max, **common))]
        base_bytes = None
        for arm, make in arms + [(f"spend_x{f:g}", f) for f in args.factors]:
            try:
                t0 = time.perf_counter()
                if callable(make):
                    n = make()
                else:
                    n = build_mbe_pasn_split(reg, domain, **route, m_max=args.m_max,
                                             byte_budget=int(make * base_bytes),
                                             sample=cal, **common)
                build_s = time.perf_counter() - t0
                row = dict(function=name, target_fn=reg, arm=arm, domain=domain,
                           route=route, target="relative", target_rel=args.target_rel,
                           build_s=round(build_s, 2), **measure(n, ev, fn))
                if arm == "greedy":
                    base_bytes = row["storage_bytes"]
                if not callable(make):
                    row["byte_budget"] = int(make * base_bytes)
                diag = getattr(n, "fit_budget_diagnostics", None) or []
                row["N_used"] = sorted({b.cfg.n_basis for b in n.bank_mods})
                row["infeasible_banks"] = sum(
                    not dd["selected"]["feasible"] for dd in diag
                    if not isinstance(dd["bank"], tuple) or dd["bank"][0] != "mag"
                    or n.router.m_of.get((dd["bank"][1], dd["bank"][2]), 0) == dd["bank"][3])
                if hasattr(n, "m_map"):
                    # Where the splits went, with each split binade's eval mass.
                    idx = n.router.route(ev)
                    mass = {}
                    for bi, sp in enumerate(n.router.banks):
                        if sp["kind"] == "mag":
                            k = (sp["sign"], sp["e"])
                            mass[k] = mass.get(k, 0.0) + float((idx == bi).float().mean())
                    row["m_map"] = {f"{k[0]:+.0f},{k[1]}": dict(m=m, mass=round(mass.get(k, 0.0), 6))
                                    for k, m in n.m_map.items()}
                    row["banks_fitted"] = len(diag)
                if args.save_dir:
                    pth = Path(args.save_dir) / f"{name}__{arm}__rel{args.target_rel:.0e}.pt"
                    torch.save(n, pth)
                    row["saved"] = str(pth).replace("\\", "/")
                print(f"{name:8s} {arm:9s} nrmse={row['eval_nrmse']:.3e} "
                      f"spikes={row['spikes_per_input']:.3f} bytes={row['storage_bytes']:5d} "
                      f"banks={row['n_banks']:3d} N={row['N_used']} infeas={row['infeasible_banks']} "
                      f"m_map={row.get('m_map', '-')} build={build_s:.0f}s", flush=True)
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
