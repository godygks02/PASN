"""E42: E39-E41 again, weighted and scored by the real GPT-2 input distribution.

E39-E41 drew inputs uniformly over each primitive's domain. A converted network
does not: GPT-2's MLP pre-activations sit around -1 (1-99%: [-3.6, 0.9]) inside a
calibrated domain of (-39, 82), and that decides which banks carry the spikes.
Samples come from ``e42_gpt2_samples.py`` (G1 calibration recipe). Each sample is
split in two: one half weights the byte-budget phase (``sample=``), the other is
the evaluation set for every arm. Routing is ``convert.py``'s, per primitive.

Arms per primitive and target:
  rule         budget="rule", N <= 4                (the frozen build's policy)
  n_adapt      budget="fit",  N in 1..4             (G1 adaptive)
  greedy       build_mbe_pasn_split, no budget      (E40)
  spend_x{f}   build_mbe_pasn_split, B = f x greedy (E41)

The fit selector holds every bank to the same absolute MSE target whatever the
input distribution; only spikes, the spend phase and the reported error are
distribution-weighted. Function level.

  python experiments/e42_gpt2_dist.py --target-mse 1e-4 --json results/e42_gpt2_t1e-4.json
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
from mbe.mbe_pasn import build_mbe_pasn, build_mbe_pasn_split  # noqa: E402
from mbe.metrics import storage_bytes  # noqa: E402

FUNCS = ["gelu", "exp2", "inv", "invsqrt"]
TARGET_FN = {"gelu": "gelu_tanh"}     # GPT-2's NewGELUActivation is the tanh form


@torch.no_grad()
def measure(n, x, fn):
    y = fn(x)
    err = n(x) - y
    mse = float(err.pow(2).mean())
    idx = n.router.route(x)
    near = [bi for bi, sp in enumerate(n.router.banks) if sp["kind"] != "mag"]
    near_mask = torch.zeros_like(idx, dtype=torch.bool)
    near_sp = 0.0
    for bi in near:
        sel = idx == bi
        near_mask |= sel
        c = int(sel.sum())
        if c:
            b = n.bank_mods[bi]
            near_sp += float(b.firing_rate(n._bank_input(x, sel, n.router.banks[bi]))) \
                * b.cfg.n_basis * b.cfg.n_steps * c
    spikes = float(n.mean_spikes(x))
    return dict(eval_mse=mse, eval_nrmse=mse ** 0.5 / float(y.pow(2).mean().sqrt()),
                eval_max_abs=float(err.abs().max()), spikes_per_input=spikes,
                near0_mass=float(near_mask.float().mean()),
                near0_spike_share=(near_sp / x.numel()) / max(spikes, 1e-12),
                storage_bytes=int(storage_bytes(n)), params=int(n.num_learnable()),
                n_banks=n.router.n_banks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", default="logs/e42_gpt2_samples.pt")
    p.add_argument("--functions", nargs="+", default=FUNCS, choices=FUNCS)
    p.add_argument("--target-mse", type=float, default=1e-4)
    p.add_argument("--factors", nargs="+", type=float, default=[1.5, 2.0, 3.0])
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
    rec = dict(experiment="e42_gpt2_dist", status="running",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:],
               samples=args.samples, sample_meta=data["meta"], threads=args.threads,
               rows=[])
    common = dict(n_steps=args.max_t, epochs=args.epochs, seed=args.seed,
                  target_mse=args.target_mse, readout_order=args.readout_order,
                  alpha_init="auto")
    for name in args.functions:
        d = data["prims"][name]
        domain, route = tuple(d["domain"]), dict(d["route"])
        s = d["sample"]
        s = s[(s >= domain[0]) & (s <= domain[1])]
        perm = torch.randperm(s.numel(), generator=torch.Generator().manual_seed(1))
        cal, ev = s[perm[: s.numel() // 2]], s[perm[s.numel() // 2:]]
        fn = functions.REGISTRY[TARGET_FN.get(name, name)][0]
        reg = TARGET_FN.get(name, name)
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
                           route=route, target_mse=args.target_mse,
                           build_s=round(build_s, 2), n_cal=cal.numel(), n_eval=ev.numel(),
                           **measure(n, ev, fn))
                if arm == "greedy":
                    base_bytes = row["storage_bytes"]
                if hasattr(n, "m_map"):
                    row["m_map"] = {f"{k[0]:+.0f},{k[1]}": m for k, m in n.m_map.items()}
                    row["banks_fitted"] = len(n.fit_budget_diagnostics)
                if not callable(make):
                    row["byte_budget"] = int(make * base_bytes)
                diag = getattr(n, "fit_budget_diagnostics", None) or []
                row["N_used"] = sorted({b.cfg.n_basis for b in n.bank_mods})
                row["infeasible_banks"] = sum(not dd["selected"]["feasible"] for dd in diag
                                              if not isinstance(dd["bank"], tuple)
                                              or dd["bank"][0] != "mag"
                                              or n.router.m_of.get((dd["bank"][1], dd["bank"][2]), 0)
                                              == dd["bank"][3])
                if args.save_dir:
                    pth = Path(args.save_dir) / f"{name}__{arm}__{args.target_mse:.0e}.pt"
                    torch.save(n, pth)
                    row["saved"] = str(pth).replace("\\", "/")
                print(f"{name:8s} {arm:10s} nrmse={row['eval_nrmse']:.3e} "
                      f"spikes={row['spikes_per_input']:.3f} bytes={row['storage_bytes']:5d} "
                      f"banks={row['n_banks']:3d} N={row['N_used']} infeas={row['infeasible_banks']} "
                      f"near0 mass={row['near0_mass']:.2f} spike-share={row['near0_spike_share']:.2f} "
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
