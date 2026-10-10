"""E46: order-1 fused readout everywhere, and energy-minimising selection.

E45: the order-2 readout's three multiplies are 44-50% of a PASN evaluation, so
order 2 loses the "replace multiplies" argument; an order-1 readout folds into the
decode kernel and costs one bias add. E45 also showed spikes are only ~1/3 of the
remaining energy -- compares (N*T, paid every timestep) and the router are the
rest -- so selecting the fewest-*spike* candidate is the wrong objective.

Arms per primitive and target (real GPT-2 inputs, hybrid target, T <= 8):
  ro2/spikes        rule, n_adapt, greedy        E44 neurons, priced here
  ro1f/spikes       rule, n_adapt, greedy, greedy spend x2
  ro1f/energy       n_adapt, greedy, greedy spend x2  (select="energy")
"ro1f" = order-1 readout, priced fused (w folded into d[t]: no multiply, one
bias add). Fusing changes no output value, only the op count. Energy is the E45
strict model including the router.

  python experiments/e46_ro1_energy_select.py --floor-k 0.1 --target-rel 1e-3 \
      --json results/e46_k0.1_r1e-3.json --save-dir logs/e46_neurons
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
from e45_energy import ann_pj, breakdown, PWL_PJ  # noqa: E402


def infeasible(n):
    diag = getattr(n, "fit_budget_diagnostics", None) or []
    out = 0
    for dd in diag:
        b = dd["bank"]
        if isinstance(b, tuple) and b[0] == "mag" and n.router.m_of.get((b[1], b[2]), 0) != b[3]:
            continue
        out += not dd["selected"]["feasible"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", default="logs/e42_gpt2_samples.pt")
    p.add_argument("--functions", nargs="+", default=FUNCS, choices=FUNCS)
    p.add_argument("--target-rel", type=float, default=1e-3)
    p.add_argument("--floor-k", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json", required=True)
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    data = torch.load(args.samples, weights_only=False)
    tag = f"hybrid_k{args.floor_k:g}_r{args.target_rel:.0e}_T8"
    out = Path(args.json)
    rec = dict(experiment="e46_ro1_energy_select", status="running", argv=sys.argv[1:],
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), setting=tag,
               threads=args.threads, rows=[])

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
        base = dict(n_steps=8, epochs=args.epochs, seed=0, target="hybrid",
                    target_rel=args.target_rel,
                    target_mse=(args.floor_k * args.target_rel * rms_dist) ** 2,
                    alpha_init="auto")

        def emit(arm, readout, select, n, build_s, fuse, extra=None):
            row = dict(function=name, setting=tag, arm=arm, readout=readout,
                       select=select, build_s=build_s, infeasible_banks=infeasible(n),
                       ann_exact_pj=ann_pj(name), pwl_pj=PWL_PJ,
                       T_used=sorted({b.cfg.n_steps for b in n.bank_mods}),
                       N_used=sorted({b.cfg.n_basis for b in n.bank_mods}),
                       **measure(n, ev, fn), **breakdown(n, ev, fuse=fuse), **(extra or {}))
            rec["rows"].append(row)
            out.write_text(json.dumps(rec, indent=1))
            print(f"{name:8s} {arm:13s} {readout:4s} {select:6s} E={row['total_pj']:6.2f} pJ "
                  f"(cmp {row['compare_pj']:.2f} ac {row['spike_ac_pj']:.2f} mul {row['readout_mul_pj']:.2f}) "
                  f"sp={row['spikes_per_input']:.3f} nrmse={row['eval_nrmse']:.2e} "
                  f"B={row['storage_bytes']} infeas={row['infeasible_banks']} "
                  f"T={row['T_used']} build={build_s}s", flush=True)
            return row

        # ro2 baselines from E44
        for arm in ("rule", "n_adapt", "greedy"):
            n = torch.load(f"logs/e44_neurons/{name}__{arm}__{tag}.pt", weights_only=False)
            emit(arm, "ro2", "spikes", n, None, fuse=False)

        def run(arm, select, make):
            t0 = time.perf_counter()
            n = make()
            b = round(time.perf_counter() - t0, 1)
            if args.save_dir:
                torch.save(n, Path(args.save_dir) / f"{name}__{arm}_ro1f_{select}__{tag}.pt")
            return n, b

        ro1 = dict(base, readout_order=1)
        n, b = run("rule", "spikes", lambda: build_mbe_pasn(
            reg, domain, **route, near0="signed", budget="rule", n_max=4, n_local=2,
            n_near0=4, **ro1))
        emit("rule", "ro1f", "spikes", n, b, fuse=True)
        for sel in ("spikes", "energy"):
            kw = dict(ro1, select=sel, fused_readout=True)
            n, b = run("n_adapt", sel, lambda: build_mbe_pasn(
                reg, domain, **route, near0="signed", budget="fit", n_max=4, n_local=2,
                n_near0=4, **kw))
            emit("n_adapt", "ro1f", sel, n, b, fuse=True)
            n, b = run("greedy", sel, lambda: build_mbe_pasn_split(
                reg, domain, **route, m_max=3, **kw))
            emit("greedy", "ro1f", sel, n, b, fuse=True)
            budget = 2 * rec["rows"][-1]["storage_bytes"]
            n, b = run("greedy_spend2", sel, lambda: build_mbe_pasn_split(
                reg, domain, **route, m_max=3, byte_budget=budget, sample=cal, **kw))
            emit("greedy_spend2", "ro1f", sel, n, b, fuse=True, extra=dict(byte_budget=budget))
    rec["status"] = "ok"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
