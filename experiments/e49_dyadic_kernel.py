"""E49: PASN-greedy with a dyadic (binary-search) kernel.

With every kernel halving per step and the leading threshold fixed at 1/2,
an N=1 bank's spike train is exactly the binary expansion of its normalised
residual and d[t] = 2^-t (``dyadic=True`` in the builder). Encoding then needs no
threshold compares -- the operand is already binary, the spikes are its bits --
and a synapse multiplying a spike by w is a shift. This measures what the
constraint costs in accuracy, spikes and memory against E46's learned-kernel
greedy, on the real GPT-2 inputs, and prices three readings of the hardware:

  sar        the E46 fused model: N*T compares, 2 AC/spike, bias add, router (6 ops)
  bitread    compares 0, 1 AC/spike (the accumulate), bias add, router (6 ops)
  bitread_r1 as bitread with the router reduced to one bit-select op

``bitread`` applies only to the dyadic neuron; a learned kernel's thresholds are
not bit positions. Same samples, split, routing, targets and settings as E46.

  python experiments/e49_dyadic_kernel.py --floor-k 0.1 --target-rel 3e-3 \
      --json results/e49_k0.1_r3e-3.json --save-dir logs/e49_neurons
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
from mbe.metrics import OP_ENERGY_PJ  # noqa: E402
from e42_gpt2_dist import FUNCS, TARGET_FN, measure  # noqa: E402
from e45_energy import ann_pj, breakdown, PWL_PJ  # noqa: E402
from e46_ro1_energy_select import infeasible  # noqa: E402

E = OP_ENERGY_PJ


def bitread_pj(spikes, router_ops):
    return E["ac"] * (spikes + 1.0) + E["cmp"] * router_ops


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", default="logs/e42_gpt2_samples.pt")
    p.add_argument("--functions", nargs="+", default=FUNCS, choices=FUNCS)
    p.add_argument("--target-rel", type=float, default=3e-3)
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
    rec = dict(experiment="e49_dyadic_kernel", status="running", argv=sys.argv[1:],
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
        kw = dict(n_steps=8, epochs=args.epochs, seed=0, target="hybrid",
                  target_rel=args.target_rel,
                  target_mse=(args.floor_k * args.target_rel * rms_dist) ** 2,
                  alpha_init="auto", readout_order=1, fused_readout=True)

        def emit(arm, kernel, n, build_s, extra=None):
            meas = measure(n, ev, fn)
            sar = breakdown(n, ev, fuse=True)
            row = dict(function=name, setting=tag, arm=arm, kernel=kernel,
                       build_s=build_s, infeasible_banks=infeasible(n),
                       T_used=sorted({b.cfg.n_steps for b in n.bank_mods}),
                       ann_exact_pj=ann_pj(name), pwl_pj=PWL_PJ,
                       sar_pj=sar["total_pj"], sar_breakdown={k: sar[k] for k in
                       ("compare_pj", "spike_ac_pj", "readout_mul_pj", "router_pj")},
                       **meas, **(extra or {}))
            if kernel == "dyadic":
                row["bitread_pj"] = bitread_pj(meas["spikes_per_input"], 6.0)
                row["bitread_r1_pj"] = bitread_pj(meas["spikes_per_input"], 1.0)
            rec["rows"].append(row)
            out.write_text(json.dumps(rec, indent=1))
            br = (f" bitread {row['bitread_pj']:.2f} / r1 {row['bitread_r1_pj']:.2f}"
                  if kernel == "dyadic" else "")
            print(f"{name:8s} {arm:13s} {kernel:7s} nrmse={meas['eval_nrmse']:.3e} "
                  f"sp={meas['spikes_per_input']:.3f} B={meas['storage_bytes']} "
                  f"params={meas['params']} infeas={row['infeasible_banks']} "
                  f"sar={row['sar_pj']:.2f}{br} pJ  T={row['T_used']} build={build_s}s",
                  flush=True)
            return row

        for arm in ("greedy", "greedy_spend2"):
            n = torch.load(f"logs/e46_neurons/{name}__{arm}_ro1f_spikes__{tag}.pt",
                           weights_only=False)
            emit(arm, "learned", n, None)

        t0 = time.perf_counter()
        n = build_mbe_pasn_split(reg, domain, **route, m_max=3, dyadic=True, **kw)
        b = round(time.perf_counter() - t0, 1)
        if args.save_dir:
            torch.save(n, Path(args.save_dir) / f"{name}__greedy_dyadic__{tag}.pt")
        base = emit("greedy", "dyadic", n, b)
        budget = 2 * base["storage_bytes"]
        t0 = time.perf_counter()
        n = build_mbe_pasn_split(reg, domain, **route, m_max=3, dyadic=True,
                                 byte_budget=budget, sample=cal, **kw)
        b = round(time.perf_counter() - t0, 1)
        if args.save_dir:
            torch.save(n, Path(args.save_dir) / f"{name}__greedy_spend2_dyadic__{tag}.pt")
        emit("greedy_spend2", "dyadic", n, b, extra=dict(byte_budget=budget))
    rec["status"] = "ok"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
