"""E45: what one PASN evaluation costs in energy, and what the order-2 readout costs.

MBE exists to replace exact FP arithmetic with spike-driven accumulates (AC, 0.9 pJ)
instead of multiply-accumulates (MAC, 4.6 pJ). An order-2 readout
``w1*o + w2*o^2 + b`` puts three FP multiplies back into every evaluation
(o^2, w1*o, w2*o^2), which is 13.8 pJ before a single spike is counted.

Part 1 prices the E44 neurons (order 2, real GPT-2 inputs) with the repo's strict
op model (``metrics.neuron_op_cost`` / ``op_energy_pj``, Horowitz 45 nm):
per element, ``cmp`` = N*T threshold compares, ``ac`` = 2 per spike,
``mac`` = N*order readout MACs, ``poly`` = N*(order-1) powers, ``router`` = 6.
Part 2 builds the greedy neuron with an order-1 readout at the same targets and
prices it two ways: *unfused* (one w*o MAC) and *fused* (w folded into the decode
kernel d[t], so the readout is accumulates only plus one bias add).
Part 3 prices the exact ANN computation of each primitive from an explicit fp32
op recipe (mul 3.7 pJ, add 0.9 pJ), and a piecewise-linear LUT (one MAC; index by
bit select). Memory reads are not priced on either side.

  python experiments/e45_energy.py --json results/e45_energy.json --save-dir logs/e45_neurons
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
from mbe.metrics import OP_ENERGY_PJ, neuron_op_cost, op_energy_pj  # noqa: E402
from e42_gpt2_dist import FUNCS, TARGET_FN, measure  # noqa: E402

E_MUL, E_ADD = 3.7, 0.9          # Horowitz 45 nm fp32; MAC = 4.6 = mul + add

# Exact fp32 recipes, per element. (muls, adds); bit selects are free.
#   EXP   : n=floor(x), f=x-n (2 add), degree-6 Horner (6 mul, 6 add), 2^n bit op
#   RECIP : seed by bit select, 2 Newton steps y(2-xy) (2 mul, 1 add each)
#   RSQRT : 0.5x (1 mul), seed, 2 Newton steps y(1.5-0.5x*y*y) (3 mul, 1 add each)
#   TANH  : 1 - 2/(exp(2z)+1) = mul + EXP + add + RECIP + mul + add
#   GELU_tanh: x^2, x^3 (2 mul), 0.044715x^3 + x (1 mul 1 add), *c (1 mul),
#              TANH, +1 (1 add), 0.5x (1 mul), product (1 mul)
EXP = (6, 8)
RECIP = (4, 2)
RSQRT = (7, 2)
TANH = (2 + EXP[0] + RECIP[0], 2 + EXP[1] + RECIP[1])
GELU = (6 + TANH[0], 2 + TANH[1])
ANN_RECIPE = {"gelu": GELU, "exp2": EXP, "inv": RECIP, "invsqrt": RSQRT}
ANN_NOTE = {"gelu": "tanh-GELU: 6 mul + 2 add + tanh(exp, recip)",
            "exp2": "range-reduced degree-6 polynomial",
            "inv": "bit-select seed + 2 Newton steps",
            "invsqrt": "bit-select seed + 2 Newton steps"}
PWL_PJ = OP_ENERGY_PJ["mac"]      # one slope*x + intercept; segment index = bit select

SETTINGS = [dict(k=1.0, r=3e-3), dict(k=0.1, r=1e-3)]


def ann_pj(name):
    m, a = ANN_RECIPE[name]
    return m * E_MUL + a * E_ADD


def breakdown(n, x, fuse=False):
    ops = neuron_op_cost(n, x)
    if fuse:
        # order-1 readout folded into the decode kernel: no multiply, one bias add
        ops = dict(ops, mac=0.0, poly=0.0, ac=ops["ac"] + 1.0)
    e = OP_ENERGY_PJ
    parts = dict(compare_pj=ops["cmp"] * e["cmp"], spike_ac_pj=ops["ac"] * e["ac"],
                 readout_mul_pj=(ops["mac"] + ops["poly"]) * e["mac"],
                 router_pj=ops["router"] * e["cmp"])
    return dict(ops=ops, total_pj=op_energy_pj(ops), **parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", default="logs/e42_gpt2_samples.pt")
    p.add_argument("--functions", nargs="+", default=FUNCS, choices=FUNCS)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--json", required=True)
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    data = torch.load(args.samples, weights_only=False)
    out = Path(args.json)
    rec = dict(experiment="e45_energy", status="running", argv=sys.argv[1:],
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), op_energy_pj=OP_ENERGY_PJ,
               e_mul=E_MUL, e_add=E_ADD, pwl_pj=PWL_PJ,
               ann={k: dict(muls=v[0], adds=v[1], pj=ann_pj(k), recipe=ANN_NOTE[k])
                    for k, v in ANN_RECIPE.items()},
               rows=[])
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
        for st in SETTINGS:
            tag = f"hybrid_k{st['k']:g}_r{st['r']:.0e}_T8"
            # Part 1: the E44 order-2 neurons as built
            for arm in ("rule", "n_adapt", "greedy"):
                n = torch.load(f"logs/e44_neurons/{name}__{arm}__{tag}.pt", weights_only=False)
                row = dict(function=name, setting=tag, arm=arm, readout="order2",
                           **measure(n, ev, fn), **breakdown(n, ev))
                rec["rows"].append(row)
                print(f"{name:8s} {tag:22s} {arm:8s} ro2        "
                      f"{row['total_pj']:6.2f} pJ (cmp {row['compare_pj']:.2f} ac {row['spike_ac_pj']:.2f} "
                      f"mul {row['readout_mul_pj']:.2f} rt {row['router_pj']:.2f}) "
                      f"nrmse {row['eval_nrmse']:.2e}", flush=True)
            # Part 2: greedy with an order-1 readout, same target
            t0 = time.perf_counter()
            n1 = build_mbe_pasn_split(
                reg, domain, **route, m_max=3, n_steps=8, epochs=args.epochs, seed=0,
                target="hybrid", target_rel=st["r"],
                target_mse=(st["k"] * st["r"] * rms_dist) ** 2,
                readout_order=1, alpha_init="auto")
            build_s = time.perf_counter() - t0
            if args.save_dir:
                torch.save(n1, Path(args.save_dir) / f"{name}__greedy_ro1__{tag}.pt")
            meas = measure(n1, ev, fn)
            diag = n1.fit_budget_diagnostics
            infeas = sum(not dd["selected"]["feasible"] for dd in diag
                         if dd["bank"][0] != "mag"
                         or n1.router.m_of.get((dd["bank"][1], dd["bank"][2]), 0) == dd["bank"][3])
            for fuse in (False, True):
                row = dict(function=name, setting=tag, arm="greedy",
                           readout="order1_fused" if fuse else "order1",
                           build_s=round(build_s, 1), infeasible_banks=infeas,
                           **meas, **breakdown(n1, ev, fuse=fuse))
                rec["rows"].append(row)
                print(f"{name:8s} {tag:22s} greedy   {row['readout']:10s} "
                      f"{row['total_pj']:6.2f} pJ (cmp {row['compare_pj']:.2f} ac {row['spike_ac_pj']:.2f} "
                      f"mul {row['readout_mul_pj']:.2f} rt {row['router_pj']:.2f}) "
                      f"nrmse {row['eval_nrmse']:.2e} bytes {row['storage_bytes']} infeas {infeas}",
                      flush=True)
            out.write_text(json.dumps(rec, indent=1))
        print(f"{name:8s} ANN exact fp32 {ann_pj(name):.1f} pJ  ({ANN_NOTE[name]}); "
              f"PWL LUT {PWL_PJ:.1f} pJ", flush=True)
    rec["status"] = "ok"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
