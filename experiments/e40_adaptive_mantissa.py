"""E40: per-binade mantissa depth m_j -- pay for banks only where they buy something.

E39 showed that splitting a binade into ``2^m`` N=1 banks beats adding bases at
equal bytes (5.4-7.9x fewer spikes), but a *uniform* m multiplies storage ~1.9x
per bit, and only a few top binades need it. Two ways to choose m_j:

  greedy    ``build_mbe_pasn_split``: start at m=0, deepen only binades that miss
            the target, up to m_max. Built for real -- fit time is measured.
  frontier  offline, from the E39 ``n1_m{0..3}`` neurons (same fits, same T
            selection): per binade, options m=0..3 with mass-weighted spikes and
            bytes; a multiple-choice knapsack gives the min-spike m_j map for
            every byte budget. Selected points are re-assembled into real neurons
            and measured, so the frontier is not a model.

Feasibility rule for the frontier: a binade may only take depths whose every
sub-bank meets the target; if none does, every depth is allowed and the cell is
flagged. Mass = the E39 evaluation draw (20,000 uniform samples, seed 913).
Function level. Tied identity is not covered (per-binade depth is not wired for
tied prototypes).

  python experiments/e40_adaptive_mantissa.py greedy --target-mse 1e-4 \
      --json results/e40_greedy_t1e-4.json --save-dir logs/e40_neurons
  python experiments/e40_adaptive_mantissa.py frontier --json results/e40_frontier.json
  python experiments/e40_adaptive_mantissa.py time --json results/e40_greedy_t1e-4.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.mbe_pasn import (MBEPASNNeuron, PrefixRouter,  # noqa: E402
                          build_mbe_pasn_split, fill_unreachable)
from mbe.metrics import storage_bytes  # noqa: E402
from e38_n_basis_value import route_for  # noqa: E402

FUNCS = ["gelu", "silu", "exp2", "invsqrt", "inv"]
TARGETS = ["1e-03", "1e-04", "1e-05"]


def eval_draw(name, m=20000):
    fn, domain = functions.REGISTRY[name]
    x, y, _ = functions.sample(name, m=m, seed=913, domain=domain)
    return x, y


@torch.no_grad()
def measure(neuron, name):
    x, y = eval_draw(name)
    err = neuron(x) - y
    mse = float(err.pow(2).mean())
    return dict(eval_mse=mse,
                eval_nrmse=mse ** 0.5 / float(y.pow(2).mean().sqrt()),
                eval_max_abs=float(err.abs().max()),
                spikes_per_input=float(neuron.mean_spikes(x)),
                storage_bytes=int(storage_bytes(neuron)),
                params=int(neuron.num_learnable()),
                n_banks=neuron.router.n_banks)


def make_router(name, m_bits):
    _, domain = functions.REGISTRY[name]
    r = route_for(name)
    return PrefixRouter(domain[0], domain[1], e_min=r["e_min"], e_max=r["e_max"],
                        near0="signed", beta=r.get("beta", 0.0), m_bits=m_bits)


# --------------------------------------------------------------------------- greedy

def cmd_greedy(args):
    torch.set_num_threads(args.threads)
    out = Path(args.json)
    rec = dict(experiment="e40_adaptive_mantissa", part="greedy", status="running",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:],
               threads=args.threads, rows=[])
    for name in args.functions:
        _, domain = functions.REGISTRY[name]
        try:
            t0 = time.perf_counter()
            n = build_mbe_pasn_split(
                name, domain, **route_for(name), m_max=args.m_max, n_steps=args.max_t,
                epochs=args.epochs, seed=args.seed, target_mse=args.target_mse,
                readout_order=args.readout_order, alpha_init="auto")
            build_s = time.perf_counter() - t0
            row = dict(function=name, arm=f"greedy_m{args.m_max}",
                       target_mse=args.target_mse, epochs=args.epochs,
                       readout_order=args.readout_order, max_t=args.max_t,
                       seed=args.seed, build_s=round(build_s, 2), **measure(n, name),
                       m_map={f"{s:+.0f},{e}": m for (s, e), m in n.m_map.items()},
                       banks_fitted=len(n.fit_budget_diagnostics),
                       infeasible_banks=sum(
                           not d["selected"]["feasible"]
                           for d in n.fit_budget_diagnostics
                           if d["bank"][0] != "mag" or
                           n.router.m_of.get((d["bank"][1], d["bank"][2]), 0) == d["bank"][3]))
            if args.save_dir:
                p = Path(args.save_dir) / f"{name}__greedy__{args.target_mse:.0e}.pt"
                torch.save(n, p)
                row["saved"] = str(p).replace("\\", "/")
            print(f"{name:8s} nrmse={row['eval_nrmse']:.3e} spikes={row['spikes_per_input']:.3f} "
                  f"bytes={row['storage_bytes']} banks={row['n_banks']} m_map={row['m_map']} "
                  f"infeas={row['infeasible_banks']} build={build_s:.1f}s", flush=True)
        except Exception as e:
            row = dict(function=name, status="failed", error=repr(e))
            print(f"{name} FAILED {e!r}", flush=True)
        rec["rows"].append(row)
        out.write_text(json.dumps(rec, indent=1))
    rec["status"] = ("ok" if all(r.get("status") != "failed" for r in rec["rows"])
                     else "partial")
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


# ------------------------------------------------------------------------- frontier

def load_e39(name, arm, tgt):
    return torch.load(f"logs/e39_neurons/{name}__{arm}__{tgt}.pt", weights_only=False)


@torch.no_grad()
def binade_options(name, tgt):
    """Per binade and depth m: (mass-weighted spikes, bytes, feasible, modules)."""
    x, _ = eval_draw(name)
    total = x.numel()
    opts: dict[tuple, dict] = {}
    near0 = None
    for m in range(4):
        old = load_e39(name, f"n1_m{m}", tgt)
        r = make_router(name, m)              # fresh router: same layout as `old`
        assert len(r.banks) == len(old.router.banks)
        feas = {}
        for d in old.fit_budget_diagnostics:
            feas[d["bank"]] = d["selected"]["feasible"]
        idx = r.route(x)
        for bi, spec in enumerate(r.banks):
            mod = old.bank_mods[bi]
            sel = idx == bi
            cnt = int(sel.sum())
            sp = 0.0
            if cnt:
                inp = r.to_key(x[sel]).abs()
                sp = float(mod.firing_rate(inp)) * mod.cfg.n_basis * mod.cfg.n_steps * cnt / total
            reach = r.reachable(bi, *functions.REGISTRY[name][1]) is not None
            if spec["kind"] != "mag":
                if m == 0:
                    near0 = near0 or dict(spikes=0.0, bytes=0, modules={})
                    near0["spikes"] += sp
                    near0["bytes"] += storage_bytes(mod) if reach else 0
                    near0["modules"][(spec["kind"], spec["sign"])] = mod
                continue
            key = (spec["sign"], spec["e"])
            o = opts.setdefault(key, {}).setdefault(
                m, dict(spikes=0.0, bytes=0, feasible=True, modules={}, reach=False))
            o["spikes"] += sp
            if reach:
                o["bytes"] += storage_bytes(mod)
                o["feasible"] &= bool(feas.get(bi, True))
                o["reach"] = True
            o["modules"][spec.get("sub", 0)] = mod
    opts = {k: v for k, v in opts.items() if any(o["reach"] for o in v.values())}
    return opts, near0


def knapsack(opts, budget_units, unit=32):
    """Min total spikes, one depth per binade, total bytes <= budget."""
    keys = sorted(opts)
    INF = float("inf")
    best = {0: (0.0, {})}
    for k in keys:
        choices = opts[k]
        allowed = [m for m, o in choices.items() if o["feasible"]] or list(choices)
        nxt = {}
        for used, (sp, pick) in best.items():
            for m in allowed:
                o = choices[m]
                u = used + o["bytes"] // unit
                if u > budget_units:
                    continue
                cand = sp + o["spikes"]
                if cand < nxt.get(u, (INF,))[0]:
                    nxt[u] = (cand, {**pick, k: m})
        best = nxt
    if not best:
        return None
    u, (sp, pick) = min(best.items(), key=lambda kv: kv[1][0])
    return dict(spikes=sp, units=u, m_map=pick,
                flagged=[k for k in keys
                         if not any(o["feasible"] for o in opts[k].values())])


def assemble(name, m_map, opts, near0):
    r = make_router(name, {k: m for k, m in m_map.items()})
    _, domain = functions.REGISTRY[name]
    banks = []
    for bi, spec in enumerate(r.banks):
        if r.reachable(bi, *domain) is None:
            banks.append(None)
        elif spec["kind"] != "mag":
            banks.append(near0["modules"][(spec["kind"], spec["sign"])])
        else:
            o = opts[(spec["sign"], spec["e"])][spec["m"]]
            banks.append(o["modules"][spec["sub"]])
    return MBEPASNNeuron(r, fill_unreachable(r, banks))


def cmd_frontier(args):
    torch.set_num_threads(args.threads)
    rec = dict(experiment="e40_adaptive_mantissa", part="frontier", status="running",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:], cells=[])
    for name in args.functions:
        for tgt in TARGETS:
            opts, near0 = binade_options(name, tgt)
            lo_units = sum(min(o["bytes"] for o in v.values()) for v in opts.values()) // 32
            hi_units = sum(max(o["bytes"] for o in v.values()) for v in opts.values()) // 32
            points, last = [], None
            for b in range(lo_units, hi_units + 1):
                s = knapsack(opts, b)
                if s is None or (last and s["spikes"] >= last["spikes"] - 1e-12):
                    continue
                last = s
                n = assemble(name, s["m_map"], opts, near0)
                meas = measure(n, name)
                points.append(dict(budget_bytes=b * 32, predicted_spikes=s["spikes"] + near0["spikes"],
                                   m_map={f"{k[0]:+.0f},{k[1]}": m for k, m in s["m_map"].items() if m},
                                   flagged=[f"{k[0]:+.0f},{k[1]}" for k in s["flagged"]],
                                   **meas))
            cell = dict(function=name, target_mse=float(tgt), points=points)
            rec["cells"].append(cell)
            print(f"{name:8s} {tgt}: {len(points)} frontier pts, bytes "
                  f"{points[0]['storage_bytes']}..{points[-1]['storage_bytes']}, spikes "
                  f"{points[0]['spikes_per_input']:.3f}..{points[-1]['spikes_per_input']:.3f}",
                  flush=True)
            Path(args.json).write_text(json.dumps(rec, indent=1))
    rec["status"] = "ok"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.json).write_text(json.dumps(rec, indent=1))


# ----------------------------------------------------------------------------- time

@torch.no_grad()
def cmd_time(args):
    torch.set_num_threads(args.threads)
    out = Path(args.json)
    rec = json.loads(out.read_text())
    for row in rec["rows"]:
        if "saved" not in row:
            continue
        n = torch.load(row["saved"], weights_only=False)
        lo, hi = functions.REGISTRY[row["function"]][1]
        x = torch.rand(1 << 20, generator=torch.Generator().manual_seed(7)) * (hi - lo) + lo
        n(x[:4096])
        fw, rt = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter(); n(x); fw.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); n.router.route(x); rt.append(time.perf_counter() - t0)
        row["forward_ms_1M"] = round(1e3 * statistics.median(fw), 2)
        row["route_ms_1M"] = round(1e3 * statistics.median(rt), 2)
        row["time_threads"] = args.threads
        print(f"{row['function']:8s} fwd={row['forward_ms_1M']:.1f}ms route={row['route_ms_1M']:.1f}ms",
              flush=True)
    out.write_text(json.dumps(rec, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["greedy", "frontier", "time"])
    p.add_argument("--json", required=True)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--functions", nargs="+", default=FUNCS, choices=FUNCS)
    p.add_argument("--target-mse", type=float, default=1e-4)
    p.add_argument("--m-max", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-t", type=int, default=8)
    p.add_argument("--readout-order", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    {"greedy": cmd_greedy, "frontier": cmd_frontier, "time": cmd_time}[args.cmd](args)


if __name__ == "__main__":
    main()
