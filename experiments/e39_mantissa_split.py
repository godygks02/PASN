"""E39: buy resolution with mantissa-prefix banks instead of bases.

E38 found that a routed bank almost never wants ``N > 1``: where it did, the bank
was a wide-output top binade pinned at the ``T`` cap, i.e. it needed *bits*.
``PrefixRouter(m_bits=m)`` reads the top ``m`` mantissa bits as well as the
exponent, cutting every binade into ``2^m`` banks. That is still a bit-read, so
the extra resolution should cost stored banks, not spikes. This compares, per
primitive and error target:

  rule        budget="rule"  (the frozen build's policy), m=0, N<=4
  n_adapt     budget="fit"   N in 1..4, m=0          (the G1 adaptive arm)
  n1_m{0..3}  budget="fit"   N fixed at 1, m bits    (bank count is the knob)

all at ``T <= 8``, G1 routing and readout. Function level only. Build time is
recorded here; inference time is measured afterwards, one neuron at a time, by
``--time`` on the saved neurons, so concurrent builds cannot pollute it.

  python experiments/e39_mantissa_split.py build --target-mse 1e-3 \
      --json results/e39_mantissa_split_t1e-3.json --save-dir logs/e39_neurons [--resume]
  python experiments/e39_mantissa_split.py time --json <record>
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

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402
from mbe.metrics import storage_bytes  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from e38_n_basis_value import route_for  # noqa: E402  (G1 routing, shared)

ARMS = {
    "rule": dict(budget="rule", m_bits=0, n_max=4),
    "n_adapt": dict(budget="fit", m_bits=0, n_max=4),
    "n1_m0": dict(budget="fit", m_bits=0, n_fixed=1),
    "n1_m1": dict(budget="fit", m_bits=1, n_fixed=1),
    "n1_m2": dict(budget="fit", m_bits=2, n_fixed=1),
    "n1_m3": dict(budget="fit", m_bits=3, n_fixed=1),
}


def build(name, arm, args):
    fn, domain = functions.REGISTRY[name]
    x, y, _ = functions.sample(name, m=args.eval_samples,
                                seed=args.seed + 913, domain=domain)
    kw = dict(ARMS[arm])
    t0 = time.perf_counter()
    neuron = build_mbe_pasn(
        name, domain, **route_for(name), near0="signed",
        n_steps=args.max_t, n_local=2, n_near0=2, epochs=args.epochs,
        seed=args.seed, target_mse=args.target_mse,
        readout_order=args.readout_order, alpha_init="auto", **kw,
    )
    build_s = time.perf_counter() - t0
    with torch.no_grad():
        err = neuron(x) - y
        mse = float(err.pow(2).mean())
        nrmse = mse ** 0.5 / float(y.pow(2).mean().sqrt())
        max_abs = float(err.abs().max())
        spikes = float(neuron.mean_spikes(x))
    diag = getattr(neuron, "fit_budget_diagnostics", None) or []
    usage = neuron.bank_usage(x)
    budgets = neuron.budget_table()
    used = [b for b in budgets if usage.get(b[0], 0)]
    row = dict(
        function=name, arm=arm, **{k: v for k, v in kw.items()},
        target_mse=args.target_mse, epochs=args.epochs, seed=args.seed,
        readout_order=args.readout_order, max_t=args.max_t,
        eval_mse=mse, eval_nrmse=nrmse, eval_max_abs=max_abs,
        spikes_per_input=spikes,
        storage_bytes=int(storage_bytes(neuron)),
        params=int(neuron.num_learnable()),
        stored_bases=int(neuron.stored_bases()),
        n_banks=neuron.router.n_banks,
        mean_T_used=statistics.fmean(b[3] for b in used) if used else None,
        N_used=sorted({b[2] for b in used}),
        infeasible_banks=sum(not d["selected"]["feasible"] for d in diag),
        build_s=round(build_s, 2),
    )
    if args.save_dir:
        p = Path(args.save_dir) / f"{name}__{arm}__{args.target_mse:.0e}.pt"
        torch.save(neuron, p)
        row["saved"] = str(p)
    print(f"{name:9s} {arm:8s} nrmse={nrmse:.3e} spikes={spikes:6.3f} "
          f"bytes={row['storage_bytes']:6d} banks={row['n_banks']:3d} "
          f"N={row['N_used']} infeas={row['infeasible_banks']} "
          f"build={build_s:6.1f}s", flush=True)
    return row


def cmd_build(args):
    torch.set_num_threads(args.threads)
    out = Path(args.json)
    if args.resume and out.exists():
        # Continue an interrupted record: keep its finished cells, rerun failures.
        rec = json.loads(out.read_text())
        rec["rows"] = [r for r in rec["rows"] if r.get("status") != "failed"]
        rec.setdefault("resumed", []).append(
            dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv[1:]))
        rec["status"] = "running"
    else:
        rec = dict(experiment="e39_mantissa_split", status="running",
                   started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   argv=sys.argv[1:], threads=args.threads, rows=[])
    done = {(r["function"], r["arm"]) for r in rec["rows"]}
    for name in args.functions:
        for arm in args.arms:
            if (name, arm) in done:
                print(f"{name:9s} {arm:8s} (kept from record)", flush=True)
                continue
            try:
                rec["rows"].append(build(name, arm, args))
            except Exception as e:  # a failed cell stays in the record
                rec["rows"].append(dict(function=name, arm=arm,
                                        status="failed", error=repr(e)))
                print(f"{name} {arm} FAILED: {e!r}", flush=True)
            out.write_text(json.dumps(rec, indent=1))
    rec["status"] = ("ok" if all(r.get("status") != "failed" for r in rec["rows"])
                     else "partial")
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


@torch.no_grad()
def cmd_time(args):
    """Median forward / route time on 2^20 uniform inputs, one neuron at a time.

    The PyTorch forward loops over banks with a mask each, so its wall time grows
    with the bank count; a hardware router indexes one bank per input. Both the
    full forward and the routing alone are recorded so the two can be told apart.
    """
    torch.set_num_threads(args.threads)
    out = Path(args.json)
    rec = json.loads(out.read_text())
    for row in rec["rows"]:
        if "saved" not in row:
            continue
        neuron = torch.load(row["saved"], weights_only=False)
        lo, hi = functions.REGISTRY[row["function"]][1]
        g = torch.Generator().manual_seed(7)
        x = torch.rand(1 << 20, generator=g) * (hi - lo) + lo
        neuron(x[:4096])                                     # warm-up
        fw, rt = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter(); neuron(x); fw.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); neuron.router.route(x); rt.append(time.perf_counter() - t0)
        row["forward_ms_1M"] = round(1e3 * statistics.median(fw), 2)
        row["route_ms_1M"] = round(1e3 * statistics.median(rt), 2)
        row["time_threads"] = args.threads
        print(f"{row['function']:9s} {row['arm']:8s} fwd={row['forward_ms_1M']:8.1f}ms "
              f"route={row['route_ms_1M']:6.1f}ms", flush=True)
    out.write_text(json.dumps(rec, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["build", "time"])
    p.add_argument("--json", required=True)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--functions", nargs="+",
                   default=["gelu", "silu", "exp2", "invsqrt", "inv", "identity"])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--target-mse", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-t", type=int, default=8)
    p.add_argument("--readout-order", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=20000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--resume", action="store_true",
                   help="build: keep finished cells of an existing --json record")
    args = p.parse_args()
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    (cmd_build if args.cmd == "build" else cmd_time)(args)


if __name__ == "__main__":
    main()
