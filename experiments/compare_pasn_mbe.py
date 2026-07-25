"""MBE vs PASN: approximation MSE vs mean spikes per input (headline metric).

For each nonlinearity, fits the global MBE neuron, the strongest MBE-family
baseline (signed/polarity-split MBE), and PASN (fixed and adaptive per-binade
budgets), then reports each point's MSE against its mean spike count per input.
PASN's frontier should dominate on wide / curved domains (GELU, SiLU); on
already-narrow primitive domains (1/x, 2^x) the advantage shrinks -- an honest
picture. All CPU.

Usage:  python experiments/compare_pasn_mbe.py [--fns gelu silu ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import (MBENeuron, functions, fit_model, build_pasn,  # noqa: E402
                 spikes_per_input)

# per-function (domain, e_min, e_max) for the PASN router; domains follow the
# paper's usage, trimmed where the far tail is numerically flat (cheaper CPU run).
FN_SETUP = {
    "gelu":    dict(domain=(-8.0, 8.0),  e_min=-2, e_max=3, signed_base=True),
    "silu":    dict(domain=(-8.0, 8.0),  e_min=-2, e_max=3, signed_base=True),
    "inv":     dict(domain=(0.5, 1.0),   e_min=-2, e_max=0, signed_base=False),
    "exp2":    dict(domain=(0.0, 1.0),   e_min=-3, e_max=0, signed_base=False),
    "invsqrt": dict(domain=(0.5, 2.0),   e_min=-2, e_max=1, signed_base=False),
}


def eval_mbe(name, domain, n_basis, n_steps, epochs, seed):
    x, y, _ = functions.sample(name, m=4000, seed=seed, domain=domain)
    cfg = functions.make_config(name, n_basis=n_basis, n_steps=n_steps, domain=domain)
    m = MBENeuron(cfg)
    fit_model(m, x, y, seed=seed, epochs=epochs)
    with torch.no_grad():
        mse = (m(x) - y).pow(2).mean().item()
    return mse, spikes_per_input(m, x), n_basis


def eval_signed_mbe(name, domain, n_side, n_steps, epochs, seed):
    x, y, _ = functions.sample(name, m=4000, seed=seed, domain=domain)
    sm = functions.make_signed(name, n_pos=n_side, n_neg=n_side, pivot=0.0, n_steps=n_steps)
    fit_model(sm, x, y, seed=seed, epochs=epochs)
    with torch.no_grad():
        mse = (sm(x) - y).pow(2).mean().item()
    xp, xn = sm._split(x)
    spikes = (sm.pos.firing_rate(xp) + sm.neg.firing_rate(xn)) * n_side * n_steps
    return mse, spikes, 2 * n_side


def eval_pasn(name, setup, n_steps, epochs, seed, **kw):
    p = build_pasn(name, setup["domain"], e_min=setup["e_min"], e_max=setup["e_max"],
                   n_steps=n_steps, epochs=epochs, seed=seed, **kw)
    x, y, _ = functions.sample(name, m=4000, seed=seed + 1, domain=setup["domain"])
    with torch.no_grad():
        mse = (p(x) - y).pow(2).mean().item()
    return mse, p.mean_spikes(x), p.stored_bases()


def run_fn(name, steps, epochs, setup, rows):
    print(f"\n### {name}  domain={setup['domain']}  T={steps}")
    print(f"{'model':22s} {'MSE':>11s} {'spikes/in':>10s} {'stored N':>9s}")
    print("-" * 56)

    def add(tag, family, mse, sp, N):
        print(f"  {tag:20s} {mse:11.3e} {sp:10.3f} {N:9d}")
        rows.append(dict(fn=name, model=tag, family=family, mse=mse,
                         spikes=sp, stored=N))

    for n in [4, 8]:
        add("MBE N=%d" % n, "mbe", *eval_mbe(name, setup["domain"], n, steps, epochs, 0))
    if setup["signed_base"]:
        for n in [2, 4]:
            add("SignedMBE %dx2" % n, "signed",
                *eval_signed_mbe(name, setup["domain"], n, steps, epochs, 0))
    for nl in [1, 2]:
        add("PASN n_local=%d" % nl, "pasn",
            *eval_pasn(name, setup, steps, epochs, 0, n_local=nl, n_near0=max(nl + 2, 4)))
    add("PASN adaptive", "pasn",
        *eval_pasn(name, setup, steps, epochs, 0, adaptive=True, target_mse=1e-4,
                   n_max=6, n_near0=6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fns", nargs="+", default=["gelu", "silu", "inv", "exp2", "invsqrt"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--domain", type=float, nargs=2, default=None,
                    help="override domain (applies to all listed fns)")
    ap.add_argument("--emin", type=int, default=None)
    ap.add_argument("--emax", type=int, default=None)
    ap.add_argument("--json", default=None, help="dump results to this JSON path")
    args = ap.parse_args()
    print("MBE vs PASN -- MSE vs mean spikes/input (lower-left = better)")
    rows: list[dict] = []
    for fn in args.fns:
        setup = dict(FN_SETUP[fn])
        if args.domain is not None:
            setup["domain"] = tuple(args.domain)
        if args.emin is not None:
            setup["e_min"] = args.emin
        if args.emax is not None:
            setup["e_max"] = args.emax
        run_fn(fn, args.steps, args.epochs, setup, rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
