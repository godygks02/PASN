"""Three-way neuron comparison on 1-D function approximation.

For each nonlinearity, compares:
  * MBE          -- one global multi-basis neuron (and signed/polarity-split, the
                    strongest MBE-family baseline for GELU/SiLU),
  * MBE-PASN     -- prefix binade banks of MBE neurons (the MBE-improvement variant),
  * PASN         -- the standalone successive-approximation neuron.

Reports three axes per point: approximation MSE (accuracy), mean spikes per input
(energy), and stored parameters (memory). Dumps JSON for plotting/report. All CPU.

Usage:  python experiments/compare_neurons.py --json results/neuron_compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import (MBENeuron, functions, fit_model, build_mbe_pasn,  # noqa: E402
                 build_pasn)
from mbe.metrics import neuron_cost, neuron_params  # noqa: E402

FN_SETUP = {
    "gelu":    dict(domain=(-8.0, 8.0), e_min=-2, e_max=4, signed=True),
    "silu":    dict(domain=(-8.0, 8.0), e_min=-2, e_max=4, signed=True),
    "inv":     dict(domain=(0.5, 1.0),  e_min=-2, e_max=0, signed=False),
    "exp2":    dict(domain=(0.0, 1.0),  e_min=-4, e_max=0, signed=False),
    "invsqrt": dict(domain=(0.5, 2.0),  e_min=-2, e_max=1, signed=False),
}


def _data(name, domain, seed):
    x, y, _ = functions.sample(name, m=4000, seed=seed, domain=domain)
    return x, y


def _record(rows, fn, tag, family, neuron, xte, yte):
    with torch.no_grad():
        mse = (neuron(xte) - yte).pow(2).mean().item()
    c = neuron_cost(neuron, xte)
    p = neuron_params(neuron)
    rows.append(dict(fn=fn, model=tag, family=family, mse=mse,
                     spikes=c["spikes"], params=p))
    print(f"  {tag:16s} MSE={mse:.2e}  spikes/in={c['spikes']:7.2f}  params={p}",
          flush=True)


def run_fn(name, epochs, rows):
    s = FN_SETUP[name]
    dom = s["domain"]
    xte, yte = _data(name, dom, seed=7)
    xtr, ytr = _data(name, dom, seed=0)
    print(f"\n### {name}  domain={dom}")

    for N in (4, 8):                                   # MBE global
        m = MBENeuron(functions.make_config(name, n_basis=N, n_steps=16, domain=dom))
        fit_model(m, xtr, ytr, seed=0, epochs=epochs)
        _record(rows, name, f"MBE N={N}", "mbe", m, xte, yte)

    if s["signed"]:                                    # strongest MBE baseline
        sm = functions.make_signed(name, n_pos=4, n_neg=4, domain=dom)
        fit_model(sm, xtr, ytr, seed=0, epochs=epochs)
        _record(rows, name, "SignedMBE 4x2", "signed", sm, xte, yte)

    mp = build_mbe_pasn(name, dom, e_min=s["e_min"], e_max=s["e_max"],
                        n_local=2, n_near0=4, epochs=epochs, seed=0)
    _record(rows, name, "MBE-PASN nloc=2", "mbe_pasn", mp, xte, yte)

    for T, order in ((4, 1), (6, 1), (6, 2)):          # standalone PASN (SAR)
        p = build_pasn(name, dom, e_min=-6, e_max=s["e_max"], T=T, order=order, seed=0)
        _record(rows, name, f"PASN T={T} o={order}", "pasn", p, xte, yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fns", nargs="+",
                    default=["gelu", "silu", "inv", "exp2", "invsqrt"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--json", default="results/neuron_compare.json")
    args = ap.parse_args()
    print("MBE vs MBE-PASN vs PASN -- accuracy (MSE) / energy (spikes) / memory (params)")
    rows = []
    for fn in args.fns:
        run_fn(fn, args.epochs, rows)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
