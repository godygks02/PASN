"""How deep should the router be for the *identity* primitives?

Once Softmax is routed, the spike-driven FP multiply's operand reconstructions are the
largest consumer of a converted Transformer's spikes (LayerNorm 34-42%, attention
matmuls 26-29%, against the activation's 8-10%). Their operands -- ``e^x``, ``1/S``,
``x-mu``, ``1/std`` -- span many decades, so with one shared ``e_min`` everything below
``2^e_min`` collapses into a single near-zero bank that has to represent the identity
across that whole range.

This sweeps ``pasn_id_e_min`` (identity primitives only; the activation keeps
``pasn_e_min``) and reports forward error, total spikes measured by instrumenting the
pass, the per-op split, and stored parameters. All CPU.

Usage:  python experiments/sweep_identity_router.py
        python experiments/sweep_identity_router.py --backends mbe_pasn_s --e-mins -3 -8
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.metrics import neuron_params  # noqa: E402
from mbe.toy import make_toy, make_inputs  # noqa: E402


def rel_err(out, ref):
    return float((out - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6))


def identity_params(model) -> int:
    """Stored parameters across every identity primitive (what deeper routing buys
    its accuracy with)."""
    total = 0
    for name, kind, neuron in cv._spiking_primitives(model):
        if name.endswith((".idn", ".idn2", ".id_dev", ".id_istd")):
            total += neuron_params(neuron)
    return total


def run(backend, id_e_min, model, calib, test, ann, epochs):
    m = copy.deepcopy(model)
    rec = cv.calibrate(m, calib)
    cfg = cv.ConvertConfig(epochs=epochs, spike_mult=True, backend=backend,
                           pasn_n_local=2, pasn_e_min=-3, pasn_id_e_min=id_e_min,
                           pasn_T=6, pasn_order=2,
                           pasn_s_n_shared=2, pasn_s_restarts=2)
    t0 = time.perf_counter()
    cv.convert(m, rec, cfg=cfg)
    secs = time.perf_counter() - t0
    with torch.no_grad():
        out = m(test)
    rep = cv.spiking_cost_report(m, test)
    tot = max(rep["total_spikes"], 1e-30)
    return dict(err=rel_err(out, ann), spikes=rep["spikes_per_input"],
                pct={k: 100.0 * v / tot for k, v in rep["by_kind"].items()},
                id_params=identity_params(m), secs=secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+",
                    default=["mbe_pasn", "mbe_pasn_s", "pasn"])
    ap.add_argument("--e-mins", nargs="+", type=int, default=[-3, -6, -10, -14])
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(0)
    D, n_layers = 16, 2
    model = make_toy(seed=0, d_model=D, n_heads=2, n_layers=n_layers)
    calib = make_inputs(4, batch=8, seq=16, d_model=D, seed=1)
    test = make_inputs(1, batch=8, seq=16, d_model=D, seed=99)[0]
    with torch.no_grad():
        ann = model(test)
    print(f"toy: {n_layers} layers, d={D}   identity-router sweep "
          f"(activation stays at pasn_e_min=-3)\n")

    for backend in args.backends:
        print(f"### {backend}")
        print(f"{'id_e_min':>9s} {'fwd rel|err|':>13s} {'TOTAL spk/in':>13s} "
              f"{'LN%':>6s} {'softmax%':>9s} {'matmul%':>8s} {'id params':>10s} "
              f"{'build s':>8s}")
        print("-" * 82)
        base = None
        for e in args.e_mins:
            r = run(backend, e, model, calib, test, ann, args.epochs)
            base = base or r
            print(f"{e:9d} {r['err']:13.3e} {r['spikes']:13.2f} "
                  f"{r['pct'].get('layernorm', 0):5.1f}% "
                  f"{r['pct'].get('softmax', 0):8.1f}% "
                  f"{r['pct'].get('matmul', 0):7.1f}% {r['id_params']:10d} "
                  f"{r['secs']:8.1f}", flush=True)
        print()


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
