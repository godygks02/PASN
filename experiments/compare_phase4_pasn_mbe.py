"""Phase 4, three-way converted toy Transformer: MBE vs MBE-PASN vs PASN.

Converts the same toy Transformer once per neuron backend and compares the
forward error vs the full-precision ANN together with the spike cost of the
dominant nonlinearity (the GELU activation):

  * ``mbe``      -- one global MBE neuron per primitive,
  * ``mbe_pasn`` -- FP-prefix binade banks of MBE neurons,
  * ``pasn``     -- the same router, successive-approximation (SAR) banks.

The backend drives the activation, the LayerNorm primitives and the
activation*activation matmul identities; Softmax has no routed variant and stays
MBE in all three, so it contributes an identical error floor. ``mbe_pasn`` and
``pasn`` share ``pasn_e_min``, so their difference isolates the encoder. All CPU.

Usage:  python experiments/compare_phase4_pasn_mbe.py
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.metrics import spikes_per_input  # noqa: E402
from mbe.toy import make_toy, make_inputs  # noqa: E402


def rel_err(out, ref):
    return float((out - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6))


def convert_backend(model, calib, backend, epochs):
    m = copy.deepcopy(model)
    rec = cv.calibrate(m, calib)
    cfg = cv.ConvertConfig(epochs=epochs, spike_mult=True, backend=backend,
                           pasn_n_local=2, pasn_e_min=-3,
                           pasn_T=6, pasn_order=2,
                           pasn_s_n_shared=2, pasn_s_restarts=2)
    cv.convert(m, rec, cfg=cfg)
    # mean GELU-activation spikes across every activation module (all layers)
    acts = [n for n, mod in m.named_modules()
            if isinstance(mod, cv._SpikingActModule)]
    sp = [spikes_per_input(m.get_submodule(n).act.neuron, rec.ranges[n][0].sample)
          for n in acts]
    return m, sum(sp) / len(sp), cv.spiking_cost_report(m, calib[0])


def main():
    torch.manual_seed(0)
    D, n_layers = 16, 2
    model = make_toy(seed=0, d_model=D, n_heads=2, n_layers=n_layers)
    calib = make_inputs(4, batch=8, seq=16, d_model=D, seed=1)
    test = make_inputs(1, batch=8, seq=16, d_model=D, seed=99)[0]
    with torch.no_grad():
        ann = model(test)
    print(f"toy: {n_layers} layers, d={D}   ANN mean|y|={ann.abs().mean():.4f}\n")
    print(f"{'backend':10s} {'fwd rel|err|':>13s} {'GELU spk/in':>12s} "
          f"{'TOTAL spk/in':>13s} {'act%':>6s} {'LN%':>6s} {'softmax%':>9s} "
          f"{'matmul%':>8s}")
    print("-" * 76)
    for backend in ["mbe", "mbe_pasn", "mbe_pasn_s", "pasn"]:
        m, act_spikes, rep = convert_backend(model, calib, backend, epochs=200)
        with torch.no_grad():
            out = m(test)
        tot = max(rep["total_spikes"], 1e-30)
        pct = {k: 100.0 * v / tot for k, v in rep["by_kind"].items()}
        print(f"{backend:10s} {rel_err(out, ann):13.3e} {act_spikes:12.3f} "
              f"{rep['spikes_per_input']:13.2f} "
              f"{pct.get('activation', 0):5.1f}% {pct.get('layernorm', 0):5.1f}% "
              f"{pct.get('softmax', 0):8.1f}% {pct.get('matmul', 0):7.1f}%")
    print("\nTOTAL spk/in counts every spiking primitive per element of the model's "
          "input, measured by instrumenting the forward pass -- the GELU column is a "
          "fraction of it. Softmax has no routed variant, so it is identical MBE in "
          "all four; mbe_pasn / mbe_pasn_s / pasn share the router and differ only in "
          "what a bank contains.")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
