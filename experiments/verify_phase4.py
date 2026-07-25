"""Phase 4 verification: end-to-end ANN->SNN conversion of a toy Transformer.

Builds a small module-structured Transformer, calibrates it on a few minibatches,
converts every nonlinearity to its spike-driven MBE form, and reports the output
error vs the full-precision ANN. Also isolates each op type (convert only
LayerNorm / Softmax / activation / matmul) so a wiring bug shows up as one op
dominating the error. All CPU.

Usage:  python experiments/verify_phase4.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import copy  # noqa: E402
import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.toy import make_toy, make_inputs  # noqa: E402


def rel_err(out, ref):
    denom = ref.abs().mean().clamp(min=1e-6)
    return float((out - ref).abs().mean() / denom)


def run(model, x):
    with torch.no_grad():
        return model(x)


def convert_copy(model, recorder_inputs, cfg, only=None, verbose=False):
    """Deep-copy the ANN, calibrate on the same inputs, convert (optionally a
    subset of op kinds), and return the converted model."""
    m = copy.deepcopy(model)
    rec = cv.calibrate(m, recorder_inputs)
    cv.convert(m, rec, cfg=cfg, only=only, verbose=verbose)
    return m


def main():
    torch.manual_seed(0)
    D = 16
    model = make_toy(seed=0, d_model=D, n_heads=2, n_layers=1)
    calib = make_inputs(4, batch=8, seq=16, d_model=D, seed=1)
    test = make_inputs(1, batch=8, seq=16, d_model=D, seed=99)[0]

    ann = run(model, test)
    print(f"ANN output: shape={tuple(ann.shape)}  mean|y|={ann.abs().mean():.4f}\n",
          flush=True)

    cfg = cv.ConvertConfig(epochs=150, spike_mult=True)

    print("== per-op isolation (convert one kind at a time) ==", flush=True)
    for kind in ["layernorm", "activation", "softmax", "matmul"]:
        m = convert_copy(model, calib, cfg, only={kind})
        out = run(m, test)
        print(f"{kind:12s} rel|err|={rel_err(out, ann):.3e}", flush=True)

    print("\n== exact-mult vs spike-mult (full conversion) ==", flush=True)
    cfg_exact = cv.ConvertConfig(epochs=150, spike_mult=False)
    m_exact = convert_copy(model, calib, cfg_exact)
    print(f"full (exact-mult) rel|err|={rel_err(run(m_exact, test), ann):.3e}"
          "   (primitive-approx error only)")

    m_full = convert_copy(model, calib, cfg, verbose=True)
    out_full = run(m_full, test)
    print(f"full (spike-mult) rel|err|={rel_err(out_full, ann):.3e}"
          "   (full spike-driven path)")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
