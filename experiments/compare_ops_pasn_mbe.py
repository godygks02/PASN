"""MBE vs PASN for the Phase-3 assembled ops: FP-mult / Softmax / LayerNorm.

Each op is built from calibrated 1-D primitives (identity for the FP multiply,
2^x + 1/x for softmax, 1/sqrt for LayerNorm). We swap those primitives between
MBE and PASN and report BOTH:

  * the driver primitive's accuracy vs spikes (this is what the op inherits), and
  * the assembled op's output error vs the exact torch op.

PASN's biggest win is the FP multiply: its identity primitive must reconstruct a
*wide* operand range, where a global MBE_Id spreads its few bases thin while PASN
gives each binade its own bank. Softmax/LayerNorm primitives live on narrow ranges
([0,1], [0.5,2]), so the gap is smaller -- an honest picture. All CPU.

Usage:  python experiments/compare_ops_pasn_mbe.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mbe import functions, build_pasn, spikes_per_input  # noqa: E402
from mbe import spiking_ops as so  # noqa: E402


def prim_mse_spikes(neuron, name, domain):
    fn, _ = functions.REGISTRY[name]
    x, y, _ = functions.sample(name, m=4000, seed=1, domain=domain)
    with torch.no_grad():
        pred = neuron(x)
    return (pred - y).pow(2).mean().item(), spikes_per_input(neuron, x)


def row(tag, mse, spikes, extra=""):
    print(f"  {tag:22s} MSE/err={mse:11.3e}  spikes/in={spikes:8.3f}  {extra}")


# --------------------------------------------------------------------------

def compare_fp_mult(hi=64.0, epochs=250):
    print(f"\n== FP multiply (identity reconstruction over [0,{hi:.0f}]) ==")
    idn_mbe = so.calibrate_identity(0.0, hi, n_basis=8, epochs=epochs)
    idn_pasn = build_pasn("identity", (0.0, hi), e_min=0,
                          e_max=int(hi).bit_length(), n_local=2, n_near0=2,
                          epochs=epochs)
    # primitive: reconstruct identity on [0,hi]
    x = torch.rand(4000) * hi
    for tag, idn in [("MBE_Id", idn_mbe), ("PASN_Id", idn_pasn)]:
        with torch.no_grad():
            mse = (idn.reconstruct(x) - x).pow(2).mean().item()
        sp = spikes_per_input(idn, x)
        # assembled product (two operands) rel error
        x1, x2 = torch.rand(3000) * hi, torch.rand(3000) * hi
        prod = so.spiking_multiply(idn, x1, x2, signed=False)
        rel = ((prod - x1 * x2).abs() / (x1 * x2).clamp(min=1e-3)).mean().item()
        row(tag, mse, 2 * sp, f"product rel|err|={rel:.2e}  (spikes = 2 operands)")


def compare_softmax(epochs=250):
    print("\n== Softmax (2^x on [0,1], 1/x on [0.5,1]) ==")
    torch.manual_seed(0)
    logits = torch.randn(128, 16) * 3.0
    ref = torch.softmax(logits, dim=-1)
    # MBE primitives
    sm_mbe = so.build_softmax(logits, n_basis=8, spike_mult=True)
    # PASN primitives
    exp_p = build_pasn("exp2", (0.0, 1.0), e_min=-3, e_max=0, n_local=2, n_near0=4, epochs=epochs)
    inv_p = build_pasn("inv", (0.5, 1.0), e_min=-2, e_max=0, n_local=2, n_near0=2, epochs=epochs)
    id_p = build_pasn("identity", (0.0, 1.0), e_min=-3, e_max=0, n_local=2, n_near0=2, epochs=epochs)
    sm_pasn = so.SpikingSoftmax(exp_p, inv_p, id_p, spike_mult=True)
    for tag, sm, expn in [("MBE softmax", sm_mbe, sm_mbe.exp),
                          ("PASN softmax", sm_pasn, exp_p)]:
        with torch.no_grad():
            out = sm(logits, dim=-1)
        err = (out - ref).abs().mean().item()
        # driver primitive = 2^x; report its spikes on the fractional input [0,1]
        frac = torch.rand(4000)
        row(tag, err, spikes_per_input(expn, frac), "(err vs torch; spikes = 2^x prim)")


def _build_ln_pasn(sample_x, eps=1e-5, epochs=250):
    mu = sample_x.mean(dim=-1, keepdim=True)
    dev = sample_x - mu
    dev_max = float(dev.abs().max()) * 1.1 + 1e-3
    var = (dev * dev).mean(dim=-1, keepdim=True) + eps
    istd_max = float((1.0 / var.sqrt()).max()) * 1.1 + 1e-3
    rsqrt = build_pasn("invsqrt", (0.5, 2.0), e_min=-2, e_max=1, n_local=2, n_near0=4, epochs=epochs)
    id_dev = build_pasn("identity", (0.0, dev_max), e_min=-2,
                        e_max=int(max(dev_max, 2)).bit_length(), n_local=2, n_near0=2, epochs=epochs)
    id_istd = build_pasn("identity", (0.0, istd_max), e_min=-2,
                         e_max=int(max(istd_max, 2)).bit_length(), n_local=2, n_near0=2, epochs=epochs)
    return so.SpikingLayerNorm(rsqrt, id_dev, id_istd, eps=eps, spike_mult=True), rsqrt


def compare_layernorm(epochs=250):
    print("\n== LayerNorm (1/sqrt on [0.5,2]) ==")
    torch.manual_seed(0)
    D = 64
    x = torch.randn(80, D) * 2.0 + 1.0
    gamma, beta = torch.randn(D) * 0.5 + 1, torch.randn(D) * 0.1
    ref = F.layer_norm(x, (D,), weight=gamma, bias=beta, eps=1e-5)
    ln_mbe = so.build_layernorm(x, n_basis=8, spike_mult=True)
    ln_pasn, rsqrt_p = _build_ln_pasn(x, epochs=epochs)
    rsqrt_m = ln_mbe.rsqrt
    for tag, ln, rs in [("MBE layernorm", ln_mbe, rsqrt_m),
                        ("PASN layernorm", ln_pasn, rsqrt_p)]:
        with torch.no_grad():
            out = ln(x, weight=gamma, bias=beta)
        err = (out - ref).abs().mean().item()
        m = torch.rand(4000) * 1.5 + 0.5      # mantissa range [0.5, 2]
        row(tag, err, spikes_per_input(rs, m), "(err vs torch; spikes = 1/sqrt prim)")


def main():
    print("MBE vs PASN on assembled Transformer ops (Phase 3 primitives)")
    compare_fp_mult()
    compare_softmax()
    compare_layernorm()
    print("\nTakeaway: PASN's edge is largest on the wide-range FP-mult identity; "
          "narrow softmax/LN primitives show a smaller gap.")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
