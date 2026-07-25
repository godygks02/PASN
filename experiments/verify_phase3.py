"""Phase 3 verification: spike-driven FP-mult / Softmax / LayerNorm vs exact ops.

Assembles each Transformer primitive from calibrated MBE neurons and reports the
error against the exact PyTorch operation. All CPU.

Usage:  python experiments/verify_phase3.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mbe import spiking_ops as so  # noqa: E402


def rep(name, out, ref, extra=""):
    err = (out - ref).abs()
    print(f"{name:28s} mean|err|={err.mean():.3e}  max|err|={err.max():.3e}  {extra}")


def main():
    torch.manual_seed(0)

    print("== FP multiplication (Eq. 9-12, 22-27) ==")
    idn = so.calibrate_identity(0.0, 8.0, n_basis=8)
    x1, x2 = torch.rand(2000) * 8, torch.rand(2000) * 8
    rep("FP-mult non-negative", so.spiking_multiply(idn, x1, x2, signed=False),
        x1 * x2, f"rel={((so.spiking_multiply(idn,x1,x2,signed=False)-x1*x2).abs()/(x1*x2).clamp(min=1e-3)).mean():.2e}")
    ids = so.calibrate_identity(0.0, 6.0, n_basis=8)
    a, b = (torch.rand(2000) - 0.5) * 12, (torch.rand(2000) - 0.5) * 12
    rep("FP-mult signed", so.spiking_multiply(ids, a, b, signed=True), a * b)
    # explicit outer-product == separable form (paper's D (x) S Hadamard sum)
    oe = so.multiply_outer(idn, 3.3, 5.1)
    sep = float(idn.reconstruct(torch.tensor([3.3])) * idn.reconstruct(torch.tensor([5.1])))
    print(f"{'outer-product == separable':28s} {oe:.5f} == {sep:.5f}  (true 16.830)")

    print("\n== Softmax (Eq. 13, Table VIII) ==")
    logits = torch.randn(128, 16) * 3.0
    sm = so.build_softmax(logits, spike_mult=True)
    ref = torch.softmax(logits, dim=-1)
    out = sm(logits, dim=-1)
    rep("Softmax spike-mult", out, ref, f"row-sum={out.sum(-1).mean():.4f}")
    sm0 = so.build_softmax(logits, spike_mult=False)
    rep("Softmax exact-mult", sm0(logits, dim=-1), ref, "(exp+recip only)")

    print("\n== LayerNorm (Fig. 5c) ==")
    D = 64
    x = torch.randn(80, D) * 2.0 + 1.0
    gamma, beta = torch.randn(D) * 0.5 + 1, torch.randn(D) * 0.1
    ln = so.build_layernorm(x, spike_mult=True)
    ref = F.layer_norm(x, (D,), weight=gamma, bias=beta, eps=1e-5)
    rep("LayerNorm spike-mult", ln(x, weight=gamma, bias=beta), ref)
    ln0 = so.build_layernorm(x, spike_mult=False)
    rep("LayerNorm exact-mult", ln0(x, weight=gamma, bias=beta), ref, "(invsqrt only)")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
