"""P0.4 Block B follow-up — is the global MBE's *fit* what fails at high n_basis?

GPT-2-medium gave ppl 12903 / 611 / 589287 at (act, ln) basis (4,6) / (6,8) /
(8,12). Adding capacity made the converted model 964x worse, which invites the
reading that our baseline build destabilises -- and that reading would mean we
never gave the baseline its best shot, which is the first thing a reviewer
attacks (trap 3, one level down: numerical rather than hyperparameter).

This settles it in 1-D, where a sweep costs minutes instead of hours. It fits the
two primitives the GPT-2 baseline is made of -- the signed GELU activation and
the whole global LayerNorm -- across n_basis, and reports how well each matches
its target.

The answer (2026-08-01): **the fits do not destabilise.** Both primitives are
*more* accurate at the 8/12 setting than at 6/8, while the end-to-end perplexity
is 964x worse. Error and perplexity move in opposite directions, so a broken fit
cannot be the explanation. What is left is that ppl past ~600 (against an ANN's
21.7) is not an ordered signal at all: the model is already destroyed, and the
map from primitive error to perplexity is not monotone there. Report the three
baseline settings as "all diverge", never as a frontier.

Caveat: synthetic gaussian samples, not GPT-2's calibration distributions. The
conclusion is directional ("more bases does not fit worse"), which is insensitive
to that, but the absolute errors are not the converted model's.

    python experiments/baseline_nbasis_sweep.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mbe import spiking_ops as so  # noqa: E402


def sweep_activation(grid_n, epochs: int, seed: int) -> None:
    """Signed-MBE GELU fit error vs n_basis, on two input widths."""
    torch.manual_seed(seed)
    x = torch.linspace(-12, 12, 4001)
    ref = F.gelu(x)
    for tag, scale in (("gauss(0,2)", 2.0), ("gauss(0,6)", 6.0)):
        s = torch.randn(20000) * scale
        lo, hi = float(s.min()), float(s.max())
        m = (x >= lo) & (x <= hi)
        print(f"\n=== GELU vs n_basis   sample={tag}  "
              f"range {lo:.1f}..{hi:.1f} ===")
        print(f"{'N':>3} {'in-domain MSE':>15} {'max|err|':>10}")
        for n in grid_n:
            act = so.build_activation("gelu", s, n_basis=n, n_steps=16,
                                      seed=seed, epochs=epochs, device="cpu")
            with torch.no_grad():
                y = act.neuron(x[m])
            print(f"{n:3d} {float(((y - ref[m]) ** 2).mean()):15.6e} "
                  f"{float((y - ref[m]).abs().max()):10.4f}")


def sweep_layernorm(grid_n, epochs: int, seed: int) -> None:
    """Whole global SpikingLayerNorm error vs n_basis, at gpt2-medium width."""
    torch.manual_seed(seed)
    D = 1024
    sample = torch.randn(256, D) * 1.5 + 0.3
    ref = F.layer_norm(sample, (D,), eps=1e-5)
    denom = ref.abs().mean().clamp(min=1e-6)
    print(f"\n=== LayerNorm vs n_basis   D={D} ===")
    print(f"{'n_ln':>5} {'rel|err| vs torch':>19} {'max|err|':>10}")
    for n in grid_n:
        ln = so.build_layernorm(sample, eps=1e-5, n_basis=n, n_steps=16,
                                epochs=epochs, seed=seed, spike_mult=True,
                                readout_order=2, log_sample=True, device="cpu")
        with torch.no_grad():
            y = ln(sample)
        print(f"{n:5d} {float((y - ref).abs().mean() / denom):19.6e} "
              f"{float((y - ref).abs().max()):10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=300,
                    help="same as the GPT-2 runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act-basis", type=int, nargs="+",
                    default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--ln-basis", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--skip", choices=["activation", "layernorm"], default=None)
    a = ap.parse_args()
    if a.skip != "activation":
        sweep_activation(a.act_basis, a.epochs, a.seed)
    if a.skip != "layernorm":
        sweep_layernorm(a.ln_basis, a.epochs, a.seed)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
