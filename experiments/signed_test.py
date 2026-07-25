"""Test the user's signed-basis idea for the GELU / non-monotonic-bend gap.

Hypothesis: split the input at a pivot (x=0). Positive bases encode relu(x-pivot),
negative bases encode relu(pivot-x) (the negative magnitude). Each side is a
standard MBE neuron on a non-negative input; the negative side's weights carry
the sign. This dedicates resolution to the negative region and puts the pivot at
GELU's high-curvature point.

Compares, at equal total N:
  - baseline : one MBENeuron (make_config)
  - signed   : SignedMBE (pos + neg halves)

Usage:  python experiments/signed_test.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mbe import MBENeuron, MBEConfig, functions, fit_function  # noqa: E402


class SignedMBE(nn.Module):
    """Positive/negative split MBE neuron: f(x) = pos(relu(x-c)) + neg(relu(c-x)) + b."""

    def __init__(self, pos_cfg: MBEConfig, neg_cfg: MBEConfig, pivot: float = 0.0):
        super().__init__()
        self.pos = MBENeuron(pos_cfg)     # inner bias disabled (see build_signed)
        self.neg = MBENeuron(neg_cfg)
        self.pivot = pivot
        self.bias = nn.Parameter(torch.zeros(1))
        # expose cfg so fit_function's surrogate-annealing can reach both halves
        self.cfg = pos_cfg

    def forward(self, x):
        xp = torch.relu(x - self.pivot)
        xn = torch.relu(self.pivot - x)
        out = self.pos(xp) + self.neg(xn) + self.bias
        # keep both halves' surrogate sharpness in sync with the fit schedule
        self.neg.cfg.surrogate_alpha = self.pos.cfg.surrogate_alpha
        return out

    def firing_rate(self, x):
        xp = torch.relu(x - self.pivot); xn = torch.relu(self.pivot - x)
        return 0.5 * (self.pos.firing_rate(xp) + self.neg.firing_rate(xn))

    def num_learnable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_signed(fn, lo, hi, n_pos, n_neg, pivot=0.0, n_steps=16):
    span_p, span_n = hi - pivot, pivot - lo
    a_pos = functions.curvature_alpha(lambda t: fn(pivot + t), 0.0, span_p, n_pos)
    a_neg = functions.curvature_alpha(lambda t: fn(pivot - t), 0.0, span_n, n_neg)
    pos_cfg = MBEConfig(n_basis=n_pos, n_steps=n_steps, x_min=0.0, x_scale=span_p,
                        alpha_v=a_pos, use_bias=False)
    neg_cfg = MBEConfig(n_basis=n_neg, n_steps=n_steps, x_min=0.0, x_scale=span_n,
                        alpha_v=a_neg, use_bias=False)
    return SignedMBE(pos_cfg, neg_cfg, pivot=pivot)


def fit_model(model, x, y, lr=0.03, epochs=1000, gamma=0.999,
              lbfgs_alphas=(6.0, 12.0, 20.0), seed=0):
    """Same Adam + LBFGS-anneal recipe as fit_function, for a generic module."""
    torch.manual_seed(seed)
    with torch.no_grad():
        if hasattr(model, "bias") and model.bias is not None:
            model.bias.fill_(y.mean().item())
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    for _ in range(epochs):
        opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward()
        opt.step(); sched.step()
    best = loss_fn(model(x), y).item()
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for sa in lbfgs_alphas:
        model.cfg.surrogate_alpha = sa
        opt = torch.optim.LBFGS(model.parameters(), lr=0.3, max_iter=80,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); l = loss_fn(model(x), y); l.backward(); return l
        try:
            opt.step(closure); cur = loss_fn(model(x), y).item()
        except RuntimeError:
            cur = float("inf")
        if math.isfinite(cur) and cur < best:
            best = cur; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(best_state)
    model.load_state_dict(best_state)
    return best


def run():
    fn = functions.REGISTRY["gelu"][0]
    for dom in [(-5, 5), (-120, 10)]:
        x, y, _ = functions.sample("gelu", m=4000, seed=0, domain=dom)
        # baseline: single neuron N=8
        base_cfg = functions.make_config("gelu", n_basis=8, n_steps=16, domain=dom)
        _, br = fit_function(x, y, base_cfg, seed=0)
        # signed: 4 pos + 4 neg (equal total N=8)
        sm = build_signed(fn, dom[0], dom[1], n_pos=4, n_neg=4, pivot=0.0)
        base_params = MBENeuron(base_cfg).num_learnable()
        sr = fit_model(sm, x, y, seed=0)
        print(f"GELU {str(dom):10s}  baseline(N=8)={br.mse:.2e}   "
              f"signed(4+4)={sr:.2e}   (params: base={base_params}, signed={sm.num_learnable()})")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    run()
