"""Target nonlinear functions and their sampling domains.

Domains follow the paper / appendix G.1:
  - GELU:    (-120, 10)   [N=4, T=16]
  - Tanh:    activation range
  - 2^x:     [0, 1]       (fractional part of the change-of-base exponent)
  - 1/x:     [0.5, 1]
  - 1/sqrt:  [0.5, 2]
  - SiLU:    Table VII intervals [-8,-2], [-2,5], [2,12]
"""
from __future__ import annotations

import math

import torch

# --- target functions ----------------------------------------------------

def gelu(x: torch.Tensor) -> torch.Tensor:
    # exact (erf) GELU
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # tanh-approx GELU (HF ``NewGELUActivation``, used by GPT-2): calibrate the
    # spiking neuron to the *same* activation the ANN uses.
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def reciprocal(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / x


def inv_sqrt(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.sqrt(x)


def exp2(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, x)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


# name -> (callable, (domain_lo, domain_hi))
REGISTRY = {
    "gelu": (gelu, (-120.0, 10.0)),
    "gelu_tanh": (gelu_tanh, (-120.0, 10.0)),
    "tanh": (tanh, (-5.0, 5.0)),
    "silu": (silu, (-8.0, 12.0)),
    "inv": (reciprocal, (0.5, 1.0)),
    "invsqrt": (inv_sqrt, (0.5, 2.0)),
    "exp2": (exp2, (0.0, 1.0)),
    "identity": (identity, (0.0, 1.0)),
}


def make_config(name: str, n_basis: int, n_steps: int = 16, decay: bool = True,
                domain: tuple[float, float] | None = None, spread: float = 6.0,
                **overrides):
    """Build a domain-appropriate :class:`MBEConfig` for a target function.

    The input domain ``[lo, hi]`` is affine-normalised to ``[0, 1]`` for the
    membrane (``x_min=lo``, ``x_scale=hi-lo``); the DC bias supplies any non-zero
    value at ``x=lo``. Signed and positive domains are handled identically.
    Per-basis leading thresholds are log-spread over ``[2^-spread, 1]``.
    """
    from .neuron import MBEConfig

    fn, default_domain = REGISTRY[name]
    lo, hi = domain if domain is not None else default_domain

    alpha_v = curvature_alpha(fn, lo, hi, n_basis, spread)
    cfg = dict(
        n_basis=n_basis, n_steps=n_steps, decay=decay,
        x_min=float(lo), x_scale=float(hi - lo),
        alpha_v=alpha_v,
    )
    cfg.update(overrides)
    return MBEConfig(**cfg)


def make_signed(name: str, n_pos: int, n_neg: int, pivot: float = 0.0,
                n_steps: int = 16, decay: bool = True,
                domain: tuple[float, float] | None = None, spread: float = 6.0):
    """Build a :class:`SignedMBENeuron` for a target function (polarity split).

    The positive bank encodes ``relu(x-pivot)`` over ``[0, hi-pivot]`` and the
    negative bank encodes ``relu(pivot-x)`` over ``[0, pivot-lo]``; each side's
    leading thresholds are curvature-placed for its own restriction of ``f``.
    """
    from .neuron import MBEConfig
    from .signed import SignedMBENeuron

    fn, default_domain = REGISTRY[name]
    lo, hi = domain if domain is not None else default_domain
    span_p, span_n = hi - pivot, pivot - lo
    pos_cfg = MBEConfig(
        n_basis=n_pos, n_steps=n_steps, decay=decay, x_min=0.0, x_scale=span_p,
        alpha_v=curvature_alpha(lambda t: fn(pivot + t), 0.0, span_p, n_pos, spread),
    )
    neg_cfg = MBEConfig(
        n_basis=n_neg, n_steps=n_steps, decay=decay, x_min=0.0, x_scale=span_n,
        alpha_v=curvature_alpha(lambda t: fn(pivot - t), 0.0, span_n, n_neg, spread),
    )
    return SignedMBENeuron(pos_cfg, neg_cfg, pivot=pivot)


def curvature_alpha(fn, lo, hi, n_basis, spread=6.0):
    """Curvature-weighted per-basis leading thresholds (Algorithm 1, step 3).

    Places each basis's leading threshold at the quantiles of |f'| of the target
    over the normalised input ``u=(x-lo)/(hi-lo) in (0,1]``, so resolution
    concentrates on high-curvature regions instead of flat tails. Returns a list
    of length ``n_basis`` sorted descending.
    """
    grid = torch.linspace(lo, hi, 4000)
    u = (grid - lo) / (hi - lo)
    var = fn(grid).diff().abs()
    var = var / var.sum().clamp(min=1e-12)
    cdf = torch.cumsum(var, dim=0)
    q = (torch.arange(n_basis, dtype=torch.float32) + 0.5) / n_basis
    pos = torch.tensor([
        torch.interp(qi, cdf, u[1:]) if hasattr(torch, "interp")
        else _interp(qi, cdf, u[1:]) for qi in q
    ])
    alpha_v = pos.clamp(min=2.0 ** (-spread), max=1.0)
    return torch.sort(alpha_v, descending=True).values.tolist()


def _interp(xq, xp, fp):
    """Scalar linear interpolation (fallback if torch.interp is unavailable)."""
    xq = float(xq)
    idx = int(torch.searchsorted(xp, torch.tensor(xq)).clamp(1, len(xp) - 1))
    x0, x1 = float(xp[idx - 1]), float(xp[idx])
    y0, y1 = float(fp[idx - 1]), float(fp[idx])
    if x1 == x0:
        return y1
    return y0 + (y1 - y0) * (xq - x0) / (x1 - x0)


def sample(name: str, m: int = 10_000, domain: tuple[float, float] | None = None,
           seed: int | None = 0):
    """Uniformly sample ``m`` points from a target function's domain.

    Returns ``(x, y, (lo, hi))``.
    """
    fn, default_domain = REGISTRY[name]
    lo, hi = domain if domain is not None else default_domain
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(m, generator=g) * (hi - lo) + lo
    else:
        x = torch.rand(m) * (hi - lo) + lo
    x, _ = torch.sort(x)
    y = fn(x)
    return x, y, (lo, hi)
