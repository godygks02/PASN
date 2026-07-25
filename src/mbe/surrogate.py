"""Surrogate gradient functions for the Heaviside spike nonlinearity.

The MBE / FS neuron uses a hard threshold ``s = H(u - Vth)`` in the forward
pass. The paper (Wang et al., AAAI-26) does *not* specify which surrogate
gradient is used for the backward pass (this is flagged as an open issue in
``MBE_Implementation_Plan.md`` section 5). We therefore expose several standard
SNN surrogates and default to the ATan surrogate, which is a common, robust
choice (Fang et al., 2021).

Forward:  H(x) = 1 if x >= 0 else 0
Backward: replaced by a smooth bump centred at x = 0, controlled by ``alpha``.
"""
from __future__ import annotations

import torch


class _ATanHeaviside(torch.autograd.Function):
    """Heaviside with the arctan surrogate gradient.

    grad = alpha / (2 * (1 + (pi/2 * alpha * x)^2))
    (derivative of  1/pi * atan(pi/2 * alpha * x) + 1/2 )
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        sg = alpha / (2.0 * (1.0 + (torch.pi / 2.0 * alpha * x) ** 2))
        return grad_output * sg, None


class _SigmoidHeaviside(torch.autograd.Function):
    """Heaviside with the fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        sig = torch.sigmoid(alpha * x)
        sg = alpha * sig * (1.0 - sig)
        return grad_output * sg, None


class _TriangleHeaviside(torch.autograd.Function):
    """Heaviside with a triangular surrogate gradient (compact support)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        sg = torch.clamp(1.0 - alpha * x.abs(), min=0.0) * alpha
        return grad_output * sg, None


_SURROGATES = {
    "atan": _ATanHeaviside,
    "sigmoid": _SigmoidHeaviside,
    "triangle": _TriangleHeaviside,
}


def heaviside(x: torch.Tensor, kind: str = "atan", alpha: float = 2.0) -> torch.Tensor:
    """Spike function with a surrogate gradient.

    Args:
        x: pre-activation ``u - Vth``.
        kind: one of ``atan``, ``sigmoid``, ``triangle``.
        alpha: sharpness of the surrogate (larger = closer to a true step).
    """
    try:
        fn = _SURROGATES[kind]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown surrogate '{kind}', choose from {list(_SURROGATES)}"
        ) from exc
    return fn.apply(x, alpha)
