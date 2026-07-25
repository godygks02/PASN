"""Fit an MBE neuron to a 1-D target function.

Training protocol
-----------------
The paper (Appendix G.1) states only: learning rate 0.01, 200 epochs, an
"exponential optimizer with decay rate 0.99". It does not specify the optimizer,
the surrogate gradient, or the parameter initialisation (all flagged as open
issues in ``MBE_Implementation_Plan.md`` section 5). We use:

  1. Adam warm-up with ``ExponentialLR(gamma)`` -- the paper's stated schedule.
  2. An optional LBFGS refinement with a *sharpening* surrogate (annealing
     ``surrogate_alpha`` upward) so the soft training surrogate converges toward
     the true Heaviside used at inference. This second-order polish is what
     reliably reaches the paper's ~1e-4..1e-5 approximation floor on CPU; it is
     an implementation choice, not from the paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .neuron import MBENeuron, MBEConfig


@dataclass
class FitResult:
    mse: float
    final_lr: float
    firing_rate: float
    num_params: int
    loss_history: list = field(repr=False)


@torch.no_grad()
def solve_readout(model, x, y, ridge: float = 1e-8):
    """Set the linear readout ``(w, bias)`` to its exact least-squares optimum,
    given the current non-readout parameters.

    ``forward = readout_features(x) @ w (+ bias)`` is linear in ``(w, bias)`` once
    the spike-generating parameters are fixed, so the optimal readout has a
    closed form. Interleaving this with gradient steps on the shape parameters
    lets the fit approach the representation ceiling far more reliably than
    surrogate-gradient descent on ``w`` alone.
    """
    feats = model.readout_features(x)               # (M, K)
    has_bias = getattr(model, "bias", None) is not None
    if has_bias:
        feats = torch.cat([feats, torch.ones(feats.shape[0], 1, dtype=feats.dtype)], 1)
    # Ridge-regularised normal equations: (F^T F + lambda I) c = F^T y. K is tiny
    # (<= ~9), so this is cheap, and it avoids the MKL ``gelsd`` lstsq path that is
    # buggy for some shapes on Windows. The Tikhonov term keeps it well-posed even
    # when bases are collinear (e.g. the decay-off ablation); ridge is escalated on
    # a (rare) solve failure rather than crashing.
    K = feats.shape[1]
    G = feats.t() @ feats                            # (K, K)
    rhs = feats.t() @ y                              # (K,)
    eye = torch.eye(K, dtype=feats.dtype)
    lam = max(ridge, 0.0)
    for _ in range(6):
        try:
            coeffs = torch.linalg.solve(G + lam * eye, rhs)
            break
        except RuntimeError:
            lam = max(lam, 1e-12) * 100.0
    else:
        coeffs = torch.linalg.pinv(G) @ rhs          # last-resort pseudo-inverse
    model.set_readout(coeffs)


def fit_function(
    x: torch.Tensor,
    y: torch.Tensor,
    config: MBEConfig,
    seed: int = 0,
    **kwargs,
) -> tuple[MBENeuron, FitResult]:
    """Build an MBE neuron from ``config`` and fit it to ``y = f(x)``."""
    torch.manual_seed(seed)
    model = MBENeuron(config)
    result = fit_model(model, x, y, seed=seed, **kwargs)
    return model, result


def fit_model(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.03,
    epochs: int = 1000,
    gamma: float = 0.999,
    solve_every: int = 50,
    lbfgs_refine: bool = True,
    lbfgs_alphas: tuple = (6.0, 12.0, 20.0),
    lbfgs_iters: int = 80,
    ridge: float = 1e-8,
    log_every: int = 0,
    seed: int = 0,
) -> FitResult:
    """Fit any neuron exposing ``readout_features`` / ``set_readout`` / ``cfg``.

    Works for both :class:`MBENeuron` and :class:`SignedMBENeuron`. The readout
    ``(w, bias)`` is periodically solved in closed form (:func:`solve_readout`)
    while gradient descent optimises the spike-generating (shape) parameters.
    """
    torch.manual_seed(seed)
    loss_fn = torch.nn.MSELoss()

    # Start from the optimal readout for the initial (random) shape parameters.
    solve_readout(model, x, y, ridge)

    # Adam optimises only the shape parameters; the readout is handled by the
    # closed-form solve, which is exact and avoids surrogate-gradient noise on w.
    shape_params = [p for name, p in model.named_parameters()
                    if p.requires_grad and not _is_readout(name)]
    opt = torch.optim.Adam(shape_params, lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        if solve_every and (epoch + 1) % solve_every == 0:
            solve_readout(model, x, y, ridge)
        history.append(loss.item())
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [adam] epoch {epoch:4d}  mse={loss.item():.3e}  "
                  f"lr={sched.get_last_lr()[0]:.4e}")

    solve_readout(model, x, y, ridge)

    # LBFGS polish on shape params with a sharpening surrogate; re-solve the
    # readout after each round and roll back rounds that do not improve.
    if lbfgs_refine:
        best_mse = loss_fn(model(x), y).item()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for sa in lbfgs_alphas:
            model.cfg.surrogate_alpha = sa
            opt = torch.optim.LBFGS(shape_params, lr=0.3, max_iter=lbfgs_iters,
                                    line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                return loss

            try:
                opt.step(closure)
                solve_readout(model, x, y, ridge)
                cur = loss_fn(model(x), y).item()
            except RuntimeError:
                cur = float("inf")
            if math.isfinite(cur) and cur < best_mse:
                best_mse = cur
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                model.load_state_dict(best_state)
            history.append(best_mse)
        model.load_state_dict(best_state)

    with torch.no_grad():
        final_mse = loss_fn(model(x), y).item()
    return FitResult(
        mse=final_mse,
        final_lr=sched.get_last_lr()[0],
        firing_rate=model.firing_rate(x),
        num_params=model.num_learnable(),
        loss_history=history,
    )


def _is_readout(param_name: str) -> bool:
    """True for the linear-readout parameters (``w`` / ``bias``) of either
    neuron type -- these are solved in closed form, not by the gradient step."""
    leaf = param_name.rsplit(".", 1)[-1]
    return leaf in ("w", "bias")
