"""PASN -- Prefix-Adaptive Spiking Neuron (standalone successive-approximation).

A distinct training-free ANN->SNN conversion neuron. It does **not** reuse the MBE
neuron dynamics (that variant is :mod:`mbe.mbe_pasn`); instead it is a deterministic
hard-routed spiking mixture-of-experts:

  1. **Route (free, parameter-free).** Read the FP sign + exponent bits of ``x`` to
     pick a *binade* bank ``|x| in [2^e, 2^{e+1})`` (log-dense near zero). The
     exponent supplies the magnitude scale ``2^e`` exactly, at zero spike cost; only
     the intra-binade residual ``rho in [0,1)`` remains.
  2. **Encode (spiking, fixed).** A successive-approximation (SAR) process spike-codes
     ``rho`` over ``T`` steps: at each step compare the running residual to a fixed
     halving threshold (1/2, 1/4, ...), emit a binary spike, subtract. This is a
     canonical few-spike temporal code (FS-neuron family) -- a genuine spiking
     neuron (threshold / fire / reset), and most inputs emit only a few spikes.
  3. **Decode (learned, per bank).** A small per-bank linear (optionally low-order
     polynomial) readout maps the spike code to ``f(x)``. The readout is the *only*
     learnable part and is solved in closed form (least squares) -- no
     surrogate-gradient training of any spike dynamics.

Because each bank sees a narrow, near-linear slice of ``f`` (and near zero the
binades are exponentially dense), a low-order readout suffices, so accuracy is high
where a single global neuron fails (GELU near zero) while spikes/memory stay small.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import functions


# --------------------------------------------------------------------------
# Successive-approximation (SAR) spike code
# --------------------------------------------------------------------------

@torch.no_grad()
def sar_encode(rho: torch.Tensor, T: int):
    """Spike-code ``rho in [0,1)`` over ``T`` halving steps.

    Returns ``(recon, spikes)``: ``recon`` is the ``T``-bit reconstruction of
    ``rho`` (sum of fired levels), ``spikes`` is the per-element spike (1-bit)
    count -- the energy-relevant quantity.
    """
    u = rho.clone()
    recon = torch.zeros_like(rho)
    spikes = torch.zeros_like(rho)
    lvl = 0.5
    for _ in range(T):
        s = (u >= lvl).to(rho.dtype)
        u = u - s * lvl
        recon = recon + s * lvl
        spikes = spikes + s
        lvl *= 0.5
    return recon, spikes


def _features(recon: torch.Tensor, order: int) -> torch.Tensor:
    """Polynomial readout features ``[1, r, r^2, ...]`` of the reconstructed
    residual, shape ``(..., order+1)``. ``order=1`` -> piecewise-linear."""
    cols = [torch.ones_like(recon), recon]
    for p in range(2, order + 1):
        cols.append(recon ** p)
    return torch.stack(cols, dim=-1)


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

class PrefixRouter:
    """Deterministic FP sign+exponent router that also yields the intra-binade
    residual ``rho``. Parameter-free.

    Bank 0 = near-zero (signed interval ``(-2^{e_min}, 2^{e_min})``); the rest are
    magnitude binades ``[2^e, 2^{e+1})`` per sign.
    """

    def __init__(self, lo: float, hi: float, e_min: int = -6,
                 e_max: int | None = None):
        self.signed = lo < 0.0
        max_mag = max(abs(lo), abs(hi), 2.0 ** (e_min + 1))
        if e_max is None:
            e_max = int(math.ceil(math.log2(max_mag)))
        if e_max <= e_min:
            e_max = e_min + 1
        self.e_min, self.e_max = int(e_min), int(e_max)
        self.near0_span = 2.0 ** e_min

        self.banks: list[dict] = [dict(kind="near0", e=0, sign=0.0)]
        self.key_to_idx: dict[tuple[float, int], int] = {}
        signs = [-1.0, 1.0] if self.signed else [1.0]
        for s in signs:
            for e in range(self.e_min, self.e_max):
                self.key_to_idx[(s, e)] = len(self.banks)
                self.banks.append(dict(kind="mag", sign=s, e=e))
        # per-bank exponent (0 for near-zero), for vectorised residual extraction
        self._bank_e = torch.tensor([b["e"] for b in self.banks], dtype=torch.int64)

    @property
    def n_banks(self) -> int:
        return len(self.banks)

    @torch.no_grad()
    def route(self, x_flat: torch.Tensor) -> torch.Tensor:
        mag = x_flat.abs()
        near0 = mag < self.near0_span
        e = torch.floor(torch.log2(mag.clamp(min=self.near0_span)))
        e = e.clamp(self.e_min, self.e_max - 1).to(torch.int64)
        s = torch.where(x_flat < 0, -1.0, 1.0)
        idx = torch.zeros_like(x_flat, dtype=torch.int64)
        for (sk, ek), bi in self.key_to_idx.items():
            m = (~near0) & (s == sk) & (e == ek)
            idx = torch.where(m, torch.full_like(idx, bi), idx)
        return torch.where(near0, torch.zeros_like(idx), idx)

    @torch.no_grad()
    def residual(self, x_flat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Intra-binade residual ``rho in [0,1)`` for each element (given its bank)."""
        bank_e = self._bank_e.to(idx.device)
        e = bank_e[idx].to(x_flat.dtype)
        near0 = idx == 0
        rho_mag = x_flat.abs() / torch.exp2(e) - 1.0
        rho_near0 = (x_flat + self.near0_span) / (2.0 * self.near0_span)
        rho = torch.where(near0, rho_near0, rho_mag)
        return rho.clamp(0.0, 1.0 - 1e-6)


# --------------------------------------------------------------------------
# Neuron
# --------------------------------------------------------------------------

class PASNNeuron(nn.Module):
    """Prefix-routed successive-approximation spiking neuron.

    ``W`` holds the per-bank readout coefficients, shape ``(n_banks, order+1)``.
    The router and SAR encoder are deterministic; ``W`` (a buffer, not a gradient
    parameter) is the only learned state and is solved by :func:`build_pasn`.
    """

    def __init__(self, router: PrefixRouter, W: torch.Tensor, T: int, order: int):
        super().__init__()
        self.router = router
        self.T = int(T)
        self.order = int(order)
        self.register_buffer("W", W)

    def _encode(self, flat: torch.Tensor):
        idx = self.router.route(flat)
        rho = self.router.residual(flat, idx)
        recon, spikes = sar_encode(rho, self.T)
        return idx, recon, spikes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        idx, recon, _ = self._encode(flat)
        feats = _features(recon, self.order)          # (M, order+1)
        w = self.W[idx]                               # (M, order+1)
        out = (feats * w).sum(dim=-1)
        return out.reshape(x.shape)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Interface parity with the MBE neuron (identity use in FP-multiply)."""
        return self.forward(x)

    # -- cost introspection ------------------------------------------------
    @torch.no_grad()
    def mean_spikes(self, x: torch.Tensor) -> float:
        flat = x.reshape(-1)
        if flat.numel() == 0:
            return 0.0
        _, _, spikes = self._encode(flat)
        return float(spikes.mean())

    @torch.no_grad()
    def stored_params(self) -> int:
        return int(self.W.numel())

    @torch.no_grad()
    def cost(self, x: torch.Tensor) -> dict:
        return dict(stored=self.stored_params(), active=self.order + 1,
                    steps=self.T, spikes=self.mean_spikes(x))

    @torch.no_grad()
    def bank_usage(self, x: torch.Tensor) -> dict:
        idx = self.router.route(x.reshape(-1))
        return {bi: int((idx == bi).sum()) for bi in range(self.router.n_banks)}


# --------------------------------------------------------------------------
# Builder (closed-form, training-free)
# --------------------------------------------------------------------------

def _solve_ls(feats: torch.Tensor, y: torch.Tensor, ridge: float = 1e-8):
    """Ridge normal-equations least squares (robust; avoids the MKL gelsd path)."""
    G = feats.t() @ feats
    rhs = feats.t() @ y
    K = G.shape[0]
    eye = torch.eye(K, dtype=feats.dtype, device=feats.device)
    lam = max(ridge, 0.0)
    for _ in range(6):
        try:
            return torch.linalg.solve(G + lam * eye, rhs)
        except RuntimeError:
            lam = max(lam, 1e-12) * 100.0
    return torch.linalg.pinv(G) @ rhs


@torch.no_grad()
def _sample_bank(spec: dict, router: PrefixRouter, m: int, device) -> torch.Tensor:
    if spec["kind"] == "near0":
        span = router.near0_span
        return (torch.rand(m, device=device) * 2.0 - 1.0) * span
    e, sign = spec["e"], spec["sign"]
    mag = torch.rand(m, device=device) * (2.0 ** (e + 1) - 2.0 ** e) + 2.0 ** e
    return sign * mag


def build_pasn(name: str, domain: tuple[float, float], e_min: int = -6,
               e_max: int | None = None, T: int = 6, order: int = 1,
               m_per_bank: int = 3000, seed: int = 0, verbose: bool = False,
               device: torch.device | str = "cpu") -> PASNNeuron:
    """Build + fit a standalone PASN neuron for target ``name`` on ``domain``.

    Each binade bank's readout is fit by closed-form least squares on samples drawn
    within that bank (no gradient/surrogate training). ``T`` = SAR bits (spike
    budget); ``order`` = readout polynomial order (1 = piecewise-linear).
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    fn, _ = functions.REGISTRY[name]
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max)
    W = torch.zeros(router.n_banks, order + 1, device=device)
    for bi, spec in enumerate(router.banks):
        xs = _sample_bank(spec, router, m_per_bank, device)
        idx = router.route(xs)
        rho = router.residual(xs, idx)
        recon, _ = sar_encode(rho, T)
        feats = _features(recon, order)
        W[bi] = _solve_ls(feats, fn(xs))
        if verbose:
            k = "near0" if spec["kind"] == "near0" else \
                f"mag s={spec['sign']:+.0f} e={spec['e']}"
            print(f"    bank {bi:2d}/{router.n_banks - 1:2d} {k:16s}", flush=True)
    return PASNNeuron(router, W, T, order)
