"""PASN -- Prefix-Adaptive Spiking Neuron (v2: adaptive-budget prefix routing).

Concretization of ``PASN_method.md`` sections 9-13. A PASN neuron is a bank of
per-range MBE neurons plus a deterministic, parameter-free FP-exponent router:

  * ``PrefixRouter``  -- reads sign + exponent bits, i.e. routes ``x`` into a
    *binade* ``|x| in [2^e, 2^{e+1})`` (log-dense near zero); a single near-zero
    bank covers the signed region ``|x| < 2^{e_min}``.
  * each **magnitude bank** ``(sigma, e)`` approximates only the intra-binade
    residual ``g(rho) = f(sigma 2^e (1+rho))`` on ``rho in [0,1)`` -- the exponent
    ``2^e`` is supplied exactly by routing at zero spike cost.
  * heterogeneous per-bank budget ``N_j`` (flat tails -> N=1; curved near-zero ->
    larger N).

Key properties: deterministic routing, one active bank per input, and R=1 (single
binade, one sign) reduces bit-identically to the baseline MBE neuron.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .neuron import MBENeuron, MBEConfig
from . import functions


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

class PrefixRouter:
    """Deterministic FP sign+exponent router (parameter-free).

    Bank 0 is always the near-zero bank (signed interval ``(-2^{e_min},
    2^{e_min})``). The remaining banks are magnitude binades ``[2^e, 2^{e+1})`` for
    each sign, keyed by ``(sign, e)``.
    """

    def __init__(self, lo: float, hi: float, e_min: int = -2,
                 e_max: int | None = None):
        self.signed = lo < 0.0
        max_mag = max(abs(lo), abs(hi), 2.0 ** (e_min + 1))
        if e_max is None:
            e_max = int(math.ceil(math.log2(max_mag)))
        if e_max <= e_min:
            e_max = e_min + 1
        self.e_min, self.e_max = int(e_min), int(e_max)
        self.near0_span = 2.0 ** e_min

        # bank 0 = near-zero; then (sign, e) magnitude binades
        self.banks: list[dict] = [dict(kind="near0",
                                       x_min=-self.near0_span,
                                       x_scale=2.0 * self.near0_span)]
        self.key_to_idx: dict[tuple[float, int], int] = {}
        signs = [-1.0, 1.0] if self.signed else [1.0]
        for s in signs:
            for e in range(self.e_min, self.e_max):
                self.key_to_idx[(s, e)] = len(self.banks)
                self.banks.append(dict(kind="mag", sign=s, e=e,
                                       x_min=2.0 ** e, x_scale=2.0 ** e))

    @property
    def n_banks(self) -> int:
        return len(self.banks)

    @torch.no_grad()
    def route(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Bank index (LongTensor) for each element of a 1-D tensor."""
        mag = x_flat.abs()
        near0 = mag < self.near0_span
        e = torch.floor(torch.log2(mag.clamp(min=self.near0_span)))
        e = e.clamp(self.e_min, self.e_max - 1).to(torch.int64)
        s = torch.where(x_flat < 0, -1.0, 1.0)
        idx = torch.zeros_like(x_flat, dtype=torch.int64)   # default -> near0 (0)
        for (sk, ek), bi in self.key_to_idx.items():
            m = (~near0) & (s == sk) & (e == ek)
            idx = torch.where(m, torch.full_like(idx, bi), idx)
        idx = torch.where(near0, torch.zeros_like(idx), idx)
        return idx


# --------------------------------------------------------------------------
# PASN neuron
# --------------------------------------------------------------------------

class PASNNeuron(nn.Module):
    """A router + a bank of MBE neurons (one per routed range).

    ``banks[i]`` corresponds to ``router.banks[i]``. Magnitude banks are fed the
    magnitude ``|x|`` (their config normalises it to the intra-binade residual);
    the near-zero bank is fed the raw signed ``x``.
    """

    def __init__(self, router: PrefixRouter, banks: list[MBENeuron]):
        super().__init__()
        assert len(banks) == router.n_banks
        self.router = router
        self.bank_mods = nn.ModuleList(banks)

    def _bank_input(self, flat: torch.Tensor, mask: torch.Tensor, spec: dict):
        sub = flat[mask]
        return sub if spec["kind"] == "near0" else sub.abs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        idx = self.router.route(flat)
        out = torch.zeros_like(flat)
        for bi, bank in enumerate(self.bank_mods):
            mask = idx == bi
            if bool(mask.any()):
                out[mask] = bank(self._bank_input(flat, mask, self.router.banks[bi]))
        return out.reshape(x.shape)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Spike-decoded reconstruction (== forward). Interface parity with
        :meth:`MBENeuron.reconstruct` so a PASN neuron is a drop-in identity for the
        spike-driven FP multiply (:func:`spiking_ops.spiking_multiply`)."""
        return self.forward(x)

    # -- cost introspection (the headline metric) --------------------------
    @torch.no_grad()
    def mean_spikes(self, x: torch.Tensor) -> float:
        """Mean number of spikes emitted per input element.

        For each input only its routed bank runs; that bank emits
        ``firing_rate * N_j * T_j`` spikes. Averaging over inputs gives the
        energy-relevant spike count PASN trades against MBE.
        """
        flat = x.reshape(-1)
        idx = self.router.route(flat)
        total = 0.0
        for bi, bank in enumerate(self.bank_mods):
            mask = idx == bi
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            inp = self._bank_input(flat, mask, self.router.banks[bi])
            fr = bank.firing_rate(inp)                    # fraction of (cnt,N,T)
            total += fr * cnt * bank.cfg.n_basis * bank.cfg.n_steps
        return total / max(flat.numel(), 1)

    @torch.no_grad()
    def stored_bases(self) -> int:
        return sum(b.cfg.n_basis for b in self.bank_mods)

    @torch.no_grad()
    def bank_usage(self, x: torch.Tensor) -> dict:
        flat = x.reshape(-1)
        idx = self.router.route(flat)
        return {bi: int((idx == bi).sum()) for bi in range(self.router.n_banks)}


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

def _fit_bank(fn, sign, feed_lo, feed_hi, n_basis, n_steps, epochs, seed,
              device):
    """Fit one MBE bank and return ``(bank, mse)``. ``[feed_lo,feed_hi]`` is the
    range of the value actually fed to the neuron (magnitude for mag banks); the
    true target is ``fn(sign*feed)`` (``sign=None`` for the raw near-zero bank)."""
    from .fit import fit_model
    device = torch.device(device)
    g = torch.Generator(device=device).manual_seed(seed)
    feed = torch.rand(4000, generator=g, device=device) * (
        feed_hi - feed_lo
    ) + feed_lo
    feed, _ = torch.sort(feed)
    y = fn(sign * feed) if sign is not None else fn(feed)
    alpha = functions.curvature_alpha(
        (lambda t: fn(sign * t)) if sign is not None else fn,
        feed_lo, feed_hi, n_basis)
    cfg = MBEConfig(n_basis=n_basis, n_steps=n_steps, x_min=float(feed_lo),
                    x_scale=float(feed_hi - feed_lo), alpha_v=alpha, use_bias=True)
    bank = MBENeuron(cfg).to(device)
    res = fit_model(bank, feed, y, seed=seed, epochs=epochs)
    return bank, res.mse


def _fit_bank_adaptive(fn, sign, feed_lo, feed_hi, target_mse, n_max,
                       n_steps, epochs, seed, device):
    """Smallest-N bank reaching ``target_mse`` (flat tails settle at N=1). Returns
    the first N whose MSE is under target, else the best over ``1..n_max``."""
    best = None
    for N in range(1, n_max + 1):
        bank, mse = _fit_bank(
            fn, sign, feed_lo, feed_hi, N, n_steps, epochs, seed, device
        )
        if best is None or mse < best[1]:
            best = (bank, mse)
        if mse <= target_mse:
            return bank, mse
    return best


def build_pasn(name: str, domain: tuple[float, float], e_min: int = -2,
               e_max: int | None = None, n_local: int = 2, n_near0: int = 4,
               n_steps: int = 16, epochs: int = 300, seed: int = 0,
               adaptive: bool = False, target_mse: float = 1e-4, n_max: int = 6,
               verbose: bool = False,
               device: torch.device | str = "cpu") -> PASNNeuron:
    """Build + fit a PASN neuron for target ``name`` on ``domain``.

    Fixed budget: ``n_local`` bases per magnitude binade, ``n_near0`` near-zero.
    ``adaptive=True``: each bank takes the smallest N (up to ``n_max``) reaching
    ``target_mse`` -- flat tails collapse to N=1, cutting stored bases and spikes.
    """
    fn, _ = functions.REGISTRY[name]
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max)
    banks: list[MBENeuron] = []
    for spec in router.banks:
        if spec["kind"] == "near0":
            span, sign, flo, fhi = router.near0_span, None, -router.near0_span, router.near0_span
            cap = n_near0
        else:
            s, e = spec["sign"], spec["e"]
            sign, flo, fhi = s, 2.0 ** e, 2.0 ** (e + 1)     # magnitude binade
            cap = n_local
        if adaptive:
            bank, mse = _fit_bank_adaptive(fn, sign, flo, fhi, target_mse,
                                           n_max, n_steps, epochs, seed, device)
        else:
            bank, mse = _fit_bank(
                fn, sign, flo, fhi, cap, n_steps, epochs, seed, device
            )
        banks.append(bank)
        if verbose:
            k = "near0" if spec["kind"] == "near0" else \
                f"mag s={spec['sign']:+.0f} e={spec['e']}"
            print(
                f"    bank {len(banks)-1:2d}/{router.n_banks - 1:2d} "
                f"{k:16s} N={bank.cfg.n_basis} mse={mse:.2e}",
                flush=True,
            )
    return PASNNeuron(router, banks)


@torch.no_grad()
def neuron_cost(neuron, x: torch.Tensor) -> dict:
    """Unified cost metrics for MBE / signed-MBE / PASN neurons on inputs ``x``.

    Returns a dict with:
      * ``stored``  -- total bases stored (memory / parameters),
      * ``active``  -- mean bases *evaluated per input* (compute breadth; PASN
        runs only the routed bank, MBE/signed run all bases),
      * ``steps``   -- timesteps T,
      * ``spikes``  -- mean spikes emitted per input (SOPs, the energy proxy).
    """
    from .signed import SignedMBENeuron
    if isinstance(neuron, PASNNeuron):
        steps = neuron.bank_mods[0].cfg.n_steps
        flat = x.reshape(-1)
        idx = neuron.router.route(flat)
        counts = torch.tensor([b.cfg.n_basis for b in neuron.bank_mods],
                              device=idx.device, dtype=torch.float32)
        active = float(counts[idx].mean()) if flat.numel() else 0.0
        return dict(stored=neuron.stored_bases(), active=active, steps=steps,
                    spikes=neuron.mean_spikes(x))
    if isinstance(neuron, SignedMBENeuron):
        Np, Nn = neuron.pos.cfg.n_basis, neuron.neg.cfg.n_basis
        steps = neuron.pos.cfg.n_steps
        xp, xn = neuron._split(x)
        spikes = (neuron.pos.firing_rate(xp) * Np
                  + neuron.neg.firing_rate(xn) * Nn) * steps
        # both polarity banks run for every input, so all bases are active
        return dict(stored=Np + Nn, active=Np + Nn, steps=steps, spikes=spikes)
    # plain MBE neuron
    N, steps = neuron.cfg.n_basis, neuron.cfg.n_steps
    return dict(stored=N, active=N, steps=steps,
                spikes=neuron.firing_rate(x) * N * steps)


@torch.no_grad()
def spikes_per_input(neuron, x: torch.Tensor) -> float:
    """Mean spikes per input element for any neuron type (unified cost meter)."""
    return neuron_cost(neuron, x)["spikes"]
