"""MBE-PASN -- prefix-routed banks of MBE neurons (the MBE-neuron *improvement*).

This is the original PASN: it keeps the MBE neuron dynamics unchanged and only
organises MBE bases into per-range banks selected by a deterministic FP-exponent
router. It is retained as the "MBE + prefix banks" variant and benchmarked against
the standalone :mod:`mbe.pasn` neuron (a distinct successive-approximation design).

A MBE-PASN neuron is a bank of per-range MBE neurons plus a parameter-free router:

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

    ``near0`` selects how the near-zero region is handled:

    * ``"single"`` (default, the original behaviour) -- one bank covering the whole
      signed interval, fed the raw signed ``x``.
    * ``"signed"`` -- the sign bit is read for the near-zero region too, giving one
      bank per polarity over ``[0, 2^{e_min})``, each fed ``|x|``. Every bank then
      sees a single-sign, monotone slice. This matters because targets whose bend
      sits *at* zero (GELU/SiLU/ReLU) put that bend in the middle of the single
      near-zero bank, where a monotone staircase code cannot resolve it -- measured
      as a hard error floor there (see the 연구일지 실험 1/2).

    ``spec["raw"]`` marks the banks that are fed the signed ``x`` rather than
    ``|x|``; consumers should use it instead of testing the bank index.

    **Placement (``beta``, ``gamma_log2``).** The binade grid is anchored at zero,
    so the log-dense ranges sit at zero whether or not that is where the target is
    hard. Routing on the *key* ``t = (x - beta) / 2^{gamma_log2}`` moves the dense
    point to ``beta``. This matters for primitives whose argument never approaches
    zero: ``1/S`` on ``[0.5, 1)`` spans a single binade -- the router degenerates to
    a global neuron -- but its curvature ``2/S^3`` peaks at ``S = 0.5``, and
    ``beta = 0.5`` re-expands exactly that end into many binades. ``beta`` also
    doubles as the polarity-split point (the ``near0="signed"`` boundary), which is
    the same place for the same reason.

    ``gamma`` is restricted to powers of two **on purpose**: dividing by ``2^g`` is
    an integer subtract on the exponent field, so the router stays a bit-read. A
    real-valued gamma would put a multiply in front of every routed input. Note
    that gamma cannot change how many binades a domain spans (scaling maps
    ``[a,b]`` to ``[ga,gb]``, the same ratio) -- it only slides the domain across
    the grid, which is what ``e_min`` already does discretely. ``beta`` is the
    parameter that actually adds ranges.

    Neither is learnable by gradient descent: the bank index is piecewise constant
    in both, so the gradient is zero almost everywhere. They are *calibrated* --
    scored on the same objective used to pick a budget.
    """

    def __init__(self, lo: float, hi: float, e_min: int = -2,
                 e_max: int | None = None, near0: str = "single",
                 beta: float = 0.0, gamma_log2: int = 0):
        if near0 not in ("single", "signed"):
            raise ValueError(f"near0 must be 'single' or 'signed', got {near0!r}")
        self.near0 = near0
        self.beta = float(beta)
        self.gamma_log2 = int(gamma_log2)
        self.gamma = 2.0 ** self.gamma_log2

        t_lo, t_hi = self.to_key(lo), self.to_key(hi)
        self.signed = t_lo < 0.0
        max_mag = max(abs(t_lo), abs(t_hi), 2.0 ** (e_min + 1))
        if e_max is None:
            e_max = int(math.ceil(math.log2(max_mag)))
        if e_max <= e_min:
            e_max = e_min + 1
        self.e_min, self.e_max = int(e_min), int(e_max)
        self.near0_span = 2.0 ** e_min

        signs = [-1.0, 1.0] if self.signed else [1.0]
        self.near0_idx: dict[float, int] = {}
        if near0 == "single":
            self.banks: list[dict] = [dict(kind="near0", raw=True,
                                           x_min=-self.near0_span,
                                           x_scale=2.0 * self.near0_span)]
        else:
            self.banks = []
            for s in signs:
                self.near0_idx[s] = len(self.banks)
                self.banks.append(dict(kind="near0s", raw=False, sign=s,
                                       x_min=0.0, x_scale=self.near0_span))
        self.key_to_idx: dict[tuple[float, int], int] = {}
        for s in signs:
            for e in range(self.e_min, self.e_max):
                self.key_to_idx[(s, e)] = len(self.banks)
                self.banks.append(dict(kind="mag", raw=False, sign=s, e=e,
                                       x_min=2.0 ** e, x_scale=2.0 ** e))

    # -- affine placement ---------------------------------------------------
    def to_key(self, x):
        """``x`` -> routing key ``t = (x - beta) / gamma`` (gamma a power of two)."""
        return (x - self.beta) / self.gamma

    def from_key(self, t):
        """Routing key -> ``x``."""
        return t * self.gamma + self.beta

    @property
    def n_banks(self) -> int:
        return len(self.banks)

    def key_interval(self, bi: int) -> tuple[float, float]:
        """The **key-space** interval bank ``bi`` owns (what its neuron sees)."""
        spec = self.banks[bi]
        if spec["kind"] == "near0":
            return -self.near0_span, self.near0_span
        if spec["kind"] == "near0s":
            return ((0.0, self.near0_span) if spec["sign"] > 0
                    else (-self.near0_span, 0.0))
        e, s = spec["e"], spec["sign"]
        lo, hi = 2.0 ** e, 2.0 ** (e + 1)
        return (lo, hi) if s > 0 else (-hi, -lo)

    def bank_interval(self, bi: int) -> tuple[float, float]:
        """The ``x``-space interval bank ``bi`` is responsible for."""
        a, b = self.key_interval(bi)
        return self.from_key(a), self.from_key(b)      # gamma > 0: order kept

    def reachable(self, bi: int, lo: float, hi: float,
                  eps: float = 1e-12) -> tuple[float, float] | None:
        """Intersection of bank ``bi``'s interval with the calibration domain
        ``[lo, hi]``, or ``None`` when the bank cannot be reached.

        The router's binade grid does not align with an arbitrary calibrated
        domain, so some banks lie entirely outside it (e.g. the signed near-zero
        bank for a positive-only domain such as ``1/sqrt(x)`` on ``[0.5, 2]``).
        Fitting those on their nominal interval evaluates the target outside its
        valid range -- which produced NaN parameters, harmless only because no
        input ever routes there. Callers must use this to clamp or skip.
        """
        a, b = self.bank_interval(bi)
        a, b = max(a, lo), min(b, hi)
        return None if b - a <= eps else (a, b)

    @torch.no_grad()
    def route(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Bank index (LongTensor) for each element of a 1-D tensor."""
        x_flat = self.to_key(x_flat)
        mag = x_flat.abs()
        near0 = mag < self.near0_span
        e = torch.floor(torch.log2(mag.clamp(min=self.near0_span)))
        e = e.clamp(self.e_min, self.e_max - 1).to(torch.int64)
        s = torch.where(x_flat < 0, -1.0, 1.0)
        idx = torch.zeros_like(x_flat, dtype=torch.int64)   # default -> near0 (0)
        for (sk, ek), bi in self.key_to_idx.items():
            m = (~near0) & (s == sk) & (e == ek)
            idx = torch.where(m, torch.full_like(idx, bi), idx)
        if self.near0 == "single":
            return torch.where(near0, torch.zeros_like(idx), idx)
        for sk, bi in self.near0_idx.items():
            idx = torch.where(near0 & (s == sk), torch.full_like(idx, bi), idx)
        return idx


# --------------------------------------------------------------------------
# PASN neuron
# --------------------------------------------------------------------------

class MBEPASNNeuron(nn.Module):
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
        """The routing *key* the bank's neuron sees (the router owns the affine)."""
        sub = self.router.to_key(flat[mask])
        return sub if spec.get("raw", spec["kind"] == "near0") else sub.abs()

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
        """Bases actually stored, counting each shared basis set once.

        Two kinds of sharing occur: unreachable banks reuse a fitted neighbour
        (:func:`fill_unreachable`), and tied banks all reference one prototype
        (:class:`TiedBank`)."""
        seen = {}
        for b in self.bank_mods:
            core = getattr(b, "core", b)
            seen[id(core)] = core.cfg.n_basis
        return sum(seen.values())

    @torch.no_grad()
    def num_learnable(self) -> int:
        """Stored parameters, de-duplicated across shared bank modules."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def budget_table(self) -> list:
        """``(bank index, kind, N, T)`` per bank -- what the budget rule chose."""
        out = []
        for bi, b in enumerate(self.bank_mods):
            spec = self.router.banks[bi]
            out.append((bi, spec["kind"], b.cfg.n_basis, b.cfg.n_steps))
        return out

    @torch.no_grad()
    def bank_usage(self, x: torch.Tensor) -> dict:
        flat = x.reshape(-1)
        idx = self.router.route(flat)
        return {bi: int((idx == bi).sum()) for bi in range(self.router.n_banks)}


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

#: Timestep grid the budget rule snaps to (see :func:`rule_budget`).
T_GRID = (2, 3, 4, 6, 8, 12, 16)


def _grid(lo, hi, m, offset=0.0, device="cpu"):
    """Deterministic midpoint grid; ``offset`` shifts by a fraction of a cell.

    A midpoint grid has strictly lower discrepancy than a random draw for a known
    deterministic target on a known interval, and it removes the run-to-run spread
    entirely (measured: up to 34x before, none after).
    """
    i = torch.arange(m, dtype=torch.float32, device=device)
    u = ((i + 0.5 + offset) / m).clamp(0.0, 1.0)
    return lo + u * (hi - lo)


def _uniform_alpha(n: int) -> list:
    """Leading thresholds at even quantiles of ``rho in [0,1)``.

    A *routed* residual is ~uniform in every bank, so the global neuron's
    log-spread placement saturates bases there (measured per-basis firing rates
    ``[0.88, 0.17, 0.71, 0.95]`` at N=4, i.e. 1-2 useful bases out of 4).
    """
    return [float((n - i - 0.5) / n) for i in range(n)]


#: Effective-bit law ``bits = a + c*log2(T)`` per ``(readout_order, learn_tau)``.
#: Measured on real bank targets (연구일지 실험 1/6/7). The decoder is what binds:
#: a linear readout of an exact binary-search code can only express an affine
#: function of the residual, so the fit distorts the staircase to manufacture
#: curvature and spends resolution doing it. Give the decoder an ``o^2`` term and
#: the same spike train is worth far more.
BIT_LAW = {
    (1, True): (3.29, 0.64),
    (1, False): (3.18, 0.77),
    (2, True): (2.86, 1.20),
    (2, False): (1.69, 2.02),
    (3, True): (2.69, 1.31),
    (3, False): (0.00, 3.10),
}


def bit_law(readout_order: int = 1, learn_tau: bool = True) -> tuple[float, float]:
    """``(a, c)`` of the effective-bit law for a decoder/kernel combination."""
    return BIT_LAW.get((readout_order, bool(learn_tau)), (2.0, 1.7))


def rule_budget(delta: float, target_mse: float, n_max: int = 4,
                t_grid=T_GRID, a: float = 2.0, c: float = 1.7,
                n_cut: tuple = (7.5, 10.0)) -> tuple[int, int]:
    """Per-bank ``(N, T)`` from the bank's dynamic range -- the 실험 1 rule.

    With ``b = log2(delta / sqrt(target_mse))`` the *required resolution in bits*
    and a bank delivering ``a + c*log2(T)`` of them, invert to
    ``T = 2^((b-a)/c)`` and snap up to ``t_grid``. The original constants
    ``(a, c) = (2, 1.7)`` were measured over 463 (bank, target) pairs on five
    activations at ``readout_order=1`` (76% exact, 97% within 2x); use
    :func:`bit_law` to get the constants for another decoder.

    ``N`` is *not* predicted by curvature: 73% of those pairs are best served by a
    single basis, and extra bases only pay once ``T`` runs out of bits. Curvature
    based selection scored worse than always answering 1. ``n_cut`` are the ``b``
    thresholds at which a second and third basis start paying.
    """
    if delta <= 0.0:                      # constant bank -- the bias alone does it
        return 1, t_grid[0]
    b = math.log2(delta / math.sqrt(max(target_mse, 1e-30)))
    t = 2.0 ** ((b - a) / c) if b > a else t_grid[0]
    T = next((g for g in t_grid if g >= t), t_grid[-1])
    N = 1 + int(b >= n_cut[0]) + int(b >= n_cut[1])
    return min(N, n_max), T


def _fit_bank(tgt, feed_lo, feed_hi, n_basis, n_steps, epochs, seed,
              device, alpha_init="auto", m=1024, readout_order=1,
              learn_tau=True, tau_range=None, spike_lambda=0.0):
    """Fit one MBE bank and return ``(bank, mse)``.

    ``[feed_lo, feed_hi]`` is the range of the **routing key** actually fed to the
    neuron (its magnitude, for magnitude banks); ``tgt(v)`` maps that fed value back
    to the target value, i.e. it already composes the router's affine and the bank's
    sign.

    The reported MSE is measured on an **offset grid** that shares no point with
    the fitting grid -- scored on the fitting grid a staircase can nail the grid
    midpoints while drifting between them.
    """
    from .fit import fit_model
    device = torch.device(device)
    xf = _grid(feed_lo, feed_hi, m, 0.0, device)
    xe = _grid(feed_lo, feed_hi, m, 0.25, device)
    yf, ye = tgt(xf), tgt(xe)

    inits = (["uniform", "logspread"] if alpha_init == "auto"
             else [alpha_init])
    best = None
    for kind in inits:
        alpha = (_uniform_alpha(n_basis) if kind == "uniform"
                 else functions.curvature_alpha(tgt, feed_lo, feed_hi, n_basis))
        extra = ({} if tau_range is None
                 else dict(tau_min=tau_range[0], tau_max=tau_range[1]))
        cfg = MBEConfig(n_basis=n_basis, n_steps=n_steps, x_min=float(feed_lo),
                        x_scale=float(feed_hi - feed_lo), alpha_v=alpha,
                        use_bias=True, readout_order=readout_order,
                        learn_tau=learn_tau, **extra)
        bank = MBENeuron(cfg).to(device)
        fit_model(bank, xf, yf, seed=seed, epochs=epochs,
                  spike_lambda=spike_lambda)
        with torch.no_grad():
            mse = float((bank(xe) - ye).pow(2).mean())
        if best is None or mse < best[1]:
            best = (bank, mse)
    return best


class TiedBank(nn.Module):
    """A magnitude bank that **is** the shared prototype, scaled by the routed key.

    For a positively homogeneous target the bank's job factorises into a scale that
    the router already knows and a unit-interval shape that every bank shares
    (:func:`_build_tied`). Holding a reference to the one prototype -- rather than a
    copy of its parameters -- is what turns that identity into an actual storage
    saving: ``nn.Module.parameters()`` de-duplicates by object, so ``R`` banks cost
    one basis set plus ``R`` scalars, and those scalars are the exponents the router
    reads for free.
    """

    def __init__(self, proto: MBENeuron, x_min: float, x_scale: float,
                 scale: float):
        super().__init__()
        self.proto = proto
        self.x_min, self.x_scale, self.scale = (float(x_min), float(x_scale),
                                                float(scale))

    @property
    def core(self) -> MBENeuron:
        return self.proto

    @property
    def cfg(self):
        return self.proto.cfg

    def _rho(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_min) / self.x_scale

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.proto.features(self._rho(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.features(x)
        out = (o * self.proto.w).sum(dim=-1)
        if self.proto.bias is not None:
            out = out + self.proto.bias
        return (out * self.scale).reshape(x.shape)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @torch.no_grad()
    def firing_rate(self, x: torch.Tensor) -> float:
        return self.proto.firing_rate(self._rho(x))


def _tie_scale(g: torch.Tensor, h: torch.Tensor) -> tuple[float, float]:
    """Least-squares scale ``s`` with ``g ~= s*h``, and its relative residual."""
    denom = float((h * h).sum())
    s = float((g * h).sum()) / denom if denom > 0 else 0.0
    rms = float(g.pow(2).mean().sqrt())
    resid = float((g - s * h).pow(2).mean().sqrt()) / max(rms, 1e-30)
    return s, resid


def _build_tied(fn, router, domain, reach, N, T, epochs, seed, device,
                alpha_init, tol, m, verbose):
    """One fitted prototype + a per-bank scale, for **positively homogeneous** f.

    If ``f(lambda x) = lambda^k f(x)`` (identity, ``1/x``, ``1/sqrt(x)``, ``x^2``,
    ...), then every magnitude bank's target factorises exactly::

        g_j(rho) = f(sigma 2^e (1+rho)) = [sigma^k 2^{ek}] * f(1+rho)
                 = s_j * h(rho)

    The bracket is a function of the routed key alone -- the sign and exponent the
    router already read at **zero spike and zero parameter cost**. So the whole
    family needs *one* fitted unit-interval approximation, and each bank is that
    prototype with its linear readout scaled by ``s_j``. Storage stops growing with
    the number of ranges, and the relative error is identical in every binade.

    The identity is the important instance: the spike-driven FP multiply
    reconstructs both operands through an ``MBE_Id``, and those reconstructions are
    the largest single consumer of spikes in a converted Transformer.

    ``tol`` guards the assumption -- a target that is not homogeneous will not
    factorise, and this raises instead of silently returning a bad fit.
    """
    from .fit import fit_model
    device = torch.device(device)
    mags = [bi for bi in reach if router.banks[bi]["kind"] == "mag"]
    if not mags:
        raise ValueError("tied=True needs at least one reachable magnitude bank")

    rho = _grid(0.0, 1.0, m, 0.0, device)

    def key_span(bi):
        a, b = reach[bi]
        ta, tb = router.to_key(a), router.to_key(b)
        return min(abs(ta), abs(tb)), max(abs(ta), abs(tb))

    ref = mags[0]
    lo_r, _ = key_span(ref)
    h = fn(router.from_key(router.banks[ref]["sign"] * (lo_r * (1.0 + rho))))
    h = h / max(float(h.pow(2).mean().sqrt()), 1e-30)      # 단위 스케일로 정규화

    cfg = MBEConfig(n_basis=N, n_steps=T, x_min=0.0, x_scale=1.0,
                    alpha_v=_uniform_alpha(N), use_bias=True)
    proto = MBENeuron(cfg).to(device)
    fit_model(proto, rho, h, seed=seed, epochs=epochs)

    banks: list = [None] * router.n_banks
    for bi in mags:
        lo_b, _ = key_span(bi)
        g = fn(router.from_key(router.banks[bi]["sign"] * (lo_b * (1.0 + rho))))
        s, resid = _tie_scale(g, h)
        if resid > tol:
            raise ValueError(
                f"target is not self-similar across binades (bank {bi}: relative "
                f"residual {resid:.2e} > tol {tol:.0e}); tied=True only applies to "
                f"positively homogeneous targets")
        banks[bi] = TiedBank(proto, lo_b, lo_b, s)
        if verbose:
            print(f"    bank {bi:2d} tied scale={s:+.4g} resid={resid:.1e}",
                  flush=True)
    return banks, proto


def _fit_bank_adaptive(tgt, feed_lo, feed_hi, target_mse, n_max,
                       n_steps, epochs, seed, device):
    """Smallest-N bank reaching ``target_mse`` (flat tails settle at N=1). Returns
    the first N whose MSE is under target, else the best over ``1..n_max``."""
    best = None
    for N in range(1, n_max + 1):
        bank, mse = _fit_bank(
            tgt, feed_lo, feed_hi, N, n_steps, epochs, seed, device
        )
        if best is None or mse < best[1]:
            best = (bank, mse)
        if mse <= target_mse:
            return bank, mse
    return best


def build_mbe_pasn(name: str, domain: tuple[float, float], e_min: int = -2,
                   e_max: int | None = None, n_local: int = 2, n_near0: int = 4,
                   n_steps: int = 16, epochs: int = 300, seed: int = 0,
                   adaptive: bool = False, target_mse: float = 1e-4, n_max: int = 6,
                   near0: str = "signed", budget: str = "fixed",
                   target: str = "absolute", target_rel: float = 1e-3,
                   tied: bool = False, tied_tol: float = 1e-3,
                   beta: float = 0.0, gamma_log2: int = 0,
                   readout_order: int = 1, learn_tau: bool = True,
                   tau_range: tuple | None = None, rule_ac: tuple | None = None,
                   spike_lambda: float = 0.0, t_fixed: int | None = None,
                   alpha_init: str = "auto", verbose: bool = False,
                   device: torch.device | str = "cpu") -> MBEPASNNeuron:
    """Build + fit an MBE-PASN neuron for target ``name`` on ``domain``.

    ``t_fixed`` overrides every bank's ``T`` *after* the budget rule has run,
    leaving the rule's ``N_j`` alone. This is the knob for a **timestep-matched
    comparison**: the MBE paper reports a single global ``T=16``, while the rule
    normally gives most banks far fewer steps, so a bare accuracy comparison is
    open to "you used fewer timesteps". Setting ``t_fixed=16`` removes that
    objection at a higher spike cost -- it cannot help accuracy less than the
    rule's choice, since every bank gets at least as many steps as it asked for.

    ``budget`` selects how each bank's ``(N, T)`` is chosen:

    * ``"fixed"`` -- ``n_local`` bases per magnitude binade, ``n_near0`` near-zero,
      ``n_steps`` everywhere (the original behaviour).
    * ``"rule"`` -- :func:`rule_budget` derives ``(N_j, T_j)`` from the bank's own
      dynamic range and ``target_mse``. Flat tails collapse to ``(1, 2)``; only
      banks that genuinely need the resolution get a long spike train. This is
      where the energy saving comes from -- ``T`` was a global constant before, and
      it is a *linear* factor in the spike count.

    ``target`` says what the rule's error budget *means*:

    * ``"absolute"`` -- every bank is held to the same MSE ``target_mse``.
    * ``"relative"`` -- every bank is held to the same **relative** RMS error
      ``target_rel``, i.e. ``eps_j = (target_rel * rms(g_j))^2``. This is the right
      budget whenever the output feeds a *product*: the spike-driven FP multiply
      reconstructs both operands and multiplies, so the product's relative error is
      what the operands' relative errors make it, and an absolute budget starves
      exactly the small operands. For ``f(x) = x`` the two coincide up to a
      constant per bank, and the required resolution becomes
      ``b = log2(delta / (r*rms)) = log2(1/r)`` -- **independent of the binade**, so
      every magnitude bank gets the same budget and the relative error is flat
      across decades.

    ``readout_order`` / ``learn_tau`` / ``tau_range`` are per-leaf decoder and
    kernel settings passed straight to :class:`MBEConfig`. ``readout_order > 1``
    buys accuracy at **zero extra spikes** (see the config docstring); freezing
    ``tau`` at ``dt/ln2`` makes each basis an exact binary-search encoder.

    ``beta`` / ``gamma_log2`` place the binade grid (see :class:`PrefixRouter`):
    routing on ``(x - beta) / 2^{gamma_log2}`` puts the log-dense ranges at ``beta``
    rather than at zero. Calibrate them, do not train them.

    ``tied=True`` fits **one** prototype on the unit residual and gives each
    magnitude bank that prototype with its readout scaled by the routed key -- exact
    for positively homogeneous targets, and the storage then stops growing with the
    number of ranges (see :func:`_build_tied`). Raises if the target does not
    factorise to within ``tied_tol``.

    ``adaptive=True`` is the older, more expensive variant: sweep ``N`` per bank
    and keep the smallest reaching ``target_mse`` (``T`` stays global).

    ``near0="signed"`` (default) splits the near-zero region by polarity, so **no
    bank straddles zero** -- see :class:`PrefixRouter`. This is what removes the
    error floor that a single near-zero bank has on targets whose bend sits at 0.
    """
    if budget not in ("fixed", "rule"):
        raise ValueError(f"budget must be 'fixed' or 'rule', got {budget!r}")
    if tied and budget == "rule" and target == "absolute":
        # Not a coding gap -- the two are incompatible. Tying fits one prototype on
        # the *unit* residual and hands every magnitude bank that same prototype
        # scaled by the routed key, so the relative error is flat across banks by
        # construction. An absolute budget means "every bank held to the same MSE",
        # which for a scale-sigma bank is a relative target proportional to 1/sigma
        # -- a different (N, T) per bank, which one shared prototype cannot express.
        #
        # This used to be silent: the tied branch read target_rel unconditionally, so
        # target="absolute" fell through to a relative budget and built a bit-identical
        # neuron. P0.4's headline arms (pasn-both-abs vs pasn-both-rel-1e-2) came back
        # equal to the last digit because of it. Compare the two targets at
        # tied=False, where the budget really is per-bank.
        raise ValueError(
            "tied=True cannot honour target='absolute': the tied prototype is fitted "
            "once on the unit residual, so its error budget is relative by "
            "construction. Use tied=False to budget per bank on an absolute MSE, or "
            "target='relative' to keep the tying."
        )
    fn, _ = functions.REGISTRY[name]
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max,
                          near0=near0, beta=beta, gamma_log2=gamma_log2)
    reach = {bi: sp for bi in range(router.n_banks)
             if (sp := router.reachable(bi, domain[0], domain[1])) is not None}

    def bank_target(bi):
        """``(tgt, feed_lo, feed_hi)`` in *key* space for a reachable bank.

        The neuron sees the routing key; ``tgt`` maps a fed key value back through
        the router's affine to the value ``fn`` is evaluated at.
        """
        a, b = reach[bi]
        ta, tb = router.to_key(a), router.to_key(b)
        spec = router.banks[bi]
        if spec["kind"] == "near0":                    # raw signed key
            return (lambda v: fn(router.from_key(v))), min(ta, tb), max(ta, tb)
        s = spec["sign"]
        return ((lambda v, s=s: fn(router.from_key(s * v))),
                min(abs(ta), abs(tb)), max(abs(ta), abs(tb)))

    if tied:
        # One prototype for every magnitude bank; the routed key supplies the scale.
        # (N, T) come from the rule applied to the *unit* residual, so they are the
        # same for every bank -- which is the point.
        rho = _grid(0.0, 1.0, 512, 0.0)
        h = fn(rho + 1.0)
        h = h / max(float(h.pow(2).mean().sqrt()), 1e-30)
        ra, rc = rule_ac or bit_law(readout_order, learn_tau)
        N_t, T_t = (rule_budget(float(h.max() - h.min()),
                                (target_rel * float(h.pow(2).mean().sqrt())) ** 2,
                                n_max=min(n_max, 4), a=ra, c=rc)
                    if budget == "rule" else (n_local, n_steps))
        if t_fixed is not None:
            T_t = t_fixed
        banks, _ = _build_tied(fn, router, domain, reach, N_t, T_t, epochs, seed,
                               device, alpha_init, tied_tol, 512, verbose)
        for bi in reach:                      # near-zero banks are not homogeneous
            if banks[bi] is None:
                tgt, flo, fhi = bank_target(bi)
                banks[bi], _ = _fit_bank(tgt, flo, fhi, n_near0,
                                         t_fixed or n_steps,
                                         epochs, seed, device,
                                         alpha_init=alpha_init,
                                         readout_order=readout_order,
                                         learn_tau=learn_tau,
                                         tau_range=tau_range,
                                         spike_lambda=spike_lambda)
        return MBEPASNNeuron(router, fill_unreachable(router, banks))

    banks: list[MBENeuron | None] = []
    for bi, spec in enumerate(router.banks):
        # Clamp every fit to the calibrated domain. Banks the domain never reaches
        # are filled in afterwards from their nearest reachable neighbour -- an
        # unfitted bank holds random parameters, and while no *calibration* input
        # routes there, a domain endpoint or any out-of-range input at inference
        # does, and then the output is garbage rather than merely extrapolated.
        span = router.reachable(bi, domain[0], domain[1])
        if span is None:
            banks.append(None)
            if verbose:
                print(f"    bank {bi:2d}/{router.n_banks - 1:2d} "
                      f"{'unreachable':16s} (outside {domain})", flush=True)
            continue
        tgt, flo, fhi = bank_target(bi)
        cap = n_local if spec["kind"] == "mag" else n_near0
        steps = n_steps
        if budget == "rule":
            y = tgt(_grid(flo, fhi, 512, 0.0))
            eps = target_mse
            if target == "relative":
                rms = float(y.pow(2).mean().sqrt())
                eps = (target_rel * max(rms, 1e-30)) ** 2
            ra, rc = rule_ac or bit_law(readout_order, learn_tau)
            cap, steps = rule_budget(float(y.max() - y.min()), eps,
                                     n_max=min(n_max, 4), a=ra, c=rc)
        if t_fixed is not None:
            steps = t_fixed
        if adaptive:
            bank, mse = _fit_bank_adaptive(tgt, flo, fhi, target_mse,
                                           n_max, steps, epochs, seed, device)
        else:
            bank, mse = _fit_bank(
                tgt, flo, fhi, cap, steps, epochs, seed, device,
                alpha_init=alpha_init, readout_order=readout_order,
                learn_tau=learn_tau, tau_range=tau_range,
                spike_lambda=spike_lambda
            )
        banks.append(bank)
        if verbose:
            if spec["kind"] == "near0":
                k = "near0"
            elif spec["kind"] == "near0s":
                k = f"near0 s={spec['sign']:+.0f}"
            else:
                k = f"mag s={spec['sign']:+.0f} e={spec['e']}"
            print(
                f"    bank {len(banks)-1:2d}/{router.n_banks - 1:2d} "
                f"{k:16s} N={bank.cfg.n_basis} T={bank.cfg.n_steps} "
                f"mse={mse:.2e}",
                flush=True,
            )
    return MBEPASNNeuron(router, fill_unreachable(router, banks))


def fill_unreachable(router: PrefixRouter, banks: list) -> list:
    """Point every unreachable (``None``) bank at its nearest reachable neighbour.

    The router's binade grid does not align with a calibrated domain, so some banks
    lie entirely outside it and have nothing to fit. Leaving them unfitted is only
    safe as long as no input ever routes there -- but a domain *endpoint* does
    (``x = hi`` has ``floor(log2|x|)`` one binade above the last fitted one), and so
    does any out-of-range input at inference. Sharing the nearest fitted neuron
    turns that from random garbage into ordinary extrapolation. The module is
    shared, not copied, so the stored-parameter count is unchanged.
    """
    reachable = [i for i, b in enumerate(banks) if b is not None]
    if not reachable:
        raise ValueError("no bank of the router intersects the domain")

    def centre(i):
        a, b = router.bank_interval(i)
        return 0.5 * (a + b)

    return [b if b is not None
            else banks[min(reachable, key=lambda j: abs(centre(j) - centre(i)))]
            for i, b in enumerate(banks)]
