"""Spike-driven Transformer primitives assembled from MBE neurons (Phase 3).

Implements the conversion framework's non-spiking components as spike-based
operations (paper "Conversion Framework" section + Appendix F):

  * spiking FP multiplication      -- Eq. 9-12, 22-27, Fig. 9
  * spiking Softmax                -- Eq. 13, Table VIII
  * spiking LayerNorm              -- Fig. 5(c)

Each nonlinear sub-function is approximated by a calibrated MBE neuron
(`MBE_Id`, `MBE_exp`, `MBE_inv`, `MBE_invsqrt`); multiplications are performed by
the outer-product spike scheme. Integer power-of-two factors (2^floor, 2^-E) are
exact bit operations.

These modules are for CPU numerical verification: each is checked against the
exact PyTorch op in `experiments/verify_phase3.py` / the unit tests.
"""
from __future__ import annotations

import math

import torch

from .neuron import MBENeuron, MBEConfig
from . import functions
from .fit import fit_model


# --------------------------------------------------------------------------
# Calibrated primitive builders
# --------------------------------------------------------------------------

# Fit cache: many primitives share a fixed domain (exp2 on [0,1], inv on [0.5,1],
# invsqrt on [0.5,2]) and identity ranges cluster, so the same neuron is fitted
# repeatedly across modules / isolation runs. Cache by (name, rounded range,
# shape hyper-params) -- neurons are used read-only at inference, so sharing is
# safe. Clear with ``clear_cache()`` if a fresh fit is wanted.
_FIT_CACHE: dict = {}


def clear_cache() -> None:
    _FIT_CACHE.clear()


def _cache_key(name, lo, hi, n_basis, n_steps, epochs, seed, device):
    return (name, round(float(lo), 4), round(float(hi), 4), n_basis, n_steps,
            epochs, seed, str(torch.device(device)))


def calibrate_identity(lo: float, hi: float, n_basis: int = 8, n_steps: int = 16,
                       epochs: int = 600, seed: int = 0,
                       use_cache: bool = True, readout_order: int = 1,
                       log_sample: bool = False,
                       device: torch.device | str = "cpu") -> MBENeuron:
    """MBE_Id: approximate the identity map on ``[lo, hi]`` with **no bias**, so
    ``reconstruct(x) = sum_{n,t} a_{n,t} s_n[t] ~= x`` is a pure spike sum.

    ``log_sample`` draws the calibration points log-uniformly instead of
    uniformly. A uniform draw puts almost every point in the top decade, so the
    fit optimises absolute error there and the *relative* error at small operands
    -- which is what a product cares about -- goes unbounded. This is the fairest
    matched form of the routed identity's relative-error budget for a single
    global neuron, which has no per-range structure to target."""
    device = torch.device(device)
    key = _cache_key(
        "identity", lo, hi, n_basis, n_steps, epochs, seed, device
    ) + (readout_order, log_sample)
    if use_cache and key in _FIT_CACHE:
        return _FIT_CACHE[key]
    if log_sample:
        g = torch.Generator().manual_seed(seed)
        lo_pos = max(lo, hi * 1e-4)
        u = torch.rand(4000, generator=g)
        x = torch.exp(u * (math.log(hi) - math.log(lo_pos)) + math.log(lo_pos))
        x, _ = torch.sort(x)
        y = x.clone()
    else:
        x, y, _ = functions.sample("identity", m=4000, seed=seed, domain=(lo, hi))
    x, y = x.to(device), y.to(device)
    cfg = functions.make_config("identity", n_basis=n_basis, n_steps=n_steps,
                                domain=(lo, hi), use_bias=False,
                                readout_order=readout_order)
    m = MBENeuron(cfg).to(device)
    fit_model(m, x, y, seed=seed, epochs=epochs)
    if use_cache:
        _FIT_CACHE[key] = m
    return m


def calibrate(name: str, lo: float, hi: float, n_basis: int = 8, n_steps: int = 16,
              epochs: int = 600, seed: int = 0, use_cache: bool = True,
              readout_order: int = 1,
              device: torch.device | str = "cpu") -> MBENeuron:
    """Calibrate a primitive MBE neuron (``exp2`` / ``inv`` / ``invsqrt`` / ...)
    on ``[lo, hi]``."""
    device = torch.device(device)
    key = _cache_key(name, lo, hi, n_basis, n_steps, epochs, seed,
                     device) + (readout_order,)
    if use_cache and key in _FIT_CACHE:
        return _FIT_CACHE[key]
    x, y, _ = functions.sample(name, m=4000, seed=seed, domain=(lo, hi))
    x, y = x.to(device), y.to(device)
    cfg = functions.make_config(name, n_basis=n_basis, n_steps=n_steps,
                                domain=(lo, hi), readout_order=readout_order)
    m = MBENeuron(cfg).to(device)
    fit_model(m, x, y, seed=seed, epochs=epochs)
    if use_cache:
        _FIT_CACHE[key] = m
    return m


# --------------------------------------------------------------------------
# Spike-driven FP multiplication  (Eq. 9-12, 22-27)
# --------------------------------------------------------------------------

@torch.no_grad()
def _recon(idn: MBENeuron, x: torch.Tensor) -> torch.Tensor:
    """Spike-sum reconstruction of non-negative ``x`` via MBE_Id."""
    return idn.reconstruct(x)


def _routes_sign(idn) -> bool:
    """True when the identity's own router reads the operand's sign.

    A prefix router whose key range straddles zero puts each polarity in its own
    bank, so the operand can be fed signed and the ``relu`` split is redundant.
    """
    router = getattr(idn, "router", None)
    return bool(router is not None and getattr(router, "signed", False))


@torch.no_grad()
def spiking_multiply(idn: MBENeuron, x1: torch.Tensor, x2: torch.Tensor,
                     signed: bool | None = True,
                     idn2: MBENeuron | None = None) -> torch.Tensor:
    """Approximate the element-wise product ``x1 * x2`` with spikes.

    Each operand is turned into a spike train by an ``MBE_Id`` (``idn`` for x1,
    ``idn2`` for x2 -- different identities let the operands use different
    calibration ranges, Eq. 11 ``D = d^T d'`` allows ``d != d'``). The product is
    the Hadamard sum of the intensity outer product ``D`` and the spike outer
    product ``S`` (Eq. 12/27); that double sum is separable, so it equals
    ``recon(x1) * recon(x2)`` -- the bulk form used here
    (:func:`multiply_outer` builds the explicit ``D (x) S`` for verification).

    ``signed``: a plain MBE_Id maps non-negative inputs, so signed operands are
    split as ``x = relu(x) - relu(-x)`` and the product expanded into four
    non-negative reconstructions. ``signed=None`` asks instead -- an identity whose
    router reads the sign takes the operand directly, halving the reconstructions
    (the spike count is unchanged, since the two relu halves carry disjoint
    elements; what halves is the number of invocations).
    """
    idn2 = idn2 or idn
    if signed is None:
        signed = not (_routes_sign(idn) and _routes_sign(idn2))
    if not signed:
        return _recon(idn, x1) * _recon(idn2, x2)
    p1, n1 = torch.relu(x1), torch.relu(-x1)
    r1 = _recon(idn, p1) - _recon(idn, n1)
    # Squaring is common in LayerNorm. Reuse the same reconstruction instead of
    # running the temporal neuron a second time for an identical tensor.
    if x1 is x2 and idn2 is idn:
        r2 = r1
    else:
        p2, n2 = torch.relu(x2), torch.relu(-x2)
        r2 = _recon(idn2, p2) - _recon(idn2, n2)
    return r1 * r2


@torch.no_grad()
def spiking_matmul(idn: MBENeuron, A: torch.Tensor, B: torch.Tensor,
                   signed: bool | None = True,
                   idn2: MBENeuron | None = None) -> torch.Tensor:
    """Spike-driven matrix product ``A @ B`` for activation*activation matmuls.

    A matmul ``C[i,j] = sum_k A[i,k] B[k,j]`` is a sum of element-wise products,
    so the separable FP-multiply (``recon(a) * recon(b)``, see
    :func:`spiking_multiply`) lifts to ``recon(A) @ recon(B)``: each operand is
    reconstructed from its ``MBE_Id`` spike train and the (exact, weight-free)
    accumulation is the ordinary matmul. Only activation*activation products need
    this; activation*weight matmuls stay as native accumulation (training-free).

    Batched leading dims are supported (``A @ B`` broadcasts). ``signed`` uses the
    ``x = relu(x) - relu(-x)`` split, expanding into four non-negative matmuls.
    """
    idn2 = idn2 or idn
    if signed is None:                       # see :func:`spiking_multiply`
        signed = not (_routes_sign(idn) and _routes_sign(idn2))
    if not signed:
        return _recon(idn, A) @ _recon(idn2, B)
    Ap, An = torch.relu(A), torch.relu(-A)
    Bp, Bn = torch.relu(B), torch.relu(-B)
    rAp, rAn = _recon(idn, Ap), _recon(idn, An)
    rBp, rBn = _recon(idn2, Bp), _recon(idn2, Bn)
    return rAp @ rBp - rAp @ rBn - rAn @ rBp + rAn @ rBn


@torch.no_grad()
def multiply_outer(idn: MBENeuron, x1: float, x2: float) -> float:
    """Explicit outer-product form for two non-negative scalars (Eq. 11-12, 26-27).

    Builds intensity matrix ``D = a (x) a`` and spike matrix ``S = b(x1) (x) b(x2)``
    then returns ``sum(D ⊙ S)``. Used to confirm the separable bulk path equals
    the paper's Hadamard-product formulation.
    """
    a = idn.intensities().reshape(-1)               # (K,)  precomputed
    sample = dict(device=a.device, dtype=a.dtype)
    b1 = idn.spike_train(torch.tensor([float(x1)], **sample)).reshape(-1)  # (K,)
    b2 = idn.spike_train(torch.tensor([float(x2)], **sample)).reshape(-1)
    D = torch.outer(a, a)                            # (K, K) precomputable
    S = torch.outer(b1, b2)                          # (K, K) binary
    return float((D * S).sum())


# --------------------------------------------------------------------------
# Spike-driven activation (GELU / SiLU / Tanh) -- a single calibrated MBE neuron
# --------------------------------------------------------------------------

class SpikingActivation(torch.nn.Module):
    """Element-wise activation approximated by one MBE neuron (Phase 2).

    The neuron's decoded output ``f_hat(x) = sum_n w_n o_n(T)`` is itself the
    spike-driven accumulation, so a calibrated ``MBE_GELU`` *is* the spiking
    activation -- no extra assembly needed. Kept as a thin callable so the
    conversion framework can treat every nonlinearity uniformly.
    """

    def __init__(self, neuron: MBENeuron):
        super().__init__()
        self.neuron = neuron

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron(x)


# Non-monotone activations (a near-zero bend/dip) are unrepresentable by a single
# global MBE neuron -- its linear-readout ceiling on GELU is ~0.5. They require the
# polarity-split (signed) MBE, which is the paper-faithful GELU/SiLU handler. Using
# the plain neuron here makes a converted Transformer diverge (perplexity blow-up).
_NONMONOTONE_ACTS = {"gelu", "gelu_tanh", "silu"}


def build_activation(name: str, sample_x: torch.Tensor, n_basis: int = 4,
                     n_steps: int = 16, seed: int = 0, margin: float = 0.05,
                     epochs: int = 600, device: torch.device | str = "cpu",
                     signed: bool | None = None,
                     readout_order: int = 1) -> SpikingActivation:
    """Calibrate a :class:`SpikingActivation` for ``name`` (``gelu`` / ``silu`` /
    ``tanh``) on the range measured in ``sample_x`` (padded by ``margin``).

    ``signed`` (default: auto) uses the polarity-split MBE for non-monotone
    activations whose range crosses zero -- required for GELU/SiLU, where a single
    global neuron cannot represent the near-zero bend. Pass ``signed=False`` to
    force the (failing) plain neuron for an ablation.

    .. warning::
       ``readout_order`` is **ignored on the signed path**, which is the path GELU and
       SiLU take. :class:`~mbe.signed.SignedMBENeuron` has no order-2 readout and
       :func:`~mbe.functions.make_signed` does not take the argument, so orders 1 and 2
       build a bit-identical neuron (verified 2026-07-28: fingerprint 19.523948072, 85
       parameters either way, against 37 -> 43 on the unsigned path). It is honoured
       only when ``signed=False``. Any "matched-readout-order" comparison against a
       routed backend is therefore **not matched on the activation** -- see
       ``experiments/P0.4_GPT2_HANDOFF.md`` §4.2.
    """
    lo = float(sample_x.min())
    hi = float(sample_x.max())
    span = max(hi - lo, 1e-6)
    lo -= margin * span
    hi += margin * span
    if signed is None:
        signed = name in _NONMONOTONE_ACTS and lo < 0.0 < hi
    if signed:
        neuron = _fit_signed_activation(name, lo, hi, n_basis, n_steps,
                                        epochs, seed, device, readout_order)
    else:
        neuron = calibrate(name, lo, hi, n_basis=n_basis, n_steps=n_steps,
                           epochs=epochs, seed=seed, device=device,
                           readout_order=readout_order)
    return SpikingActivation(neuron)


def _fit_signed_activation(name, lo, hi, n_basis, n_steps, epochs, seed, device,
                           readout_order=1):
    """Polarity-split (signed) MBE fit of ``name`` on ``[lo, hi]`` (crosses 0).

    Each side gets ``n_basis`` bases, curvature-placed for its restriction of the
    activation; a single joint readout combines both banks (see
    :class:`SignedMBENeuron`).

    ``readout_order`` is accepted and **discarded** -- the joint readout is order-1 and
    ``make_signed`` has no parameter to raise it. Kept for signature parity with the
    unsigned path. Giving the signed neuron a real order-2 readout is a change to
    :mod:`mbe.signed`, not a wiring fix; until then a caller cannot match a routed
    backend's decoder order here, and P0.4's ``mbe-*-matched`` rows do not."""
    device = torch.device(device)
    sm = functions.make_signed(name, n_pos=n_basis, n_neg=n_basis, pivot=0.0,
                               n_steps=n_steps, domain=(lo, hi)).to(device)
    # (the joint readout lives on SignedMBENeuron, so readout_order applies to
    # its sub-banks' features only via make_signed; kept for signature parity)
    x, y, _ = functions.sample(name, m=4000, seed=seed, domain=(lo, hi))
    fit_model(sm, x.to(device), y.to(device), seed=seed, epochs=epochs)
    return sm


# --------------------------------------------------------------------------
# Spike-driven Softmax  (Eq. 13, Table VIII)
# --------------------------------------------------------------------------

_LOG2E = 1.4426950408889634         # log2(e)


#: Mantissa bits per float dtype, for locating the smallest subnormal.
_MANTISSA_BITS = {torch.float32: 23, torch.float64: 52,
                  torch.float16: 10, torch.bfloat16: 7}


def _exp_underflow_floor(dtype) -> float:
    """Largest ``x`` for which ``exp(x)`` is exactly ``0`` in ``dtype``.

    Below the smallest subnormal (``smallest_normal * 2**-mantissa_bits``) there
    is no representable nonzero value left, so clamping a softmax shift here
    changes nothing about the result -- it only keeps the intermediate
    ``x * log2(e)`` inside the exponent range. fp32: about -103.97.
    """
    fi = torch.finfo(dtype)
    mant = _MANTISSA_BITS.get(dtype, 23)
    return math.log(fi.smallest_normal) - (mant + 1) * math.log(2.0)


class SpikingSoftmax(torch.nn.Module):
    """Softmax decomposed into exp / sum / reciprocal / FP-multiply.

    * exp:  e^x = 2^(x log2 e) = 2^floor * 2^frac; the integer power is an exact
            shift, the fractional part 2^frac (frac in [0,1)) uses ``MBE_exp``.
    * sum:  spike accumulation over the axis (exact reduce).
    * 1/S:  IEEE-754 split S = m * 2^e (m in [0.5,1)); 1/m via ``MBE_inv``,
            times the exact factor 2^-e.
    * out:  e^x * (1/S) via the spiking FP multiply.
    """

    def __init__(self, exp_neuron: MBENeuron, inv_neuron: MBENeuron,
                 id_neuron: MBENeuron, spike_mult: bool = True):
        super().__init__()
        self.exp = exp_neuron          # 2^x on [0, 1]
        self.inv = inv_neuron          # 1/x on [0.5, 1]
        self.idn = id_neuron           # identity on [0, 1] (for the final mult)
        self.spike_mult = spike_mult

    @torch.no_grad()
    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # numerical-stability shift (does not change softmax); makes x_m <= 0,
        # floored where exp() underflows to exactly zero anyway.
        #
        # The floor is what makes a *masked* attention score survive. A causal
        # mask adds ``finfo.min`` (-3.4e38 in fp32) to the score; ``x_m * log2(e)``
        # then overflows to ``-inf``, and ``-inf - floor(-inf)`` is **NaN**, which
        # propagates through the whole head. Clamping is exact rather than a
        # tolerance: below this point the entry contributes 0 to the sum and 0 to
        # the output, which is what softmax gives it regardless.
        x_m = (x - x.max(dim=dim, keepdim=True).values).clamp_min(
            _exp_underflow_floor(x.dtype))
        z = x_m * _LOG2E
        z_floor = torch.floor(z)
        frac = z - z_floor                          # in [0, 1)
        two_frac = self.exp(frac)                   # ~ 2^frac
        exp_x = torch.ldexp(two_frac, z_floor.to(torch.int64))  # 2^frac * 2^floor
        S = exp_x.sum(dim=dim, keepdim=True)        # spike accumulation
        # reciprocal of the denominator via IEEE-754 mantissa/exponent split
        m, e = torch.frexp(S)                       # m in [0.5,1), S = m * 2^e
        inv_m = self.inv(m)                         # ~ 1/m
        inv_S = torch.ldexp(inv_m, -e)              # 1/S = (1/m) * 2^-e
        # Left at one value per row, **not** broadcast to the full matrix. The
        # multiply reconstructs each operand through the identity, and a
        # reconstruction is element-wise, so recon(expand(v)) == expand(recon(v)):
        # broadcasting first would encode the same S values S times over. On a
        # causal attention matrix that is the difference between S and S*S
        # invocations of the identity -- 256x measured at S=256, bit-identical
        # output either way. The product broadcasts on its own.
        if self.spike_mult:
            return spiking_multiply(self.idn, exp_x, inv_S, signed=False)
        return exp_x * inv_S


# --------------------------------------------------------------------------
# Spike-driven LayerNorm  (Fig. 5(c))
# --------------------------------------------------------------------------

class SpikingLayerNorm(torch.nn.Module):
    """LayerNorm decomposed into square / inverse-sqrt / FP-multiply.

    LN(x) = γ (x-μ) / sqrt( Σ(x-μ)^2 / n + ε ) + β

    * μ:            mean over the normalised axis (exact reduce).
    * (x-μ)^2 / n:  spiking FP multiply of the deviation with itself; folding the
                    1/n into the intensity (``D`` pre-scaled by 1/n) is equivalent
                    to averaging the squared deviations (done here by ``mean``).
    * 1/sqrt(var):  IEEE-754 split var = M·2^E with E made even (parity fix), then
                    (1/sqrt(M)) via ``MBE_invsqrt`` times the exact factor 2^(-E/2).
    * normalise:    (x-μ) * invstd via the spiking FP multiply.
    * affine:       γ, β.
    """

    def __init__(self, invsqrt_neuron: MBENeuron, id_dev: MBENeuron,
                 id_istd: MBENeuron, eps: float = 1e-5, spike_mult: bool = True):
        super().__init__()
        self.rsqrt = invsqrt_neuron    # 1/sqrt(x) on [0.5, 2]
        self.id_dev = id_dev           # identity over the deviation range
        self.id_istd = id_istd         # identity over the inverse-std range
        self.eps = eps
        self.spike_mult = spike_mult

    @torch.no_grad()
    def _square_over_n(self, dev, n):
        if self.spike_mult:
            sq = spiking_multiply(self.id_dev, dev, dev, signed=None)
        else:
            sq = dev * dev
        return sq.sum(dim=-1, keepdim=True) / n

    @torch.no_grad()
    def forward(self, x, weight=None, bias=None):
        n = x.shape[-1]
        mu = x.mean(dim=-1, keepdim=True)
        dev = x - mu
        var = self._square_over_n(dev, n) + self.eps        # (..., 1)
        # inverse sqrt via IEEE-754 mantissa/exponent with even-exponent parity fix
        m, e = torch.frexp(var)                             # m in [0.5,1)
        odd = (e % 2 != 0)
        m = torch.where(odd, m * 2.0, m)                    # -> m in [0.5, 2)
        e = torch.where(odd, e - 1, e)                      # e now even
        inv_sqrt_m = self.rsqrt(m)                          # ~ 1/sqrt(m)
        invstd = torch.ldexp(inv_sqrt_m, (-e // 2))         # * 2^(-E/2)
        invstd = invstd.expand_as(dev)
        if self.spike_mult:
            normed = spiking_multiply(self.id_dev, dev, invstd, signed=None,
                                      idn2=self.id_istd)
        else:
            normed = dev * invstd
        if weight is not None:
            normed = normed * weight
        if bias is not None:
            normed = normed + bias
        return normed


# --------------------------------------------------------------------------
# One-shot builders that calibrate all primitives on measured ranges
# --------------------------------------------------------------------------

def build_softmax(sample_logits: torch.Tensor, dim: int = -1, n_basis: int = 8,
                  n_steps: int = 16, epochs: int = 600, seed: int = 0,
                  spike_mult: bool = True, readout_order: int = 1,
                  log_sample: bool = False,
                  device: torch.device | str = "cpu") -> SpikingSoftmax:
    """Calibrate a :class:`SpikingSoftmax` for logits like ``sample_logits``."""
    exp_n = calibrate(
        "exp2", 0.0, 1.0, n_basis=n_basis, n_steps=n_steps,
        epochs=epochs, seed=seed, device=device,
        readout_order=readout_order
    )
    inv_n = calibrate(
        "inv", 0.5, 1.0, n_basis=n_basis, n_steps=n_steps,
        epochs=epochs, seed=seed, device=device,
        readout_order=readout_order
    )
    id_n = calibrate_identity(
        0.0, 1.0, n_basis=n_basis, n_steps=n_steps,
        epochs=epochs, seed=seed, device=device, log_sample=log_sample
    )
    return SpikingSoftmax(exp_n, inv_n, id_n, spike_mult=spike_mult)


def build_layernorm(sample_x: torch.Tensor, eps: float = 1e-5, n_basis: int = 8,
                    n_steps: int = 16, epochs: int = 600, seed: int = 0,
                    spike_mult: bool = True, readout_order: int = 1,
                    log_sample: bool = False,
                    device: torch.device | str = "cpu") -> SpikingLayerNorm:
    """Calibrate a :class:`SpikingLayerNorm` using the ranges seen in ``sample_x``."""
    mu = sample_x.mean(dim=-1, keepdim=True)
    dev = sample_x - mu
    dev_max = float(dev.abs().max()) * 1.1 + 1e-3
    var = (dev * dev).mean(dim=-1, keepdim=True) + eps
    istd_max = float((1.0 / var.sqrt()).max()) * 1.1 + 1e-3
    rsqrt = calibrate(
        "invsqrt", 0.5, 2.0, n_basis=n_basis, n_steps=n_steps,
        epochs=epochs, seed=seed, device=device,
        readout_order=readout_order
    )
    id_dev = calibrate_identity(
        0.0, dev_max, n_basis=n_basis, n_steps=n_steps,
        epochs=epochs, seed=seed, device=device, log_sample=log_sample
    )
    id_istd = calibrate_identity(
        0.0, istd_max, n_basis=n_basis, n_steps=n_steps,
        epochs=epochs, seed=seed, device=device, log_sample=log_sample
    )
    return SpikingLayerNorm(rsqrt, id_dev, id_istd, eps=eps, spike_mult=spike_mult)
