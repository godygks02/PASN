"""Multi-Basis Exponential Decay (MBE) neuron.

Reference: Wang et al., "Training-Free ANN-to-SNN Conversion for High-Performance
Spiking Transformers", AAAI-26 (paper + appendix in ``paper/``).

Core equations reproduced here
------------------------------
Parameter update (Eq. 4), per basis ``n`` and time step ``t`` in [0, T-1]::

    Para(tau_n, t) = alpha * exp(-t * dt / tau_n),   tau_n in {tau_d, tau_r, tau_Vth}

so the three time-varying kernels of basis ``n`` are::

    d_n[t]   = exp(-t * dt_n / tau_d_n)                 # spike intensity  (scaled by w)
    r_n[t]   = alpha_v_n * exp(-t * dt_n / tau_r_n)     # reset magnitude
    Vth_n[t] = alpha_v_n * exp(-t * dt_n / tau_Vth_n)   # firing threshold

Basis dynamics (Eq. 5-7), with membrane initial state ``u_n[0] = x_norm``::

    s_n[t]   = H(u_n[t] - Vth_n[t])
    u_n[t+1] = u_n[t] - s_n[t] * r_n[t]
    o_n[t+1] = o_n[t] + s_n[t] * d_n[t]

Decoded output (Eq. 8)::

    f_hat(x) = sum_n  w_n * o_n[T]  (+ bias)

Design decisions not fixed by the paper (documented; see plan section 5 and
MBE_RESULTS.md)
---------------------------------------------------------------------------
1. Surrogate gradient: configurable, default ATan (see ``surrogate.py``).
2. Input normalisation. The membrane input is mapped to O(1) by an affine
   calibration ``u[0] = (x - x_min) / x_scale`` so all internal quantities are
   order-one regardless of the raw domain (e.g. GELU's (-120, 10)). This is a
   numerical-stability transform, not a learned parameter; ``x_min`` / ``x_scale``
   come from the calibration range and are shared by every PASN bank.
3. Per-basis, learnable threshold scale ``alpha_v_n``. The paper's "multi-basis
   encoding" adaptively focuses resolution on high-curvature regions (its fix for
   the GSO problem). A *shared* leading threshold forces all bases into nested,
   collinear codes (verified empirically). We therefore give each basis its own
   ``alpha_v_n`` (leading threshold), initialised log-spread across the range and
   -- by default -- learnable, so bases migrate to where the target curves most.
   The paper treats ``alpha`` as a fixed hyperparameter (5N learnable params); a
   learnable per-basis ``alpha_v`` adds N params (6N total). Set
   ``learn_alpha_v=False`` to recover the fixed-alpha, 5N-parameter form.
4. Output DC bias: a single constant offset supplying the non-zero baseline of
   targets like 1/x, 2^x (analogue of the affine beta in LayerNorm/GELU handling).
5. Signed inputs are handled by the ``x_min`` shift so ``u[0] >= 0``; the DC bias
   provides any non-zero value at the domain minimum. This mechanism is applied
   identically inside every PASN bank (see ``PASN_method.md`` section 6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .surrogate import heaviside


@dataclass
class MBEConfig:
    """Configuration for an :class:`MBENeuron`."""

    n_basis: int = 4                 # N: number of temporal bases
    n_steps: int = 16                # T: number of time steps
    dt_init: float = 1.0             # initial discrete time step
    # Leading threshold scale (per basis). Scalar -> log-spread across bases when
    # alpha_v_spread > 0, else shared. A length-N sequence sets bases explicitly.
    alpha_v: float | list = 1.0
    alpha_v_spread: float = 6.0      # log2 span of the initial per-basis alpha_v
    learn_alpha_v: bool = True       # if False, alpha_v is a fixed hyperparameter (5N)
    x_min: float = 0.0               # calibration: domain lower bound (membrane shift)
    x_scale: float = 1.0             # calibration: domain span (membrane normaliser)
    decay: bool = True               # exponential decay kernels (False = ablation)
    use_bias: bool = True            # learnable output DC offset (affine baseline)
    # Decoder order. Eq. 8 reads out f_hat = sum_n w_n o_n, i.e. the decode is
    # *linear* in the accumulated features. That is the binding constraint on how
    # many bits a spike train can be worth: a basis running an exact binary search
    # produces o = rho_hat, so a linear readout can only express an affine function
    # of rho -- the fit must therefore distort the staircase itself to manufacture
    # curvature, and spends resolution doing it (measured: effective bits grow like
    # 2 + 1.7*log2(T), not like T). order > 1 appends o^2, o^3, ... so the decoder
    # supplies the curvature instead. Costs stored coefficients only -- the spike
    # count is unchanged. Keep order 1 for MBE_Id: the FP-multiply factorisation
    # needs the output to stay a pure spike sum.
    readout_order: int = 1
    learn_tau: bool = True           # False freezes the kernels at their init
    tau_min: float = 1.0             # smallest init time constant (fastest decay)
    surrogate: str = "atan"          # surrogate gradient kind
    surrogate_alpha: float = 2.0     # surrogate sharpness
    tau_max: float = 8.0             # largest init time constant (slowest decay)


class MBENeuron(nn.Module):
    """A vectorised MBE neuron approximating a scalar function ``f: R -> R``.

    A single neuron holds ``N`` bases. ``forward`` accepts an arbitrary input
    tensor and applies the neuron element-wise (broadcasting over the ``N`` bases
    and ``T`` time steps), returning a tensor of the same shape.
    """

    def __init__(self, config: MBEConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = MBEConfig(**kwargs)
        elif kwargs:
            raise ValueError("pass either a MBEConfig or kwargs, not both")
        self.cfg = config
        N = config.n_basis

        # -- per-basis leading threshold alpha_v (in normalised input units) ---
        if isinstance(config.alpha_v, (list, tuple)):
            av = torch.tensor(list(config.alpha_v), dtype=torch.float32)
            if av.numel() != N:
                raise ValueError(f"alpha_v sequence must have length N={N}")
        elif config.alpha_v_spread > 0 and N > 1:
            hi = float(config.alpha_v)
            av = torch.logspace(
                math.log10(hi),
                math.log10(hi / (2.0 ** config.alpha_v_spread)),
                N,
            )
        else:
            av = torch.full((N,), float(config.alpha_v))
        log_alpha_v = av.clamp(min=1e-6).log()
        if config.learn_alpha_v:
            self.log_alpha_v = nn.Parameter(log_alpha_v)
        else:
            self.register_buffer("log_alpha_v", log_alpha_v)

        # -- time constants (log-space, positive) ------------------------------
        # Log-spaced across bases -> multi-resolution set of decay rates.
        log_tau_init = torch.linspace(
            math.log(config.tau_min), math.log(config.tau_max), N
        )
        if config.learn_tau:
            self.log_tau_r = nn.Parameter(log_tau_init.clone())
            self.log_tau_vth = nn.Parameter(log_tau_init.clone())
            self.log_tau_d = nn.Parameter(log_tau_init.clone())
            self.log_dt = nn.Parameter(torch.full((N,), math.log(config.dt_init)))
        else:
            # Frozen kernels. With tau = dt/ln2 every step halves, i.e. the basis
            # is an exact binary-search (SAR) encoder -- the paper's own reading of
            # Eq. 4. Freezing lets us ask whether the *learned* kernels are worth
            # their cost once the decoder can be nonlinear (see readout_order).
            self.register_buffer("log_tau_r", log_tau_init.clone())
            self.register_buffer("log_tau_vth", log_tau_init.clone())
            self.register_buffer("log_tau_d", log_tau_init.clone())
            self.register_buffer(
                "log_dt", torch.full((N,), math.log(config.dt_init)))

        # -- basis weights w (order blocks: [o, o^2, ...]) ----------------------
        self.w = nn.Parameter(torch.randn(N * config.readout_order) * 0.1)

        # -- output DC bias ----------------------------------------------------
        if config.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("bias", None)

    # -- kernels -----------------------------------------------------------
    def _kernels(self):
        """Return the (T, N) kernels d, r, Vth following Eq. 4."""
        cfg = self.cfg
        T = cfg.n_steps
        dtype = self.w.dtype
        device = self.w.device
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T,1)

        alpha_v = self.log_alpha_v.exp()          # (N,)
        tau_r = self.log_tau_r.exp()
        tau_vth = self.log_tau_vth.exp()
        tau_d = self.log_tau_d.exp()
        dt = self.log_dt.exp()

        if cfg.decay:
            decay_d = torch.exp(-t * dt / tau_d)      # (T,N)
            decay_r = torch.exp(-t * dt / tau_r)
            decay_vth = torch.exp(-t * dt / tau_vth)
        else:
            # Ablation: kernels are constant across time (no exponential decay).
            ones = torch.ones(T, cfg.n_basis, device=device, dtype=dtype)
            decay_d = decay_r = decay_vth = ones

        d = decay_d                                   # intensity scaled later by w
        r = alpha_v * decay_r                         # (N,) broadcast over (T,N)
        vth = alpha_v * decay_vth
        return d, r, vth

    # -- forward -----------------------------------------------------------
    def _normalise(self, x_flat: torch.Tensor) -> torch.Tensor:
        return (x_flat - self.cfg.x_min) / self.cfg.x_scale

    def features(self, x: torch.Tensor, with_rate: bool = False):
        """Per-basis accumulated outputs ``o_n(T)``, shape ``(M, N)``.

        These are the linear-readout features: ``forward = features @ w (+ bias)``.
        Exposing them lets the optimiser solve the readout ``(w, bias)`` in closed
        form (see ``fit.solve_readout``).

        ``with_rate=True`` also returns the mean surrogate spike fraction over the
        same pass -- differentiable, so it can be penalised in the loss. Note the
        readout ``(w, bias)`` does not enter the spike dynamics at all, so adding a
        firing-rate term leaves the closed-form readout solve exactly optimal.
        """
        cfg = self.cfg
        T, N = cfg.n_steps, cfg.n_basis
        x_flat = x.reshape(-1)
        d, r, vth = self._kernels()                 # each (T, N)
        u = self._normalise(x_flat).unsqueeze(1).expand(-1, N).contiguous()  # (M,N)
        o = torch.zeros_like(u)
        fired = None
        for t in range(T):
            s = heaviside(u - vth[t], cfg.surrogate, cfg.surrogate_alpha)
            u = u - s * r[t]
            o = o + s * d[t]
            if with_rate:
                fired = s.mean() if fired is None else fired + s.mean()
        return (o, fired / T) if with_rate else o

    def forward_with_rate(self, x: torch.Tensor):
        """``(output, mean firing rate)`` from one pass -- both differentiable."""
        o, rate = self.features(x, with_rate=True)
        p = self.cfg.readout_order
        feats = o if p == 1 else torch.cat([o ** k for k in range(1, p + 1)], 1)
        out = (feats * self.w).sum(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(x.shape), rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.readout_features(x)             # (M, N*order)
        out = (feats * self.w).sum(dim=1)            # Eq. 8 (order 1) / extended
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(x.shape)

    # -- linear readout interface (for closed-form (w, bias) solves) -------
    def readout_features(self, x: torch.Tensor) -> torch.Tensor:
        """``[o, o^2, ..., o^order]`` concatenated, shape ``(M, N*order)``.

        Still *linear in the coefficients*, so the closed-form readout solve
        (:func:`fit.solve_readout`) needs no change -- only the design matrix gets
        wider. The spike train is untouched, so the energy cost is unchanged.
        """
        o = self.features(x)
        p = self.cfg.readout_order
        if p == 1:
            return o
        return torch.cat([o ** k for k in range(1, p + 1)], dim=1)

    @torch.no_grad()
    def set_readout(self, coeffs: torch.Tensor):
        """Assign the readout from a solved vector ``[w (N*order,), bias]``."""
        self.w.copy_(coeffs[:self.w.numel()])
        if self.bias is not None:
            self.bias.copy_(coeffs[-1:])

    # -- spike-level access (for spike-driven FP multiplication) -----------
    @torch.no_grad()
    def spike_train(self, x: torch.Tensor) -> torch.Tensor:
        """Hard binary spikes ``s_n[t]``, shape ``(M, N, T)``.

        Unlike :meth:`features` (which returns the intensity-weighted sum), this
        exposes the raw spikes so that the spike-driven FP multiplication can form
        the outer product ``S = b(x1) (x) b(x2)`` (Eq. 11, 26).
        """
        cfg = self.cfg
        T, N = cfg.n_steps, cfg.n_basis
        x_flat = x.reshape(-1)
        _, r, vth = self._kernels()
        u = self._normalise(x_flat).unsqueeze(1).expand(-1, N).contiguous()
        S = torch.zeros(x_flat.shape[0], N, T, device=u.device, dtype=u.dtype)
        for t in range(T):
            s = (u - vth[t] >= 0).to(u.dtype)
            S[:, :, t] = s
            u = u - s * r[t]
        return S

    @torch.no_grad()
    def intensities(self) -> torch.Tensor:
        """Per-``(n, t)`` intensities ``a_{n,t} = w_n * d_n[t]``, shape ``(N, T)``.

        Input-independent (depends only on trained parameters), so it is the
        precomputable intensity matrix ``D`` of Eq. 11 / Appendix F.1.

        Only defined for a linear decoder: the FP-multiply factorisation needs the
        output to *be* the spike sum, which a polynomial readout is not.
        """
        if self.cfg.readout_order != 1:
            raise ValueError(
                "intensities()/reconstruct() need readout_order=1 -- the "
                "spike-driven multiply factorises only a pure spike sum")
        d, _, _ = self._kernels()                    # d: (T, N)
        return self.w.unsqueeze(1) * d.t()           # (N, T)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Spike-sum reconstruction ``sum_{n,t} a_{n,t} s_n[t] (+bias)`` ~= f(x).

        For an identity-calibrated neuron this returns ~= x, purely from spikes
        and precomputed intensities."""
        S = self.spike_train(x)                       # (M, N, T)
        a = self.intensities()                        # (N, T)
        out = (S * a).sum(dim=(1, 2))
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(x.shape)

    # -- introspection -----------------------------------------------------
    @torch.no_grad()
    def firing_rate(self, x: torch.Tensor) -> float:
        """Average firing rate (fraction of the (M,N,T) spikes that fire)."""
        cfg = self.cfg
        T, N = cfg.n_steps, cfg.n_basis
        x_flat = x.reshape(-1)
        _, r, vth = self._kernels()
        u = self._normalise(x_flat).unsqueeze(1).expand(-1, N).contiguous()
        total = 0.0
        for t in range(T):
            s = (u - vth[t] >= 0).to(u.dtype)
            total += s.sum().item()
            u = u - s * r[t]
        return total / (x_flat.numel() * N * T)

    def num_learnable(self) -> int:
        """Number of learnable parameters (5N, or 6N with learnable alpha_v,
        plus 1 for the optional bias)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
