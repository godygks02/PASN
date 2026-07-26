"""MBE-PASN-S -- one *shared* MBE basis set, prefix-routed linear readout.

Where :mod:`mbe.mbe_pasn` gives every routed range its own complete MBE neuron
(``R`` independent basis sets, ``sum_v 6 N_v`` stored parameters), this variant
keeps a **single** MBE neuron and lets the FP-prefix router select only the
*decoder*:

    j    = z(x)                        # FP sign+exponent, parameter-free
    u[0] = (|x| - 2^e) / 2^e = rho     # routed affine, exactly MBENeuron._normalise
    f(x) = features(rho) @ W[j] + b[j] # one spike train, R different readouts

Consequences (the reason for the design):

  * **Spikes.** Every input runs the same ``N`` bases for ``T`` steps, so the
    energy accounting is identical to a single global MBE neuron of the same
    ``(N, T)`` -- routing costs zero spikes.
  * **Capacity.** A global MBE spans one ``(N+1)``-dimensional function space; here
    the same features carry ``R`` independent readouts, i.e. a piecewise element of
    that space over ``R`` pieces -- ``R(N+1)`` dimensions for ``R(N+1)`` extra
    scalars, versus ``R x 6N`` for independent banks.
  * **Memory.** ``5N`` shared shape parameters + ``R(N+1)`` readout coefficients.
  * **Build cost.** The shape parameters are fitted once; each additional range is
    a linear solve, not another surrogate-gradient fit. Adding ranges is therefore
    cheap enough to route on more than the exponent.

The router is imported from :mod:`mbe.mbe_pasn` unchanged, so a flat-vs-shared
comparison differs only in what a bank contains.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import functions
from .mbe_pasn import PrefixRouter
from .neuron import MBENeuron, MBEConfig


# --------------------------------------------------------------------------
# Neuron
# --------------------------------------------------------------------------

class MBEPASNSNeuron(nn.Module):
    """Shared-basis, prefix-routed-readout MBE neuron.

    ``core`` is an ordinary :class:`MBENeuron` with identity normalisation (the
    routed residual ``rho`` is fed to it directly) and no bias -- the DC term lives
    in the per-bank readout. ``W`` has shape ``(n_banks, N+1)``: ``N`` basis weights
    plus the bank's DC offset.

    Exposes ``readout_features`` / ``set_readout`` / ``cfg`` / ``firing_rate``, so
    :func:`fit.fit_model` trains it with no changes: the ``R`` readouts are solved
    jointly in closed form while Adam/LBFGS move the shared shape parameters.
    """

    def __init__(self, router: PrefixRouter, core: MBENeuron, W: torch.Tensor):
        super().__init__()
        assert core.bias is None, "the DC term belongs to the per-bank readout"
        self.router = router
        self.core = core
        # fit_model anneals cfg.surrogate_alpha -- share the object, don't copy it.
        self.cfg = core.cfg
        self.register_buffer("W", W)
        self.register_buffer(
            "bank_min",
            torch.tensor([b["x_min"] for b in router.banks], dtype=W.dtype),
        )
        self.register_buffer(
            "bank_scale",
            torch.tensor([b["x_scale"] for b in router.banks], dtype=W.dtype),
        )

    # -- routing -----------------------------------------------------------
    def _route(self, flat: torch.Tensor):
        """``(bank index, intra-bank residual rho)`` for a flat input.

        Magnitude banks see ``|x|`` and the near-zero bank the signed ``x`` (as in
        :class:`mbe_pasn.MBEPASNNeuron`); the affine is then exactly the per-bank
        ``MBENeuron._normalise``, left unclamped for the same reason.
        """
        idx = self.router.route(flat)
        val = torch.where(idx == 0, flat, flat.abs())
        rho = (val - self.bank_min[idx]) / self.bank_scale[idx]
        return idx, rho

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        idx, rho = self._route(flat)
        feats = self.core.features(rho)              # (M, N) -- one spike train
        w = self.W[idx]                              # (M, N+1)
        N = self.core.cfg.n_basis
        out = (feats * w[:, :N]).sum(dim=1) + w[:, N]
        return out.reshape(x.shape)

    # -- linear readout interface (joint closed-form solve) ----------------
    def readout_features(self, x: torch.Tensor) -> torch.Tensor:
        """Block-structured features, shape ``(M, R(N+1))``.

        Row ``i`` holds ``[features(rho_i), 1]`` in the block of its own bank and
        zeros elsewhere. The normal equations are therefore block diagonal, so one
        least-squares solve over ``R(N+1)`` unknowns is *exactly* the ``R``
        independent per-bank solves -- and :func:`fit.solve_readout` needs no
        special case. (Dense here; for very large ``R`` the blocks would be solved
        separately.)
        """
        flat = x.reshape(-1)
        idx, rho = self._route(flat)
        feats = self.core.features(rho)              # (M, N)
        M, N = feats.shape
        K = N + 1
        vals = torch.cat([feats, torch.ones(M, 1, dtype=feats.dtype,
                                            device=feats.device)], dim=1)
        cols = idx.unsqueeze(1) * K + torch.arange(K, device=feats.device)
        blocks = torch.zeros(M, self.router.n_banks * K, dtype=feats.dtype,
                             device=feats.device)
        return blocks.scatter(1, cols, vals)

    @torch.no_grad()
    def set_readout(self, coeffs: torch.Tensor):
        self.W.copy_(coeffs.reshape(self.router.n_banks, -1))

    # -- spike-level access (parity with MBENeuron) ------------------------
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Spike-sum reconstruction ``sum_{n,t} W[j,n] d_n[t] s_n[t] + W[j,N]``.

        Same pure-spike form as :meth:`MBENeuron.reconstruct`, with the intensity
        matrix selected by the router -- so this neuron is a drop-in operand for
        the spike-driven FP multiply.
        """
        flat = x.reshape(-1)
        idx, rho = self._route(flat)
        S = self.core.spike_train(rho)               # (M, N, T)
        d, _, _ = self.core._kernels()               # (T, N)
        N = self.core.cfg.n_basis
        w = self.W[idx]                              # (M, N+1)
        a = d.t().unsqueeze(0) * w[:, :N].unsqueeze(-1)      # (M, N, T)
        out = (S * a).sum(dim=(1, 2)) + w[:, N]
        return out.reshape(x.shape)

    # -- cost introspection ------------------------------------------------
    @torch.no_grad()
    def spike_counts(self, x: torch.Tensor) -> torch.Tensor:
        """Spikes emitted **per input element**, same shape as ``x``.

        Element-wise rather than averaged, so a cost can be taken under a weighting
        (see the width-weighted selection metric in :func:`build_mbe_pasn_s`).
        """
        cfg = self.core.cfg
        _, rho = self._route(x.reshape(-1))
        _, r, vth = self.core._kernels()
        u = rho.unsqueeze(1).expand(-1, cfg.n_basis).contiguous()
        cnt = torch.zeros_like(rho)
        for t in range(cfg.n_steps):
            s = (u - vth[t] >= 0).to(u.dtype)
            cnt = cnt + s.sum(dim=1)
            u = u - s * r[t]
        return cnt.reshape(x.shape)

    @torch.no_grad()
    def mean_spikes(self, x: torch.Tensor) -> float:
        if x.numel() == 0:
            return 0.0
        return float(self.spike_counts(x).mean())

    @torch.no_grad()
    def firing_rate(self, x: torch.Tensor) -> float:
        cfg = self.core.cfg
        return self.mean_spikes(x) / (cfg.n_basis * cfg.n_steps)

    @torch.no_grad()
    def stored_bases(self) -> int:
        """One shared basis set, regardless of the number of ranges."""
        return int(self.core.cfg.n_basis)

    def num_learnable(self) -> int:
        """``5N`` shared shape parameters + ``R(N+1)`` readout coefficients.

        ``core.w`` is excluded: the per-bank ``W`` replaces it (it is left in place
        only so :func:`fit._is_readout` keeps it out of the gradient step).
        """
        shape = sum(p.numel() for n, p in self.core.named_parameters()
                    if p.requires_grad and n.rsplit(".", 1)[-1] not in ("w", "bias"))
        return int(shape + self.W.numel())

    @torch.no_grad()
    def bank_usage(self, x: torch.Tensor) -> dict:
        idx = self.router.route(x.reshape(-1))
        return {bi: int((idx == bi).sum()) for bi in range(self.router.n_banks)}


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

def _sample_bank(router: PrefixRouter, bi: int, domain: tuple[float, float],
                 m: int, device, offset: float = 0.5) -> torch.Tensor | None:
    """Grid over routed range ``bi``, **clamped to the domain**.

    ``None`` when the range lies entirely outside the calibrated domain -- such a
    bank is unreachable and must not contribute training samples (fitting it would
    evaluate the target outside its valid range; see
    :meth:`mbe_pasn.PrefixRouter.reachable`).

    A *grid*, not a random draw: the target is a known deterministic function on a
    known interval, so a stratified grid has strictly lower discrepancy than
    sampling -- and it removes the calibration set's seed dependence. That
    dependence was the dominant source of run-to-run spread (the shape-parameter
    init is deterministic and the readout is closed-form, so a random draw was the
    only thing the seed actually changed: 34x MSE spread on GELU at N=2).

    ``offset`` positions the samples inside each cell: ``0.5`` is the midpoint grid
    used for fitting, ``0.25`` an offset grid used only for *selecting* between
    candidates. Two grids with the same step and different in-cell offsets share no
    point, so selecting on the second cannot reward fitting the first's midpoints.
    """
    span = router.reachable(bi, domain[0], domain[1])
    if span is None:
        return None
    a, b = span
    step = (b - a) / m
    return a + step * (torch.arange(m, device=device, dtype=torch.float32) + offset)


def uniform_alpha(n_shared: int) -> list:
    """Leading thresholds that split ``rho in [0,1)`` at even quantiles.

    A global MBE log-spreads ``alpha_v`` over ``[2^-spread, 1]`` because the
    target's curvature concentrates at one end of its domain
    (:func:`functions.curvature_alpha`). A *routed* residual is different: ``rho``
    is ~uniform on ``[0,1)`` in **every** bank and each ``g_v`` is smooth there, so
    a basis whose leading threshold sits near 0 fires at almost every step and its
    feature is nearly constant in ``rho`` -- it carries no information.

    Measured on GELU with the log-spread init: ``N=4`` gave per-basis firing rates
    ``[0.88, 0.17, 0.71, 0.95]`` (effectively 1-2 useful bases) and an MSE *worse*
    than ``N=2``. Even quantiles keep every basis informative.
    """
    return [(n_shared - n) / (n_shared + 1) for n in range(n_shared)]


def build_mbe_pasn_s(name: str, domain: tuple[float, float], e_min: int = -2,
                     e_max: int | None = None, n_shared: int | list = 4,
                     n_steps: int = 16, epochs: int = 300,
                     m_per_bank: int = 800, seed: int = 0,
                     normalize_banks: bool = True, alpha_init: str = "auto",
                     restarts: int = 3, spike_budget: float | None = None,
                     verbose: bool = False,
                     device: torch.device | str = "cpu") -> MBEPASNSNeuron:
    """Build + fit an MBE-PASN-S neuron for target ``name`` on ``domain``.

    One surrogate-gradient fit of the shared dynamics, interleaved with joint
    closed-form solves of all ``R`` readouts (:func:`fit.fit_model`). Training
    samples are drawn per range so no bank is starved.

    ``normalize_banks`` (default on) fits the shared dynamics against *per-bank
    scale-normalised* targets ``g_v(rho)/s_v``, then folds ``s_v`` back into that
    bank's readout row. This is exact -- the readout is linear and per-bank, so
    ``s_v W'_v`` reproduces the unnormalised target -- and it matters: on a tail
    like GELU's the targets grow as ``2^e``, so an unweighted pooled loss lets the
    largest binades dictate where the *shared* staircase puts its steps, and the
    small banks inherit a placement tuned for someone else. Normalising equalises
    each range's contribution, i.e. the shared basis is fitted for uniform
    *relative* accuracy. Pass ``False`` for the ablation.

    ``alpha_init``: which leading-threshold placement to use for the shared bases.
    ``"uniform"`` puts them at even quantiles of ``rho`` (see :func:`uniform_alpha`);
    ``"logspread"`` keeps a global MBE's placement. Neither wins outright -- measured
    across four targets they select different points on the accuracy-energy frontier
    (uniform is far cheaper, e.g. GELU N=2: 3.7 vs 15.2 spikes, and repairs the N=4
    pathology; log-spread reaches lower MSE at the high-spike end). ``"auto"``
    (default) fits both and keeps the better one, so the choice is no longer a
    per-target hand-tune.

    ``restarts``: fits per candidate init, from jittered shape parameters (restart 0
    is always the principled init). The **selection grid is what makes these work**:
    scored on the *fitting* grid the very same restarts made things worse (GELU N=8
    test MSE 4.7e-5 -> 1.3e-4, since a fit can place staircase breakpoints to nail
    grid midpoints while drifting between them). Cost is linear: ``2 * restarts``
    fits with ``alpha_init="auto"``.

    ``spike_budget``: maximum mean spikes per input. Candidates over budget are
    discarded and the most accurate survivor is kept; if none fit, the cheapest
    candidate is returned (and reported). **Selection is otherwise accuracy-only, and
    accuracy is bought with spikes** -- on SiLU the candidate pool spans 9.6e-5 at 3.9
    spikes to 6.8e-6 at 22.1, and min-MSE always takes the expensive end. Since spikes
    are the energy currency, pass a budget whenever energy matters.

    ``n_shared`` accepts a list, in which case every value is a candidate. This is
    what makes a budget useful: the spike cost is dominated by ``N``, so a pool at one
    ``N`` cannot span much of the frontier.

    ``seed`` does **not** affect the result: the calibration grids, the threshold
    placement, the restart jitter and the readout solve are all deterministic. It is
    accepted for interface parity with the other builders.

    The returned neuron carries ``selection_trace``: every candidate's
    ``(n_shared, alpha_init, restart, mse, spikes, params, feasible, chosen)`` on the
    offset grid, so the caller can plot the frontier from a single build.
    """
    from .fit import fit_model

    device = torch.device(device)
    torch.manual_seed(seed)
    fn, _ = functions.REGISTRY[name]
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max)

    def grid(offset):
        parts = {bi: p for bi in range(router.n_banks)
                 if (p := _sample_bank(router, bi, domain, m_per_bank, device,
                                       offset)) is not None}
        if not parts:
            raise ValueError(f"no routed range intersects domain {domain}")
        return parts

    fit_parts = grid(0.5)                       # midpoints: what we fit on
    xs = torch.cat(list(fit_parts.values()))
    ys_true = fn(xs)
    sel_parts = grid(0.25)                      # offset: what we select on
    xsel = torch.cat(list(sel_parts.values()))
    ysel = fn(xsel)

    scale = torch.ones(router.n_banks, device=device)
    if normalize_banks:
        for bi, part in fit_parts.items():
            yv = fn(part)
            # Centre-free scale: the readout's DC term absorbs any offset, so only
            # the spread of the target over the range has to be equalised.
            s = float((yv - yv.mean()).abs().max())
            scale[bi] = s if s > 1e-12 else 1.0
    ys = ys_true / scale[router.route(xs)] if normalize_banks else ys_true

    # Selection weights. The grid puts the *same* number of points in every range
    # regardless of its width, so a plain mean over it over-weights the narrow
    # near-zero binades. Weighting each range by its width makes the offset-grid
    # loss an unbiased estimate of the metric actually reported (MSE under a uniform
    # draw over the domain) -- and it must be that metric, not the per-range
    # relative loss the *fit* uses: selecting on the relative loss picked the worse
    # candidate on 3 of 12 configurations (SiLU N=4 by 18x), because a model can win
    # on error averaged across ranges while losing on error averaged across inputs.
    # (When a real activation distribution is available, these weights are where its
    # measured per-range visit probabilities belong.)
    spans = {bi: router.reachable(bi, domain[0], domain[1]) for bi in sel_parts}
    total_w = sum(b - a for a, b in spans.values())
    sel_w = torch.cat([
        torch.full((p.numel(),),
                   (spans[bi][1] - spans[bi][0]) / total_w / p.numel(),
                   device=device)
        for bi, p in sel_parts.items()
    ])

    @torch.no_grad()
    def score(cand) -> tuple:
        """Width-weighted (squared error, spikes per input) on the offset grid.

        ``sel_w`` sums to one, so both are expectations under the same distribution
        the MSE column reports -- the accuracy and the energy of a candidate are
        measured against each other consistently.
        """
        mse = float(((cand(xsel) - ysel).pow(2) * sel_w).sum())
        spikes = float((cand.spike_counts(xsel) * sel_w).sum())
        return mse, spikes

    n_list = [n_shared] if isinstance(n_shared, int) else list(n_shared)
    inits = ["uniform", "logspread"] if alpha_init == "auto" else [alpha_init]
    candidates = [(n, ai, uniform_alpha(n) if ai == "uniform" else 1.0)
                  for n in n_list for ai in inits]

    trace = []
    for n_basis, ai, av in candidates:
        for k in range(max(restarts, 1)):
            # Jitter is keyed on the restart index *only*, never on ``seed``. Its
            # job is to explore basins, not to be random, and keying it on the seed
            # turned restarts into a lottery: the median over 3 seeds was no better
            # than no restarts at all while one lucky seed looked 7x better.
            torch.manual_seed(1000 * k)
            # Identity normalisation: the routed affine already maps input to rho.
            cfg = MBEConfig(n_basis=n_basis, n_steps=n_steps, x_min=0.0,
                            x_scale=1.0, alpha_v=av, use_bias=False)
            core = MBENeuron(cfg).to(device)
            # Restart 0 is the principled init; later restarts jitter it in log
            # space. Without this a restart is a no-op: the shape parameters are
            # initialised deterministically (alpha_v list + linspace tau) and the
            # only random tensor, the readout w, is overwritten by the closed-form
            # solve on the first step -- every "restart" was byte-identical.
            if k > 0:
                with torch.no_grad():
                    for p in (core.log_alpha_v, core.log_tau_r, core.log_tau_vth,
                              core.log_tau_d, core.log_dt):
                        p.add_(torch.randn_like(p) * 0.3)
            W = torch.zeros(router.n_banks, n_basis + 1, device=device)
            cand = MBEPASNSNeuron(router, core, W).to(device)
            fit_model(cand, xs, ys, seed=0, epochs=epochs)
            if normalize_banks:
                with torch.no_grad():
                    cand.W.mul_(scale.unsqueeze(1))   # undo the per-bank rescaling
            mse, spikes = score(cand)
            trace.append(dict(model=cand, n_shared=n_basis, alpha_init=ai,
                              restart=k, mse=mse, spikes=spikes,
                              params=cand.num_learnable(),
                              feasible=spike_budget is None
                              or spikes <= spike_budget, chosen=False))

    feasible = [c for c in trace if c["feasible"]]
    if feasible:
        pick = min(feasible, key=lambda c: c["mse"])
    else:
        # Budget unsatisfiable at any candidate: return the cheapest, don't pretend.
        pick = min(trace, key=lambda c: c["spikes"])
    pick["chosen"] = True
    model = pick["model"]
    model.selection_trace = [{k: v for k, v in c.items() if k != "model"}
                             for c in trace]
    if verbose:
        tag = "relative" if normalize_banks else "absolute"
        budget = ("none" if spike_budget is None
                  else f"{spike_budget:g}"
                       f"{'' if feasible else ' (UNSATISFIABLE)'}")
        print(f"    shared T={n_steps} over {router.n_banks} banks, {tag} fit, "
              f"spike budget {budget}: chose N={pick['n_shared']} "
              f"{pick['alpha_init']}/r{pick['restart']} -> mse={pick['mse']:.2e} "
              f"spikes={pick['spikes']:.2f} params={pick['params']}", flush=True)
        for c in trace:
            print(f"      {'*' if c['chosen'] else ' '} N={c['n_shared']} "
                  f"{c['alpha_init']:9s}/r{c['restart']} mse={c['mse']:.2e} "
                  f"spikes={c['spikes']:6.2f}"
                  f"{'' if c['feasible'] else '  over budget'}", flush=True)
    return model


def pareto_front(trace: list) -> list:
    """Candidates from a ``selection_trace`` that no other candidate beats on both
    accuracy and spikes. Use to report the frontier rather than one operating point."""
    out = []
    for c in trace:
        if not any(o is not c and o["mse"] <= c["mse"] and o["spikes"] <= c["spikes"]
                   and (o["mse"] < c["mse"] or o["spikes"] < c["spikes"])
                   for o in trace):
            out.append(c)
    return sorted(out, key=lambda c: c["spikes"])
