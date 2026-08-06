"""Accuracy-spike-memory Pareto per operator: global MBE vs PASN, same conditions.

Every previous head-to-head in this repo fixed one operating point per backend and
compared the numbers (``experiments/compare_ops_pasn_mbe.py``). That cannot answer
the question the method actually rests on -- *at the same accuracy, what does each
backend cost* -- because a single point is free to be better on accuracy by
spending more spikes. This sweeps both backends over their own budget knob, builds
the non-dominated front for each, and then reads the cost off at matched accuracy.

Five operator families, each priced end-to-end:

  ``activation``   gelu / silu / tanh -- one neuron, the decoded output *is* the op
  ``fp_multiply``  signed operands through the spike-driven product (Eq. 9-12)
  ``softmax``      exp / sum / reciprocal / multiply (Eq. 13)
  ``layernorm``    square / inverse-sqrt / multiply (Fig. 5(c))
  ``attention``    QK^T -> softmax -> AV, the three assembled back to back

**Spikes are metered, not derived.** :class:`~mbe.op_cost.SpikeMeter` wraps every
primitive an op holds and tallies what a real forward pass emits, normalised per
*output element* of the op. This matters because the driver primitive is not the
op: softmax's reciprocal runs once per row while its ``2^x`` runs once per
element, and the FP multiply's signed split reconstructs each operand twice --
unless the identity's own router reads the sign, which is one of the things being
compared. Multiplicities are reported in ``mult_by_prim`` so the totals can be
audited.

**Four arms, so the allocation is isolated rather than assumed.**

  ``mbe``          one global ``(N, T)`` over the whole domain -- the paper's
                   uniform setting, and the only shape a single neuron has.
  ``pasn``         prefix banks, every bank on the *same* ``(n_local, T)``. No
                   allocation at all, so this front measures **routing alone**.
  ``pasn_rule_T``  the budget rule picks ``N_j`` per bank; ``T`` is then pinned
                   to the MBE arm's ``T``. Timestep-matched, which answers "you
                   used fewer timesteps than the paper" at a higher spike cost.
  ``pasn_rule``    the rule picks ``N_j`` **and** ``T_j`` per bank -- fully
                   dynamic, PASN as the method actually is. ``T`` is a *linear*
                   factor in the spike count, so pinning it gives away a real
                   part of the saving; this arm is the headline one.

Chaining them (``pasn`` -> ``pasn_rule_T`` -> ``pasn_rule``) separates what
routing buys from what allocating ``N_j`` across the routes buys from what
allocating ``T_j`` adds, each measured at matched accuracy. ``budget_tables`` is
recorded per build so the allocation can be inspected rather than trusted: if
every bank came out at the same ``(N, T)`` the rule degenerated to the uniform
arm, and the rows are not measuring different things.

**What "same conditions" means here.** All arms see the same eval tensors, the
same fit budget (``--epochs``), the same seed, the same metering code, and the
same reference op. What differs is only the neuron. Each backend is built the way
it is meant to be built -- the signed activation handler for MBE's GELU/SiLU,
sign routing for PASN's identities -- so the comparison is between methods, not
between crippled variants of one method.

**On the baseline.** The sweep needs a knob, so the MBE arm here is *our*
reimplementation, not the paper's published tables (which report single operating
points). Per the 2026-08-01 decision that reimplementation is an internal
ablation, never a headline baseline -- so the paper's own numbers are overlaid
separately, at primitive level, where they are directly comparable:
:data:`PAPER_TABLE_X` (MSE vs N at T=16) and :data:`PAPER_TABLE_XI` (firing rates
-> spikes per element). Read those as the external reference and the swept MBE
front as the controlled one.

Usage::

    python experiments/op_pareto.py --smoke                     # ~2 min, wiring
    python experiments/op_pareto.py --json results/op_pareto.json
    python experiments/op_pareto.py --ops softmax layernorm     # the isolated pair
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mbe import functions  # noqa: E402
from mbe import spiking_ops as so  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402
from mbe.op_cost import SpikeMeter, op_memory  # noqa: E402


# --------------------------------------------------------------------------
# The paper's published operator numbers (external reference points)
# --------------------------------------------------------------------------

#: Table X -- MSE vs number of bases at T=16, per primitive. Directly comparable
#: to our primitive-level accuracy because it is the same quantity on the same
#: functions; the sampling domain is the paper's own and may differ from ours,
#: which is why it is drawn as reference points and not as a swept arm.
PAPER_TABLE_X = {
    "gelu":    {1: 7.1e-3, 2: 4.1e-3, 4: 2.3e-4, 6: 1.7e-4, 8: 1.0e-4},
    "invsqrt": {1: 1.4e-3, 2: 1.2e-3, 4: 3.1e-4, 6: 1.0e-4, 8: 4.9e-5},
    "inv":     {1: 8.6e-3, 2: 2.1e-4, 4: 1.1e-3, 6: 2.2e-3, 8: 4.4e-5},
    "exp2":    {1: 8.9e-4, 2: 4.5e-4, 4: 4.0e-4, 6: 2.4e-4, 8: 5.3e-5},
}

#: Table XI -- per-primitive firing rate (%) at T=16 on ViT-M/16, with the basis
#: count from appendix G.1. Spikes per element is ``rate/100 * N * T``: a firing
#: rate alone is not comparable across different N, and their own energy model is
#: ``T * eta * N``, so the product is the honest column.
#:
#: **Cross-model.** Table XI is measured during ViT-M/16 inference; our numbers
#: come from the isolated ops here. The two rows whose argument is an IEEE field
#: rather than an activation (``2^x`` sees ``frac(x log2 e)``, ``1/x`` sees a
#: mantissa) are close to architecture-independent and are the ones worth reading
#: closely -- see ``experiments/firing_rates.py``, which makes the same caveat.
PAPER_TABLE_XI = {
    # label:                (rate %, N, our op, our primitive)
    "2^x":                  (46.94, 8, "softmax",   "exp"),
    "1/x":                  (3.74,  8, "softmax",   "inv"),
    "1/sqrt(x)":            (25.27, 8, "layernorm", "rsqrt"),
    "LayerNorm_input_id":   (27.61, 8, "layernorm", "id_dev"),
    "LayerNorm_1/x_id":     (38.46, 8, "layernorm", "id_istd"),
    "GELU":                 (38.22, 4, "activation", "act"),
    "Attention_score":      (8.31,  8, "attention", "id_qk"),
}
PAPER_T = 16


def paper_table_xi_spikes() -> dict:
    """Table XI rows as spikes per input element (``rate * N * T``)."""
    return {k: dict(rate=r, n=n, t=PAPER_T, spikes=r / 100.0 * n * PAPER_T,
                    op=op, prim=prim)
            for k, (r, n, op, prim) in PAPER_TABLE_XI.items()}


# --------------------------------------------------------------------------
# Error metrics -- one scale-free primary axis for every op
# --------------------------------------------------------------------------

def errors(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Accuracy of ``out`` against the exact op's ``ref``.

    ``nrmse`` (RMS error over RMS signal) is the **primary axis** for every op:
    the five ops differ in output scale by orders of magnitude -- a softmax row
    sums to 1, a product over [-64,64] does not -- so a raw MSE front cannot be
    read across ops, and a per-op axis cannot be read at all. ``mse`` is kept
    because it is what the paper's Table X reports, and ``rel``/``mae`` because
    the FP multiply is judged on relative error.
    """
    d = out - ref
    ref_rms = float(ref.pow(2).mean().sqrt())
    return dict(
        nrmse=float(d.pow(2).mean().sqrt()) / max(ref_rms, 1e-12),
        mse=float(d.pow(2).mean()),
        mae=float(d.abs().mean()),
        rel=float((d.abs() / ref.abs().clamp(min=1e-6)).mean()),
        max_abs=float(d.abs().max()),
    )


# --------------------------------------------------------------------------
# PASN construction helpers
# --------------------------------------------------------------------------

_PASN_CACHE: dict = {}


#: Per-target router origin, taken from the conversion config so the sweep and
#: the shipped converter cannot drift apart. ``inv`` needs one because ``[0.5, 1)``
#: is a single binade -- see ``experiments/inv_router_fix.py`` and
#: ``ConvertConfig.pasn_beta``.
def _beta_table() -> dict:
    from mbe.convert import ConvertConfig
    return dict(ConvertConfig().pasn_beta or {})


BETA = _beta_table()


def _binades(lo: float, hi: float, span: int,
             beta: float = 0.0) -> tuple[int, int]:
    """Router exponent range covering ``[lo, hi]`` with ``span`` binades.

    ``e_max`` is set by the largest magnitude present; ``e_min`` cuts the ladder
    off ``span`` binades below it, and everything smaller falls into the
    near-zero bank. ``span`` is the memory knob of the routed side (one bank per
    binade per sign), held **fixed** across the sweep so that the swept axis stays
    ``(n_local, T)`` -- the same two quantities the MBE arm sweeps.

    With ``beta`` the magnitudes are measured from the shifted origin, since that
    is what the router keys on. Shifting lets ``|x - beta|`` reach arbitrarily
    close to zero *inside* the domain, so the usable binade count stops being a
    property of the domain and becomes the ``span`` choice -- which is exactly why
    it is a fixed knob here rather than a swept one.
    """
    lo, hi = float(lo) - beta, float(hi) - beta
    mag = max(abs(lo), abs(hi), 1e-6)
    e_max = int(math.ceil(math.log2(mag)))
    return e_max - int(span), e_max


def pasn(name: str, domain: tuple[float, float], n: int, t: int, epochs: int,
         seed: int, span: int, **kw):
    """Build (and memoise) a uniform-budget PASN neuron for ``name`` on ``domain``.

    ``n_local == n_near0 == n``: a *uniform* budget across banks, deliberately
    mirroring the MBE arm's single global ``N``. Neither arm gets per-region
    allocation here, so the front measures what **routing alone** buys. The
    allocation on top is the separate ``pasn_rule`` arm.
    """
    beta = BETA.get(name, 0.0)
    e_min, e_max = _binades(*domain, span=span, beta=beta)
    key = (name, round(domain[0], 6), round(domain[1], 6), e_min, e_max, n, t,
           epochs, seed, beta, tuple(sorted(kw.items())))
    if key not in _PASN_CACHE:
        _PASN_CACHE[key] = build_mbe_pasn(
            name, domain, e_min=e_min, e_max=e_max, n_local=n, n_near0=n,
            n_steps=t, epochs=epochs, seed=seed, beta=beta, **kw)
    return _PASN_CACHE[key]


def pasn_rule(name: str, domain: tuple[float, float], t: int | None,
              target_rel: float, epochs: int, seed: int, span: int):
    """PASN with the budget **rule**: each bank's ``(N_j, T_j)`` solved from its
    own dynamic range against a common relative-error target.

    ``t`` controls how much of that allocation survives, and the difference is the
    point of having two arms:

    * ``t=None`` -- **fully dynamic**. The rule picks ``N_j`` *and* ``T_j`` per
      bank; a flat tail collapses to ``(1, 2)`` while a bank that needs
      resolution gets a long train. This is PASN as the method actually is, and
      ``T`` is a *linear* factor in the spike count, so it is not a detail.
    * ``t=<int>`` -- ``t_fixed`` pins every bank's ``T`` **after** the rule has
      run, keeping ``N_j`` and discarding ``T_j``. Every bank then gets at least
      as many steps as it asked for, so this cannot help accuracy less than the
      full rule; it costs more spikes and exists only to answer the objection
      "you used fewer timesteps than the paper's global T".

    Running both isolates the two halves of the allocation: ``pasn`` -> ``rule_T``
    is what ``N_j`` buys, ``rule_T`` -> ``rule`` is what ``T_j`` adds on top.
    """
    beta = BETA.get(name, 0.0)
    e_min, e_max = _binades(*domain, span=span, beta=beta)
    key = ("rule", name, round(domain[0], 6), round(domain[1], 6), e_min, e_max,
           t, target_rel, epochs, seed, beta)
    if key not in _PASN_CACHE:
        kw = {} if t is None else dict(t_fixed=t)
        _PASN_CACHE[key] = build_mbe_pasn(
            name, domain, e_min=e_min, e_max=e_max, budget="rule",
            target="relative", target_rel=target_rel, beta=beta,
            epochs=epochs, seed=seed, **kw)
    return _PASN_CACHE[key]


#: The routed arms, in order of how much per-bank allocation they are allowed.
#: ``pasn`` gets none (uniform banks, so the front measures *routing alone*),
#: ``pasn_rule_T`` gets ``N_j`` only, ``pasn_rule`` gets ``N_j`` and ``T_j``.
PASN_ARMS = ("pasn", "pasn_rule_T", "pasn_rule")


def maker(backend: str, b, t: int | None, cfg):
    """``mk(name, domain) -> neuron`` for whichever routed arm is being built.

    One dispatch point for all five operator specs: each op only knows the
    (target, domain) pairs its decomposition needs, never which arm it is on.
    """
    if backend == "pasn":
        return lambda nm, dom: pasn(nm, dom, int(b), t, cfg.epochs, cfg.seed,
                                    cfg.binades)
    if backend == "pasn_rule_T":        # rule N_j, T pinned to the MBE arm's T
        return lambda nm, dom: pasn_rule(nm, dom, t, b, cfg.epochs, cfg.seed,
                                         cfg.binades)
    if backend == "pasn_rule":          # rule N_j and T_j -- fully dynamic
        return lambda nm, dom: pasn_rule(nm, dom, None, b, cfg.epochs, cfg.seed,
                                         cfg.binades)
    raise ValueError(f"not a routed arm: {backend!r}")


# --------------------------------------------------------------------------
# Operator specs
#   data(cfg)                -> dict of eval tensors + `ref` + `n_out`
#   build(backend, b, t, cfg, data) -> dict of primitives (metered)
#   run(prims, data)         -> the op's output
# --------------------------------------------------------------------------

def _act_domain(name: str) -> tuple[float, float]:
    return functions.REGISTRY[name][1]


def make_activation(name: str):
    """One calibrated neuron; its decoded output is the op (Phase 2)."""

    def data(cfg):
        lo, hi = _act_domain(name)
        fn, _ = functions.REGISTRY[name]
        # held-out draw (seed+1): fitting and scoring on the same points hides a
        # staircase that only matches at its own grid nodes
        x, y, _ = functions.sample(name, m=cfg.m_eval, seed=cfg.seed + 1,
                                   domain=(lo, hi))
        return dict(x=x, ref=y, n_out=x.numel(), domain=(lo, hi))

    def build(backend, b, t, cfg, d):
        lo, hi = d["domain"]
        if backend == "mbe":
            # build_activation picks the polarity-split handler for gelu/silu --
            # the paper-faithful path; a plain global neuron cannot represent the
            # near-zero bend at all (spiking_ops._NONMONOTONE_ACTS).
            sa = so.build_activation(name, torch.tensor([lo, hi]), n_basis=b,
                                     n_steps=t, epochs=cfg.epochs, seed=cfg.seed,
                                     margin=0.0)
            return dict(act=sa.neuron)
        return dict(act=maker(backend, b, t, cfg)(name, (lo, hi)))

    def run(prims, d):
        return prims["act"](d["x"])

    return dict(name=f"activation:{name}", fn=name, data=data, build=build,
                run=run, primary="nrmse")


def make_fp_multiply(hi: float = 64.0):
    """Spike-driven product of two **signed** operands (Eq. 9-12, 22-27).

    Signed rather than the [0, hi] case measured before, because that is what a
    transformer actually multiplies -- and because the sign is where the two
    backends structurally differ: ``spiking_multiply(signed=None)`` asks the
    identity whether its router reads the sign, so MBE takes the four-term
    ``relu`` split and a sign-routing PASN takes the operands directly. Both
    compute the same product; the invocation count is what differs, and metering
    catches it.
    """

    def data(cfg):
        g = torch.Generator().manual_seed(cfg.seed + 1)
        x1 = (torch.rand(cfg.m_eval, generator=g) * 2 - 1) * hi
        x2 = (torch.rand(cfg.m_eval, generator=g) * 2 - 1) * hi
        return dict(x1=x1, x2=x2, ref=x1 * x2, n_out=x1.numel())

    def build(backend, b, t, cfg, d):
        if backend == "mbe":
            # a plain MBE_Id maps non-negative inputs, so its domain is [0, hi]
            # and the signed split supplies the polarity
            return dict(idn=so.calibrate_identity(0.0, hi, n_basis=b, n_steps=t,
                                                  epochs=cfg.epochs,
                                                  seed=cfg.seed))
        return dict(idn=maker(backend, b, t, cfg)("identity", (-hi, hi)))

    def run(prims, d):
        return so.spiking_multiply(prims["idn"], d["x1"], d["x2"], signed=None)

    return dict(name="fp_multiply", fn="identity", data=data, build=build,
                run=run, primary="rel")


def make_softmax(rows: int = 128, cols: int = 64, scale: float = 3.0):
    """exp / sum / reciprocal / multiply over a row of ``cols`` logits."""

    def data(cfg):
        g = torch.Generator().manual_seed(cfg.seed + 1)
        logits = torch.randn(rows, cols, generator=g) * scale
        return dict(logits=logits, ref=torch.softmax(logits, dim=-1),
                    n_out=logits.numel())

    def build(backend, b, t, cfg, d):
        if backend == "mbe":
            sm = so.build_softmax(d["logits"], n_basis=b, n_steps=t,
                                  epochs=cfg.epochs, seed=cfg.seed,
                                  spike_mult=True)
            return dict(exp=sm.exp, inv=sm.inv, idn=sm.idn)
        mk = maker(backend, b, t, cfg)
        # the three fixed domains the decomposition guarantees: frac in [0,1),
        # mantissa in [0.5,1), and the multiply's operands in [0,1]
        return dict(exp=mk("exp2", (0.0, 1.0)), inv=mk("inv", (0.5, 1.0)),
                    idn=mk("identity", (0.0, 1.0)))

    def run(prims, d):
        sm = so.SpikingSoftmax(prims["exp"], prims["inv"], prims["idn"],
                               spike_mult=True)
        return sm(d["logits"], dim=-1)

    return dict(name="softmax", fn="exp2", data=data, build=build, run=run,
                primary="nrmse")


def make_layernorm(rows: int = 80, dim: int = 64):
    """square / inverse-sqrt / multiply, with the affine applied."""

    def data(cfg):
        g = torch.Generator().manual_seed(cfg.seed + 1)
        x = torch.randn(rows, dim, generator=g) * 2.0 + 1.0
        gamma = torch.randn(dim, generator=g) * 0.5 + 1.0
        beta = torch.randn(dim, generator=g) * 0.1
        ref = F.layer_norm(x, (dim,), weight=gamma, bias=beta, eps=1e-5)
        return dict(x=x, gamma=gamma, beta=beta, ref=ref, n_out=x.numel())

    def _ranges(x, eps=1e-5):
        dev = x - x.mean(dim=-1, keepdim=True)
        dev_max = float(dev.abs().max()) * 1.1 + 1e-3
        var = (dev * dev).mean(dim=-1, keepdim=True) + eps
        istd_max = float((1.0 / var.sqrt()).max()) * 1.1 + 1e-3
        return dev_max, istd_max

    def build(backend, b, t, cfg, d):
        if backend == "mbe":
            ln = so.build_layernorm(d["x"], n_basis=b, n_steps=t,
                                    epochs=cfg.epochs, seed=cfg.seed,
                                    spike_mult=True)
            return dict(rsqrt=ln.rsqrt, id_dev=ln.id_dev, id_istd=ln.id_istd)
        dev_max, istd_max = _ranges(d["x"])
        mk = maker(backend, b, t, cfg)
        # the deviation and inverse-std identities are given *signed* domains so
        # the router splits polarity itself (spiking_multiply(signed=None))
        return dict(rsqrt=mk("invsqrt", (0.5, 2.0)),
                    id_dev=mk("identity", (-dev_max, dev_max)),
                    id_istd=mk("identity", (-istd_max, istd_max)))

    def run(prims, d):
        ln = so.SpikingLayerNorm(prims["rsqrt"], prims["id_dev"],
                                 prims["id_istd"], eps=1e-5, spike_mult=True)
        return ln(d["x"], weight=d["gamma"], bias=d["beta"])

    return dict(name="layernorm", fn="invsqrt", data=data, build=build, run=run,
                primary="nrmse")


def make_attention(heads: int = 4, seq: int = 32, dh: int = 16):
    """QK^T -> softmax -> AV: the three assembled ops back to back.

    Both matmuls are activation x activation, so both go through the spike-driven
    path; only these need it (activation x weight stays native accumulation).
    This is the op that dominates the real spike budget -- 86.5% of GPT-2's --
    and the one where ``av_matmul`` was the largest per-site error in the layer
    profile, so it is the one worth pricing whole.
    """

    def data(cfg):
        g = torch.Generator().manual_seed(cfg.seed + 1)
        q = torch.randn(heads, seq, dh, generator=g)
        k = torch.randn(heads, seq, dh, generator=g)
        v = torch.randn(heads, seq, dh, generator=g)
        scale = 1.0 / math.sqrt(dh)
        ref = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v
        return dict(q=q, k=k, v=v, scale=scale, ref=ref, n_out=ref.numel())

    def build(backend, b, t, cfg, d):
        qk_max = float(max(d["q"].abs().max(), d["k"].abs().max())) * 1.1
        v_max = float(d["v"].abs().max()) * 1.1
        if backend == "mbe":
            sm = so.build_softmax(d["q"] @ d["k"].transpose(-1, -2) * d["scale"],
                                  n_basis=b, n_steps=t, epochs=cfg.epochs,
                                  seed=cfg.seed, spike_mult=True)
            com = dict(epochs=cfg.epochs, seed=cfg.seed)
            return dict(
                exp=sm.exp, inv=sm.inv, idn=sm.idn,
                id_qk=so.calibrate_identity(0.0, qk_max, n_basis=b, n_steps=t,
                                            **com),
                # the attention matrix is in [0,1]; V is not
                id_a=so.calibrate_identity(0.0, 1.0, n_basis=b, n_steps=t, **com),
                id_v=so.calibrate_identity(0.0, v_max, n_basis=b, n_steps=t,
                                           **com))
        mk = maker(backend, b, t, cfg)
        return dict(exp=mk("exp2", (0.0, 1.0)), inv=mk("inv", (0.5, 1.0)),
                    idn=mk("identity", (0.0, 1.0)),
                    id_qk=mk("identity", (-qk_max, qk_max)),
                    id_a=mk("identity", (0.0, 1.0)),
                    id_v=mk("identity", (-v_max, v_max)))

    def run(prims, d):
        scores = so.spiking_matmul(prims["id_qk"], d["q"],
                                   d["k"].transpose(-1, -2),
                                   signed=None) * d["scale"]
        sm = so.SpikingSoftmax(prims["exp"], prims["inv"], prims["idn"],
                               spike_mult=True)
        attn = sm(scores, dim=-1)
        # attention weights are non-negative; V is signed, so the split is driven
        # by id_v's router (spiking_matmul asks both operands)
        return so.spiking_matmul(prims["id_a"], attn, d["v"], signed=None,
                                 idn2=prims["id_v"])

    return dict(name="attention", fn="identity", data=data, build=build, run=run,
                primary="nrmse")


def build_ops(names: list[str], acts: list[str]) -> list[dict]:
    out = []
    for n in names:
        if n == "activation":
            out.extend(make_activation(a) for a in acts)
        elif n == "fp_multiply":
            out.append(make_fp_multiply())
        elif n == "softmax":
            out.append(make_softmax())
        elif n == "layernorm":
            out.append(make_layernorm())
        elif n == "attention":
            out.append(make_attention())
        else:
            raise ValueError(f"unknown op {n!r}")
    return out


# --------------------------------------------------------------------------
# One (op, backend, budget, T) cell
# --------------------------------------------------------------------------

def run_cell(op, backend, b, t, cfg, d) -> dict:
    # NOT under no_grad: the build *fits* the primitives and needs the graph.
    # Only the metered forward pass is inference.
    t0 = time.time()
    prims = op["build"](backend, b, t, cfg, d)
    fit_s = time.time() - t0

    with torch.no_grad(), SpikeMeter() as meter:
        meter.attach_all(prims)
        out = op["run"](prims, d)

    n_out = d["n_out"]
    cost = op_memory(prims)
    per_prim = meter.per_element(n_out)
    rec = dict(op=op["name"], backend=backend, budget=b, T=t,
               spikes=per_prim["_total"],
               spikes_by_prim={k: v for k, v in per_prim.items()
                               if k != "_total"},
               mult_by_prim=meter.invocations(n_out),
               calls=dict(meter.calls),
               fit_seconds=fit_s, **cost, **allocation(prims))
    rec.update(errors(out, d["ref"]))
    return rec


def allocation(prims: dict) -> dict:
    """What the budget rule actually gave each bank, per routed primitive.

    Recorded rather than assumed: the whole claim of the ``pasn_rule`` arm is
    that ``(N_j, T_j)`` *varies* across banks, and a table showing every bank at
    the same ``(N, T)`` would mean the rule had degenerated to the uniform arm
    and the two rows are not measuring different things. ``t_span`` /``n_span``
    are the min-max across reachable banks -- 1 means no allocation happened.
    """
    from mbe.mbe_pasn import MBEPASNNeuron

    tables, ns, ts = {}, [], []
    for name, p in prims.items():
        if not isinstance(p, MBEPASNNeuron):
            continue
        tab = p.budget_table()
        tables[name] = [list(r) for r in tab]
        ns += [r[2] for r in tab]
        ts += [r[3] for r in tab]
    if not tables:
        return dict(budget_tables={}, n_span=1.0, t_span=1.0,
                    n_mean=None, t_mean=None)
    return dict(budget_tables=tables,
                n_span=max(ns) / max(min(ns), 1), t_span=max(ts) / max(min(ts), 1),
                n_mean=sum(ns) / len(ns), t_mean=sum(ts) / len(ts))


# --------------------------------------------------------------------------
# Pareto front + iso-accuracy read-off
# --------------------------------------------------------------------------

def front(points: list[dict], cost_key: str, err_key: str) -> list[dict]:
    """Non-dominated points on (cost, error), both minimised, cheapest first.

    Ties on cost keep the more accurate point; a point is dropped when some other
    point is at least as cheap **and** at least as accurate.
    """
    pts = sorted(points, key=lambda p: (p[cost_key], p[err_key]))
    out: list[dict] = []
    best = float("inf")
    for p in pts:
        if p[err_key] < best - 1e-15:
            out.append(p)
            best = p[err_key]
    return out


def iso_accuracy(ref_pts: list[dict], cand_pts: list[dict], err_key: str,
                 keys=("spikes", "bytes", "params")) -> list[dict]:
    """For each reference point, the cheapest candidate at least as accurate.

    "Cheapest" is by spikes, which is the energy axis; ``bytes`` and ``params``
    are then reported *for that same build* rather than minimised separately, so
    every row is one realisable neuron and not a mix of three. Reference points
    no candidate can match are kept with ``matched=False`` -- dropping them would
    quietly restrict the comparison to the range where the candidate wins.
    """
    rows = []
    for r in ref_pts:
        ok = [c for c in cand_pts if c[err_key] <= r[err_key]]
        if not ok:
            rows.append(dict(ref_err=r[err_key], ref_budget=r["budget"],
                             ref_T=r["T"], matched=False))
            continue
        c = min(ok, key=lambda p: p["spikes"])
        row = dict(ref_err=r[err_key], ref_budget=r["budget"], ref_T=r["T"],
                   cand_err=c[err_key], cand_budget=c["budget"], cand_T=c["T"],
                   matched=True)
        for k in keys:
            row[f"ref_{k}"] = r[k]
            row[f"cand_{k}"] = c[k]
            row[f"{k}_ratio"] = r[k] / c[k] if c[k] else float("inf")
        rows.append(row)
    return rows


def summarise(records: list[dict]) -> dict:
    """Per-op fronts and the MBE-vs-PASN iso-accuracy tables."""
    by_op: dict[str, list[dict]] = {}
    for r in records:
        by_op.setdefault(r["op"], []).append(r)

    out = {}
    for op_name, pts in by_op.items():
        primary = pts[0]["primary"]
        arms = {}
        for b in sorted({p["backend"] for p in pts}):
            sel = [p for p in pts if p["backend"] == b]
            arms[b] = dict(
                spike_front=front(sel, "spikes", primary),
                byte_front=front(sel, "bytes", primary),
                best_err=min(p[primary] for p in sel),
                min_spikes=min(p["spikes"] for p in sel),
            )
        iso = {}
        base = [p for p in pts if p["backend"] == "mbe"]
        base_front = front(base, "spikes", primary)
        for b in arms:
            if b == "mbe":
                continue
            cand = [p for p in pts if p["backend"] == b]
            iso[b] = iso_accuracy(base_front, cand, primary)

        # What each half of the per-bank allocation buys, measured the same way:
        # uniform banks -> +N_j -> +T_j. Read against the MBE-vs-PASN table, this
        # separates "routing helps" from "allocating across the routes helps".
        decomp = {}
        for a, c in (("pasn", "pasn_rule_T"), ("pasn_rule_T", "pasn_rule"),
                     ("pasn", "pasn_rule")):
            if a in arms and c in arms:
                decomp[f"{a}->{c}"] = iso_accuracy(
                    arms[a]["spike_front"],
                    [p for p in pts if p["backend"] == c], primary)

        out[op_name] = dict(primary=primary, arms=arms, iso_accuracy=iso,
                            alloc_decomp=decomp, n_points=len(pts))
    return out


# --------------------------------------------------------------------------

def tlabel(t) -> str:
    """``T`` for a fixed-T arm, ``rule`` when the budget rule solved it per bank."""
    return "rule" if t is None else str(t)


def _ratio_line(tag: str, rows: list, width: int = 22) -> str | None:
    ok = [r for r in rows if r["matched"]]
    if not ok:
        return f"  {tag:{width}s} no reference point matched"
    med = lambda v: sorted(v)[len(v) // 2]      # noqa: E731
    sr = sorted(r["spikes_ratio"] for r in ok)
    br = sorted(r["bytes_ratio"] for r in ok)
    return (f"  {tag:{width}s} ({len(ok)}/{len(rows)}) "
            f"spikes {med(sr):.2f}x [{sr[0]:.2f}, {sr[-1]:.2f}], "
            f"bytes {med(br):.2f}x [{br[0]:.2f}, {br[-1]:.2f}]")


def _print_op(name: str, s: dict) -> None:
    primary = s["primary"]
    print(f"\n== {name}  (primary={primary}) ==")
    for b, a in s["arms"].items():
        pts = a["spike_front"]
        print(f"  {b:12s} spike front ({len(pts)} pts): " +
              "  ".join(f"[b={p['budget']},T={tlabel(p['T'])}] "
                        f"{p[primary]:.2e}@{p['spikes']:.1f}sp/{p['bytes']}B"
                        for p in pts[:5]))
    # ratios are reference / candidate, so >1 means the candidate is cheaper at
    # the same accuracy and <1 means it costs more
    print("  -- at matched accuracy, ref/cand (>1 = cand cheaper) --")
    for b, rows in s["iso_accuracy"].items():
        line = _ratio_line(f"MBE / {b}", rows)
        if line:
            print(line)
    for tag, rows in s.get("alloc_decomp", {}).items():
        line = _ratio_line(f"  {tag}", rows)
        if line:
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="+",
                    default=["activation", "fp_multiply", "softmax",
                             "layernorm", "attention"])
    ap.add_argument("--acts", nargs="+", default=["gelu", "silu", "tanh"])
    ap.add_argument("--backends", nargs="+",
                    default=["mbe", "pasn", "pasn_rule_T", "pasn_rule"])
    ap.add_argument("--mbe-N", nargs="+", type=int, default=[1, 2, 4, 6, 8],
                    help="global basis counts for the MBE arm")
    ap.add_argument("--pasn-N", nargs="+", type=int, default=[1, 2, 3, 4],
                    help="per-bank basis counts for the uniform PASN arm")
    ap.add_argument("--rule-targets", nargs="+", type=float,
                    default=[3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3],
                    help="relative-error targets: the budget rule's own knob, "
                         "and the only knob the fully dynamic arm has")
    ap.add_argument("--Ts", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--rule-Ts", nargs="+", type=int, default=[16],
                    help="T values to pin the pasn_rule_T arm at. Defaults to "
                         "the paper's global T=16, which is the comparison that "
                         "arm exists to make; the fully dynamic pasn_rule arm "
                         "ignores this and solves T_j per bank.")
    ap.add_argument("--binades", type=int, default=6,
                    help="binades below the max magnitude the router resolves")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--m-eval", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid, for checking the wiring")
    ap.add_argument("--json", default="results/op_pareto.json")
    cfg = ap.parse_args()

    if cfg.smoke:
        cfg.mbe_N, cfg.pasn_N, cfg.Ts = [2, 4], [2], [8]
        cfg.rule_Ts, cfg.rule_targets = [8], [1e-1, 1e-2]
        cfg.epochs, cfg.m_eval = 40, 512

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(os.path.dirname(cfg.json) or ".", exist_ok=True)

    budgets = dict(mbe=cfg.mbe_N, pasn=cfg.pasn_N,
                   pasn_rule_T=cfg.rule_targets, pasn_rule=cfg.rule_targets)

    def cells_for(backend):
        """(budget, T) pairs to build for an arm.

        The fully dynamic arm solves ``T_j`` per bank, so the outer ``T`` loop
        has nothing to vary -- sweeping it would build the same neuron three
        times and plot three identical points as if they were evidence.
        """
        if backend == "pasn_rule":
            return [(b, None) for b in budgets[backend]]
        ts = cfg.rule_Ts if backend == "pasn_rule_T" else cfg.Ts
        return [(b, t) for b in budgets[backend] for t in ts]
    records: list[dict] = []
    t_start = time.time()

    def save():
        """Write what exists so far. Called after **every cell**, not just at the
        end: this sweep runs for hours, a single op can be 30 min of it, and a
        lost run is a rerun rather than a gap. Serialising a few hundred records
        is milliseconds against a build measured in seconds."""
        res = dict(meta=dict(vars(cfg), elapsed_s=time.time() - t_start,
                             torch=torch.__version__),
                   paper=dict(table_x=PAPER_TABLE_X,
                              table_xi=paper_table_xi_spikes()),
                   records=records, summary=summarise(records))
        with open(cfg.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        return res

    for op in build_ops(cfg.ops, cfg.acts):
        d = op["data"](cfg)
        print(f"\n[{op['name']}] n_out={d['n_out']}", flush=True)
        for backend in cfg.backends:
            for b, t in cells_for(backend):
                tag = f"  {backend:12s} b={b:<6g} T={tlabel(t):>4s}"
                try:
                    rec = run_cell(op, backend, b, t, cfg, d)
                except Exception as exc:                # noqa: BLE001
                    # one dead cell must not lose the sweep: a build can fail
                    # legitimately (a rule target no budget reaches), and the
                    # front is still well defined without it
                    print(f"{tag}  FAILED: {type(exc).__name__}: {exc}",
                          flush=True)
                    continue
                rec["primary"] = op["primary"]
                records.append(rec)
                # n_mean/t_mean expose what the rule allocated: if they equal the
                # nominal budget everywhere, the rule degenerated to the uniform
                # arm and the two rows are not measuring different things
                alloc = ("" if rec["n_mean"] is None else
                         f"  <N>={rec['n_mean']:.2f}(x{rec['n_span']:.0f}) "
                         f"<T>={rec['t_mean']:.1f}(x{rec['t_span']:.0f})")
                print(f"{tag}  {op['primary']}={rec[op['primary']]:.3e}  "
                      f"spikes={rec['spikes']:8.2f}  "
                      f"bytes={rec['bytes']:6d}  params={rec['params']:5d}"
                      f"{alloc}  ({rec['fit_seconds']:.0f}s)", flush=True)
                save()

    summary = save()["summary"]

    print("\n" + "=" * 72)
    for name, s in summary.items():
        _print_op(name, s)
    print(f"\nwrote {cfg.json}  ({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
