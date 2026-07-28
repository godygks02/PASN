"""ANN->SNN conversion framework (Phase 4, Algorithm 1).

Turns a *module-structured* ANN into its spike-driven MBE form without touching
any weights (training-free). The pipeline is the paper's Algorithm 1:

  1. **Calibrate** -- run a few minibatches through the ANN and record the input
     range actually seen by every nonlinear op (:class:`CalibrationRecorder`).
  2. **Build**     -- fit the MBE primitives on those measured ranges.
  3. **Replace**   -- swap each nonlinearity for its spiking module in place;
     Linear/weight matmuls are left untouched (they are native accumulation).
  4. **Evaluate**  -- the caller runs the converted model through its metric.

Conversion points are marked by the op modules below (:class:`Activation`,
:class:`Softmax`, :class:`MatMulAA`) plus standard ``nn.LayerNorm``. A real HF
model buries softmax / QK^T in functional calls inside ``forward``; wiring those
is Phase 5. Here the model is expressed with these op modules so the calibrate ->
build -> replace loop can be verified end-to-end on CPU (``verify_phase4.py``).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from . import functions
from . import spiking_ops as so
from .mbe_pasn import build_mbe_pasn
from .mbe_pasn_s import build_mbe_pasn_s
from .pasn import build_pasn


# --------------------------------------------------------------------------
# Conversion-point op modules (used by the ANN so swaps are explicit)
# --------------------------------------------------------------------------

class Activation(nn.Module):
    """Element-wise activation (``gelu`` / ``silu`` / ``tanh``) as a swap point."""

    def __init__(self, kind: str = "gelu"):
        super().__init__()
        self.kind = kind
        self.fn = functions.REGISTRY[kind][0]

    def forward(self, x):
        return self.fn(x)


class Softmax(nn.Module):
    """Softmax as a swap point (functional softmax has no module to hook)."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, dim=self.dim)


class MatMulAA(nn.Module):
    """activation*activation matmul ``A @ B`` (e.g. QK^T, attn*V) as a swap point.

    Distinct from ``nn.Linear`` (activation*weight), which is native accumulation
    and must NOT be spiked -- only both-operands-are-activations products need the
    spike-driven FP multiply.
    """

    def forward(self, A, B):
        return A @ B


# --------------------------------------------------------------------------
# Spiking replacement modules (match the op-module forward signatures)
# --------------------------------------------------------------------------

class _SpikingActModule(nn.Module):
    def __init__(self, act: so.SpikingActivation):
        super().__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)


class _SpikingSoftmaxModule(nn.Module):
    def __init__(self, sm: so.SpikingSoftmax, dim: int):
        super().__init__()
        self.sm = sm
        self.dim = dim

    def forward(self, x):
        return self.sm(x, dim=self.dim)


class _SpikingLayerNormModule(nn.Module):
    def __init__(self, ln: so.SpikingLayerNorm, weight, bias):
        super().__init__()
        self.ln = ln
        self.register_buffer("weight", weight if weight is not None else None)
        self.register_buffer("bias", bias if bias is not None else None)

    def forward(self, x):
        return self.ln(x, weight=self.weight, bias=self.bias)


class _SpikingMatMulModule(nn.Module):
    def __init__(self, idn, idn2, spike_mult: bool):
        super().__init__()
        self.idn, self.idn2 = idn, idn2
        self.spike_mult = spike_mult

    def forward(self, A, B):
        if not self.spike_mult:
            return A @ B
        return so.spiking_matmul(self.idn, A, B, signed=True, idn2=self.idn2)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def classify(mod: nn.Module) -> str | None:
    """Conversion kind of ``mod``, or ``None`` if it is not a conversion point."""
    if isinstance(mod, _SPIKING_TYPES):
        return None                       # already converted
    if isinstance(mod, nn.LayerNorm):
        return "layernorm"
    if isinstance(mod, Activation):
        return "activation"
    if isinstance(mod, Softmax):
        return "softmax"
    if isinstance(mod, MatMulAA):
        return "matmul"
    return None


_SPIKING_TYPES = (_SpikingActModule, _SpikingSoftmaxModule,
                  _SpikingLayerNormModule, _SpikingMatMulModule)


@dataclass
class _Range:
    """Running per-operand statistics from calibration."""

    lo: float = float("inf")
    hi: float = float("-inf")
    absmax: float = 0.0
    sample: torch.Tensor | None = None
    # Size of the last axis. The sample is flattened, so this is what lets a
    # reduction op (softmax over ``dim=-1``) reshape it back into rows and replay its
    # own decomposition to recover each primitive's real argument distribution.
    width: int = 0

    def update(self, t: torch.Tensor, keep: int):
        t = t.detach()
        self.lo = min(self.lo, float(t.min()))
        self.hi = max(self.hi, float(t.max()))
        self.absmax = max(self.absmax, float(t.abs().max()))
        self.width = max(self.width, int(t.shape[-1]) if t.dim() else 0)
        have = 0 if self.sample is None else self.sample.numel()
        if have < keep:
            # Calibration may run on CUDA, but primitive fitting is intentionally
            # performed on CPU. Copy only the retained prefix so hooks do not keep
            # GPU tensors alive or later mix CUDA samples with CPU-fitted neurons.
            take = t.reshape(-1)[:keep - have].to(device="cpu").clone()
            self.sample = take if self.sample is None else torch.cat(
                [self.sample, take]
            )


class CalibrationRecorder:
    """Records the input range of every conversion point during forward passes.

    Register with a model, run calibration minibatches through it, then read
    ``ranges[name]`` -- a list of :class:`_Range` (one entry per operand; matmul
    has two). Use as a context manager to auto-remove the hooks.
    """

    def __init__(self, model: nn.Module, keep: int = 20_000):
        self.model = model
        self.keep = keep
        self.kinds: dict[str, str] = {}
        self.ranges: dict[str, list[_Range]] = {}
        self._handles: list = []
        for name, mod in model.named_modules():
            kind = classify(mod)
            if kind is not None:
                self.kinds[name] = kind
                self._handles.append(
                    mod.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def pre_hook(module, args):
            slots = self.ranges.setdefault(name, [])
            for i, a in enumerate(args):
                if not torch.is_tensor(a):
                    continue
                if len(slots) <= i:
                    slots.append(_Range())
                slots[i].update(a, self.keep)
        return pre_hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def calibrate(model: nn.Module, batches, keep: int = 20_000) -> CalibrationRecorder:
    """Run ``batches`` (iterable of input tensors) through ``model`` and record
    every conversion point's input range."""
    rec = CalibrationRecorder(model, keep=keep)
    was_training = model.training
    model.eval()
    device = _module_device(model)
    for xb in batches:
        xb = _batch_to_device(xb, device)
        if isinstance(xb, dict):
            model(**xb)
        elif isinstance(xb, (tuple, list)):
            model(*xb)
        else:
            model(xb)
    model.train(was_training)
    return rec


def _module_device(model: nn.Module) -> torch.device:
    """Execution device of a module, falling back to CPU for parameterless models."""
    first = next(model.parameters(), None)
    if first is not None:
        return first.device
    first = next(model.buffers(), None)
    return first.device if first is not None else torch.device("cpu")


def _batch_to_device(batch, device: torch.device):
    """Move tensor leaves in a calibration batch to the model's device."""
    if torch.is_tensor(batch):
        return batch.to(device=device)
    if isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_batch_to_device(v, device) for v in batch)
    if isinstance(batch, list):
        return [_batch_to_device(v, device) for v in batch]
    return batch


def _module_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Device and floating dtype used by newly inserted conversion modules."""
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return _module_device(model), torch.float32


def _shared_fit_slots(model: nn.Module, recorder: CalibrationRecorder) -> dict:
    """Build shared calibration inputs for repeated Transformer blocks."""
    overrides: dict[str, list[_Range]] = {}
    act_groups: dict[str, list[tuple[str, _Range]]] = {}
    ln_groups: dict[tuple, list[tuple[str, _Range]]] = {}

    for name, kind in recorder.kinds.items():
        slots = recorder.ranges.get(name)
        if not slots:
            continue
        mod = model.get_submodule(name)
        if kind == "activation":
            act_groups.setdefault(mod.kind, []).append((name, slots[0]))
        elif kind == "layernorm":
            key = (tuple(mod.normalized_shape), float(mod.eps))
            ln_groups.setdefault(key, []).append((name, slots[0]))

    # One activation fit on the union of ranges observed across all blocks. The
    # endpoints are appended so ``sample.min()/.max()`` still give that union, but the
    # retained rows are kept: a routed backend selects its operating point from the
    # *distribution* of activations, and reducing the sample to two endpoints (as this
    # did) hands it a two-point histogram at the one site that matters most.
    for group in act_groups.values():
        lo = min(slot.lo for _, slot in group)
        hi = max(slot.hi for _, slot in group)
        pooled = [slot.sample for _, slot in group if slot.sample is not None]
        pooled.append(torch.tensor([lo, hi], dtype=torch.float32))
        shared = _Range(
            lo=lo,
            hi=hi,
            absmax=max(abs(lo), abs(hi)),
            sample=torch.cat(pooled),
        )
        for name, _ in group:
            overrides[name] = [shared]

    # One LayerNorm primitive range from the retained rows of all matching LNs.
    for (shape, _eps), group in ln_groups.items():
        width = shape[0]
        samples = []
        for _, slot in group:
            usable = (slot.sample.numel() // width) * width
            if usable:
                samples.append(slot.sample[:usable])
        if not samples:
            continue
        shared = _Range(
            lo=min(slot.lo for _, slot in group),
            hi=max(slot.hi for _, slot in group),
            absmax=max(slot.absmax for _, slot in group),
            sample=torch.cat(samples),
            width=width,
        )
        for name, _ in group:
            overrides[name] = [shared]
    return overrides


# --------------------------------------------------------------------------
# Build + replace
# --------------------------------------------------------------------------

@dataclass
class ConvertConfig:
    n_basis_act: int = 6
    n_basis_ln: int = 8
    n_basis_sm: int = 8
    n_basis_mm: int = 8
    n_steps: int = 16
    epochs: int = 300
    seed: int = 0
    spike_mult: bool = True          # spike-driven FP mult inside SM/LN/MatMul
    margin: float = 0.05             # range padding for identity/activation fits
    # Neuron backend used consistently by activation and LayerNorm primitives
    # (and activation*activation matmul where a model exposes MatMulAA markers).
    # "mbe"        = one global MBE neuron;
    # "mbe_pasn"   = FP-prefix binade banks of MBE neurons;
    # "mbe_pasn_s" = one *shared* MBE basis set, FP-prefix selects the readout;
    # "pasn"       = FP-prefix binade banks + successive-approximation (SAR) code.
    # GPT-2 Stage 1 converts GELU + LayerNorm only; functional attention/Softmax
    # stays exact.
    backend: str = "mbe"
    # Router budget -- shared by every prefix-routed backend *on purpose*, so a
    # backend-vs-backend run differs only in the encoder, not in the routing.
    pasn_e_min: int = -3
    # Router depth for the *identity* primitives only (the spike-driven FP multiply's
    # operand reconstructions inside LayerNorm / Softmax / activation*activation
    # matmul). Their operands span many decades -- e^x, 1/S, x-mu, 1/std -- so with a
    # single shared e_min everything below 2^e_min collapses into one near-zero bank,
    # and those primitives are the largest spike consumers once Softmax is routed.
    # Measured to saturate at -6 on the toy Transformer: going -3 -> -6 improves the
    # forward error on all three routed backends (1.72x for pasn, 1.19x for mbe_pasn)
    # while pasn's spikes drop 4.5%, and -10 / -14 buy nothing on any backend while
    # growing identity storage linearly. All backends share the value, so they stay
    # comparable; None falls back to pasn_e_min.
    pasn_id_e_min: int | None = -6
    pasn_n_local: int = 2            # mbe_pasn: MBE bases per binade bank (budget="fixed")
    # mbe_pasn budget policy. "rule" derives each bank's (N, T) from its own dynamic
    # range and pasn_target_mse (연구일지 실험 1): flat tails collapse to (1, 2) and
    # only banks that need the resolution get a long spike train. "fixed" keeps the
    # uniform pasn_n_local / n_steps everywhere. T is a *linear* factor in the spike
    # count and was a global constant until this was added.
    pasn_budget: str = "rule"
    pasn_target_mse: float = 1e-5
    # Split the near-zero region by polarity so no bank straddles zero (실험 2).
    # A single near-zero bank has a hard error floor on targets whose bend sits at 0
    # (ReLU 1362x, GELU 61x), and it is the most-visited bank of all.
    pasn_near0: str = "signed"
    # Identity primitives only. The identity is positively homogeneous, so every
    # magnitude bank's target is the *same* unit-interval function up to the scale
    # the router already read: one fitted prototype serves them all (연구일지 실험 4).
    # And what a spike-driven multiply needs is *relative* accuracy -- an absolute
    # MSE budget starves exactly the small operands, which is why a global MBE_Id
    # lands at 15-49% relative error on a real product (실험 10).
    pasn_id_tied: bool = True
    pasn_id_target_rel: float = 1e-2
    # Decoder order for the activation. Not used for the identity: the FP-multiply
    # factorisation needs the output to stay a spike sum (실험 6).
    pasn_readout_order: int = 2
    pasn_T: int = 6                  # pasn: SAR bits (spike budget) per bank
    pasn_order: int = 2              # pasn: per-bank readout polynomial order
    # mbe_pasn_s: candidate bases (int or list) and an optional cap on mean spikes
    # per input. The budget is measured under the *calibration* distribution, since
    # the calibration sample is handed to the builder as range weights.
    pasn_s_n_shared: int | list = 4
    pasn_s_spike_budget: float | None = None
    pasn_s_restarts: int = 3
    # None -> fit on the ANN's execution device (CUDA on vast.ai, CPU locally).
    fit_device: str | None = None
    verbose_fits: bool = False
    share_fits: bool = True


def _set_submodule(root: nn.Module, name: str, new: nn.Module):
    parent_name, _, last = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if last.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def _erange(lo, hi, e_min):
    """(e_min, e_max) for a PASN router covering ``[lo, hi]``."""
    maxmag = max(abs(lo), abs(hi), 2.0 ** (e_min + 1))
    return e_min, max(int(math.ceil(math.log2(maxmag))), e_min + 1)


# Backends whose neurons are prefix-routed banks. All share the router
# (``pasn_e_min``) and differ only in what a bank is: independent MBE bases
# (mbe_pasn), one shared basis set with a routed readout (mbe_pasn_s), or a fixed
# SAR spike code with a routed readout (pasn).
_ROUTED_BACKENDS = ("mbe_pasn", "mbe_pasn_s", "pasn")


def _routed_primitive(name, domain, e_min, e_max, cfg: ConvertConfig,
                      fit_device, n_near0=None, sample=None, **pasn_kw):
    """Build one prefix-routed neuron for target ``name`` on ``domain``.

    Dispatches on ``cfg.backend``; the router bounds are passed through identically
    so the routed backends are compared at the same routing. ``sample`` is the real
    calibration input for this primitive -- ``mbe_pasn_s`` uses it to weight its
    candidate selection by measured per-range visit probabilities, which is what
    makes its spike budget the spike count the network actually spends.
    """
    if cfg.backend == "pasn":
        # The SAR code needs no per-bank basis count: the budget is (T, order).
        return build_pasn(
            name, domain, e_min=e_min, e_max=e_max,
            T=cfg.pasn_T, order=cfg.pasn_order, seed=cfg.seed,
            device=fit_device, verbose=cfg.verbose_fits,
        )
    if cfg.backend == "mbe_pasn_s":
        return build_mbe_pasn_s(
            name, domain, e_min=e_min, e_max=e_max,
            n_shared=cfg.pasn_s_n_shared, n_steps=cfg.n_steps,
            epochs=cfg.epochs, seed=cfg.seed, restarts=cfg.pasn_s_restarts,
            spike_budget=cfg.pasn_s_spike_budget, sample=sample,
            device=fit_device, verbose=cfg.verbose_fits,
        )
    kw = dict(readout_order=cfg.pasn_readout_order)
    kw.update(pasn_kw)
    return build_mbe_pasn(
        name, domain, e_min=e_min, e_max=e_max,
        n_local=cfg.pasn_n_local,
        n_near0=n_near0 or cfg.pasn_n_local,
        n_steps=cfg.n_steps, epochs=cfg.epochs, seed=cfg.seed,
        budget=cfg.pasn_budget, target_mse=cfg.pasn_target_mse,
        near0=cfg.pasn_near0,
        device=fit_device, verbose=cfg.verbose_fits, **kw,
    )


def _routed_softmax(mod, slot, cfg: ConvertConfig, fit_device):
    """Routed Softmax primitives, each fitted on its argument's real distribution.

    The three primitives do not see the logits -- they see what the op's own
    decomposition feeds them, so this replays that decomposition on the calibration
    sample: ``frac = frac(x_m log2 e)`` for ``MBE_exp``, the ``frexp`` mantissa of the
    row sum for ``MBE_inv``, and the two product operands for the identity.

    Note ``MBE_inv`` gains nothing from routing **by construction**: its argument is
    already an IEEE mantissa in ``[0.5, 1)``, a single binade, so the exponent router
    has one reachable range and reduces to a plain global neuron. The exponent split
    inside the op has done that work already. Only a mantissa-prefix router could
    subdivide it further.

    The identity's operands (``e^x`` and ``1/S``, both in ``(0, 1]``) span many decades,
    so everything below ``2^pasn_e_min`` collapses into the single near-zero bank; a
    deeper ``pasn_e_min`` is what would buy accuracy there. It is left at the shared
    value so the routed backends stay comparable at one router setting.
    """
    logits, w = slot.sample, slot.width
    frac = mant = operands = None
    if logits is not None and w > 1 and logits.numel() >= w:
        rows = logits[: (logits.numel() // w) * w].reshape(-1, w)
        z = (rows - rows.max(dim=-1, keepdim=True).values) * math.log2(math.e)
        z_floor = torch.floor(z)
        frac = (z - z_floor).reshape(-1)
        exp_x = torch.ldexp(torch.exp2(z - z_floor), z_floor.to(torch.int64))
        m_, e_ = torch.frexp(exp_x.sum(dim=-1, keepdim=True))
        mant = m_.reshape(-1)
        inv_S = torch.ldexp(1.0 / m_, -e_).expand_as(exp_x)
        operands = torch.cat([exp_x.reshape(-1), inv_S.reshape(-1)])

    e_lo, e_hi = _erange(0.0, 1.0, cfg.pasn_e_min)
    exp_n = _routed_primitive("exp2", (0.0, 1.0), e_lo, e_hi, cfg, fit_device,
                              sample=frac)
    inv_n = _routed_primitive("inv", (0.5, 1.0), -1, 0, cfg, fit_device,
                              sample=mant)
    idn = _routed_identity(1.0, cfg, fit_device, sample=operands)
    return so.SpikingSoftmax(exp_n, inv_n, idn, spike_mult=cfg.spike_mult)


def _routed_identity(hi, cfg: ConvertConfig, fit_device, sample=None):
    """Prefix-routed identity over ``[0, hi]`` for the spike-driven multiply.

    Uses ``pasn_id_e_min`` when set: identity operands span many decades, so these
    primitives benefit from a deeper router than the activation needs.

    Note the routed backends reconstruct the identity with a DC term in the
    readout, so ``reconstruct`` is not the strictly bias-free spike sum that
    :func:`spiking_ops.calibrate_identity` builds for the plain MBE backend.
    """
    id_e_min = (cfg.pasn_e_min if cfg.pasn_id_e_min is None
                else cfg.pasn_id_e_min)
    e_min, e_max = _erange(0.0, hi, id_e_min)
    return _routed_primitive(
        "identity", (0.0, hi), e_min, e_max, cfg, fit_device, sample=sample,
        # Homogeneous target + a product downstream: tie the banks and budget on
        # *relative* error. Order 1 -- the multiply factorises a spike sum only.
        tied=cfg.pasn_id_tied, target="relative",
        target_rel=cfg.pasn_id_target_rel, readout_order=1,
    )


def _build_replacement(kind, mod, slots, cfg: ConvertConfig, fit_device):
    if kind == "activation":
        s = slots[0].sample
        lo, hi = float(s.min()), float(s.max())
        span = max(hi - lo, 1e-6)
        lo, hi = lo - cfg.margin * span, hi + cfg.margin * span
        if cfg.backend in _ROUTED_BACKENDS:
            e_min, e_max = _erange(lo, hi, cfg.pasn_e_min)
            # +2 bases in the near-zero bank: it carries the GELU/SiLU bend.
            neuron = _routed_primitive(mod.kind, (lo, hi), e_min, e_max, cfg,
                                       fit_device,
                                       n_near0=cfg.pasn_n_local + 2, sample=s)
            return _SpikingActModule(so.SpikingActivation(neuron))
        act = so.build_activation(mod.kind, slots[0].sample,
                                  n_basis=cfg.n_basis_act, n_steps=cfg.n_steps,
                                  seed=cfg.seed, margin=cfg.margin,
                                  epochs=cfg.epochs, device=fit_device)
        return _SpikingActModule(act)

    if kind == "softmax":
        if cfg.backend in _ROUTED_BACKENDS:
            sm = _routed_softmax(mod, slots[0], cfg, fit_device)
        else:
            sm = so.build_softmax(
                slots[0].sample, dim=mod.dim,
                n_basis=cfg.n_basis_sm, n_steps=cfg.n_steps,
                epochs=cfg.epochs, seed=cfg.seed,
                spike_mult=cfg.spike_mult, device=fit_device
            )
        return _SpikingSoftmaxModule(sm, mod.dim)

    if kind == "layernorm":
        D = mod.normalized_shape[0]
        flat = slots[0].sample
        flat = flat[: (flat.numel() // D) * D]        # trim to a multiple of D
        sample = flat.reshape(-1, D)
        if cfg.backend in _ROUTED_BACKENDS:
            mu = sample.mean(dim=-1, keepdim=True)
            dev = sample - mu
            dev_max = float(dev.abs().max()) * 1.1 + 1e-3
            var = (dev * dev).mean(dim=-1, keepdim=True) + mod.eps
            istd_max = float((1.0 / var.sqrt()).max()) * 1.1 + 1e-3
            # The rsqrt argument is the (parity-fixed) mantissa of the variance, not
            # a tensor we hold here, so it gets no measured weights.
            rsqrt = _routed_primitive(
                "invsqrt", (0.5, 2.0), -2, 1, cfg, fit_device, n_near0=4
            )
            id_dev = _routed_identity(dev_max, cfg, fit_device,
                                      sample=dev.abs().reshape(-1))
            id_istd = _routed_identity(istd_max, cfg, fit_device,
                                       sample=(1.0 / var.sqrt()).reshape(-1))
            ln = so.SpikingLayerNorm(
                rsqrt, id_dev, id_istd, eps=mod.eps,
                spike_mult=cfg.spike_mult
            )
        else:
            ln = so.build_layernorm(
                sample, eps=mod.eps,
                n_basis=cfg.n_basis_ln, n_steps=cfg.n_steps,
                epochs=cfg.epochs, seed=cfg.seed,
                spike_mult=cfg.spike_mult, device=fit_device
            )
        return _SpikingLayerNormModule(ln, mod.weight, mod.bias)

    if kind == "matmul":
        hi_a = slots[0].absmax * (1.0 + cfg.margin) + 1e-6
        hi_b = slots[1].absmax * (1.0 + cfg.margin) + 1e-6
        if cfg.backend in _ROUTED_BACKENDS:
            # Operands are split by polarity before reconstruction, so the identity
            # sees magnitudes.
            idn = _routed_identity(hi_a, cfg, fit_device,
                                   sample=slots[0].sample.abs())
            idn2 = _routed_identity(hi_b, cfg, fit_device,
                                    sample=slots[1].sample.abs())
        else:
            idn = so.calibrate_identity(0.0, hi_a, n_basis=cfg.n_basis_mm,
                                        n_steps=cfg.n_steps, epochs=cfg.epochs,
                                        seed=cfg.seed, device=fit_device)
            idn2 = so.calibrate_identity(0.0, hi_b, n_basis=cfg.n_basis_mm,
                                         n_steps=cfg.n_steps, epochs=cfg.epochs,
                                         seed=cfg.seed, device=fit_device)
        return _SpikingMatMulModule(idn, idn2, cfg.spike_mult)

    raise ValueError(kind)


def convert(model: nn.Module, recorder: CalibrationRecorder,
            cfg: ConvertConfig | None = None, only: set[str] | None = None,
            verbose: bool = False) -> nn.Module:
    """Replace calibrated conversion points in ``model`` with spiking modules.

    Mutates and returns ``model``. ``only`` restricts conversion to a subset of
    kinds (``{"layernorm"}`` etc.) so each op's contribution can be isolated.
    """
    cfg = cfg or ConvertConfig()
    device, dtype = _module_device_dtype(model)
    fit_device = torch.device(cfg.fit_device) if cfg.fit_device else device
    if verbose or cfg.verbose_fits:
        print(f"  fitting conversion primitives on {fit_device}", flush=True)
    shared_slots = _shared_fit_slots(model, recorder) if cfg.share_fits else {}
    activation_prototypes: dict[str, nn.Module] = {}
    layernorm_prototypes: dict[tuple, so.SpikingLayerNorm] = {}
    for name, kind in list(recorder.kinds.items()):
        if only is not None and kind not in only:
            continue
        slots = recorder.ranges.get(name)
        if not slots:
            continue                      # never exercised during calibration
        mod = model.get_submodule(name)
        fit_slots = shared_slots.get(name, slots)
        prototype_key = mod.kind if kind == "activation" else None
        if prototype_key in activation_prototypes:
            if verbose or cfg.verbose_fits:
                print(
                    f"  reuse   {name:32s} (activation: {prototype_key})",
                    flush=True,
                )
            _set_submodule(
                model, name, copy.deepcopy(activation_prototypes[prototype_key])
            )
            continue
        if kind == "layernorm" and cfg.share_fits:
            ln_key = (tuple(mod.normalized_shape), float(mod.eps), cfg.backend)
            if ln_key in layernorm_prototypes:
                if verbose or cfg.verbose_fits:
                    print(
                        f"  reuse   {name:32s} "
                        f"(layernorm: {cfg.backend})",
                        flush=True,
                    )
                new = _SpikingLayerNormModule(
                    copy.deepcopy(layernorm_prototypes[ln_key]),
                    mod.weight,
                    mod.bias,
                )
                new.to(device=device, dtype=dtype)
                _set_submodule(model, name, new)
                continue
        if verbose or cfg.verbose_fits:
            print(f"  fitting {name:32s} ({kind})", flush=True)
        new = _build_replacement(kind, mod, fit_slots, cfg, fit_device)
        # Keep insertion compatible when fitting and inference devices differ.
        new.to(device=device, dtype=dtype)
        if prototype_key is not None:
            activation_prototypes[prototype_key] = new
        if kind == "layernorm" and cfg.share_fits:
            layernorm_prototypes[
                (tuple(mod.normalized_shape), float(mod.eps), cfg.backend)
            ] = new.ln
        _set_submodule(model, name, new)
        if verbose:
            rng = ", ".join(f"[{s.lo:.3g},{s.hi:.3g}]" for s in slots)
            print(f"  convert {name:22s} {kind:10s} range={rng}")
    return model


# --------------------------------------------------------------------------
# Cost metrics (fair comparison: accuracy is not enough when N differs)
# --------------------------------------------------------------------------

def _spiking_primitives(model: nn.Module) -> list:
    """Every spiking neuron a converted model actually executes.

    ``[(name, op_kind, neuron)]``. The activation is one neuron per module, but each
    LayerNorm / Softmax / activation*activation matmul runs *several*: the
    spike-driven FP multiply reconstructs each operand through its own ``MBE_Id``.
    Counting only the activations (as :func:`activation_cost_report` does) therefore
    reports a fraction of the spikes a converted Transformer emits.

    ``spike_mult=False`` disables exactly the identity-based multiplies, so those
    primitives are excluded when it is off; ``rsqrt`` / ``exp`` / ``inv`` run either
    way. Neurons shared by object identity are listed once.
    """
    out, seen = [], set()

    def add(name, kind, neuron):
        if neuron is not None and id(neuron) not in seen:
            seen.add(id(neuron))
            out.append((name, kind, neuron))

    for name, mod in model.named_modules():
        if isinstance(mod, _SpikingActModule):
            add(f"{name}", "activation", mod.act.neuron)
        elif isinstance(mod, _SpikingLayerNormModule):
            add(f"{name}.rsqrt", "layernorm", mod.ln.rsqrt)
            if mod.ln.spike_mult:
                add(f"{name}.id_dev", "layernorm", mod.ln.id_dev)
                add(f"{name}.id_istd", "layernorm", mod.ln.id_istd)
        elif isinstance(mod, _SpikingSoftmaxModule):
            add(f"{name}.exp", "softmax", mod.sm.exp)
            add(f"{name}.inv", "softmax", mod.sm.inv)
            if mod.sm.spike_mult:
                add(f"{name}.idn", "softmax", mod.sm.idn)
        elif isinstance(mod, _SpikingMatMulModule):
            if mod.spike_mult:
                add(f"{name}.idn", "matmul", mod.idn)
                add(f"{name}.idn2", "matmul", mod.idn2)
    return out


@torch.no_grad()
def _install_cost_probe(name, neuron, tally, busy, saved):
    """Patch a neuron's entry points so one invocation is tallied exactly once.

    Both ``forward`` and ``reconstruct`` are patched, because the ops use both.
    But some neurons implement one *in terms of* the other -- ``MBEPASNNeuron``
    and ``MBEPASNSNeuron`` return ``self.forward(x)`` from ``reconstruct``, and
    once patched that resolves to the wrapped forward. Tallying at every wrapped
    entry therefore charged a routed reconstruct **twice**, while a plain
    ``MBENeuron`` (whose ``reconstruct`` goes to ``spike_train`` /
    ``intensities``, neither patched) was charged once. The bias fell entirely on
    the routed backends, and the identity is 60-70% of a converted model's
    spikes. A per-neuron depth counter fixes it: only the outermost entry counts,
    and two *sequential* calls still count twice because the depth returns to 0
    in between.
    """
    from .metrics import neuron_cost

    entry = tally[name]
    state = {"depth": 0}

    def wrap(fn):
        def inner(x, *a, **k):
            outermost = state["depth"] == 0
            state["depth"] += 1
            try:
                # ``busy`` additionally guards the cost query itself.
                if (outermost and torch.is_tensor(x) and x.numel()
                        and not busy["flag"]):
                    busy["flag"] = True
                    try:
                        entry["spikes"] += (neuron_cost(neuron, x)["spikes"]
                                            * x.numel())
                        entry["elements"] += x.numel()
                        entry["calls"] += 1
                    finally:
                        busy["flag"] = False
                return fn(x, *a, **k)
            finally:
                state["depth"] -= 1
        return inner

    # ``SignedMBENeuron`` has no ``reconstruct`` (it is only ever an activation,
    # never a multiply operand), so patch what exists and restore exactly that.
    for attr in ("forward", "reconstruct"):
        orig = getattr(neuron, attr, None)
        if orig is None:
            continue
        saved.append((neuron, attr, orig))
        setattr(neuron, attr, wrap(orig))


def _remove_cost_probes(saved):
    """Undo :func:`_install_cost_probe`, dropping the instance shadow entirely."""
    for neuron, attr, orig in saved:
        try:
            delattr(neuron, attr)
        except AttributeError:
            setattr(neuron, attr, orig)


def spiking_cost_report(model: nn.Module, batch) -> dict:
    """Total spike cost of a converted model, measured by instrumenting a forward.

    Every primitive's ``forward`` and ``reconstruct`` is wrapped for one pass through
    ``batch``, and each call contributes ``mean spikes per element x elements``. The
    invocation counts are therefore *measured*, not derived: a signed FP multiply
    reconstructs four operand halves, a LayerNorm squares and then normalises, a
    softmax reduces over one axis -- multipliers that are easy to get wrong by hand.

    Returns ``primitives`` (per neuron: kind, spikes, elements, calls,
    spikes/element), ``by_kind`` (spikes summed per op kind), ``total_spikes``, and
    ``spikes_per_input`` = total spikes divided by the number of elements in the
    model's own input tensor -- spikes per token for a language model, spikes per
    input activation for the toy Transformer. That last figure is the one comparable
    across backends; a per-activation figure is not a total.
    """
    from .metrics import neuron_cost

    prims = _spiking_primitives(model)
    if not prims:
        return dict(primitives={}, by_kind={}, total_spikes=0.0,
                    spikes_per_input=0.0, input_elements=0)

    tally = {name: dict(kind=kind, spikes=0.0, elements=0, calls=0)
             for name, kind, _ in prims}
    busy = {"flag": False}
    saved = []

    for name, _, neuron in prims:
        _install_cost_probe(name, neuron, tally, busy, saved)
    try:
        device = _module_device(model)
        b = _batch_to_device(batch, device)
        was_training = model.training
        model.eval()
        if isinstance(b, dict):
            model(**b)
            n_in = next(iter(b.values())).numel()
        elif isinstance(b, (tuple, list)):
            model(*b)
            n_in = b[0].numel()
        else:
            model(b)
            n_in = b.numel()
        model.train(was_training)
    finally:
        _remove_cost_probes(saved)

    by_kind: dict[str, float] = {}
    for name, e in tally.items():
        e["spikes_per_element"] = e["spikes"] / max(e["elements"], 1)
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0.0) + e["spikes"]
    total = sum(by_kind.values())
    return dict(primitives=tally, by_kind=by_kind, total_spikes=total,
                spikes_per_input=total / max(n_in, 1), input_elements=n_in)


def format_spiking_cost_report(rep: dict, label: str = "") -> str:
    """Printable breakdown of :func:`spiking_cost_report`."""
    tag = f" {label}" if label else ""
    if not rep.get("primitives"):
        return f"[spikes{tag}] no spiking primitives"
    lines = [f"[spikes{tag}] total={rep['total_spikes']:.3e} over "
             f"{rep['input_elements']} model-input elements  ->  "
             f"{rep['spikes_per_input']:.2f} spikes per input element"]
    total = max(rep["total_spikes"], 1e-30)
    for kind, s in sorted(rep["by_kind"].items(), key=lambda kv: -kv[1]):
        n = sum(1 for e in rep["primitives"].values() if e["kind"] == kind)
        lines.append(f"    {kind:10s} {s / total * 100:5.1f}%  "
                     f"{s:.3e} spikes over {n} primitive(s)")
    return "\n".join(lines)


@torch.no_grad()
def activation_cost_report(model: nn.Module, batch, keep: int = 20_000) -> dict:
    """Per-activation cost of a converted model, measured on a real ``batch``.

    Runs ``batch`` through ``model`` once, capturing each spiking activation's
    input, and returns ``{module_name: neuron_cost(...)}`` (stored / active bases,
    timesteps, spikes per input). Because MBE and PASN store different numbers of
    bases, these let the comparison be made at matched cost, not just accuracy.

    **Activations only** -- this is not a total. Use :func:`spiking_cost_report` for
    the model's whole spike cost; the FP-multiply identities inside LayerNorm,
    Softmax and the attention matmuls are not counted here.
    """
    from .metrics import neuron_cost
    acts = [(n, m) for n, m in model.named_modules()
            if isinstance(m, _SpikingActModule)]
    if not acts:
        return {}
    caps: dict[str, torch.Tensor] = {}
    handles = []

    def _mk(name):
        def hook(mod, args):
            if name not in caps and args and torch.is_tensor(args[0]):
                caps[name] = args[0].detach().reshape(-1)[:keep]
        return hook

    for name, mod in acts:
        handles.append(mod.register_forward_pre_hook(_mk(name)))
    device = _module_device(model)
    b = _batch_to_device(batch, device)
    was_training = model.training
    model.eval()
    if isinstance(b, dict):
        model(**b)
    elif isinstance(b, (tuple, list)):
        model(*b)
    else:
        model(b)
    model.train(was_training)
    for h in handles:
        h.remove()
    return {name: neuron_cost(mod.act.neuron, caps[name])
            for name, mod in acts if name in caps}


def format_cost_report(costs: dict, label: str = "") -> str:
    """Aggregate :func:`activation_cost_report` output into a printable summary."""
    if not costs:
        return f"[cost{(' ' + label) if label else ''}] no spiking activations"
    n = len(costs)
    stored = max(c["stored"] for c in costs.values())
    steps = max(c["steps"] for c in costs.values())
    active = sum(c["active"] for c in costs.values()) / n
    spikes = sum(c["spikes"] for c in costs.values()) / n
    tag = f" {label}" if label else ""
    return (f"[cost{tag}] activations={n}  T={steps}  "
            f"stored bases/act={stored}  active bases/input={active:.2f}  "
            f"activation spikes/input={spikes:.2f}")
