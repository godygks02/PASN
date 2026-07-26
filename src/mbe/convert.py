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

    def update(self, t: torch.Tensor, keep: int):
        t = t.detach()
        self.lo = min(self.lo, float(t.min()))
        self.hi = max(self.hi, float(t.max()))
        self.absmax = max(self.absmax, float(t.abs().max()))
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

    # One activation fit on the union of ranges observed across all blocks.
    for group in act_groups.values():
        lo = min(slot.lo for _, slot in group)
        hi = max(slot.hi for _, slot in group)
        shared = _Range(
            lo=lo,
            hi=hi,
            absmax=max(abs(lo), abs(hi)),
            sample=torch.tensor([lo, hi], dtype=torch.float32),
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
    # "mbe"      = one global MBE neuron;
    # "mbe_pasn" = FP-prefix binade banks of MBE neurons;
    # "pasn"     = FP-prefix binade banks + successive-approximation (SAR) code.
    # GPT-2 Stage 1 converts GELU + LayerNorm only; functional attention/Softmax
    # stays exact.
    backend: str = "mbe"
    # Router budget -- shared by both prefix-routed backends *on purpose*, so a
    # mbe_pasn-vs-pasn run differs only in the encoder, not in the routing.
    pasn_e_min: int = -3
    pasn_n_local: int = 2            # mbe_pasn: MBE bases per binade bank
    pasn_T: int = 6                  # pasn: SAR bits (spike budget) per bank
    pasn_order: int = 2              # pasn: per-bank readout polynomial order
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


# Backends whose neurons are prefix-routed banks. Both share the router
# (``pasn_e_min``) and differ only in what each bank does: MBE bases (mbe_pasn)
# vs a fixed SAR spike code + closed-form readout (pasn).
_ROUTED_BACKENDS = ("mbe_pasn", "pasn")


def _routed_primitive(name, domain, e_min, e_max, cfg: ConvertConfig,
                      fit_device, n_near0=None):
    """Build one prefix-routed neuron for target ``name`` on ``domain``.

    Dispatches on ``cfg.backend``; the router bounds are passed through
    identically so ``mbe_pasn`` and ``pasn`` are compared at the same routing.
    """
    if cfg.backend == "pasn":
        # The SAR code needs no per-bank basis count: the budget is (T, order).
        return build_pasn(
            name, domain, e_min=e_min, e_max=e_max,
            T=cfg.pasn_T, order=cfg.pasn_order, seed=cfg.seed,
            device=fit_device, verbose=cfg.verbose_fits,
        )
    return build_mbe_pasn(
        name, domain, e_min=e_min, e_max=e_max,
        n_local=cfg.pasn_n_local,
        n_near0=n_near0 or cfg.pasn_n_local,
        n_steps=cfg.n_steps, epochs=cfg.epochs, seed=cfg.seed,
        device=fit_device, verbose=cfg.verbose_fits,
    )


def _routed_identity(hi, cfg: ConvertConfig, fit_device):
    """Prefix-routed identity over ``[0, hi]`` for the spike-driven multiply.

    Note both routed backends reconstruct the identity with a DC term in the
    readout, so ``reconstruct`` is not the strictly bias-free spike sum that
    :func:`spiking_ops.calibrate_identity` builds for the plain MBE backend.
    """
    e_min, e_max = _erange(0.0, hi, cfg.pasn_e_min)
    return _routed_primitive("identity", (0.0, hi), e_min, e_max, cfg,
                             fit_device)


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
                                       n_near0=cfg.pasn_n_local + 2)
            return _SpikingActModule(so.SpikingActivation(neuron))
        act = so.build_activation(mod.kind, slots[0].sample,
                                  n_basis=cfg.n_basis_act, n_steps=cfg.n_steps,
                                  seed=cfg.seed, margin=cfg.margin,
                                  epochs=cfg.epochs, device=fit_device)
        return _SpikingActModule(act)

    if kind == "softmax":
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
            rsqrt = _routed_primitive(
                "invsqrt", (0.5, 2.0), -2, 1, cfg, fit_device, n_near0=4
            )
            id_dev = _routed_identity(dev_max, cfg, fit_device)
            id_istd = _routed_identity(istd_max, cfg, fit_device)
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
            idn = _routed_identity(hi_a, cfg, fit_device)
            idn2 = _routed_identity(hi_b, cfg, fit_device)
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

@torch.no_grad()
def activation_cost_report(model: nn.Module, batch, keep: int = 20_000) -> dict:
    """Per-activation cost of a converted model, measured on a real ``batch``.

    Runs ``batch`` through ``model`` once, capturing each spiking activation's
    input, and returns ``{module_name: neuron_cost(...)}`` (stored / active bases,
    timesteps, spikes per input). Because MBE and PASN store different numbers of
    bases, these let the comparison be made at matched cost, not just accuracy.
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
            f"spikes/input={spikes:.2f}")
