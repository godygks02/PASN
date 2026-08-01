"""GPT-2 -> spiking conversion adapter (Phase 4d prep).

Bridges HuggingFace GPT-2 to the conversion framework in :mod:`mbe.convert`.

Stage 1 (implemented here): convert the two nonlinearities that are plain
modules -- the MLP activation (``NewGELUActivation``) and every ``nn.LayerNorm``.
These are GPT-2's dominant conversion-error sources (the Phase-4 toy study showed
the GELU activation dominates), and it is exactly where PASN wins. Attention
(QK^T / softmax / attn·V) is computed inside ``eager_attention_forward`` as
functional ops with no module to swap, so it is left exact here and wired in
Stage 2.

Usage::

    from mbe.gpt2_convert import make_spikable, convert_gpt2
    make_spikable(model)                      # NewGELU -> Activation marker
    convert_gpt2(model, calib_input_ids, cfg) # calibrate + replace (in place)
"""
from __future__ import annotations

import torch.nn as nn

from . import convert as cv


def _is_new_gelu(mod: nn.Module) -> bool:
    return type(mod).__name__ in ("NewGELUActivation", "GELUActivation")


def make_spikable(model: nn.Module) -> int:
    """Replace HF GELU activation modules with a convertible ``Activation`` marker.

    ``NewGELUActivation`` is the tanh-approx GELU; we calibrate to the matching
    ``gelu_tanh`` target so the spiking neuron reproduces the ANN's activation.
    Returns the number of activations marked. ``nn.LayerNorm`` needs no marking --
    :func:`mbe.convert.classify` already recognises it.
    """
    n = 0
    for name, mod in list(model.named_modules()):
        if _is_new_gelu(mod):
            cv._set_submodule(model, name, cv.Activation("gelu_tanh"))
            n += 1
    return n


#: Name this module registers in HF's attention dispatch table.
SPIKING_ATTN_IMPL = "mbe_spiking"


def _spiking_attention_forward(module, query, key, value, attention_mask,
                               scaling=None, dropout=0.0, **kwargs):
    """``eager_attention_forward`` routed through the module's swap points.

    Identical arithmetic, except the two activation*activation products and the
    softmax go through :class:`~mbe.convert.MatMulAA` / :class:`~mbe.convert.Softmax`
    markers held on the attention module. Before conversion those markers *are*
    ``@`` and ``torch.softmax``, so registering this changes nothing numerically;
    after conversion they are the spiking primitives.

    The ``c_attn`` / ``c_proj`` projections stay exact on purpose: they are
    activation*weight, which is native accumulation on neuromorphic hardware and
    must not be spiked (exp 10 measured their approximation error as exactly 0).
    """
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = module.qk_matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    # The mask puts finfo.min in here; SpikingSoftmax floors the shift so those
    # entries underflow to zero instead of turning into NaN.
    attn_weights = module.attn_softmax(attn_weights)
    attn_weights = attn_weights.type(value.dtype)
    if dropout:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout,
                                             training=module.training)
    attn_output = module.av_matmul(attn_weights, value)
    return attn_output.transpose(1, 2), attn_weights


def _is_gpt2_attention(mod: nn.Module) -> bool:
    return type(mod).__name__ == "GPT2Attention"


def make_attention_spikable(model: nn.Module) -> int:
    """Stage 2: give every attention block swap points for QK^T / softmax / attn*V.

    HF computes attention in a *function*, so there is nothing to replace the way
    ``make_spikable`` replaces the GELU module. But the function is looked up
    through ``ALL_ATTENTION_FUNCTIONS``, which is a supported extension point --
    so instead of patching library code we register an implementation and hang the
    markers on each ``GPT2Attention``. The markers are ordinary submodules, so
    :func:`mbe.convert.calibrate` records their operand ranges and
    :func:`mbe.convert.convert` replaces them, both unchanged.

    Call before ``convert_gpt2``; returns the number of attention blocks marked.
    Registering is idempotent and numerically inert until conversion runs.
    """
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import (ALL_MASK_ATTENTION_FUNCTIONS,
                                            eager_mask)

    AttentionInterface.register(SPIKING_ATTN_IMPL, _spiking_attention_forward)
    # **Both** dispatches are keyed on the implementation name. Registering only
    # the attention function leaves the name unknown to the mask pipeline, which
    # reads that as "custom backend, builds its own mask" and hands back None
    # (masking_utils, "we don't need a mask!"). The model then attends to future
    # tokens with no error anywhere -- logits moved by 269 before this line
    # existed. We consume the additive float mask exactly as eager does, so we
    # take eager's mask builder.
    ALL_MASK_ATTENTION_FUNCTIONS.register(SPIKING_ATTN_IMPL, eager_mask)

    n = 0
    for mod in model.modules():
        if _is_gpt2_attention(mod):
            mod.qk_matmul = cv.MatMulAA()
            mod.attn_softmax = cv.Softmax(dim=-1)
            mod.av_matmul = cv.MatMulAA()
            # the dispatch reads self.config, which each block shares with the model
            mod.config._attn_implementation = SPIKING_ATTN_IMPL
            n += 1
    model.config._attn_implementation = SPIKING_ATTN_IMPL
    return n


def convert_gpt2(model: nn.Module, calib_input_ids, cfg: cv.ConvertConfig | None = None,
                 only: set[str] | None = None, verbose: bool = False):
    """Calibrate on ``calib_input_ids`` (iterable of ``input_ids`` tensors) and
    replace the marked nonlinearities in place. Call :func:`make_spikable` first."""
    cfg = cfg or cv.ConvertConfig()
    rec = cv.calibrate(model, calib_input_ids)
    cv.convert(model, rec, cfg=cfg, only=only, verbose=verbose)
    return model, rec
