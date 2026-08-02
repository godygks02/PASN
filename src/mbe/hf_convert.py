"""HuggingFace -> spiking conversion, for any model the framework can reach.

``gpt2_convert`` was written against GPT-2; everything in it that was not about
GPT-2 lives here. Modern HF encoders and decoders share the two hooks that matter:

* the MLP activation is a **module** (``GELUActivation`` and friends), so it can be
  swapped directly;
* attention is a **function**, but it is looked up through
  ``ALL_ATTENTION_FUNCTIONS`` and the mask through ``ALL_MASK_ATTENTION_FUNCTIONS``,
  both of which accept registrations -- so attention is reached by registering an
  implementation rather than by patching library code.

Verified identical across GPT-2, RoBERTa and ViT in transformers 5.8: the same
``eager_attention_forward(module, query, key, value, attention_mask, scaling,
dropout, **kwargs)`` signature and the same two-registry dispatch.

Weight matmuls (``c_attn``, ``query``/``key``/``value``, ``c_proj``) stay exact on
purpose: activation*weight is native accumulation, measured at exactly zero
approximation error (exp 10). Only activation*activation products are spiked.
"""
from __future__ import annotations

import torch.nn as nn

from . import convert as cv

#: HF activation module class -> our calibration target. **These are different
#: functions.** ``NewGELUActivation`` (GPT-2) is the tanh approximation;
#: ``GELUActivation`` (BERT/RoBERTa/ViT) is the exact erf form. Calibrating one
#: against the other's target injects an error the conversion never recovers,
#: and it is invisible -- both are "GELU" by name.
ACT_TARGETS = {
    "NewGELUActivation": "gelu_tanh",
    "GELUActivation": "gelu",
    "FastGELUActivation": "gelu_tanh",
    "SiLUActivation": "silu",
    "ReLUActivation": "relu",
}

#: Name this module registers in HF's two dispatch tables.
SPIKING_ATTN_IMPL = "mbe_spiking"


def make_spikable(model: nn.Module) -> int:
    """Swap every recognised HF activation module for a calibratable marker.

    Returns the number marked. ``nn.LayerNorm`` needs no marking --
    :func:`mbe.convert.classify` already recognises it.
    """
    n = 0
    for name, mod in list(model.named_modules()):
        target = ACT_TARGETS.get(type(mod).__name__)
        if target is not None:
            cv._set_submodule(model, name, cv.Activation(target))
            n += 1
    return n


def _spiking_attention_forward(module, query, key, value, attention_mask,
                               scaling=None, dropout=0.0, **kwargs):
    """``eager_attention_forward`` routed through the module's swap points.

    Identical arithmetic; the two activation*activation products and the softmax
    go through markers held on the attention module. Before conversion those
    markers *are* ``@`` and ``torch.softmax``, so registering this changes nothing
    numerically -- a property worth testing, since a silent shift here is
    indistinguishable from a bad fit later.
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


def make_attention_spikable(model: nn.Module, class_names=None) -> int:
    """Give every attention block swap points for QK^T / softmax / attn*V.

    ``class_names`` selects the attention modules by class name; the default
    matches the usual HF spellings (``GPT2Attention``, ``RobertaSelfAttention``,
    ``ViTSelfAttention``, ...). Pass an explicit tuple for a model that names
    them otherwise.

    **Both** registries have to be fed. Registering only the attention function
    leaves the name unknown to the mask pipeline, which reads that as "custom
    backend, builds its own mask" and returns ``attention_mask=None``; the model
    then attends to future tokens with no error raised anywhere. That cost us a
    269-unit logit shift before it was found, and it is pinned by
    ``test_stage2_attention_markers_are_numerically_inert``.
    """
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import (ALL_MASK_ATTENTION_FUNCTIONS,
                                            eager_mask)

    AttentionInterface.register(SPIKING_ATTN_IMPL, _spiking_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(SPIKING_ATTN_IMPL, eager_mask)

    def is_attention(mod) -> bool:
        name = type(mod).__name__
        if class_names is not None:
            return name in class_names
        return (name.endswith("SelfAttention") or name.endswith("Attention")) \
            and hasattr(mod, "config") and not name.startswith("Cross")

    n = 0
    for mod in model.modules():
        if is_attention(mod):
            mod.qk_matmul = cv.MatMulAA()
            mod.attn_softmax = cv.Softmax(dim=-1)
            mod.av_matmul = cv.MatMulAA()
            # the dispatch reads self.config, shared with the model
            mod.config._attn_implementation = SPIKING_ATTN_IMPL
            n += 1
    if n:
        model.config._attn_implementation = SPIKING_ATTN_IMPL
    return n


def convert_hf(model: nn.Module, calib_batches, cfg: cv.ConvertConfig | None = None,
               only: set[str] | None = None, verbose: bool = False):
    """Calibrate on ``calib_batches`` and replace the marked ops in place.

    ``calib_batches`` is an iterable of whatever the model's ``forward`` takes --
    a tensor of ``input_ids``, a dict of kwargs, or a tuple.
    """
    cfg = cfg or cv.ConvertConfig()
    rec = cv.calibrate(model, calib_batches)
    cv.convert(model, rec, cfg=cfg, only=only, verbose=verbose)
    return model, rec
