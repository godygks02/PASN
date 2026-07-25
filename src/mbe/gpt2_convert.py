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


def convert_gpt2(model: nn.Module, calib_input_ids, cfg: cv.ConvertConfig | None = None,
                 only: set[str] | None = None, verbose: bool = False):
    """Calibrate on ``calib_input_ids`` (iterable of ``input_ids`` tensors) and
    replace the marked nonlinearities in place. Call :func:`make_spikable` first."""
    cfg = cfg or cv.ConvertConfig()
    rec = cv.calibrate(model, calib_input_ids)
    cv.convert(model, rec, cfg=cfg, only=only, verbose=verbose)
    return model, rec
