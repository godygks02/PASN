"""GPT-2 -> spiking conversion adapter.

The mechanism is not GPT-2 specific and now lives in :mod:`mbe.hf_convert`; this
module is the GPT-2 entry point and keeps the original names importable.

Stage 1 converts the two nonlinearities that are plain modules -- the MLP
activation and every ``nn.LayerNorm``. Stage 2 adds attention (QK^T / softmax /
attn*V), which HF computes functionally and which is reached by registering an
attention implementation rather than by patching (see
:func:`~mbe.hf_convert.make_attention_spikable`).

Usage::

    from mbe.gpt2_convert import make_spikable, make_attention_spikable, convert_gpt2
    make_spikable(model)                       # GELU -> Activation marker
    make_attention_spikable(model)             # Stage 2 only
    convert_gpt2(model, calib_input_ids, cfg)  # calibrate + replace (in place)
"""
from __future__ import annotations

import torch.nn as nn

from . import convert as cv
from .hf_convert import (ACT_TARGETS, SPIKING_ATTN_IMPL,  # noqa: F401
                         _spiking_attention_forward, convert_hf,
                         make_attention_spikable, make_spikable)

#: GPT-2's activation is ``NewGELUActivation``, the **tanh approximation** --
#: distinct from the exact-erf ``GELUActivation`` that BERT/RoBERTa/ViT use.
#: ``ACT_TARGETS`` maps each to its own calibration target; conflating them would
#: fit the wrong function and never recover, invisibly, since both are "GELU".
GPT2_ACT = "NewGELUActivation"


def convert_gpt2(model: nn.Module, calib_input_ids, cfg: cv.ConvertConfig | None = None,
                 only: set[str] | None = None, verbose: bool = False):
    """Calibrate on ``calib_input_ids`` and replace the marked nonlinearities.

    Call :func:`make_spikable` (and :func:`make_attention_spikable` for Stage 2)
    first.
    """
    return convert_hf(model, calib_input_ids, cfg=cfg, only=only, verbose=verbose)
