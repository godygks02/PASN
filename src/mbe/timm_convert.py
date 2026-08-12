"""timm -> spiking conversion.

``hf_convert`` reaches HuggingFace models through two properties they all share:
the MLP activation is a *module* with a recognisable class name, and attention is
dispatched through ``ALL_ATTENTION_FUNCTIONS``. **timm has neither.**

* Its activation is plain ``torch.nn.GELU`` -- no HF wrapper class, so
  ``hf_convert.ACT_TARGETS`` (keyed on ``GELUActivation`` and friends) matches
  nothing.
* Its attention is computed **inline** in ``timm.layers.attention.Attention.forward``
  with the signature ``(self, x, attn_mask, is_causal)``. There is no registry to
  register into.

Both markers therefore return **0** on a timm model, which is worse than an
error: nothing converts, the "SNN" is the ANN, and the conversion loss prints as
0.00%. This module supplies the two missing hooks.

**``approximate`` decides which GELU it is.** ``nn.GELU()`` is the exact erf form;
``nn.GELU(approximate='tanh')`` is GPT-2's. Calibrating one against the other's
target injects an error the conversion never recovers and is invisible by name --
exactly the bug the RoBERTa adapter hit. This module reads the attribute.

**``fused_attn`` is forced off.** timm defaults it to ``True``, which routes
attention through ``F.scaled_dot_product_attention`` -- one fused kernel with no
seam to put a marker in. The explicit branch computes the same attention with
``@`` and ``.softmax()``, which is where the swap points go.

That switch *can* move the numerics, so marker inertness is checked against the
**unfused** reference rather than the fused one. Measured on ViT-M/16 the
fused/unfused gap is at most ~6e-8 and is sometimes exactly zero -- on CPU the
two can dispatch to the same kernel -- so it is bounded, not assumed nonzero.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn

from . import convert as cv


def _gelu_target(mod: nn.GELU) -> str:
    """``nn.GELU`` -> our calibration target, by its ``approximate`` flag."""
    return "gelu_tanh" if getattr(mod, "approximate", "none") == "tanh" else "gelu"


#: ``torch.nn`` activation class -> target. GELU is resolved per-instance because
#: one class covers two different functions.
_SIMPLE_TARGETS = {nn.SiLU: "silu", nn.ReLU: "relu", nn.Tanh: "tanh"}


def make_spikable(model: nn.Module) -> int:
    """Swap every recognised activation module for a calibratable marker.

    Returns the number marked. ``nn.LayerNorm`` needs no marking --
    :func:`mbe.convert.classify` already recognises it.
    """
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.GELU):
            target = _gelu_target(mod)
        else:
            target = _SIMPLE_TARGETS.get(type(mod))
        if target is not None:
            cv._set_submodule(model, name, cv.Activation(target))
            n += 1
    return n


def _spiking_attention_forward(self, x: torch.Tensor, attn_mask=None,
                               is_causal: bool = False) -> torch.Tensor:
    """``timm.layers.attention.Attention.forward`` with the swap points in.

    Mirrors the library's **unfused** branch exactly; the two
    activation*activation products and the softmax go through markers held on the
    module. Before conversion those markers *are* ``@`` and ``.softmax()``, so
    binding this changes nothing against the unfused reference -- pinned by
    ``tests/test_timm_convert.py``.
    """
    from timm.layers.attention import maybe_add_mask, resolve_self_attn_mask

    B, N, C = x.shape
    gate = self.gate(x).sigmoid() if self.gate is not None else None
    qkv = (self.qkv(x)
           .reshape(B, N, 3, self.num_heads, self.head_dim)
           .permute(2, 0, 3, 1, 4))
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = self.qk_matmul(q, k.transpose(-2, -1))
    attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
    attn = maybe_add_mask(attn, attn_bias)
    attn = self.attn_softmax(attn)
    attn = self.attn_drop(attn)
    x = self.av_matmul(attn, v)

    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    if gate is not None:
        x = x * gate
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def make_attention_spikable(model: nn.Module) -> int:
    """Give every timm attention block swap points for QK^T / softmax / attn*V.

    Returns the number marked. Forces ``fused_attn=False`` on each, which is a
    numerical change and the reason this cannot be silent -- see the module
    docstring.
    """
    from timm.layers.attention import Attention

    n = 0
    for mod in model.modules():
        if isinstance(mod, Attention):
            mod.fused_attn = False
            mod.qk_matmul = cv.MatMulAA()
            mod.attn_softmax = cv.Softmax(dim=-1)
            mod.av_matmul = cv.MatMulAA()
            mod.forward = types.MethodType(_spiking_attention_forward, mod)
            n += 1
    return n


def unfuse_attention(model: nn.Module) -> int:
    """Force the unfused attention path **without** adding markers.

    The reference a marked model should be compared against: it isolates the
    fused-to-unfused numerical change from the marking itself.
    """
    from timm.layers.attention import Attention
    n = 0
    for mod in model.modules():
        if isinstance(mod, Attention):
            mod.fused_attn = False
            n += 1
    return n


def convert_timm(model: nn.Module, calib_batches,
                 cfg: cv.ConvertConfig | None = None,
                 only: set[str] | None = None, verbose: bool = False):
    """Calibrate on ``calib_batches`` and replace the marked ops in place."""
    cfg = cfg or cv.ConvertConfig()
    rec = cv.calibrate(model, calib_batches)
    cv.convert(model, rec, cfg=cfg, only=only, verbose=verbose)
    return model, rec
