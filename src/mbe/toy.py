"""A tiny module-structured Transformer for Phase 4 conversion verification.

Every nonlinearity is expressed with the conversion-point op modules from
``convert.py`` (``Activation``, ``Softmax``, ``MatMulAA``) plus ``nn.LayerNorm``,
so :func:`convert.convert` can swap them all. Weights are ordinary ``nn.Linear``
(activation*weight = native accumulation, left untouched by conversion).

Operates on already-embedded inputs ``(B, S, D)`` -- the point is the nonlinear
wiring, not tokenisation. This is the ANN whose converted (spiking) forward is
compared against it in ``experiments/verify_phase4.py``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .convert import Activation, Softmax, MatMulAA


class ToyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.qk = MatMulAA()            # Q @ K^T   (activation*activation)
        self.av = MatMulAA()            # attn @ V  (activation*activation)
        self.softmax = Softmax(dim=-1)

    def _split(self, x):
        B, S, D = x.shape
        return x.view(B, S, self.h, self.dh).transpose(1, 2)   # (B,H,S,Dh)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self._split(self.q(x)), self._split(self.k(x)), self._split(self.v(x))
        scores = self.qk(q, k.transpose(-2, -1)) * self.scale  # (B,H,S,S)
        attn = self.softmax(scores)
        ctx = self.av(attn, v)                                 # (B,H,S,Dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, D)
        return self.o(ctx)


class ToyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                 act: str = "gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ToyAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_ratio * d_model)
        self.act = Activation(act)
        self.fc2 = nn.Linear(mlp_ratio * d_model, d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return x


class ToyTransformer(nn.Module):
    """Pre-norm Transformer stack over ``(B, S, D)`` embeddings."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2,
                 mlp_ratio: int = 4, act: str = "gelu"):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ToyBlock(d_model, n_heads, mlp_ratio, act) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


def make_toy(seed: int = 0, d_model: int = 32, n_heads: int = 4,
             n_layers: int = 2) -> ToyTransformer:
    torch.manual_seed(seed)
    model = ToyTransformer(d_model, n_heads, n_layers)
    model.eval()
    return model


def make_inputs(n_batches: int, batch: int = 8, seq: int = 16, d_model: int = 32,
                seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, seq, d_model, generator=g) for _ in range(n_batches)]
