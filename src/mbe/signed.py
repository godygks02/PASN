"""Signed (polarity-split) MBE neuron.

The plain MBE neuron encodes a non-negative membrane and struggles with the
non-monotonic near-zero bend of GELU / SiLU (the negative dip): the linear-readout
representation ceiling on GELU's wide (-120, 10) domain is ~0.5, i.e. essentially
unrepresentable.

``SignedMBENeuron`` implements the signed-magnitude decomposition that
``PASN_method.md`` section 6 refers to. The input is split at a fixed pivot ``c``
into a positive magnitude ``relu(x - c)`` and a negative magnitude
``relu(c - x)``; each is encoded by an independent MBE bank (a plain
:class:`MBENeuron`), and a single linear readout combines both banks' bases::

    f_hat(x) = [pos.features(relu(x-c)) , neg.features(relu(c-x))] @ w  +  bias

Splitting at ``c = 0`` places the pivot exactly at GELU's high-curvature point and
gives the negative region dedicated resolution. Empirically this drops the GELU
representation ceiling from ~0.5 to ~3e-5 (paper level).

Per PASN's consistency rule, this signed mechanism is a property of the MBE
baseline; every PASN bank must apply it identically after range selection.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .neuron import MBENeuron, MBEConfig


class SignedMBENeuron(nn.Module):
    """Polarity-split MBE neuron with a single shared linear readout."""

    def __init__(self, pos_cfg: MBEConfig, neg_cfg: MBEConfig, pivot: float = 0.0):
        super().__init__()
        # Sub-banks are feature generators only: disable their private readout so
        # all readout weight lives in this module (one joint (w, bias) to solve).
        pos_cfg = _feature_only(pos_cfg)
        neg_cfg = _feature_only(neg_cfg)
        self.pos = MBENeuron(pos_cfg)
        self.neg = MBENeuron(neg_cfg)
        # The sub-banks contribute *features* only; their private readout weights
        # are unused here, so freeze them (keeps the param count honest).
        self.pos.w.requires_grad_(False)
        self.neg.w.requires_grad_(False)
        self.pivot = float(pivot)
        n = pos_cfg.n_basis + neg_cfg.n_basis
        self.w = nn.Parameter(torch.randn(n) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        # cfg alias so the fit loop's surrogate-annealing reaches this neuron;
        # a property keeps both sub-banks in sync.
        self._cfg = pos_cfg

    # keep both sub-banks' surrogate sharpness in lock-step with the schedule
    @property
    def cfg(self):
        return _SyncCfg(self)

    def _split(self, x):
        return torch.relu(x - self.pivot), torch.relu(self.pivot - x)

    def readout_features(self, x: torch.Tensor) -> torch.Tensor:
        xp, xn = self._split(x)
        return torch.cat([self.pos.features(xp), self.neg.features(xn)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.readout_features(x)            # (M, Npos+Nneg)
        out = (feats * self.w).sum(dim=1) + self.bias
        return out.reshape(x.shape)

    @torch.no_grad()
    def set_readout(self, coeffs: torch.Tensor):
        self.w.copy_(coeffs[:-1])
        self.bias.copy_(coeffs[-1:])

    @torch.no_grad()
    def firing_rate(self, x: torch.Tensor) -> float:
        xp, xn = self._split(x)
        return 0.5 * (self.pos.firing_rate(xp) + self.neg.firing_rate(xn))

    def num_learnable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _feature_only(cfg: MBEConfig) -> MBEConfig:
    """Copy a config with the private readout disabled (bias off)."""
    import dataclasses
    return dataclasses.replace(cfg, use_bias=False)


class _SyncCfg:
    """Proxy so ``model.cfg.surrogate_alpha = x`` sets it on both sub-banks."""

    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(self._owner.pos.cfg, name)

    def __setattr__(self, name, value):
        setattr(self._owner.pos.cfg, name, value)
        setattr(self._owner.neg.cfg, name, value)
