"""Unified cost metrics across neuron types (energy = spikes, plus memory/compute).

Accuracy alone is not comparable when neurons store different numbers of
parameters, so every neuron type reports the same cost dict:

  * ``stored``  -- stored parameters / bases (memory),
  * ``active``  -- units evaluated per input (compute breadth),
  * ``steps``   -- timesteps T,
  * ``spikes``  -- mean spikes emitted per input (SOPs; the SNN energy proxy).

Handles MBE, signed-MBE, MBE-PASN (prefix banks of MBE neurons), and the
standalone PASN (successive-approximation) neuron.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def neuron_cost(neuron, x: torch.Tensor) -> dict:
    from .signed import SignedMBENeuron
    from .mbe_pasn import MBEPASNNeuron
    from .mbe_pasn_s import MBEPASNSNeuron
    from .pasn import PASNNeuron

    if isinstance(neuron, PASNNeuron):
        return neuron.cost(x)

    if isinstance(neuron, MBEPASNSNeuron):
        # One shared basis set runs for every input, so the spike accounting is
        # that of a single global MBE neuron of the same (N, T): routing is free.
        cfg = neuron.core.cfg
        return dict(stored=neuron.stored_bases(), active=cfg.n_basis,
                    steps=cfg.n_steps, spikes=neuron.mean_spikes(x))

    if isinstance(neuron, MBEPASNNeuron):
        steps = neuron.bank_mods[0].cfg.n_steps
        flat = x.reshape(-1)
        idx = neuron.router.route(flat)
        counts = torch.tensor([b.cfg.n_basis for b in neuron.bank_mods],
                              device=idx.device, dtype=torch.float32)
        active = float(counts[idx].mean()) if flat.numel() else 0.0
        return dict(stored=neuron.stored_bases(), active=active, steps=steps,
                    spikes=neuron.mean_spikes(x))

    if isinstance(neuron, SignedMBENeuron):
        Np, Nn = neuron.pos.cfg.n_basis, neuron.neg.cfg.n_basis
        steps = neuron.pos.cfg.n_steps
        xp, xn = neuron._split(x)
        spikes = (neuron.pos.firing_rate(xp) * Np
                  + neuron.neg.firing_rate(xn) * Nn) * steps
        # both polarity banks run for every input, so all bases are active
        return dict(stored=Np + Nn, active=Np + Nn, steps=steps, spikes=spikes)

    # plain MBE neuron
    N, steps = neuron.cfg.n_basis, neuron.cfg.n_steps
    return dict(stored=N, active=N, steps=steps,
                spikes=neuron.firing_rate(x) * N * steps)


@torch.no_grad()
def spikes_per_input(neuron, x: torch.Tensor) -> float:
    """Mean spikes per input element for any neuron type (unified cost meter)."""
    return neuron_cost(neuron, x)["spikes"]


@torch.no_grad()
def neuron_params(neuron) -> int:
    """Stored floating-point parameters (memory) for any neuron type.

    PASN stores only per-bank readout coefficients (the SAR thresholds are fixed
    constants); MBE / signed-MBE / MBE-PASN store learned spike-dynamics + readout.
    """
    from .mbe_pasn import MBEPASNNeuron
    from .pasn import PASNNeuron
    if isinstance(neuron, PASNNeuron):
        return int(neuron.W.numel())
    if isinstance(neuron, MBEPASNNeuron):
        return int(sum(b.num_learnable() for b in neuron.bank_mods))
    # MBE, signed-MBE and MBE-PASN-S all report their own learnable count
    # (MBE-PASN-S: 5N shared shape parameters + R(N+1) routed readout).
    return int(neuron.num_learnable())
