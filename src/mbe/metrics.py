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
    from .pasn import PASNNeuron
    if isinstance(neuron, PASNNeuron):
        return int(neuron.W.numel())
    # MBE, signed-MBE, MBE-PASN and MBE-PASN-S all report their own learnable
    # count. For MBE-PASN that count is **de-duplicated**: unreachable banks share
    # a fitted neighbour and tied banks all reference one prototype, so summing
    # per-bank counts would charge the same basis set many times over.
    return int(neuron.num_learnable())


def _tensors(module) -> "list[torch.Tensor]":
    """Every tensor a module stores -- parameters *and* buffers.

    ``state_dict()`` rather than ``parameters()``: ``learn_tau=False`` moves the
    taus into buffers, which are still stored (trap 7).
    """
    return [t for t in module.state_dict().values() if torch.is_tensor(t)]


@torch.no_grad()
def storage_bytes(modules) -> int:
    """Real stored footprint of one module or an iterable of them, in bytes.

    **De-duplicated by storage, not by state-dict key.** ``TiedBank`` holds a
    *reference* to one prototype and :func:`~.mbe_pasn.fill_unreachable` points
    unreachable banks at a fitted neighbour, so a routed neuron's ``state_dict()``
    lists the same tensor once per bank that shares it. Summing those values
    charges the sharing as if it were copies -- which is precisely the saving the
    sharing exists to produce, counted backwards. A global MBE shares nothing, so
    the same sum is exact for the baseline and inflated for the routed side only:
    the comparison, not the measurement, is what breaks.

    ``neuron_params`` already de-duplicates (it goes through ``parameters()``),
    which is why the two axes disagreed on GPT-2 -- 8792 params vs 9249 for a
    *larger* byte count. Same object, two de-duplication rules.

    Sharing survives serialisation: ``torch.save`` records storages once, so the
    de-duplicated number is what a checkpoint and a weight-loading accelerator
    both actually pay.
    """
    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    seen: dict[int, int] = {}
    for m in modules:
        for t in _tensors(m):
            s = t.untyped_storage()
            seen[s.data_ptr()] = s.nbytes()
    return int(sum(seen.values()))


@torch.no_grad()
def storage_breakdown(modules) -> dict:
    """:func:`storage_bytes` split by what the tensor is for, plus the naive sum.

    Roles follow the neuron's own names: ``alpha`` (per-basis gains), ``tau`` (the
    three time constants and ``dt`` -- buffers when ``learn_tau=False``),
    ``readout`` (``w``, ``N*readout_order`` of them), ``bias``, and ``other``.
    ``bytes_naive`` is the un-de-duplicated sum the P0.4 tables reported, kept so
    those rows stay interpretable; ``shared_factor`` is how much of it was the
    same tensor counted twice.
    """
    if isinstance(modules, torch.nn.Module):
        modules = [modules]

    def role(key: str) -> str:
        leaf = key.rsplit(".", 1)[-1]
        if "alpha" in leaf:
            return "alpha"
        if leaf.startswith("log_tau") or leaf == "log_dt":
            return "tau"
        if leaf == "w":
            return "readout"
        if leaf == "bias":
            return "bias"
        return "other"

    by_role: dict[str, int] = {}
    naive = 0
    seen: set[int] = set()
    for m in modules:
        for key, t in m.state_dict().items():
            if not torch.is_tensor(t):
                continue
            naive += t.numel() * t.element_size()
            s = t.untyped_storage()
            if s.data_ptr() in seen:
                continue
            seen.add(s.data_ptr())
            by_role[role(key)] = by_role.get(role(key), 0) + s.nbytes()
    total = int(sum(by_role.values()))
    return dict(bytes=total, bytes_naive=int(naive),
                shared_factor=naive / max(total, 1), by_role=by_role)
