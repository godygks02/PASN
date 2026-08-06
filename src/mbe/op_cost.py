"""Whole-op cost accounting: spikes, storage and parameters for an assembled op.

``metrics.spikes_per_input`` prices **one neuron on one tensor**. An assembled op
(softmax, LayerNorm, an attention block) is several neurons invoked a different
number of times each, on tensors of different shapes: the softmax reciprocal runs
once per *row* while its ``2^x`` runs once per *element*, and the FP multiply's
signed split reconstructs each operand twice unless the identity's own router
reads the sign. Quoting the driver primitive's spikes -- which is what
``experiments/compare_ops_pasn_mbe.py`` does -- therefore reports a number no
hardware ever pays, and it is not even wrong in a constant way: the ratio between
the driver and the whole op is different for the two backends, because the number
of *invocations* is itself something PASN changes.

:class:`SpikeMeter` measures it instead of deriving it. It wraps the bound
``forward`` / ``reconstruct`` of every primitive an op holds, runs the op for
real, and tallies ``spikes_per_input(neuron, x) * x.numel()`` at each call. The
total, divided by the op's output elements, is the spikes per output element the
op costs -- with the invocation counts, the shapes, and the signed-split
behaviour all included because they actually happened.

Instance-attribute patching is what makes this work for ``reconstruct``:
``nn.Module.__call__`` resolves ``self.forward`` through the instance dict, so
one mechanism covers the decoded path (activations) and the spike-sum path (the
FP multiply's identities) alike.

Usage::

    with SpikeMeter() as m:
        m.attach("exp", sm.exp); m.attach("inv", sm.inv); m.attach("idn", sm.idn)
        out = sm(logits, dim=-1)
    m.per_element(out.numel())      # -> {"exp": ..., "inv": ..., "_total": ...}
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .metrics import neuron_params, spikes_per_input, storage_bytes

#: Methods a primitive may be invoked through. ``forward`` is the decoded output
#: (an activation, ``2^x``, ``1/sqrt``); ``reconstruct`` is the pure spike sum the
#: FP multiply uses. Both emit the same spike train, so both are metered.
_METERED = ("forward", "reconstruct")


class SpikeMeter:
    """Tally spikes emitted by a set of neurons over a real forward pass.

    Not reentrant and not thread-safe: it mutates the neurons it is given. Always
    use it as a context manager so the patches come off, including on exception --
    a leaked patch would double-count the neuron in every later measurement and
    keep the meter alive through the fit cache.
    """

    def __init__(self) -> None:
        self.spikes: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.elements: dict[str, float] = defaultdict(float)
        self._neurons: dict[str, torch.nn.Module] = {}
        self._depth: dict[str, int] = defaultdict(int)
        self._undo: list = []

    # -- attachment ---------------------------------------------------------

    def attach(self, name: str, neuron: torch.nn.Module) -> torch.nn.Module:
        """Meter ``neuron`` under ``name``. Returns it, so it can wrap a builder.

        Attach only the primitives the op invokes **directly**. A routed or signed
        neuron drives sub-banks internally; metering those as well would count the
        same spikes twice.
        """
        if name in self._neurons:
            raise ValueError(f"{name!r} already attached")
        self._neurons[name] = neuron
        for meth in _METERED:
            fn = getattr(neuron, meth, None)
            if fn is None:
                continue
            self._patch(neuron, meth, fn, name)
        return neuron

    def attach_all(self, prims: dict) -> dict:
        for name, neuron in prims.items():
            self.attach(name, neuron)
        return prims

    def _patch(self, neuron, meth: str, orig, name: str) -> None:
        had_own = meth in neuron.__dict__

        def wrapped(x, *a, **k):
            # Count the OUTERMOST metered entry only. Some neurons implement one
            # metered method in terms of another -- ``MBEPASNNeuron.reconstruct``
            # is literally ``return self.forward(x)`` -- so a naive tally charges
            # such a call twice, while ``MBENeuron.reconstruct`` builds the spike
            # sum directly and is charged once. That is an asymmetry *between the
            # two backends being compared*, i.e. exactly the measurement error
            # that would invent a 2x difference out of an implementation detail.
            if self._depth[name]:
                return orig(x, *a, **k)
            # Priced on the tensor as passed: a reciprocal called on one value per
            # row costs one row's worth, not one matrix's worth.
            self.spikes[name] += spikes_per_input(neuron, x) * x.numel()
            self.elements[name] += x.numel()
            self.calls[name] += 1
            self._depth[name] += 1
            try:
                return orig(x, *a, **k)
            finally:
                self._depth[name] -= 1

        setattr(neuron, meth, wrapped)
        self._undo.append((neuron, meth, orig, had_own))

    def detach(self) -> None:
        for neuron, meth, orig, had_own in reversed(self._undo):
            if had_own:
                setattr(neuron, meth, orig)
            else:                       # restore lookup to the class method
                neuron.__dict__.pop(meth, None)
        self._undo.clear()

    def __enter__(self) -> "SpikeMeter":
        return self

    def __exit__(self, *exc) -> None:
        self.detach()

    # -- readout ------------------------------------------------------------

    @property
    def total(self) -> float:
        return float(sum(self.spikes.values()))

    def per_element(self, n_out: int) -> dict:
        """Spikes per **output element** of the op, per primitive plus ``_total``."""
        n = max(int(n_out), 1)
        out = {k: v / n for k, v in self.spikes.items()}
        out["_total"] = self.total / n
        return out

    def invocations(self, n_out: int) -> dict:
        """Input elements each primitive saw, per output element of the op.

        This is the multiplicity the driver-primitive view drops. A softmax over
        rows of width ``C`` gives ``2^x`` a multiplicity of 1 and ``1/x`` one of
        ``1/C``; the signed FP multiply gives each identity 2.
        """
        n = max(int(n_out), 1)
        return {k: v / n for k, v in self.elements.items()}


def op_memory(prims: dict) -> dict:
    """Stored footprint of the primitives making up one op.

    ``bytes`` is de-duplicated by storage (:func:`~.metrics.storage_bytes`), so a
    tied or shared basis set is charged once -- which is the whole point of
    sharing, and charging it per bank would report the saving backwards.
    ``params`` is the learnable count, which de-duplicates by a different rule
    (see the ``storage_bytes`` docstring); the two are reported side by side
    rather than reconciled, because they answer different questions.
    """
    mods = list(prims.values())
    return dict(bytes=storage_bytes(mods),
                params=int(sum(neuron_params(m) for m in mods)),
                n_prims=len(mods))


@torch.no_grad()
def measure_op(run, prims: dict, n_out: int | None = None) -> dict:
    """Run ``run(prims)`` under a meter and return cost + the op's output.

    ``run`` takes the (already built) primitive dict and returns the op's output
    tensor. ``n_out`` defaults to that tensor's element count.
    """
    with SpikeMeter() as m:
        m.attach_all(prims)
        out = run(prims)
    n = out.numel() if n_out is None else n_out
    cost = op_memory(prims)
    cost.update(spikes=m.per_element(n)["_total"],
                spikes_by_prim=m.per_element(n),
                mult_by_prim=m.invocations(n),
                calls=dict(m.calls))
    return dict(out=out, cost=cost)
