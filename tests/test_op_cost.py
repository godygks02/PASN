"""Unit tests for the whole-op spike meter (:mod:`mbe.op_cost`).

The meter is the measuring instrument behind ``experiments/op_pareto.py``, so a
silent error in it does not produce a wrong-looking result -- it produces a
plausible one. These tests pin the two properties the comparison depends on:

  * **multiplicity** -- each primitive is charged once per input element it
    actually saw, so a per-row reciprocal costs 1/C of a per-element ``2^x``; and
  * **backend symmetry** -- the same op costs the same *number of invocations*
    whichever neuron implements it, so a spike ratio between backends reflects
    the neurons and not their internal call structure.

Run:  python tests/test_op_cost.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import spiking_ops as so  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402
from mbe.op_cost import SpikeMeter, op_memory  # noqa: E402
from mbe.metrics import spikes_per_input  # noqa: E402

EPOCHS = 30          # these tests check accounting, not fit quality


def _softmax_prims_mbe(logits, n=2, t=8):
    sm = so.build_softmax(logits, n_basis=n, n_steps=t, epochs=EPOCHS)
    return sm, dict(exp=sm.exp, inv=sm.inv, idn=sm.idn)


def _softmax_prims_pasn(n=2, t=8):
    mk = lambda nm, dom, e0, e1: build_mbe_pasn(       # noqa: E731
        nm, dom, e_min=e0, e_max=e1, n_local=n, n_near0=n, n_steps=t,
        epochs=EPOCHS, seed=0)
    exp = mk("exp2", (0.0, 1.0), -4, 0)
    inv = mk("inv", (0.5, 1.0), -2, 0)
    idn = mk("identity", (0.0, 1.0), -4, 0)
    return so.SpikingSoftmax(exp, inv, idn), dict(exp=exp, inv=inv, idn=idn)


def _meter(op, prims, n_out):
    with torch.no_grad(), SpikeMeter() as m:
        m.attach_all(prims)
        out = op()
    return out, m.per_element(n_out), m.invocations(n_out), dict(m.calls)


def test_multiplicity_matches_the_decomposition():
    """2^x runs per element, 1/x per row, the identity once for each operand."""
    torch.manual_seed(0)
    R, C = 16, 8
    logits = torch.randn(R, C) * 3.0
    sm, prims = _softmax_prims_mbe(logits)
    _, _, mult, calls = _meter(lambda: sm(logits, dim=-1), prims, R * C)
    assert abs(mult["exp"] - 1.0) < 1e-9, mult
    assert abs(mult["inv"] - 1.0 / C) < 1e-9, mult
    # the final multiply reconstructs exp_x (per element) and inv_S (per row);
    # inv_S is deliberately NOT broadcast first (see SpikingSoftmax.forward)
    assert abs(mult["idn"] - (1.0 + 1.0 / C)) < 1e-9, mult
    assert calls == {"exp": 1, "inv": 1, "idn": 2}, calls


def test_backends_agree_on_invocation_counts():
    """A routed neuron must not be charged extra invocations for the same op.

    ``MBEPASNNeuron.reconstruct`` is implemented as ``self.forward(x)``, so a
    meter that patches both methods without a re-entrancy guard charges the PASN
    identity twice and the MBE identity once -- manufacturing a 2x spike gap out
    of nothing. This is the regression test for that.
    """
    torch.manual_seed(0)
    R, C = 16, 8
    logits = torch.randn(R, C) * 3.0
    sm_m, pm = _softmax_prims_mbe(logits)
    sm_p, pp = _softmax_prims_pasn()
    _, _, mult_m, calls_m = _meter(lambda: sm_m(logits, dim=-1), pm, R * C)
    _, _, mult_p, calls_p = _meter(lambda: sm_p(logits, dim=-1), pp, R * C)
    assert calls_m == calls_p, (calls_m, calls_p)
    for k in mult_m:
        assert abs(mult_m[k] - mult_p[k]) < 1e-9, (k, mult_m, mult_p)


def test_total_equals_hand_computed_spikes():
    """The tally is exactly ``spikes_per_input * numel`` summed over calls."""
    torch.manual_seed(0)
    R, C = 12, 4
    logits = torch.randn(R, C) * 2.0
    sm, prims = _softmax_prims_mbe(logits)
    _, per, _, _ = _meter(lambda: sm(logits, dim=-1), prims, R * C)

    # replay the decomposition by hand and price each tensor the meter saw
    with torch.no_grad():
        x_m = logits - logits.max(dim=-1, keepdim=True).values
        z = x_m * 1.4426950408889634
        frac = z - torch.floor(z)
        two_frac = sm.exp(frac)
        exp_x = torch.ldexp(two_frac, torch.floor(z).to(torch.int64))
        S = exp_x.sum(dim=-1, keepdim=True)
        m, e = torch.frexp(S)
        inv_S = torch.ldexp(sm.inv(m), -e)
        want = (spikes_per_input(sm.exp, frac) * frac.numel()
                + spikes_per_input(sm.inv, m) * m.numel()
                + spikes_per_input(sm.idn, exp_x) * exp_x.numel()
                + spikes_per_input(sm.idn, inv_S) * inv_S.numel())
    assert abs(per["_total"] - want / (R * C)) < 1e-6, (per["_total"], want)


def test_detach_restores_methods():
    """A leaked patch would double-count every later measurement."""
    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    sm, prims = _softmax_prims_mbe(logits)
    before = {k: (v.forward, getattr(v, "reconstruct", None)) for k, v in
              prims.items()}
    with SpikeMeter() as m:
        m.attach_all(prims)
        assert prims["exp"].forward is not before["exp"][0]
    for k, v in prims.items():
        assert v.forward == before[k][0], k
        assert getattr(v, "reconstruct", None) == before[k][1], k
    # and "forward" must not be left behind as an instance attribute shadowing
    # the class method
    assert "forward" not in prims["exp"].__dict__


def test_detach_survives_an_exception():
    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    _, prims = _softmax_prims_mbe(logits)
    orig = prims["exp"].forward
    try:
        with SpikeMeter() as m:
            m.attach_all(prims)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert prims["exp"].forward == orig


def test_op_memory_dedups_shared_primitives():
    """An op reusing one neuron at two sites pays for it once."""
    torch.manual_seed(0)
    idn = so.calibrate_identity(0.0, 4.0, n_basis=2, n_steps=8, epochs=EPOCHS)
    one = op_memory(dict(a=idn))
    two = op_memory(dict(a=idn, b=idn))
    assert two["bytes"] == one["bytes"], (one, two)
    assert two["n_prims"] == 2


def test_attach_rejects_duplicate_names():
    idn = so.calibrate_identity(0.0, 4.0, n_basis=2, n_steps=8, epochs=EPOCHS)
    with SpikeMeter() as m:
        m.attach("a", idn)
        try:
            m.attach("a", idn)
        except ValueError:
            return
    raise AssertionError("duplicate attach should raise")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
