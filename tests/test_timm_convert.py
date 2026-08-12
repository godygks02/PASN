"""timm wiring: the markers must reach every site and change nothing.

``hf_convert``'s hooks do not exist in timm -- its activation is plain
``nn.GELU`` and its attention is computed inline with no registry -- so both
markers return **0** on a timm model. That failure is silent in the worst way:
nothing converts, the "SNN" is the ANN, and the conversion loss prints as 0.00%.
These tests pin the replacement.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _skip():
    try:
        import timm  # noqa: F401
        return False
    except ImportError:                                       # pragma: no cover
        print("  (skipped: timm not installed)")
        return True


def _model(seed=0):
    import timm
    torch.manual_seed(seed)
    return timm.create_model("vit_medium_patch16_reg4_gap_256",
                             pretrained=False, num_classes=10).eval()


def test_hf_markers_do_not_reach_timm():
    """The reason this module exists: hf_convert marks nothing on timm.

    If this ever starts failing, timm has grown HF-compatible hooks and
    ``timm_convert`` may be redundant -- check before deleting it.
    """
    if _skip():
        return
    from mbe import hf_convert as hc
    m = _model()
    assert hc.make_spikable(m) == 0
    assert hc.make_attention_spikable(m) == 0


def test_markers_reach_every_site():
    if _skip():
        return
    from mbe import timm_convert as tc
    m = _model()
    n_blocks = len(m.blocks)
    assert tc.make_spikable(m) == n_blocks, "one activation per block"
    assert tc.make_attention_spikable(m) == n_blocks, "one attention per block"


def test_gelu_target_follows_the_approximate_flag():
    """``nn.GELU`` is two different functions and only the flag tells them apart.

    Calibrating the exact erf form against the tanh target (or the reverse)
    injects an error the conversion never recovers, and it is invisible by name.
    That bug shipped once already, on RoBERTa.
    """
    if _skip():
        return
    from mbe import timm_convert as tc
    assert tc._gelu_target(torch.nn.GELU()) == "gelu"
    assert tc._gelu_target(torch.nn.GELU(approximate="none")) == "gelu"
    assert tc._gelu_target(torch.nn.GELU(approximate="tanh")) == "gelu_tanh"
    # and the model we actually convert takes the exact form
    m = _model()
    gelus = [x for x in m.modules() if isinstance(x, torch.nn.GELU)]
    assert gelus and all(tc._gelu_target(g) == "gelu" for g in gelus)


def test_markers_are_numerically_inert():
    """Marking must not change the ANN by one bit, against the *unfused* model.

    timm defaults ``fused_attn=True``, and the fused kernel is where the marker
    cannot go, so ``make_attention_spikable`` forces it off. That switch can move
    the numerics, so it must not be folded into the inertness claim: the
    reference is the **unfused** model, and the fused/unfused gap is bounded
    separately.

    That gap is **not asserted to be nonzero** -- on CPU, SDPA can dispatch to
    the same math kernel and agree exactly. Measured on this model it is either 0
    or ~6e-8 depending on the input, so only the upper bound is a property.
    """
    if _skip():
        return
    from mbe import timm_convert as tc
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        y_fused = _model()(x)
        m_unfused = _model(); tc.unfuse_attention(m_unfused)
        y_unfused = m_unfused(x)
        m_marked = _model(); tc.make_attention_spikable(m_marked)
        y_marked = m_marked(x)
        m_act = _model(); tc.make_spikable(m_act)
        y_act = m_act(x)

    assert torch.equal(y_unfused, y_marked), "attention markers must be exact"
    assert torch.equal(y_fused, y_act), "activation markers must be exact"
    gap = (y_fused - y_unfused).abs().max().item()
    assert gap < 1e-5, f"fused/unfused gap {gap:.3e} exceeds float32 noise"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
