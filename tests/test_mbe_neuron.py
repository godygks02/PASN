"""Unit tests for the MBE neuron core.

Run:  python -m pytest tests/ -q     (from the project root, SNN env)
   or  python tests/test_mbe_neuron.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from mbe import (  # noqa: E402
    MBENeuron, MBEConfig, SignedMBENeuron, functions, fit_function, fit_model,
    solve_readout, spiking_ops, PrefixRouter, PASNNeuron, build_pasn,
)


def test_forward_shape_and_finiteness():
    m = MBENeuron(MBEConfig(n_basis=4, n_steps=16))
    for shape in [(10,), (3, 5), (2, 4, 6)]:
        x = torch.randn(*shape)
        y = m(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_param_count_5N_and_6N():
    # learnable alpha_v (default) + bias  ->  6N + 1
    m = MBENeuron(MBEConfig(n_basis=4, learn_alpha_v=True, use_bias=True))
    assert m.num_learnable() == 6 * 4 + 1
    # fixed alpha_v (paper's convention) + bias  ->  5N + 1
    m = MBENeuron(MBEConfig(n_basis=4, learn_alpha_v=False, use_bias=True))
    assert m.num_learnable() == 5 * 4 + 1
    # fixed alpha_v, no bias  ->  exactly 5N (paper Table VI)
    m = MBENeuron(MBEConfig(n_basis=4, learn_alpha_v=False, use_bias=False))
    assert m.num_learnable() == 5 * 4


def test_gradients_flow():
    m = MBENeuron(MBEConfig(n_basis=4, n_steps=16))
    x = torch.linspace(0.5, 2.0, 64)
    loss = (m(x) - 1.0 / torch.sqrt(x)).pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item()
                    for p in m.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_decay_off_is_constant_kernel():
    # With decay disabled, the kernels must be constant across time steps.
    m = MBENeuron(MBEConfig(n_basis=3, n_steps=8, decay=False))
    d, r, vth = m._kernels()
    for k in (d, r, vth):
        assert torch.allclose(k[0], k[-1])


def test_r1_reduces_to_single_bank():
    """PASN with R=1 is bit-identical to the MBE neuron (PASN_method.md section 13).

    A single-binade, single-sign router sends every input to one magnitude bank
    fed the magnitude ``|x|``. With ``x >= 0`` and the bank's ``x_min/x_scale``
    matching, the PASN output must equal that MBE neuron's output exactly.
    """
    torch.manual_seed(0)
    # domain [1,2): router has one active magnitude binade at e=0 (plus an unused
    # near-zero bank), so all inputs route to a single MBE bank.
    router = PrefixRouter(1.0, 2.0, e_min=0, e_max=1)
    assert router.signed is False
    mbe = MBENeuron(MBEConfig(n_basis=4, n_steps=16, x_min=1.0, x_scale=1.0))
    dummy = MBENeuron(MBEConfig(n_basis=1, n_steps=16))     # near-zero bank, unused
    pasn = PASNNeuron(router, [dummy, mbe])
    x = torch.linspace(1.0, 2.0, 50)[:-1]                   # in [1,2)
    assert torch.equal(pasn(x), mbe(x))
    # confirm only the single magnitude bank is ever used
    usage = pasn.bank_usage(x)
    assert usage[0] == 0 and usage[1] == len(x)


def test_prefix_router_routes_by_binade():
    """Router places inputs in the correct sign+exponent binade, near-zero first."""
    r = PrefixRouter(-8.0, 8.0, e_min=-2, e_max=3)          # near0 span = 0.25
    x = torch.tensor([0.1, -0.1, 1.5, 3.0, -1.5, -6.0])
    idx = r.route(x)
    assert int(idx[0]) == 0 and int(idx[1]) == 0            # |x|<0.25 -> near0
    # same-binade same-sign share a bank; different sign / exponent differ
    assert int(idx[2]) == r.key_to_idx[(1.0, 0)]            # 1.5 in [1,2)
    assert int(idx[3]) == r.key_to_idx[(1.0, 1)]            # 3.0 in [2,4)
    assert int(idx[4]) == r.key_to_idx[(-1.0, 0)]           # -1.5 in [-2,-1)
    assert int(idx[5]) == r.key_to_idx[(-1.0, 2)]           # -6.0 in [-8,-4)


def test_pasn_dispatch_matches_per_bank():
    """PASN forward == each element routed through its own bank (dispatch check)."""
    torch.manual_seed(0)
    router = PrefixRouter(-4.0, 4.0, e_min=-1, e_max=2)
    banks = [MBENeuron(MBEConfig(n_basis=2, n_steps=8,
                                 x_min=b.get("x_min", -0.5),
                                 x_scale=b.get("x_scale", 1.0)))
             for b in router.banks]
    pasn = PASNNeuron(router, banks)
    x = torch.linspace(-4.0, 4.0, 60)[:-1]
    out = pasn(x)
    idx = router.route(x)
    for i, xi in enumerate(x):
        bi = int(idx[i])
        spec = router.banks[bi]
        feed = xi if spec["kind"] == "near0" else xi.abs()
        assert torch.allclose(out[i], banks[bi](feed.reshape(1)).reshape(()))


def test_can_fit_monotone_function():
    """End-to-end: MBE fits 1/sqrt(x) to well below the N=1 paper floor."""
    x, y, _ = functions.sample("invsqrt", m=2000, seed=0)
    cfg = functions.make_config("invsqrt", n_basis=8, n_steps=16)
    _, res = fit_function(x, y, cfg, seed=0, epochs=400)
    assert res.mse < 1e-3  # paper N=8 target 4.9e-5; loose bound for a fast run


def test_solve_readout_is_optimal():
    """Closed-form readout must beat (or match) any single gradient state for w."""
    torch.manual_seed(0)
    x, y, _ = functions.sample("inv", m=1000, seed=0)
    m = MBENeuron(functions.make_config("inv", n_basis=8, n_steps=16))
    solve_readout(m, x, y)
    mse_solved = (m(x) - y).pow(2).mean().item()
    # perturbing the readout can only make MSE worse (LS optimality)
    with torch.no_grad():
        m.w.add_(0.1)
    mse_perturbed = (m(x) - y).pow(2).mean().item()
    assert mse_solved <= mse_perturbed + 1e-12


def test_signed_neuron_fits_gelu():
    """SignedMBENeuron reproduces GELU on the paper domain (base neuron cannot)."""
    x, y, _ = functions.sample("gelu", m=2000, seed=0, domain=(-120, 10))
    sm = functions.make_signed("gelu", n_pos=4, n_neg=4, pivot=0.0)
    res = fit_model(sm, x, y, seed=0, epochs=400)
    assert sm.num_learnable() == 6 * 4 + 6 * 4 + 1  # 6N per bank (learn alpha) + bias
    assert res.mse < 1e-3  # paper GELU N=8 = 1.0e-4; loose bound for a fast run


def test_fp_mult_outer_equals_separable():
    """The explicit D (x) S Hadamard sum equals the separable recon product."""
    idn = spiking_ops.calibrate_identity(0.0, 8.0, n_basis=8, epochs=300)
    outer = spiking_ops.multiply_outer(idn, 3.3, 5.1)
    sep = float(idn.reconstruct(torch.tensor([3.3]))
                * idn.reconstruct(torch.tensor([5.1])))
    assert abs(outer - sep) < 1e-4
    assert abs(outer - 3.3 * 5.1) < 0.5  # spike approx of the true product


def test_spiking_softmax_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(32, 12) * 3.0
    sm = spiking_ops.build_softmax(logits, n_basis=8)
    out = sm(logits, dim=-1)
    ref = torch.softmax(logits, dim=-1)
    assert (out - ref).abs().mean() < 5e-3
    assert abs(out.sum(-1).mean().item() - 1.0) < 0.02  # rows ~normalised


def test_spiking_layernorm_matches_torch():
    torch.manual_seed(0)
    D = 32
    x = torch.randn(40, D) * 2.0 + 1.0
    ln = spiking_ops.build_layernorm(x, n_basis=8)
    out = ln(x)
    ref = F.layer_norm(x, (D,), eps=1e-5)
    assert (out - ref).abs().mean() < 2e-2


def test_spiking_matmul_approximates_product():
    """Spike-driven A@B (reconstruct-and-accumulate) approximates the true matmul."""
    torch.manual_seed(0)
    idn = spiking_ops.calibrate_identity(0.0, 4.0, n_basis=8, epochs=300)
    A = (torch.rand(6, 5) - 0.5) * 6
    B = (torch.rand(5, 4) - 0.5) * 6
    out = spiking_ops.spiking_matmul(idn, A, B, signed=True)
    ref = A @ B
    assert out.shape == ref.shape
    assert ((out - ref).abs().mean() / ref.abs().mean()) < 0.05


def _convert_tiny(spike_mult, epochs=120):
    """Build + calibrate + convert a tiny toy Transformer (fresh each call)."""
    import copy
    from mbe import convert as cv
    from mbe.toy import make_toy, make_inputs
    D = 8
    ann = make_toy(seed=0, d_model=D, n_heads=2, n_layers=1)
    calib = make_inputs(2, batch=4, seq=8, d_model=D, seed=1)
    test = make_inputs(1, batch=4, seq=8, d_model=D, seed=7)[0]
    snn = copy.deepcopy(ann)
    rec = cv.calibrate(snn, calib)
    cv.convert(snn, rec, cfg=cv.ConvertConfig(epochs=epochs, spike_mult=spike_mult,
                                              n_basis_act=8))
    return ann, snn, test


def test_phase4_conversion_replaces_only_nonlinearities():
    """convert() swaps every nonlinearity for a spiking module and leaves the
    weight matmuls (nn.Linear) untouched (training-free)."""
    import torch.nn as nn
    from mbe import convert as cv
    _, snn, _ = _convert_tiny(spike_mult=True)
    assert isinstance(snn.get_submodule("blocks.0.ln1"), cv._SpikingLayerNormModule)
    assert isinstance(snn.get_submodule("blocks.0.act"), cv._SpikingActModule)
    assert isinstance(snn.get_submodule("blocks.0.attn.softmax"),
                      cv._SpikingSoftmaxModule)
    assert isinstance(snn.get_submodule("blocks.0.attn.qk"), cv._SpikingMatMulModule)
    # weight matmuls stay as ordinary Linear
    assert isinstance(snn.get_submodule("blocks.0.attn.q"), nn.Linear)
    assert isinstance(snn.get_submodule("blocks.0.fc1"), nn.Linear)
    # The callable primitive containers must themselves be nn.Modules so their
    # internal MBE/PASN neurons follow model.to(device/dtype) on GPU conversion.
    assert isinstance(snn.get_submodule("blocks.0.act").act, nn.Module)
    assert isinstance(snn.get_submodule("blocks.0.attn.softmax").sm, nn.Module)
    assert isinstance(snn.get_submodule("blocks.0.ln1").ln, nn.Module)
    snn.to(dtype=torch.float64)
    assert snn.get_submodule("blocks.0.act").act.neuron.w.dtype == torch.float64
    assert snn.get_submodule("blocks.0.attn.softmax").sm.exp.w.dtype == torch.float64
    assert snn.get_submodule("blocks.0.ln1").ln.rsqrt.w.dtype == torch.float64


def test_phase4_spike_path_matches_exact_wiring():
    """The spike-driven FP-mult path must add little over exact reconstruction --
    if it doesn't, the outer-product / matmul wiring is wrong (not just the fit)."""
    ann, snn_spike, test = _convert_tiny(spike_mult=True)
    _, snn_exact, _ = _convert_tiny(spike_mult=False)
    with torch.no_grad():
        y = ann(test)
        ys, ye = snn_spike(test), snn_exact(test)
    assert torch.isfinite(ys).all()
    denom = y.abs().mean().clamp(min=1e-6)
    # spike-mult and exact-mult agree closely: isolates wiring from fit quality
    assert ((ys - ye).abs().mean() / denom) < 3e-2
    # overall conversion error is at the single-op (GELU-dominated) scale
    assert ((ys - y).abs().mean() / denom) < 0.2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
