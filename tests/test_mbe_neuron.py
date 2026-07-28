"""Unit tests for the MBE neuron core.

Run:  python -m pytest tests/ -q     (from the project root, SNN env)
   or  python tests/test_mbe_neuron.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from mbe import (  # noqa: E402
    MBENeuron, MBEConfig, SignedMBENeuron, functions, fit_function, fit_model,
    solve_readout, spiking_ops, PASNNeuron, build_pasn, sar_encode,
)
from mbe.mbe_pasn import (  # noqa: E402
    PrefixRouter as MBEPrefixRouter, MBEPASNNeuron, build_mbe_pasn,
)
from mbe.mbe_pasn_s import MBEPASNSNeuron, build_mbe_pasn_s  # noqa: E402


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


def test_mbe_pasn_r1_reduces_to_single_bank():
    """MBE-PASN with R=1 is bit-identical to the MBE neuron.

    A single-binade, single-sign router sends every input to one magnitude bank
    fed ``|x|``. With ``x >= 0`` and the bank's ``x_min/x_scale`` matching, the
    MBE-PASN output must equal that MBE neuron's output exactly.
    """
    torch.manual_seed(0)
    router = MBEPrefixRouter(1.0, 2.0, e_min=0, e_max=1)
    assert router.signed is False
    mbe = MBENeuron(MBEConfig(n_basis=4, n_steps=16, x_min=1.0, x_scale=1.0))
    dummy = MBENeuron(MBEConfig(n_basis=1, n_steps=16))     # near-zero bank, unused
    pasn = MBEPASNNeuron(router, [dummy, mbe])
    x = torch.linspace(1.0, 2.0, 50)[:-1]                   # in [1,2)
    assert torch.equal(pasn(x), mbe(x))
    usage = pasn.bank_usage(x)
    assert usage[0] == 0 and usage[1] == len(x)


def test_mbe_pasn_router_routes_by_binade():
    """Router places inputs in the correct sign+exponent binade, near-zero first."""
    r = MBEPrefixRouter(-8.0, 8.0, e_min=-2, e_max=3)       # near0 span = 0.25
    x = torch.tensor([0.1, -0.1, 1.5, 3.0, -1.5, -6.0])
    idx = r.route(x)
    assert int(idx[0]) == 0 and int(idx[1]) == 0            # |x|<0.25 -> near0
    assert int(idx[2]) == r.key_to_idx[(1.0, 0)]            # 1.5 in [1,2)
    assert int(idx[3]) == r.key_to_idx[(1.0, 1)]            # 3.0 in [2,4)
    assert int(idx[4]) == r.key_to_idx[(-1.0, 0)]           # -1.5 in [-2,-1)
    assert int(idx[5]) == r.key_to_idx[(-1.0, 2)]           # -6.0 in [-8,-4)


def test_mbe_pasn_signed_near0_splits_the_polarity():
    """``near0="signed"`` reads the sign bit in the near-zero region too.

    With the default single near-zero bank the bend of GELU/SiLU/ReLU sits in the
    *middle* of that bank, where a monotone staircase code cannot resolve it (a
    measured error floor -- 연구일지 실험 1 section 6). Splitting by polarity gives
    every bank single-sign, monotone inputs; this test pins the routing, not the
    accuracy.
    """
    r = MBEPrefixRouter(-8.0, 8.0, e_min=-2, e_max=3, near0="signed")
    assert r.n_banks == MBEPrefixRouter(-8.0, 8.0, e_min=-2,
                                        e_max=3).n_banks + 1
    x = torch.tensor([-0.2, -0.01, 0.0, 0.01, 0.2, -1.5, 1.5])
    idx = r.route(x)
    neg, pos = r.near0_idx[-1.0], r.near0_idx[1.0]
    assert idx[0] == neg and idx[1] == neg                  # negatives together
    assert idx[2] == pos and idx[3] == pos and idx[4] == pos
    assert r.bank_interval(neg) == (-0.25, 0.0)
    assert r.bank_interval(pos) == (0.0, 0.25)
    # every near-zero bank now sees a single sign, and is fed |x| (raw=False)
    for bi in (neg, pos):
        assert r.banks[bi]["raw"] is False
        sub = x[idx == bi]
        assert bool((sub >= 0).all()) or bool((sub < 0).all())
    # magnitude routing is unchanged
    assert idx[5] == r.key_to_idx[(-1.0, 0)] and idx[6] == r.key_to_idx[(1.0, 0)]


def test_mbe_pasn_rule_budget_scales_with_dynamic_range():
    """The 실험 1 rule: ``T`` follows the bank's dynamic range, ``N`` stays 1.

    ``b = log2(delta / sqrt(eps))`` is the required resolution in bits, and a fitted
    MBE bank delivers ``2 + 1.7*log2(T)`` of them. A flat tail therefore needs the
    minimum budget while a full-scale bank needs the maximum -- which is the whole
    reason a per-bank budget saves spikes over a global ``T``.
    """
    from mbe.mbe_pasn import T_GRID, rule_budget
    eps = 1e-5
    flat_N, flat_T = rule_budget(1e-4, eps)          # essentially constant bank
    mid_N, mid_T = rule_budget(0.25, eps)
    big_N, big_T = rule_budget(64.0, eps)

    assert (flat_N, flat_T) == (1, T_GRID[0])        # nothing to resolve
    assert flat_T < mid_T < big_T                    # monotone in dynamic range
    assert flat_N == mid_N == 1                      # extra bases only at the top
    assert big_N > 1 and big_T == T_GRID[-1]
    assert rule_budget(0.0, eps) == (1, T_GRID[0])   # constant target: bias alone
    # A tighter target needs more resolution at the same dynamic range.
    assert rule_budget(0.25, 1e-8)[1] > mid_T


def test_mbe_pasn_beta_moves_the_dense_ranges():
    """``beta`` relocates the log-dense point; ``gamma`` provably cannot.

    The binade grid is anchored at zero, so a domain that never approaches zero
    gets one usable range. ``1/S`` on ``[0.5, 1)`` is exactly that -- a single
    binade -- yet its curvature peaks at ``S = 0.5``. Routing on ``S - 0.5``
    re-expands that end. ``gamma`` only slides the domain along the grid, which
    ``e_min`` already does, so with the exponent bounds taken relative to the key
    range it cancels exactly.
    """
    r0 = MBEPrefixRouter(0.5, 1.0, e_min=-6, e_max=0)
    rb = MBEPrefixRouter(0.5, 1.0, e_min=-6, e_max=0, beta=0.5)
    s = torch.tensor([0.5, 0.501, 0.52, 0.6, 0.8, 0.99])

    # anchored at zero every point lands in one bank; anchored at 0.5 they spread
    assert len(set(r0.route(s).tolist())) == 1
    assert len(set(rb.route(s).tolist())) >= 4

    # the routed interval still maps back to x-space through the affine
    for bi in range(rb.n_banks):
        a, b = rb.bank_interval(bi)
        ka, kb = rb.key_interval(bi)
        assert abs(rb.from_key(ka) - a) < 1e-9 and abs(rb.from_key(kb) - b) < 1e-9

    # gamma is a relabelling: same partition of the same points
    rg = MBEPrefixRouter(0.5, 1.0, e_min=-9, e_max=-3, beta=0.5, gamma_log2=3)
    def parts(r, x):
        idx = r.route(x)
        return sorted(tuple(sorted((idx == i).nonzero().flatten().tolist()))
                      for i in set(idx.tolist()))
    assert parts(rg, s) == parts(rb, s)


def test_mbe_pasn_tied_banks_share_one_prototype():
    """A homogeneous target factorises into (routed scale) x (one unit shape).

    ``f(sigma 2^e (1+rho)) = [sigma^k 2^{ek}] f(1+rho)`` for ``f(lx) = l^k f(x)``,
    and the bracket is a function of the routed key -- which the router reads for
    free. So every magnitude bank can *be* one prototype with a scaled readout:
    storage stops growing with the number of ranges while the relative error stays
    flat across binades. The identity is the case that matters, since the
    spike-driven FP multiply reconstructs both operands through an ``MBE_Id``.
    """
    from mbe.mbe_pasn import build_mbe_pasn
    lo, hi = 2.0 ** -6, 2.0 ** 6
    x = torch.exp(torch.linspace(math.log(lo), math.log(hi * 0.99), 3000))
    kw = dict(e_min=-6, e_max=6, budget="rule", target="relative",
              target_rel=1e-3, near0="single", epochs=120,
              alpha_init="uniform")
    free = build_mbe_pasn("identity", (lo, hi), tied=False, **kw)
    tied = build_mbe_pasn("identity", (lo, hi), tied=True, **kw)

    def rel(m):
        with torch.no_grad():
            return float(((m(x) - x) / x).pow(2).mean().sqrt())

    # far less storage, no accuracy given up
    assert tied.num_learnable() * 4 < free.num_learnable()
    assert rel(tied) < 2.0 * rel(free)
    assert tied.stored_bases() < free.stored_bases()

    # relative error is flat across binades -- the point of the factorisation
    per = []
    for e in range(-6, 6):
        m = (x >= 2.0 ** e) & (x < 2.0 ** (e + 1))
        with torch.no_grad():
            per.append(float(((tied(x[m]) - x[m]) / x[m]).pow(2).mean().sqrt()))
    assert max(per) < 4.0 * min(per)

    # and the guard rejects a target that does not factorise
    try:
        build_mbe_pasn("gelu", (-8.0, 8.0), e_min=-2, e_max=4, tied=True,
                       epochs=20, near0="single")
    except ValueError as exc:
        assert "self-similar" in str(exc)
    else:
        raise AssertionError("tied=True must reject a non-homogeneous target")


def test_mbe_pasn_unreachable_banks_are_not_random():
    """Domain endpoints route to a bank the calibration domain never covers.

    ``x = hi`` has ``floor(log2|x|)`` one binade above the last fitted bank, so an
    unfitted bank there returns its random initialisation -- garbage, not
    extrapolation. :func:`fill_unreachable` points those at the nearest fitted
    neighbour, and shares the module so storage is unchanged.
    """
    from mbe.mbe_pasn import build_mbe_pasn
    m = build_mbe_pasn("tanh", (-8.0, 8.0), e_min=-2, e_max=4, n_local=1,
                       n_near0=1, n_steps=4, epochs=20, near0="single")
    unreachable = [i for i in range(m.router.n_banks)
                   if m.router.reachable(i, -8.0, 8.0) is None]
    assert unreachable, "this router/domain pair should have unreachable banks"
    for i in unreachable:                       # shares a fitted neighbour's module
        assert any(m.bank_mods[i] is m.bank_mods[j]
                   for j in range(m.router.n_banks) if j not in unreachable)
    edge = torch.tensor([-8.0, 8.0])
    assert float((m(edge) - torch.tanh(edge)).abs().max()) < 0.5


def test_mbe_pasn_dispatch_matches_per_bank():
    """MBE-PASN forward == each element routed through its own bank."""
    torch.manual_seed(0)
    router = MBEPrefixRouter(-4.0, 4.0, e_min=-1, e_max=2)
    banks = [MBENeuron(MBEConfig(n_basis=2, n_steps=8,
                                 x_min=b.get("x_min", -0.5),
                                 x_scale=b.get("x_scale", 1.0)))
             for b in router.banks]
    pasn = MBEPASNNeuron(router, banks)
    x = torch.linspace(-4.0, 4.0, 60)[:-1]
    out = pasn(x)
    idx = router.route(x)
    for i, xi in enumerate(x):
        bi = int(idx[i])
        spec = router.banks[bi]
        feed = xi if spec["kind"] == "near0" else xi.abs()
        assert torch.allclose(out[i], banks[bi](feed.reshape(1)).reshape(()))


# -- standalone PASN (successive-approximation) neuron ---------------------

def test_pasn_sar_encode_reconstructs():
    """SAR code reconstructs rho to T-bit precision; spikes = bit count."""
    rho = torch.tensor([0.0, 0.5, 0.75, 0.1, 0.999])
    recon, spikes = sar_encode(rho, T=10)
    assert (recon - rho).abs().max() < 2 ** -9
    # 0.5 -> one spike; 0.75 -> two spikes (0.5 + 0.25)
    assert int(spikes[1]) == 1 and int(spikes[2]) == 2


def test_pasn_beats_global_on_gelu_near_zero_at_fewer_spikes():
    """The standalone PASN resolves GELU near zero (where a global neuron floors
    at ~0.07) with a low near-zero error and few spikes -- its whole point."""
    torch.manual_seed(0)
    p = build_pasn("gelu", (-14.3, 10.9), e_min=-6, e_max=4, T=6, order=1)
    x = (torch.randn(4000) * 3.0).clamp(-14.3, 10.9)
    y = functions.gelu(x)
    with torch.no_grad():
        pred = p(x)
    near0 = x.abs() < 3
    assert (pred[near0] - y[near0]).abs().max() < 0.05     # global MBE floored ~0.07
    assert p.mean_spikes(x) < 6                            # vs MBE ~35 spikes/in
    assert p.stored_params() < 200                         # tiny readout-only memory


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


# -- MBE-PASN-S (shared basis set, prefix-routed readout) ------------------

def test_mbe_pasn_s_reduces_to_the_baseline_neuron():
    """A single routed range must be bit-identical to a plain MBENeuron carrying
    the same shape parameters and readout -- the shared-basis variant changes only
    which readout is selected, never the dynamics."""
    torch.manual_seed(0)
    N = 3
    router = MBEPrefixRouter(1.0, 2.0, e_min=0, e_max=1)   # near0 + mag e=0
    core = MBENeuron(MBEConfig(n_basis=N, n_steps=16, x_min=0.0, x_scale=1.0,
                               use_bias=False))
    ref = MBENeuron(MBEConfig(n_basis=N, n_steps=16, x_min=1.0, x_scale=1.0,
                              use_bias=False))
    ref.load_state_dict(core.state_dict())
    W = torch.zeros(router.n_banks, N + 1)
    W[1, :N] = ref.w                       # bank 1 gets the reference readout
    s = MBEPASNSNeuron(router, core, W)
    x = torch.rand(400) + 1.0              # all in [1,2) -> routed to bank 1
    with torch.no_grad():
        assert torch.equal(s(x), ref(x))


def test_mbe_pasn_s_reconstruct_is_the_spike_sum():
    """reconstruct() must be the pure spike-sum form of forward() (so the neuron is
    a valid operand for the spike-driven FP multiply), with the intensity matrix
    selected by the router."""
    m = build_mbe_pasn_s("invsqrt", (0.5, 2.0), e_min=-2, e_max=1, n_shared=3,
                         epochs=40, seed=0)
    x = torch.rand(500) * 1.5 + 0.5
    with torch.no_grad():
        assert (m.reconstruct(x) - m(x)).abs().max() < 1e-5


def test_mbe_pasn_s_costs_one_shared_basis_set():
    """The cost meter must credit the design: one basis set stored and executed
    regardless of the number of ranges, memory = 5N + R(N+1)."""
    from mbe.metrics import neuron_cost, neuron_params
    N = 3
    m = build_mbe_pasn_s("invsqrt", (0.5, 2.0), e_min=-2, e_max=1, n_shared=N,
                         epochs=40, seed=0)
    R = m.router.n_banks
    x = torch.rand(2000) * 1.5 + 0.5
    c = neuron_cost(m, x)
    assert c["stored"] == N and c["active"] == N        # not R*N
    assert c["spikes"] <= N * c["steps"]
    assert neuron_params(m) == 5 * N + R * (N + 1)


def test_mbe_pasn_s_respects_a_spike_budget():
    """Selection must honour a spike budget: min-MSE alone always takes the
    expensive end of the candidate pool, and spikes are the energy currency."""
    from mbe.mbe_pasn_s import pareto_front
    dom = (-8.0, 8.0)
    free = build_mbe_pasn_s("silu", dom, e_min=-2, e_max=4, n_shared=[2, 4],
                            epochs=40)
    tight = build_mbe_pasn_s("silu", dom, e_min=-2, e_max=4, n_shared=[2, 4],
                             epochs=40, spike_budget=8.0)
    x = (torch.randn(3000) * 3.0).clamp(*dom)
    # the budgeted build must be cheaper, and the unbudgeted one must be the
    # pool's most accurate candidate
    assert tight.mean_spikes(x) < free.mean_spikes(x)
    chosen = [c for c in free.selection_trace if c["chosen"]][0]
    assert chosen["mse"] == min(c["mse"] for c in free.selection_trace)
    # every chosen candidate under a budget must have been marked feasible
    tc = [c for c in tight.selection_trace if c["chosen"]][0]
    assert tc["spikes"] <= 8.0 and tc["feasible"]
    # the frontier is non-dominated and ordered by spikes
    front = pareto_front(free.selection_trace)
    assert front and all(front[i]["spikes"] <= front[i + 1]["spikes"]
                         for i in range(len(front) - 1))
    assert all(front[i]["mse"] > front[i + 1]["mse"]
               for i in range(len(front) - 1))


def test_convert_mbe_pasn_s_backend_uses_measured_range_weights():
    """backend="mbe_pasn_s" must reach every routed primitive, and the builder must
    receive the real calibration sample -- otherwise its spike budget is expressed
    under a uniform draw over the domain rather than what the network spends."""
    import copy
    from mbe import convert as cv
    from mbe.toy import make_toy, make_inputs
    from mbe.mbe_pasn_s import range_weights_from_sample
    D = 8
    ann = make_toy(seed=0, d_model=D, n_heads=2, n_layers=1)
    calib = make_inputs(2, batch=4, seq=8, d_model=D, seed=1)
    snn = copy.deepcopy(ann)
    rec = cv.calibrate(snn, calib)
    # captured before conversion: the op modules are replaced in place
    act_slot = cv._shared_fit_slots(snn, rec)["blocks.0.act"][0]
    cv.convert(snn, rec, cfg=cv.ConvertConfig(
        backend="mbe_pasn_s", spike_mult=True, pasn_e_min=-3, epochs=40,
        pasn_s_n_shared=[2, 4], pasn_s_restarts=1, pasn_s_spike_budget=12.0))
    act = snn.get_submodule("blocks.0.act").act.neuron
    ln = snn.get_submodule("blocks.0.ln1").ln
    qk = snn.get_submodule("blocks.0.attn.qk")
    for neuron in (act, ln.rsqrt, ln.id_dev, ln.id_istd, qk.idn, qk.idn2):
        assert isinstance(neuron, MBEPASNSNeuron), type(neuron)
    # budget honoured, and the chosen candidate came from the multi-N pool
    chosen = [c for c in act.selection_trace if c["chosen"]][0]
    assert chosen["spikes"] <= 12.0
    assert {c["n_shared"] for c in act.selection_trace} == {2, 4}
    # The builder must have received a real distribution, not the two-endpoint
    # histogram the shared-activation path used to hand it. Activations concentrate
    # near zero, so the measured per-range probabilities must be far from the
    # range-width weighting that stands in when no sample is available.
    x = act_slot.sample
    assert x.numel() > 100, "activation sample was reduced to its endpoints"
    dom = (float(x.min()), float(x.max()))
    w = range_weights_from_sample(act.router, dom, x)
    assert w is not None
    spans = {bi: act.router.reachable(bi, *dom) for bi in w}
    spans = {bi: s for bi, s in spans.items() if s is not None}
    total = sum(b - a for a, b in spans.values())
    tv = 0.5 * sum(abs(w[bi] - (spans[bi][1] - spans[bi][0]) / total)
                   for bi in spans)
    assert tv > 0.2, f"measured weights indistinguishable from width weights ({tv=})"
    test = make_inputs(1, batch=4, seq=8, d_model=D, seed=7)[0]
    with torch.no_grad():
        assert torch.isfinite(snn(test)).all()


def test_convert_pasn_backend_installs_pasn_neurons():
    """backend="pasn" must reach every routed conversion point (activation,
    LayerNorm primitives, activation*activation matmul identities) -- otherwise a
    "PASN" run silently falls back to plain MBE and the comparison is meaningless.
    Shares the router (pasn_e_min) with mbe_pasn so only the encoder differs."""
    import copy
    from mbe import convert as cv
    from mbe.toy import make_toy, make_inputs
    D = 8
    ann = make_toy(seed=0, d_model=D, n_heads=2, n_layers=1)
    calib = make_inputs(2, batch=4, seq=8, d_model=D, seed=1)
    snn = copy.deepcopy(ann)
    rec = cv.calibrate(snn, calib)
    cv.convert(snn, rec, cfg=cv.ConvertConfig(backend="pasn", spike_mult=True,
                                              pasn_e_min=-3, pasn_T=6,
                                              pasn_order=2))
    act = snn.get_submodule("blocks.0.act").act.neuron
    ln = snn.get_submodule("blocks.0.ln1").ln
    qk = snn.get_submodule("blocks.0.attn.qk")
    for neuron in (act, ln.rsqrt, ln.id_dev, ln.id_istd, qk.idn, qk.idn2):
        assert isinstance(neuron, PASNNeuron), type(neuron)
    assert act.T == 6 and act.order == 2
    assert act.router.e_min == -3
    # forward must stay finite and the cost meter must handle the routed neuron
    test = make_inputs(1, batch=4, seq=8, d_model=D, seed=7)[0]
    with torch.no_grad():
        out = snn(test)
    assert torch.isfinite(out).all()
    costs = cv.activation_cost_report(snn, test)
    assert costs and all(c["spikes"] <= 6 for c in costs.values())


def test_identity_router_depth_is_independent_of_the_activation():
    """pasn_id_e_min must deepen the router for the identity primitives only. Their
    operands span many decades, unlike the activation's, so they want a different
    depth; the activation must keep pasn_e_min so backends stay comparable."""
    import copy
    from mbe import convert as cv
    from mbe.toy import make_toy, make_inputs
    D = 8
    ann = make_toy(seed=0, d_model=D, n_heads=2, n_layers=1)
    calib = make_inputs(2, batch=4, seq=8, d_model=D, seed=1)

    def build(id_e_min):
        m = copy.deepcopy(ann)
        rec = cv.calibrate(m, calib)
        cv.convert(m, rec, cfg=cv.ConvertConfig(
            backend="pasn", epochs=30, spike_mult=True, pasn_e_min=-3,
            pasn_id_e_min=id_e_min))
        return m

    shallow, deep = build(None), build(-10)
    for m in (shallow, deep):
        assert torch.isfinite(m(make_inputs(1, batch=4, seq=8, d_model=D,
                                            seed=7)[0])).all()
    act_s = shallow.get_submodule("blocks.0.act").act.neuron
    act_d = deep.get_submodule("blocks.0.act").act.neuron
    assert act_s.router.e_min == act_d.router.e_min == -3, "activation must not move"
    id_s = shallow.get_submodule("blocks.0.attn.qk").idn
    id_d = deep.get_submodule("blocks.0.attn.qk").idn
    assert id_s.router.e_min == -3 and id_d.router.e_min == -10
    assert id_d.router.n_banks > id_s.router.n_banks


def test_routed_softmax_fits_each_primitive_on_its_own_argument():
    """A routed Softmax must route all three primitives, and each must be fitted on
    the argument the op actually feeds it -- not on the logits. MBE_inv is the
    documented exception: its argument is an IEEE mantissa in [0.5,1), a single
    binade, so the exponent router has one reachable range there by construction."""
    import copy
    from mbe import convert as cv
    from mbe.toy import make_toy, make_inputs
    D = 8
    ann = make_toy(seed=0, d_model=D, n_heads=2, n_layers=1)
    calib = make_inputs(2, batch=4, seq=8, d_model=D, seed=1)
    snn = copy.deepcopy(ann)
    rec = cv.calibrate(snn, calib)
    # the reduction axis must be recorded, or the argument distributions are
    # unrecoverable from the flattened sample
    assert rec.ranges["blocks.0.attn.softmax"][0].width > 1
    cv.convert(snn, rec, cfg=cv.ConvertConfig(
        backend="mbe_pasn_s", epochs=40, spike_mult=True, pasn_e_min=-3,
        pasn_s_n_shared=2, pasn_s_restarts=1))
    sm = snn.get_submodule("blocks.0.attn.softmax").sm
    for neuron in (sm.exp, sm.inv, sm.idn):
        assert isinstance(neuron, MBEPASNSNeuron), type(neuron)
    reach = lambda n, dom: sum(  # noqa: E731
        1 for bi in range(n.router.n_banks) if n.router.reachable(bi, *dom))
    assert reach(sm.exp, (0.0, 1.0)) > 1        # frac spans several binades
    assert reach(sm.idn, (0.0, 1.0)) > 1        # operands span several binades
    assert reach(sm.inv, (0.5, 1.0)) == 1       # mantissa is one binade
    test = make_inputs(1, batch=4, seq=8, d_model=D, seed=7)[0]
    with torch.no_grad():
        out = snn(test)
        assert torch.isfinite(out).all()
    # routing softmax must cut its share of the spike budget, not just move it
    rep = cv.spiking_cost_report(snn, test)
    assert rep["by_kind"]["softmax"] < 0.5 * rep["total_spikes"]


def test_spiking_cost_report_counts_every_primitive():
    """The activation is a fraction of a converted Transformer's spikes: each
    LayerNorm, Softmax and activation*activation matmul runs its own MBE_Id for the
    spike-driven multiply. A report that counts only activations understates energy."""
    from mbe import convert as cv
    _, snn, test = _convert_tiny(spike_mult=True)
    rep = cv.spiking_cost_report(snn, test)
    kinds = set(rep["by_kind"])
    assert kinds == {"activation", "layernorm", "softmax", "matmul"}, kinds
    # every primitive must actually have been invoked, and the total must exceed the
    # activation share by a wide margin
    assert all(e["calls"] > 0 and e["elements"] > 0
               for e in rep["primitives"].values())
    assert rep["total_spikes"] > 3 * rep["by_kind"]["activation"]
    assert rep["spikes_per_input"] > 0
    # by_kind must sum to the total, and instrumentation must be fully removed
    assert abs(sum(rep["by_kind"].values()) - rep["total_spikes"]) < 1e-6
    act = snn.get_submodule("blocks.0.act").act.neuron
    assert "forward" not in vars(act), "forward wrapper left installed"
    assert "reconstruct" not in vars(act), "reconstruct wrapper left installed"
    # spike_mult=False removes exactly the identity-based multiplies
    _, exact, _ = _convert_tiny(spike_mult=False)
    rep2 = cv.spiking_cost_report(exact, test)
    assert "matmul" not in rep2["by_kind"]
    assert rep2["total_spikes"] < rep["total_spikes"]


def test_cost_report_counts_one_call_once():
    """One invocation must be charged once, whatever entry point it came in by.

    ``spiking_cost_report`` patches both ``forward`` and ``reconstruct``, and the
    routed neurons implement ``reconstruct`` *as* ``self.forward`` -- which, once
    patched, is the wrapped forward. Tallying at every wrapped entry charged a
    routed reconstruct twice while a plain ``MBENeuron`` (whose ``reconstruct``
    goes to ``spike_train``/``intensities``) was charged once, so the bias fell
    entirely on the routed backends, and the identity is 60-70% of a converted
    model's spikes.
    """
    from mbe import convert as cv
    from mbe.mbe_pasn import build_mbe_pasn

    routed = build_mbe_pasn("identity", (0.0, 4.0), e_min=-2, e_max=2,
                            n_local=1, n_near0=1, n_steps=4, epochs=20,
                            near0="single")
    plain = MBENeuron(MBEConfig(n_basis=1, n_steps=4, x_min=0.0, x_scale=4.0,
                                use_bias=False))
    x = torch.rand(64) * 4.0

    for neuron in (routed, plain):
        for entry in ("forward", "reconstruct"):
            tally = {"n": dict(kind="k", spikes=0.0, elements=0, calls=0)}
            busy = {"flag": False}
            saved = []
            cv._install_cost_probe("n", neuron, tally, busy, saved)
            try:
                getattr(neuron, entry)(x)
            finally:
                cv._remove_cost_probes(saved)
            assert tally["n"]["calls"] == 1, (
                f"{type(neuron).__name__}.{entry} charged "
                f"{tally['n']['calls']} times")
            assert tally["n"]["elements"] == x.numel()
        assert "forward" not in vars(neuron) and "reconstruct" not in vars(neuron)


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
