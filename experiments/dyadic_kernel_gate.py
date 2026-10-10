"""E22 Stage 0 -- does a **shift-generated** kernel survive at the function level?

The MBE kernels are exponentials sampled on a grid::

    decay_P[t] = exp(-t * dt / tau_P) = rho_P ** t,   rho_P = exp(-dt / tau_P)

i.e. each kernel is a *geometric sequence*. So the whole (T, N) table is
generated from a seed by repeated multiplication by ``rho``. If ``rho`` is
constrained to a short signed sum of powers of two, that multiplication is a
shift-add and the table never has to be stored: the neuron holds three small
integer ratio codes per basis instead of four FP32 time constants. That is the
transplant of the mul-free PSN quantiser (Xue et al., NeurIPS 2025, Eq. 13) into
a *conversion* neuron, with B-PLA's multi-term dyadic expansion supplying the
precision that a single power of two cannot (rho = 0.9 is 41% away from its
nearest single power of two; two signed terms bring it to 0.875, three to
0.90625).

This script does **not** claim any energy or memory number. It measures the one
thing that gates all of them: **how much approximation error the constraint
costs at the function level** (the Table X axis, where no scope question exists).
If two terms already ruin the fit, every downstream idea is dead and no GPU time
is spent finding that out.

Arms, all fitted identically (same seeds, same schedule, same closed-form
readout):

===========  ==========================================================
``base``     unmodified ``MBENeuron._kernels`` -- the control
``geom``     ratio-parameterised, **not** quantised. Algebraically identical
             to ``base``; it exists to prove the refactor is a no-op *and*
             to price the reparameterisation (see ``dt`` below)
``rho{k}``   ratio quantised to ``k`` signed powers of two, STE on the whole
             quantiser (the paper's Eq. 16, not Eq. 14 -- see ``_ste``)
===========  ==========================================================

The ``dt`` observation, free and independent of everything else: ``dt`` enters
only ever as ``dt / tau_P``, so the four stored tensors
(``log_tau_r``, ``log_tau_vth``, ``log_tau_d``, ``log_dt``) carry **three**
degrees of freedom. ``geom`` holds 3N ratio parameters where ``base`` holds 4N
and must reach the same MSE. That is 1/6 of the neuron's 6N parameters,
removable at zero accuracy cost, and it is measured here rather than argued.

Usage::

    python experiments/dyadic_kernel_gate.py                    # full sweep
    python experiments/dyadic_kernel_gate.py --fns gelu --n 4   # one cell
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions, fit_model  # noqa: E402
from mbe.neuron import MBENeuron  # noqa: E402

# Table X's four primitives plus the identity: identities inside matmul are 45.1%
# of a converted GPT-2's spikes, so a constraint that survives GELU but breaks the
# identity has not survived anything that matters.
FNS = ["gelu", "invsqrt", "inv", "exp2", "identity"]
N_LIST = [2, 4]
T = 16
M = 4000

#: Ratio exponent range. A ratio lives in (0, 1]; -12 is already a decay so fast
#: that the basis is dead within one step at T=16, so the clamp never binds on a
#: useful kernel -- it only keeps log2 of a degenerate residual finite.
RHO_EXP = (-12, 0)
#: Readout weights span decades (alpha_v is log-spread over 6 octaves by default),
#: so ``w`` needs a far wider window than a ratio does.
W_EXP = (-24, 8)


# -- dyadic quantiser -------------------------------------------------------

def pot_quantise(v: torch.Tensor, n_terms: int, exp_range=RHO_EXP) -> torch.Tensor:
    """Nearest sum of ``n_terms`` signed powers of two (greedy, CSD-style).

    ``n_terms=1`` is exactly Eq. 13 of the mul-free PSN paper,
    ``sign(v) * 2**round(log2|v|)``. Further terms are fitted to the *residual*,
    which is B-PLA's short signed power-of-two expansion. A term is kept only
    when it actually shrinks the residual, so more terms can never be worse.
    """
    lo, hi = exp_range
    out = torch.zeros_like(v)
    res = v
    for _ in range(n_terms):
        mag = res.abs().clamp_min(1e-30)
        e = torch.round(torch.log2(mag)).clamp(lo, hi)
        term = torch.sign(res) * torch.pow(2.0, e)
        keep = (res - term).abs() < res.abs()
        term = torch.where(keep, term, torch.zeros_like(term))
        out = out + term
        res = res - term
    return out


def _ste(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Straight-through the **whole** quantiser, not just ``round``.

    The paper's Eq. 15 shows that STE-ing ``round`` alone leaves a ``1/|x|``
    factor in the gradient that oscillates around zero and collapses the fit;
    Eq. 16 redefines dQ/dv = 1 for the composite instead. Same fix, same reason.
    """
    return v + (q - v).detach()


# -- arm state --------------------------------------------------------------
# The dyadic kernel has to reach *every* MBENeuron, including the two banks
# inside a SignedMBENeuron and the per-bank neurons of a routed one, so the arm
# is installed by patching the class rather than by subclassing.
_ARM: dict = {"n_terms": None}


def _kernels_geom(self):
    """``_kernels`` in ratio form, optionally dyadic.

    Returns the same ``(d, r, vth)`` triple as :meth:`MBENeuron._kernels`. With
    ``n_terms=None`` this is the original up to float rounding: ``exp(-t*dt/tau)``
    and ``exp(-dt/tau)**t`` are the same number computed two ways.
    """
    cfg = self.cfg
    dtype, device = self.w.dtype, self.w.device
    t = torch.arange(cfg.n_steps, device=device, dtype=dtype).unsqueeze(1)  # (T,1)

    alpha_v = self.log_alpha_v.exp()                        # (N,)

    if not cfg.decay:                                       # ablation path
        ones = torch.ones(cfg.n_steps, cfg.n_basis, device=device, dtype=dtype)
        return ones, alpha_v * ones, alpha_v * ones

    dt = self.log_dt.exp()
    n_terms = _ARM["n_terms"]

    def ratio(log_tau):
        rho = torch.exp(-dt / log_tau.exp())                # (N,) in (0,1]
        if n_terms is None:
            return rho
        q = pot_quantise(rho, n_terms).clamp(2.0 ** RHO_EXP[0], 1.0)
        return _ste(rho, q)

    d = ratio(self.log_tau_d) ** t                          # (T,N)
    r = alpha_v * (ratio(self.log_tau_r) ** t)
    vth = alpha_v * (ratio(self.log_tau_vth) ** t)
    return d, r, vth


@contextmanager
def arm(n_terms: int | None, patched: bool):
    """Install an arm for the duration of a fit, then restore the class."""
    orig = MBENeuron._kernels
    _ARM["n_terms"] = n_terms
    if patched:
        MBENeuron._kernels = _kernels_geom
    try:
        yield
    finally:
        MBENeuron._kernels = orig
        _ARM["n_terms"] = None


# -- readout weights (reported, not fitted under the constraint) ------------

@torch.no_grad()
def mse_with_dyadic_w(model, x, y, n_terms: int) -> float:
    """MSE after quantising every ``w`` to dyadic, **without refitting**.

    This is an *upper bound* on what a shift-only readout costs: the honest test
    is STE-in-the-loop, which this script does not run. Reported so the number
    exists with its caveat attached rather than being guessed later.

    ``requires_grad`` is the discriminator, not the class: a ``SignedMBENeuron``
    holds the joint readout on the *wrapper* and freezes ``pos.w``/``neg.w``
    (``signed.py:44``), which ``features()`` never reads. Quantising by class
    would touch only those two dead tensors and report "no damage" for every
    signed fit -- which is what the first run of this script did.
    """
    saved = [(m, m.w.detach().clone()) for m in model.modules()
             if isinstance(getattr(m, "w", None), torch.nn.Parameter)
             and m.w.requires_grad]
    try:
        for m, w in saved:
            m.w.copy_(pot_quantise(w, n_terms, W_EXP))
        return float(torch.nn.functional.mse_loss(model(x), y))
    finally:
        for m, w in saved:
            m.w.copy_(w)


# -- one cell ---------------------------------------------------------------

def build(name: str, n_basis: int):
    """The same model choice ``reproduce_table10.py`` makes.

    GELU is non-monotonic, so it needs the polarity-split neuron; the other four
    are monotone on their calibrated domains and use the plain one. Changing this
    per arm would make the arms incomparable, so it is a function of ``name`` only.
    """
    if name == "gelu":
        npos = (n_basis + 1) // 2
        nneg = n_basis // 2 if n_basis > 1 else 1
        return functions.make_signed(name, n_pos=npos, n_neg=nneg, pivot=0.0,
                                     n_steps=T, decay=True)
    return MBENeuron(functions.make_config(name, n_basis=n_basis, n_steps=T,
                                           decay=True))


def run_cell(name, n_basis, arm_name, n_terms, patched, seed, x, y):
    t0 = time.time()
    torch.manual_seed(seed)
    with arm(n_terms, patched):
        model = build(name, n_basis)
        res = fit_model(model, x, y, seed=seed)
        wq = {f"mse_wq{k}": mse_with_dyadic_w(model, x, y, k) for k in (2, 3)}
    return dict(fn=name, n_basis=n_basis, arm=arm_name, seed=seed,
                mse=res.mse, firing_rate=res.firing_rate,
                num_params=res.num_params, fit_s=round(time.time() - t0, 1),
                **wq)


ARMS = [("base", None, False), ("geom", None, True),
        ("rho1", 1, True), ("rho2", 2, True), ("rho3", 3, True)]


# -- gate judgement ---------------------------------------------------------
# The thresholds are idea-003's, transcribed. They live in code so the verdict is
# not re-derived by eye, and so widening a band afterwards shows up as a diff.
R_TOL = 2.0        # R1/R3: median(rho{2,3}) must stay within this x median(base)
R2_TOL = 1.2       # R2: rho1 this close to base everywhere => single-PoT suffices


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def report(path: str) -> None:
    """Apply idea-003's R0/R1/R2/R3 gates to a record and print the verdict."""
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)[0]["rows"]

    cells: dict = {}
    for r in rows:
        cells.setdefault((r["fn"], r["n_basis"]), {}).setdefault(
            r["arm"], []).append(r["mse"])

    hdr = f"{'cell':18s} {'base':>10s} {'spread':>7s} " + \
          " ".join(f"{a:>10s}" for a in ("geom", "rho1", "rho2", "rho3"))
    print(hdr)
    print("-" * len(hdr))

    r0, r1, r3, r2_all = [], [], [], True
    for (fn, n), arms in sorted(cells.items()):
        base = _median(arms["base"])
        spread = max(arms["base"]) / min(arms["base"])
        line = f"{fn + ' N=' + str(n):18s} {base:10.3e} {spread:7.2f}"
        for a in ("geom", "rho1", "rho2", "rho3"):
            if a not in arms:
                line += f"{'-':>11s}"
                continue
            ratio = _median(arms[a]) / base
            line += f"  {ratio:8.2f}x"
            if a == "geom" and not (1 / spread <= ratio <= spread):
                r0.append((fn, n, ratio, spread))
            if a == "rho1" and ratio > R2_TOL:
                r2_all = False
            if a == "rho2" and ratio > R_TOL:
                r1.append((fn, n, ratio))
            if a == "rho3" and ratio > R_TOL:
                r3.append((fn, n, ratio))
        print(line)

    print("\n-- gates (idea-003) --")
    print(f"R0 geom outside base spread : {'FAIL ' + str(r0) if r0 else 'pass'}")
    print(f"R1 rho2 > {R_TOL}x base        : "
          f"{'REFUTED ' + str(r1) if r1 else 'pass'}")
    print(f"R2 rho1 <= {R2_TOL}x everywhere : "
          f"{'HALF-HYPOTHESIS VOID' if r2_all else 'pass (single PoT is not enough)'}")
    print(f"R3 rho3 > {R_TOL}x base        : "
          f"{'REFUTED ' + str(r3) if r3 else 'pass'}")
    verdict = "supported" if not r0 and not r1 else "see gates above"
    print(f"\nverdict: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fns", nargs="+", default=FNS)
    ap.add_argument("--n", nargs="+", type=int, default=N_LIST)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--arms", nargs="+", default=[a[0] for a in ARMS])
    ap.add_argument("--out", default="results/e22_dyadic_gate.json")
    ap.add_argument("--report", metavar="RECORD",
                    help="judge an existing record instead of running")
    args = ap.parse_args()

    if args.report:
        return report(args.report)

    arms = [a for a in ARMS if a[0] in args.arms]
    rows = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_all = time.time()

    for name in args.fns:
        for n_basis in args.n:
            for seed in args.seeds:
                x, y, _ = functions.sample(name, m=M, seed=seed)
                for arm_name, n_terms, patched in arms:
                    row = run_cell(name, n_basis, arm_name, n_terms, patched,
                                   seed, x, y)
                    rows.append(row)
                    print(f"{name:9s} N={n_basis} {arm_name:5s} s{seed}  "
                          f"MSE={row['mse']:.3e}  "
                          f"wq2={row['mse_wq2']:.3e}  {row['fit_s']:.0f}s",
                          flush=True)

    record = [dict(
        tag="e22-dyadic-gate", backend="mbe", model="function-level",
        # Not a conversion build: there is no ConvertConfig, no PPL, no frozen
        # fingerprint. check_build.py reports "--" for this record by design, and
        # nothing here may be quoted beside the freeze build's numbers.
        convert_cfg=None, pasn_beta=None,
        stored_bytes=None, stored_params=None, n_primitives=None,
        n_steps=T, m_samples=M, seeds=args.seeds,
        rho_exp_range=list(RHO_EXP), w_exp_range=list(W_EXP),
        started=started, total_s=round(time.time() - t_all, 1),
        rows=rows,
    )]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {args.out}  ({len(rows)} rows, "
          f"{round(time.time() - t_all, 1)}s)")


if __name__ == "__main__":
    main()
