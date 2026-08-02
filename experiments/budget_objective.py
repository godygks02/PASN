"""Does the budget rule minimise the right thing?

``rule_budget`` picks the smallest ``(N, T)`` that delivers the required bits.
The claim it underwrites -- "we solve the per-range budget rather than hand-tuning
it" -- is only as good as the objective it solves for, and the objective is
implicitly ``(N, T)``, not spikes.

Those come apart. A bank with one basis has to fire on most timesteps to build
its value; a bank with more bases can reach the same error while each fires
rarely. Table XI made that concrete: on ``1/x`` the paper's N=8 fires 3.74% for
4.79 spikes/element where our small N fires 57% for 18.25 -- **more bases, fewer
spikes**, the opposite of the direction the rule searches.

This sweeps ``(N, T)`` for each primitive the pipeline actually builds, fits every
combination, and asks: among the settings that reach *at least* the accuracy the
rule's own choice reaches, which emits the fewest spikes? The ratio is what the
rule leaves on the table.

The rule's constants were measured over 463 (bank, target) pairs on **five
activation functions** (see ``rule_budget``). ``exp2`` / ``inv`` / ``invsqrt`` /
``identity`` -- the softmax and LayerNorm primitives, which carry ~97% of a
converted GPT-2's spikes -- were **not in that calibration set**. That is the
hypothesis under test.

    python experiments/budget_objective.py
    python experiments/budget_objective.py --epochs 300 --json results/budget_obj.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.mbe_pasn import T_GRID, _fit_bank, _grid, bit_law, rule_budget  # noqa: E402
from mbe.metrics import neuron_cost  # noqa: E402

#: ``name -> (domain, readout_order, target_mse)``. Domains follow the paper's
#: G.1 for the shared primitives and our own calibration for the rest; the
#: readout order is what the conversion pipeline gives each one.
CASES = {
    "exp2":     ((0.0, 1.0), 2, 1e-5),      # softmax exponential, frac in [0,1)
    "inv":      ((0.5, 1.0), 2, 1e-5),      # softmax reciprocal, IEEE mantissa
    "invsqrt":  ((0.5, 2.0), 2, 1e-5),      # LayerNorm rsqrt
    "identity": ((0.0, 1.0), 1, 1e-5),      # FP-multiply identity (one bank)
    "gelu":     ((-6.0, 6.0), 2, 1e-5),     # activation -- the rule's home turf
}


def fit_one(name, lo, hi, N, T, order, epochs, seed):
    """Fit one (N, T) and return ``(mse, spikes per element)``."""
    fn, _ = functions.REGISTRY[name]
    bank, mse = _fit_bank(fn, lo, hi, N, T, epochs, seed, "cpu",
                          readout_order=order)
    x = _grid(lo, hi, 2048, 0.0)
    return float(mse), float(neuron_cost(bank, x)["spikes"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=4)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {}
    for name, (dom, order, tgt_mse) in CASES.items():
        lo, hi = dom
        fn, _ = functions.REGISTRY[name]
        y = fn(_grid(lo, hi, 2048, 0.0))
        delta = float(y.max() - y.min())
        ra, rc = bit_law(order, True)
        rN, rT = rule_budget(delta, tgt_mse, n_max=a.n_max, a=ra, c=rc)

        print(f"\n=== {name}  domain [{lo}, {hi}]  order {order}  "
              f"delta {delta:.3f} ===")
        print(f"rule picks (N={rN}, T={rT})")
        print(f"{'N':>2} {'T':>3} {'mse':>11} {'spikes/elem':>12}")

        grid, rule_row = {}, None
        for N in range(1, a.n_max + 1):
            for T in T_GRID:
                mse, spk = fit_one(name, lo, hi, N, T, order, a.epochs, a.seed)
                grid[(N, T)] = (mse, spk)
                mark = "  <- rule" if (N, T) == (rN, rT) else ""
                if (N, T) == (rN, rT):
                    rule_row = (mse, spk)
                print(f"{N:2d} {T:3d} {mse:11.3e} {spk:12.2f}{mark}")

        # Self-check: at fixed N, more timesteps cannot fit worse. If it does,
        # the fits are undertrained and any ranking off them is noise, not a
        # property of the budget. (Caught a 60-epoch run reporting a bogus 3.77x.)
        noisy = [(N, T_GRID[i], T_GRID[i + 1])
                 for N in range(1, a.n_max + 1)
                 for i in range(len(T_GRID) - 1)
                 if grid[(N, T_GRID[i + 1])][0] > 1.3 * grid[(N, T_GRID[i])][0]]
        if noisy:
            print(f"  !! {len(noisy)} (N,T) pairs fit WORSE with more timesteps "
                  f"-- undertrained, rankings below are unreliable. "
                  f"e.g. {noisy[:3]}")

        if rule_row is None:                          # rule fell off the grid
            print("  (rule choice not on the swept grid)")
            continue
        rule_mse, rule_spk = rule_row
        # Anyone at least as accurate as the rule's own choice is admissible.
        better = {k: v for k, v in grid.items() if v[0] <= rule_mse}
        best = min(better.items(), key=lambda kv: kv[1][1])
        (bN, bT), (bmse, bspk) = best
        out[name] = dict(rule=[rN, rT, rule_mse, rule_spk],
                         best=[bN, bT, bmse, bspk],
                         waste=rule_spk / max(bspk, 1e-9),
                         n_noisy=len(noisy),
                         grid={f"{N},{T}": list(v) for (N, T), v in grid.items()})
        verdict = ("optimal" if (bN, bT) == (rN, rT)
                   else f"{rule_spk / max(bspk, 1e-9):.2f}x more spikes than needed")
        print(f"  rule (N={rN},T={rT}) mse {rule_mse:.3e} spikes {rule_spk:.2f}")
        print(f"  best (N={bN},T={bT}) mse {bmse:.3e} spikes {bspk:.2f}   -> {verdict}")

    print("\n\n# summary -- spikes the rule leaves on the table, "
          "at no accuracy cost")
    print(f"{'primitive':<10} {'rule (N,T)':>11} {'best (N,T)':>11} {'waste':>8}")
    print("-" * 45)
    for k, v in out.items():
        print(f"{k:<10} {'(%d,%d)' % (v['rule'][0], v['rule'][1]):>11} "
              f"{'(%d,%d)' % (v['best'][0], v['best'][1]):>11} "
              f"{v['waste']:7.2f}x")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n[json] {a.json}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
