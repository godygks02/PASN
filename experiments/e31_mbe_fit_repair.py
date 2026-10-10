"""E31 -- does repairing the global-MBE fit remove the non-monotonicity E2 found?

E2 (`results/e2_global_fit_sweep.json`) asked whether the global MBE's network
failure is a wall or a search failure, and answered **search failure**: on the
real calibrated domains the error is non-monotone in `N` on every metric --
`ln_id` is best at `N=1` and `mse_grid` goes 0.0038 -> **72.16** at `N=2` -- and
10x the epochs does not fix it. A method whose accuracy falls when you give it
more capacity is not being fitted.

Three repairs went in before this script (all default-off, so every existing
record still reproduces):

1. **`MBEConfig.alpha_on_d`** -- Eq. (4) puts `alpha` on all three of
   `{tau_d, tau_r, tau_Vth}`; the original transcription applied it to `r` and
   `Vth` only and let `w` absorb it. Mathematically a reparameterisation (the
   readout is solved in closed form), so this arm is a **neutrality check**: if it
   moves the numbers, something else is wrong.
2. **`spiking_ops.FIT_RESTARTS`** -- the global path called `fit_model` once while
   the routed path (`mbe_pasn._fit_bank`) has always tried two alpha placements
   and kept the better. Same treatment for both sides.
3. **`spiking_ops.FIT_TAU_SEARCH`** -- also try `tau = dt/ln 2` (ratio 0.5, an
   exact SAR encoder). Measured 2026-08-19 on the `ln_id` domain: median
   **13-1132x** better at `N <= 6`, and consistently 15-2540x at `N in {1,2}`.

⚠️ **This makes the baseline stronger, so our own margins can only shrink.** The
operator table's 10/10 spike win and E30's 3-axis counts were all measured against
the single-shot MBE. That is the point: a comparison against a baseline we fit
worse than our own method is not a comparison.

⚠️ **Only the grid metrics are comparable to E2.** E2's `rel_sample` /
`rel_elem_median` were taken on the *real* GPT-2 calibration draw; reproducing
that needs a forward pass of the model. This script reuses E2's **recorded
domains** (so `rel_grid` / `mse_grid` line up exactly) and scores the sample
metrics on a uniform draw instead, which is a different distribution. They are
recorded but must not be compared across the two records.

Usage::
    python experiments/e31_mbe_fit_repair.py --smoke
    python experiments/e31_mbe_fit_repair.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from mbe import functions                                            # noqa: E402
from mbe import spiking_ops as so                                    # noqa: E402
from mbe import convert as cv                                        # noqa: E402

#: E2's measured domains, read off `results/e2_global_fit_sweep.json` so the grid
#: metrics are computed on exactly the same interval.
TARGETS = {
    "ln_id": dict(kind="identity", target="identity",
                  lo=0.0, hi=4271.956541992188),
    "act": dict(kind="activation", target="gelu_tanh",
                lo=-33.48661804199219, hi=75.84541320800781),
}
N_SWEEP = [1, 2, 4, 6, 8, 12]

#: (restarts, tau_search, alpha_on_d). `baseline` reproduces E2 exactly.
ARMS = {
    "baseline":   (1, False, False),
    "tau":        (1, True,  False),
    "restart3":   (3, False, False),
    "both":       (3, True,  False),
    "both_alpha": (3, True,  True),
}


def rel_err(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def build(t, n, n_steps, epochs, alpha_on_d):
    """One global-MBE build for a target, honouring the module-level fit knobs."""
    lo, hi = t["lo"], t["hi"]
    # `alpha_on_d` is an MBEConfig field, not a builder argument -- it rides the
    # ``**cfg_overrides`` channel the calibrate helpers forward into
    # ``make_config`` / ``make_signed``. The first run of this script passed it as
    # a builder kwarg, died on the last of five arms, and took the four already
    # computed with it, because the record was only written at the end. Hence the
    # per-arm checkpoint below.
    extra = dict(alpha_on_d=True) if alpha_on_d else {}
    if t["kind"] == "activation":
        # `build_activation` reads only min/max off the sample, so a two-point
        # tensor reproduces E2's domain (margin applied inside, as there).
        smp = torch.tensor([lo, hi], dtype=torch.float32)
        neuron = so.build_activation(t["target"], smp, n_basis=n,
                                     n_steps=n_steps, epochs=epochs, **extra)
        return (lambda z: neuron(z)), cv.Activation(t["target"])
    neuron = so.calibrate_identity(lo, hi, n_basis=n, n_steps=n_steps,
                                   epochs=epochs, use_cache=False, **extra)
    return (lambda z: neuron.reconstruct(z)), (lambda z: z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--m-eval", type=int, default=20000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e31_mbe_fit_repair.json"))
    a = ap.parse_args()

    ns = N_SWEEP[:3] if a.smoke else N_SWEEP
    arms = {k: v for k, v in ARMS.items() if not a.smoke or k in ("baseline", "both")}

    rows, t0 = [], time.time()
    for arm, (restarts, tau_search, alpha_on_d) in arms.items():
        so.FIT_RESTARTS, so.FIT_TAU_SEARCH = restarts, tau_search
        so.clear_cache()
        for key, t in TARGETS.items():
            grid = torch.linspace(t["lo"], t["hi"], a.m_eval)
            smp, _, _ = functions.sample(
                t["target"] if t["kind"] == "activation" else "identity",
                m=a.m_eval, seed=7, domain=(t["lo"], t["hi"]))
            for n in ns:
                ts = time.time()
                got_fn, ref_fn = build(t, n, a.n_steps, a.epochs, alpha_on_d)
                with torch.no_grad():
                    gg, rg = got_fn(grid), ref_fn(grid)
                    gs, rs = got_fn(smp), ref_fn(smp)
                rows.append(dict(
                    arm=arm, restarts=restarts, tau_search=tau_search,
                    alpha_on_d=alpha_on_d, which=key, target=t["target"],
                    n_basis=n, n_steps=a.n_steps, lo=t["lo"], hi=t["hi"],
                    rel_grid=rel_err(gg, rg), mse_grid=float((gg - rg).pow(2).mean()),
                    rel_sample_uniform=rel_err(gs, rs),
                    fit_s=round(time.time() - ts, 1)))
                r = rows[-1]
                print(f"  {arm:<11}{key:<7}N={n:<3} rel_grid={r['rel_grid']:.3e} "
                      f"mse_grid={r['mse_grid']:.4g}  ({r['fit_s']}s)", flush=True)
        # Checkpoint after every arm. Losing four completed arms to a crash in the
        # fifth is a bookkeeping failure, not an experimental one, and the project
        # rule is that even a failed run leaves a record.
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump([dict(tag="e31-mbe-fit-repair-partial", status="running",
                            arms_done=list(arms)[:list(arms).index(arm) + 1],
                            rows=rows)], f, indent=1, ensure_ascii=False)
    so.FIT_RESTARTS, so.FIT_TAU_SEARCH = 1, False

    # ------------------------------------------------------------- monotonicity
    def monotone(vals):
        return all(b <= a * 1.0000001 for a, b in zip(vals, vals[1:]))

    verdict = {}
    for arm in arms:
        for key in TARGETS:
            sel = sorted((r for r in rows if r["arm"] == arm and r["which"] == key),
                         key=lambda r: r["n_basis"])
            if len(sel) < 2:
                continue
            for metric in ("rel_grid", "mse_grid"):
                v = [r[metric] for r in sel]
                verdict[f"{arm}|{key}|{metric}"] = dict(
                    monotone=monotone(v), first=v[0], last=v[-1],
                    best=min(v), best_n=sel[v.index(min(v))]["n_basis"],
                    worst=max(v), spread=max(v) / max(min(v), 1e-30))

    print("\n" + "=" * 92)
    print("MONOTONICITY IN N -- the thing E2 said was broken")
    print("=" * 92)
    print(f"{'arm|target|metric':<34}{'monotone':>10}{'best@N':>9}{'best':>12}"
          f"{'worst':>12}{'spread':>11}")
    print("-" * 92)
    for k, v in verdict.items():
        print(f"{k:<34}{str(v['monotone']):>10}{v['best_n']:>9}{v['best']:>12.4g}"
              f"{v['worst']:>12.4g}{v['spread']:>10.4g}x")

    base = [v for k, v in verdict.items() if k.startswith("baseline|")]
    n_mono = {arm: sum(v["monotone"] for k, v in verdict.items()
                       if k.startswith(arm + "|")) for arm in arms}
    print("\n  monotone cells per arm:", n_mono)

    rec = [dict(
        tag="e31-mbe-fit-repair" + ("-smoke" if a.smoke else ""),
        experiment="E31", backend="mbe", model="global-MBE on E2's recorded domains",
        convert_cfg=None, pasn_beta=None,
        stored_bytes=None, stored_params=None, n_primitives=None,
        smoke=bool(a.smoke), status="ok",
        started=_dt.datetime.now().isoformat(timespec="seconds"),
        total_s=round(time.time() - t0, 1),
        repairs=dict(
            alpha_on_d="MBEConfig.alpha_on_d -- Eq.(4) puts alpha on d too",
            restarts="spiking_ops.FIT_RESTARTS -- keep-best over seeds",
            tau_search="spiking_ops.FIT_TAU_SEARCH -- also try tau=dt/ln2 (ratio 0.5)"),
        arms={k: dict(restarts=v[0], tau_search=v[1], alpha_on_d=v[2])
              for k, v in arms.items()},
        compare_to=dict(
            record="results/e2_global_fit_sweep.json",
            comparable=["rel_grid", "mse_grid"],
            not_comparable=["rel_sample_uniform"],
            why=("E2 scored the sample metrics on the real GPT-2 calibration draw; "
                 "this script uses a uniform draw over the same recorded domain. "
                 "The grid metrics use the identical interval and do line up.")),
        caveat=("repairing the fit strengthens the BASELINE. The operator table's "
                "10/10 spike win and E30's domination counts were measured against "
                "the single-shot MBE and must be re-measured if these repairs are "
                "adopted as defaults."),
        n_sweep=ns, epochs=a.epochs, rows=rows, verdict=verdict,
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {a.json}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
