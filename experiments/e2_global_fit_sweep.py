"""E2 -- is the global-MBE failure a fit that cannot be found, or one we fail to find?

`experiments/layer_error_profile.py --backend mbe` established what the failure is
*not*: clean-input error tracks in-situ error (so it is not upstream drift) and
essentially nothing leaves the fitted range (so it is not domain escape). What is
left is the fit itself, on the domains the real model calibrates to:

    LayerNorm identity   [-3932, 843]    width 4775
    GELU                 [-33.8, 76.2]   width 110

But the network profile also came out **non-monotone in N** -- the `N_act=4,
N_ln=6` arm is several times *more* accurate per site than `N_act=6, N_ln=8`.
More capacity making a fit worse is not a property of the method; it is a sign
the optimiser is not finding the fit. That distinction decides whether C9 ("a
global multi-basis neuron has a structural problem on wide ranges") is supported
or whether we merely have a weak reimplementation, and the two call for opposite
things in the paper.

This script isolates it. No network, no conversion: take the **real calibrated
samples** off a GPT-2 forward pass, then fit the global MBE primitives directly
over a sweep of `N` and read the error.

  monotone in N   -> the fit is being found; the wide range is the wall (C9 holds)
  non-monotone    -> our fitting is unstable and no structural claim can rest on it

Both readings are reportable; the point is to stop confounding them.

Usage::  python experiments/e2_global_fit_sweep.py
         python experiments/e2_global_fit_sweep.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe import spiking_ops as so  # noqa: E402
from mbe.gpt2_convert import make_spikable  # noqa: E402

from gpt2_wikitext import load_wikitext_ids  # noqa: E402


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative Frobenius error -- the same metric the layer profile reports."""
    denom = ref.norm()
    return float(got.norm()) if denom == 0 else float((got - ref).norm() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--calib-offset", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--n-sweep", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    ap.add_argument("--m-eval", type=int, default=20000,
                    help="evaluation points drawn over the fitted range")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", default="results/e2_global_fit_sweep.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    t0 = time.perf_counter()

    # ------------------------------------------------------------- calibrate
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    cids = load_wikitext_ids(tok, "train", True)[: 64 * args.block]
    start = args.calib_offset * args.block
    n_blk = 2 if args.smoke else 8
    calib = [cids[i:i + args.block].unsqueeze(0)
             for i in range(start, start + n_blk * args.block, args.block)]
    make_spikable(model)
    rec = cv.calibrate(model, calib)
    print(f"[calibrate] {len(rec.kinds)} sites, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    # The pooled fit slots are what a shared prototype actually sees -- the same
    # object the converter hands the builder, so the domains match the profile.
    shared = cv._shared_fit_slots(model, rec)

    targets = {}
    for name, kind in rec.kinds.items():
        if kind == "activation" and "act" not in targets:
            s = (shared.get(name) or rec.ranges[name])[0]
            targets["act"] = dict(kind="activation", target=model.get_submodule(name).kind,
                                  sample=s.sample, lo=float(s.lo), hi=float(s.hi))
        if kind == "layernorm" and "ln_id" not in targets:
            s = (shared.get(name) or rec.ranges[name])[0]
            D = model.get_submodule(name).normalized_shape[0]
            flat = s.sample[: (s.sample.numel() // D) * D].reshape(-1, D)
            dev = flat - flat.mean(dim=-1, keepdim=True)
            # The LayerNorm identity is fitted on the deviation magnitude, which
            # is the quantity whose range the converter pads and hands over.
            hi = float(dev.abs().max()) * 1.1 + 1e-3
            targets["ln_id"] = dict(kind="identity", target="identity",
                                    sample=dev.abs().reshape(-1), lo=0.0, hi=hi)

    for k, t in targets.items():
        print(f"  {k:6s} {t['target']:10s} domain [{t['lo']:.4g}, {t['hi']:.4g}] "
              f"width {t['hi'] - t['lo']:.4g}  ({t['sample'].numel()} samples)")

    # ----------------------------------------------------------------- sweep
    # ⚠ Two evaluation sets, deliberately. A uniform grid over a range this wide
    # is dominated by its large end, so a Frobenius error taken there is the
    # *absolute* view -- exactly the target P0.4 Block C measured as far worse
    # downstream (+1.32% absolute vs -0.23% relative). The network feeds a
    # distribution concentrated near zero. Reading only the grid would reproduce
    # the trap this project already documented.
    rows = []
    for key, t in targets.items():
        smp = t["sample"].reshape(-1)
        g = torch.Generator().manual_seed(0)
        if smp.numel() > args.m_eval:
            smp = smp[torch.randperm(smp.numel(), generator=g)[: args.m_eval]]
        grid = torch.linspace(t["lo"], t["hi"], args.m_eval)
        for n in args.n_sweep:
            ts = time.perf_counter()
            if t["kind"] == "activation":
                neuron = so.build_activation(
                    t["target"], t["sample"], n_basis=n, n_steps=args.n_steps,
                    epochs=args.epochs, device=device)
                fn = cv.Activation(t["target"])
                with torch.no_grad():
                    got_g, ref_g = neuron(grid), fn(grid)
                    got_s, ref_s = neuron(smp), fn(smp)
            else:
                neuron = so.calibrate_identity(
                    t["lo"], t["hi"], n_basis=n, n_steps=args.n_steps,
                    epochs=args.epochs, device=device)
                with torch.no_grad():
                    got_g, ref_g = neuron.reconstruct(grid), grid
                    got_s, ref_s = neuron.reconstruct(smp), smp
            # Scale-free per-element view: a wide-range global fit can be fine in
            # Frobenius terms and hopeless where the mass is.
            per = ((got_s - ref_s).abs()
                   / ref_s.abs().clamp_min(1e-12)).median().item()
            row = dict(which=key, target=t["target"], n_basis=n,
                       n_steps=args.n_steps, lo=t["lo"], hi=t["hi"],
                       rel_grid=rel_err(got_g, ref_g),
                       rel_sample=rel_err(got_s, ref_s),
                       rel_elem_median=per,
                       mse_grid=float(((got_g - ref_g) ** 2).mean()),
                       fit_s=time.perf_counter() - ts)
            rows.append(row)
            print(f"  {key:6s} N={n:<3} grid={row['rel_grid']:.3e}  "
                  f"sample={row['rel_sample']:.3e}  "
                  f"per-elem-med={per:.3e}  ({row['fit_s']:.0f}s)", flush=True)

    # --------------------------------------------------------------- verdict
    print("\n=== monotone in N? (per metric) ===")
    verdict = {}
    for key in targets:
        v = sorted([r for r in rows if r["which"] == key],
                   key=lambda r: r["n_basis"])
        verdict[key] = {}
        for metric in ("rel_grid", "rel_sample", "rel_elem_median"):
            errs = [r[metric] for r in v]
            mono = all(b <= a * 1.05 for a, b in zip(errs, errs[1:]))
            best = min(v, key=lambda r: r[metric])
            verdict[key][metric] = dict(monotone=mono, best_n=best["n_basis"],
                                        best=best[metric], first=errs[0],
                                        last=errs[-1])
            print(f"  {key:6s} {metric:16s} "
                  f"{'MONOTONE    ' if mono else 'NON-MONOTONE'}  "
                  f"best N={best['n_basis']:<3} {best[metric]:.3e}   "
                  f"(N={v[0]['n_basis']} {errs[0]:.3e} -> "
                  f"N={v[-1]['n_basis']} {errs[-1]:.3e})")

    out = dict(model=args.model, epochs=args.epochs, n_steps=args.n_steps,
               calib_offset=args.calib_offset, m_eval=args.m_eval,
               smoke=args.smoke, elapsed_s=time.perf_counter() - t0,
               torch=torch.__version__, rows=rows, verdict=verdict)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.json}  ({len(rows)} fits, "
          f"{out['elapsed_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
