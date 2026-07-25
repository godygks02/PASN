"""Reproduce Table X (appendix G.3): MSE of MBE function approximation vs the
number of bases N, with and without the exponential-decay mechanism.

Paper targets (log MSE, T=16):
              N=1      N=2      N=4      N=6      N=8      N=8 (no decay)
  GELU     7.1e-3   4.1e-3   2.3e-4   1.7e-4   1.0e-4    2.8e-1
  invsqrt  1.4e-3   1.2e-3   3.1e-4   1.0e-4   4.9e-5    2.8e-3
  inv      8.6e-3   2.1e-4   1.1e-3   2.2e-3   4.4e-5    4.5e-3
  2^x      8.9e-4   4.5e-4   4.0e-4   2.4e-4   5.3e-5    9.5e-3

Usage:  python experiments/reproduce_table10.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions, fit_function, fit_model  # noqa: E402

PAPER = {
    "gelu":    {1: 7.1e-3, 2: 4.1e-3, 4: 2.3e-4, 6: 1.7e-4, 8: 1.0e-4, "8*": 2.8e-1},
    "invsqrt": {1: 1.4e-3, 2: 1.2e-3, 4: 3.1e-4, 6: 1.0e-4, 8: 4.9e-5, "8*": 2.8e-3},
    "inv":     {1: 8.6e-3, 2: 2.1e-4, 4: 1.1e-3, 6: 2.2e-3, 8: 4.4e-5, "8*": 4.5e-3},
    "exp2":    {1: 8.9e-4, 2: 4.5e-4, 4: 4.0e-4, 6: 2.4e-4, 8: 5.3e-5, "8*": 9.5e-3},
}
# 2^x is called "exp2" here; label mapping for display
LABEL = {"gelu": "GELU", "invsqrt": "invsqrt", "inv": "inv", "exp2": "2^x"}

N_LIST = [1, 2, 4, 6, 8]
M = 6000
SEED = 0


def run(m_samples=M, seed=SEED):
    results = {}
    for name in ["gelu", "invsqrt", "inv", "exp2"]:
        x, y, _ = functions.sample(name, m=m_samples, seed=seed)
        results[name] = {}
        # GELU is non-monotonic (negative dip): use the signed (polarity-split)
        # neuron with a balanced split. The three monotone primitives use the
        # plain MBE neuron.
        signed = (name == "gelu")

        def fit_N(N, decay):
            if signed:
                npos = (N + 1) // 2
                nneg = N // 2 if N > 1 else 1  # N=1 cannot split; use 1+1
                model = functions.make_signed(name, n_pos=npos, n_neg=nneg,
                                              pivot=0.0, n_steps=16, decay=decay)
                return fit_model(model, x, y, seed=seed).mse
            cfg = functions.make_config(name, n_basis=N, n_steps=16, decay=decay)
            return fit_function(x, y, cfg, seed=seed)[1].mse

        for N in N_LIST:
            t0 = time.time()
            mse = fit_N(N, decay=True)
            results[name][str(N)] = mse
            print(f"{LABEL[name]:8s} N={N} decay=T  MSE={mse:.2e}  "
                  f"(paper {PAPER[name][N]:.1e})  {time.time()-t0:.0f}s")
        t0 = time.time()
        mse = fit_N(8, decay=False)
        results[name]["8*"] = mse
        print(f"{LABEL[name]:8s} N=8 decay=F  MSE={mse:.2e}  "
              f"(paper {PAPER[name]['8*']:.1e})  {time.time()-t0:.0f}s")
        print()
    return results


def render_markdown(results):
    lines = ["# Table X reproduction (MBE MSE vs N)\n",
             "MSE on M samples, T=16. `8*` = N=8 without decay (ablation).",
             "Each cell: **ours** (paper). GELU uses the signed (polarity-split) "
             "neuron with a balanced split; the others use the plain MBE neuron. "
             "Readout (w, bias) is solved in closed form during fitting.\n",
             "| Func | N=1 | N=2 | N=4 | N=6 | N=8 | N=8* |",
             "|---|---|---|---|---|---|---|"]
    for name in ["gelu", "invsqrt", "inv", "exp2"]:
        row = [LABEL[name]]
        for key in ["1", "2", "4", "6", "8", "8*"]:
            pk = 8 if key == "8" else ("8*" if key == "8*" else int(key))
            ours = results[name][key]
            paper = PAPER[name][pk] if key != "8*" else PAPER[name]["8*"]
            row.append(f"{ours:.1e} ({paper:.1e})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    t0 = time.time()
    results = run()
    outdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "table10.json"), "w") as f:
        json.dump(results, f, indent=2)
    md = render_markdown(results)
    with open(os.path.join(outdir, "table10.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"total {time.time()-t0:.0f}s -> results/table10.json, results/table10.md")
