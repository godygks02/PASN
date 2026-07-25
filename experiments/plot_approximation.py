"""Plot MBE approximations of the target functions (Fig. 8 style).

Fits an MBE neuron to each function and saves a ground-truth-vs-approximation
figure with the achieved MSE. Saves to results/approx_<name>.png.

Usage:  python experiments/plot_approximation.py [name ...]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mbe import functions, fit_function, fit_model  # noqa: E402

# functions whose domain contains a non-monotonic near-zero bend -> signed neuron
_SIGNED = {"gelu"}


def plot_one(name, n_basis=8, m=4000, seed=0, outdir="results"):
    x, y, (lo, hi) = functions.sample(name, m=m, seed=seed)
    if name in _SIGNED:
        model = functions.make_signed(name, n_pos=n_basis // 2,
                                      n_neg=n_basis // 2, pivot=0.0)
        res = fit_model(model, x, y, seed=seed)
    else:
        cfg = functions.make_config(name, n_basis=n_basis, n_steps=16)
        model, res = fit_function(x, y, cfg, seed=seed)
    with torch.no_grad():
        pred = model(x)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, y, label="ground truth", lw=2)
    ax[0].plot(x, pred, "--", label="MBE", lw=1.5)
    ax[0].set_title(f"{name}  (N={n_basis}, T=16)  MSE={res.mse:.2e}")
    ax[0].legend(); ax[0].set_xlabel("x"); ax[0].set_ylabel("f(x)")
    ax[1].plot(x, (pred - y).abs())
    ax[1].set_title("|error|"); ax[1].set_xlabel("x"); ax[1].set_yscale("log")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"approx_{name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"{name:8s} N={n_basis}  MSE={res.mse:.2e}  -> {path}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["invsqrt", "inv", "exp2", "gelu"]
    outdir = os.path.join(os.path.dirname(__file__), "..", "results")
    for nm in names:
        plot_one(nm, outdir=outdir)
