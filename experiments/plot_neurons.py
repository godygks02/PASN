"""Figure for the three-way neuron comparison (MBE / MBE-PASN / PASN).

Per function: MSE (y, log) vs mean spikes/input (x, log); point *area* encodes
stored parameters (memory); colour encodes neuron family. Lower-left + smaller =
better on all three axes.

Usage:
  python experiments/compare_neurons.py --json results/neuron_compare.json
  python experiments/plot_neurons.py results/neuron_compare.json
"""
from __future__ import annotations

import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"mbe": "#9e9e9e", "signed": "#e08214",
          "mbe_pasn": "#2166ac", "pasn": "#1a9850"}
LABELS = {"mbe": "MBE (global)", "signed": "signed MBE",
          "mbe_pasn": "MBE-PASN", "pasn": "PASN (SAR)"}


def plot(json_path, out_path=None):
    rows = json.load(open(json_path))
    fns = []
    for r in rows:
        if r["fn"] not in fns:
            fns.append(r["fn"])
    ncol = min(3, len(fns))
    nrow = math.ceil(len(fns) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.9 * nrow),
                             squeeze=False)
    seen = set()
    for i, fn in enumerate(fns):
        ax = axes[i // ncol][i % ncol]
        for r in [x for x in rows if x["fn"] == fn]:
            fam = r["family"]
            lbl = LABELS[fam] if fam not in seen else None
            seen.add(fam)
            ax.scatter(max(r["spikes"], 1e-3), max(r["mse"], 1e-12),
                       s=20 + 3.0 * r["params"], c=COLORS[fam], alpha=0.75,
                       edgecolors="white", linewidths=0.8, label=lbl, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(fn, fontsize=11, fontweight="bold")
        ax.set_xlabel("mean spikes / input  (energy)")
        ax.set_ylabel("approximation MSE")
        ax.grid(True, which="both", alpha=0.25)
    for j in range(len(fns), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Accuracy vs energy vs memory  (lower-left better; "
                 "point area = stored params)", y=1.06, fontsize=12)
    fig.tight_layout()
    out_path = out_path or json_path.replace(".json", ".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/neuron_compare.json"
    plot(path, sys.argv[2] if len(sys.argv) > 2 else None)
