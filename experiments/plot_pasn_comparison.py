"""Paper figures: MSE vs mean-spikes/input Pareto, MBE vs PASN.

Reads the JSON dumped by ``compare_pasn_mbe.py --json`` and draws, per function, a
log-log scatter of approximation MSE against mean spikes per input, coloured by
model family (global MBE / signed MBE / PASN). PASN points sitting lower-left =
better accuracy at fewer spikes.

Usage:
  python experiments/compare_pasn_mbe.py --json results/pareto_all.json
  python experiments/plot_pasn_comparison.py results/pareto_all.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"mbe": "#888888", "signed": "#e08214", "pasn": "#2166ac"}
LABELS = {"mbe": "global MBE", "signed": "signed MBE", "pasn": "PASN"}


def plot(json_path, out_path=None):
    with open(json_path) as f:
        rows = json.load(f)
    fns = []
    for r in rows:
        if r["fn"] not in fns:
            fns.append(r["fn"])

    ncol = min(3, len(fns))
    nrow = math.ceil(len(fns) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow),
                             squeeze=False)
    seen_family = set()
    for i, fn in enumerate(fns):
        ax = axes[i // ncol][i % ncol]
        for r in [x for x in rows if x["fn"] == fn]:
            fam = r["family"]
            lbl = LABELS[fam] if fam not in seen_family else None
            seen_family.add(fam)
            ax.scatter(r["spikes"], max(r["mse"], 1e-12), c=COLORS[fam], s=70,
                       edgecolors="white", linewidths=0.8, label=lbl, zorder=3)
            ax.annotate(r["model"].replace("PASN ", "").replace("SignedMBE ", "S")
                        .replace("MBE ", ""), (r["spikes"], max(r["mse"], 1e-12)),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points",
                        color=COLORS[fam])
        ax.set_yscale("log")
        ax.set_title(fn, fontsize=11, fontweight="bold")
        ax.set_xlabel("mean spikes / input")
        ax.set_ylabel("approximation MSE")
        ax.grid(True, which="both", alpha=0.25)
    for j in range(len(fns), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("MSE vs spikes/input  (lower-left = better)", y=1.06, fontsize=12)
    fig.tight_layout()
    out_path = out_path or json_path.replace(".json", ".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/pareto_all.json"
    plot(path, sys.argv[2] if len(sys.argv) > 2 else None)
