"""Figures for ``experiments/op_pareto.py``.

Three views, each answering a different question:

``pareto``  -- accuracy against spikes and against stored bytes, per operator.
              The scatter is every swept build; the line is that arm's
              non-dominated front. Both axes are log: the arms differ by orders
              of magnitude on cost, and a linear axis hides the whole low-spike
              end where the comparison is actually decided.

``curves``  -- what the approximation looks like, MBE vs PASN, **at matched spike
              budget**. The pair is not chosen by hand: for each function the
              script takes the MBE front point and the PASN point whose metered
              spike count is closest to it, so the two curves cost the same
              energy and only the accuracy differs. A curve pair at unmatched
              budget is a picture of a budget, not of a method.

``tablex``   -- our MSE against the MBE paper's published Table X, per function
              and per basis count. The paper reports single operating points, so
              these are drawn as reference markers rather than as a swept arm.

Usage::

    python experiments/plot_op_pareto.py --json results/op_pareto.json
    python experiments/plot_op_pareto.py --what curves --funcs gelu exp2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe import spiking_ops as so  # noqa: E402
from mbe.metrics import spikes_per_input, neuron_params, storage_bytes  # noqa: E402

import op_pareto as OP  # noqa: E402

STYLE = {
    "mbe":         dict(color="#c0392b", marker="o",
                        label="global MBE (ours)"),
    "pasn":        dict(color="#1f6fb4", marker="s",
                        label="PASN uniform banks (routing only)"),
    "pasn_rule_T": dict(color="#e67e22", marker="v",
                        label="PASN rule $N_j$, T pinned=16"),
    "pasn_rule":   dict(color="#16a085", marker="^",
                        label="PASN rule $N_j$+$T_j$ (fully dynamic)"),
}
PAPER_STYLE = dict(color="#7f8c8d", marker="*", s=160, zorder=5,
                   label="MBE paper (published)")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _by_op(records):
    out = {}
    for r in records:
        out.setdefault(r["op"], []).append(r)
    return out


# --------------------------------------------------------------------------
# 1. Pareto
# --------------------------------------------------------------------------

def plot_pareto(res, out_path, cost_key="spikes", cost_label=None):
    ops = _by_op(res["records"])
    names = sorted(ops)
    ncol = min(3, len(names))
    nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow),
                             squeeze=False)
    cost_label = cost_label or {"spikes": "spikes per output element",
                                "bytes": "stored bytes"}[cost_key]

    for ax, name in zip(axes.flat, names):
        pts = ops[name]
        primary = pts[0]["primary"]
        for backend, st in STYLE.items():
            sel = [p for p in pts if p["backend"] == backend]
            if not sel:
                continue
            ax.scatter([p[cost_key] for p in sel], [p[primary] for p in sel],
                       s=22, alpha=0.32, color=st["color"], marker=st["marker"],
                       linewidths=0)
            f = OP.front(sel, cost_key, primary)
            ax.plot([p[cost_key] for p in f], [p[primary] for p in f],
                    color=st["color"], marker=st["marker"], ms=5.5, lw=1.8,
                    label=st["label"])
        _overlay_paper(ax, name, res, cost_key)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(cost_label)
        ax.set_ylabel(f"{primary}  (lower = better)")
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.legend(fontsize=7.5, framealpha=0.9)

    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Accuracy vs {cost_label}: isolated operators, same conditions",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _overlay_paper(ax, op_name, res, cost_key):
    """Table XI reference lines: what the paper's own primitives cost.

    Drawn as a **vertical band** on the spike axis, not as a point: Table XI
    gives a cost without an accuracy for the same build, so a marker at some
    height would be inventing the y-coordinate. The band says "the paper's
    primitive for this op sits at this cost", which is all it licenses.
    """
    if cost_key != "spikes":
        return
    rows = [v for v in res.get("paper", {}).get("table_xi", {}).values()
            if v["op"] == op_name]
    if not rows:
        return
    total = sum(r["spikes"] for r in rows)
    ax.axvline(total, color=PAPER_STYLE["color"], ls="--", lw=1.4,
               label=f"MBE paper Table XI ({len(rows)} prim, T=16)")


# --------------------------------------------------------------------------
# 2. Approximation curves at matched spike budget
# --------------------------------------------------------------------------

def _rebuild(rec, cfg, domain, fname):
    """Rebuild the exact neuron a record describes (builds are deterministic)."""
    b, t = rec["budget"], rec["T"]
    if rec["backend"] == "mbe":
        sa = so.build_activation(fname, torch.tensor(list(domain)), n_basis=int(b),
                                 n_steps=t, epochs=cfg["epochs"],
                                 seed=cfg["seed"], margin=0.0)
        return sa.neuron
    return OP.maker(rec["backend"], b, t, argparse.Namespace(**cfg))(
        fname, domain)


def _matched_pair(pts, primary, arm="pasn_rule"):
    """(MBE point, PASN point) closest in metered spikes, over the two fronts.

    ``arm`` defaults to the **fully dynamic** rule, so the picture shows PASN as
    the method actually is; it falls back to the uniform arm when the dynamic one
    was not swept.
    """
    m = OP.front([p for p in pts if p["backend"] == "mbe"], "spikes", primary)
    sel = [p for p in pts if p["backend"] == arm]
    if not sel:
        sel = [p for p in pts if p["backend"] == "pasn"]
    p = OP.front(sel, "spikes", primary)
    if not m or not p:
        return None, None
    return min(((a, b) for a in m for b in p),
               key=lambda ab: abs(math.log(ab[0]["spikes"] / ab[1]["spikes"])))


def plot_curves(res, out_path, funcs=None, arm="pasn_rule"):
    cfg = res["meta"]
    ops = _by_op(res["records"])
    acts = [(n, pts) for n, pts in sorted(ops.items())
            if n.startswith("activation:")]
    if funcs:
        acts = [(n, p) for n, p in acts if n.split(":", 1)[1] in funcs]
    if not acts:
        print("no activation records to plot")
        return

    n = len(acts)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 6.6), squeeze=False,
                             gridspec_kw=dict(height_ratios=[2.1, 1]))

    for col, (op_name, pts) in enumerate(acts):
        fname = op_name.split(":", 1)[1]
        primary = pts[0]["primary"]
        rec_m, rec_p = _matched_pair(pts, primary, arm)
        if rec_m is None:
            continue
        fn, domain = functions.REGISTRY[fname]
        lo, hi = domain
        x = torch.linspace(lo, hi, 3000)
        y = fn(x)
        top, bot = axes[0][col], axes[1][col]
        top.plot(x, y, color="k", lw=2.2, label="exact", zorder=1)

        for rec, key in ((rec_m, "mbe"), (rec_p, rec_p["backend"])):
            neuron = _rebuild(rec, cfg, domain, fname)
            with torch.no_grad():
                yh = neuron(x)
            st = STYLE[key]
            top.plot(x, yh, color=st["color"], lw=1.5, alpha=0.9,
                     label=f"{st['label']}\n{rec['spikes']:.1f} sp, "
                           f"{rec['params']} par, {primary}={rec[primary]:.1e}")
            bot.semilogy(x, (yh - y).abs().clamp(min=1e-12), color=st["color"],
                         lw=1.2)

        top.set_title(f"{fname}   (matched budget: "
                      f"{rec_m['spikes']:.1f} vs {rec_p['spikes']:.1f} spikes)",
                      fontsize=10)
        top.legend(fontsize=7, framealpha=0.9)
        top.grid(alpha=0.25, lw=0.4)
        bot.set_xlabel("x")
        bot.set_ylabel("|error|")
        bot.grid(alpha=0.25, which="both", lw=0.4)
        if col == 0:
            top.set_ylabel("f(x)")

    fig.suptitle("Approximation at matched spike budget: global MBE vs PASN",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# 3. Table X overlay (MSE vs N, ours vs the paper)
# --------------------------------------------------------------------------

def plot_tablex(res, out_path):
    """Our MSE against the paper's Table X, per function.

    The x-axis is the arm's own budget knob -- ``N`` for MBE and the paper,
    ``n_local`` per bank for PASN -- so the curves are **not** aligned on the
    x-axis and must not be read as "PASN at N=2 beats MBE at N=2". The
    comparable reading is vertical position at equal *cost*, which is what the
    Pareto figure is for; this one exists to place our MBE arm against the
    paper's published numbers and confirm the reimplementation is in the right
    regime.
    """
    ops = _by_op(res["records"])
    table_x = res.get("paper", {}).get("table_x", {})
    names = [n for n in sorted(ops) if n.startswith("activation:")
             and n.split(":", 1)[1] in table_x]
    if not names:
        print("no functions with a Table X entry in this run")
        return

    ncol = min(4, len(names))
    nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.8 * nrow),
                             squeeze=False)
    for ax, name in zip(axes.flat, names):
        fname = name.split(":", 1)[1]
        pts = ops[name]
        # compare at the paper's T=16; the fully dynamic arm has T=None (it
        # solves T_j per bank) and has no place on a fixed-T axis anyway
        t_max = max(p["T"] for p in pts if p["T"] is not None)
        for backend in ("mbe", "pasn"):      # the two arms whose knob *is* a
            st = STYLE[backend]              # basis count, so an x-axis exists
            sel = sorted((p for p in pts
                          if p["backend"] == backend and p["T"] == t_max),
                         key=lambda p: p["budget"])
            if not sel:
                continue
            ax.plot([p["budget"] for p in sel], [p["mse"] for p in sel],
                    color=st["color"], marker=st["marker"], lw=1.6, ms=5,
                    label=st["label"])
        pap = table_x[fname]
        ax.scatter([int(k) for k in pap], list(pap.values()), **PAPER_STYLE)
        ax.set_yscale("log")
        ax.set_title(f"{fname}  (T={t_max})", fontsize=11)
        ax.set_xlabel("basis count (arm's own knob)")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.legend(fontsize=7.5)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("MSE vs basis count, against the MBE paper's Table X",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/op_pareto.json")
    ap.add_argument("--what", nargs="+",
                    default=["pareto", "curves", "tablex"])
    ap.add_argument("--funcs", nargs="+", default=None)
    ap.add_argument("--curve-arm", default="pasn_rule",
                    help="which PASN arm the curve figure draws against MBE")
    ap.add_argument("--prefix", default="results/op_pareto")
    args = ap.parse_args()

    res = _load(args.json)
    torch.set_num_threads(os.cpu_count() or 4)

    if "pareto" in args.what:
        plot_pareto(res, f"{args.prefix}_spikes.png", "spikes")
        plot_pareto(res, f"{args.prefix}_bytes.png", "bytes")
    if "curves" in args.what:
        plot_curves(res, f"{args.prefix}_curves.png", args.funcs,
                    args.curve_arm)
    if "tablex" in args.what:
        plot_tablex(res, f"{args.prefix}_tablex.png")


if __name__ == "__main__":
    main()
