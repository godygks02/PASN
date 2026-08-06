"""Markdown report for ``experiments/op_pareto.py``.

Turns the sweep JSON into the three tables the experiment exists to produce:

  1. **Best achievable accuracy** per operator per arm, with what it cost.
  2. **Iso-accuracy** -- at each accuracy the MBE front reaches, the cheapest
     PASN build that matches it, as a spike / byte / parameter ratio. This is
     the table that answers "same accuracy, what does it cost", and it is the
     only one where a ratio is meaningful.
  3. **Against the paper** -- our per-primitive spikes beside Table XI's, and our
     MSE beside Table X's, with the caveats that make those comparisons legal.

Ratios are printed as **MBE / PASN**, so >1 means PASN is cheaper.

Usage::  python experiments/op_pareto_report.py --json results/op_pareto.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import op_pareto as OP  # noqa: E402

ARM_LABEL = {"mbe": "global MBE (ours)",
             "pasn": "PASN uniform banks (routing only)",
             "pasn_rule_T": "PASN rule N_j, T pinned",
             "pasn_rule": "PASN rule N_j+T_j (fully dynamic)"}
ARM_ORDER = ("mbe", "pasn", "pasn_rule_T", "pasn_rule")


def _fmt(v, spec=".3g"):
    return "—" if v is None else format(v, spec)


def _med(v):
    return statistics.median(v) if v else None


def _tl(t):
    return "rule" if t is None else t


def best_table(res) -> str:
    """Best accuracy each arm reaches, and the build that reaches it."""
    lines = ["| operator | arm | best err | at spikes | bytes | params | "
             "build | per-bank alloc |", "|---|---|---|---|---|---|---|---|"]
    by_op = {}
    for r in res["records"]:
        by_op.setdefault(r["op"], []).append(r)
    for op in sorted(by_op):
        pts = by_op[op]
        primary = pts[0]["primary"]
        for arm in ARM_ORDER:
            sel = [p for p in pts if p["backend"] == arm]
            if not sel:
                continue
            b = min(sel, key=lambda p: p[primary])
            alloc = ("—" if b.get("n_mean") is None else
                     f"⟨N⟩={b['n_mean']:.2f} (×{b['n_span']:.0f}), "
                     f"⟨T⟩={b['t_mean']:.1f} (×{b['t_span']:.0f})")
            lines.append(
                f"| {op} | {ARM_LABEL[arm]} | {b[primary]:.3e} ({primary}) | "
                f"{b['spikes']:.2f} | {b['bytes']} | {b['params']} | "
                f"b={b['budget']:g}, T={_tl(b['T'])} | {alloc} |")
    return "\n".join(lines)


def alloc_table(res) -> str:
    """What each half of the per-bank allocation buys, at matched accuracy.

    The chain is uniform banks -> ``+N_j`` -> ``+T_j``. If the ``+T_j`` step is
    ~1x it means the rule wanted the pinned ``T`` anyway on this operator, which
    is a result about the operator, not a failure of the arm.
    """
    lines = ["| operator | step | spikes ref/cand | bytes ref/cand | "
             "matched |", "|---|---|---|---|---|"]
    label = {"pasn->pasn_rule_T": "routing → + allocate $N_j$",
             "pasn_rule_T->pasn_rule": "+ allocate $T_j$ (fully dynamic)",
             "pasn->pasn_rule": "routing → full allocation"}
    for op, s in sorted(res["summary"].items()):
        for step, rows in s.get("alloc_decomp", {}).items():
            ok = [r for r in rows if r["matched"]]
            if not ok:
                lines.append(f"| {op} | {label.get(step, step)} | — | — | "
                             f"0/{len(rows)} |")
                continue
            lines.append(
                f"| {op} | {label.get(step, step)} | "
                f"**{_med([r['spikes_ratio'] for r in ok]):.2f}x** | "
                f"{_med([r['bytes_ratio'] for r in ok]):.2f}x | "
                f"{len(ok)}/{len(rows)} |")
    return "\n".join(lines)


def iso_table(res, arm="pasn_rule") -> str:
    """Cheapest build of ``arm`` matching each MBE front accuracy."""
    out = []
    for op, s in sorted(res["summary"].items()):
        primary = s["primary"]
        rows = s["iso_accuracy"].get(arm, [])
        if not rows:
            continue
        out.append(f"\n**{op}** (accuracy axis: `{primary}`)\n")
        out.append("| MBE front point | MBE err | PASN build | PASN err | "
                   "spikes MBE/PASN | bytes MBE/PASN | params MBE/PASN |")
        out.append("|---|---|---|---|---|---|---|")
        for r in rows:
            if not r["matched"]:
                out.append(f"| N={r['ref_budget']:g}, T={_tl(r['ref_T'])} | "
                           f"{r['ref_err']:.3e} | *unmatched* | — | — | — | — |")
                continue
            out.append(
                f"| N={r['ref_budget']:g}, T={_tl(r['ref_T'])} | "
                f"{r['ref_err']:.3e} | "
                f"r={r['cand_budget']:g}, T={_tl(r['cand_T'])} | "
                f"{r['cand_err']:.3e} | "
                f"**{r['spikes_ratio']:.2f}x** | {r['bytes_ratio']:.2f}x | "
                f"{r['params_ratio']:.2f}x |")
        ok = [r for r in rows if r["matched"]]
        if ok:
            out.append(
                f"\nmedian at matched accuracy: spikes "
                f"**{_med([r['spikes_ratio'] for r in ok]):.2f}x**, bytes "
                f"{_med([r['bytes_ratio'] for r in ok]):.2f}x, params "
                f"{_med([r['params_ratio'] for r in ok]):.2f}x "
                f"({len(ok)}/{len(rows)} MBE points matched)")
    return "\n".join(out)


def paper_table(res) -> str:
    """Our per-primitive spikes beside the paper's Table XI."""
    xi = res.get("paper", {}).get("table_xi", {})
    if not xi:
        return "_no Table XI reference in this run_"
    by_op = {}
    for r in res["records"]:
        by_op.setdefault(r["op"], []).append(r)

    lines = ["| paper primitive | paper spikes/elem | our op | our primitive | "
             "MBE spikes | PASN spikes | PASN vs paper |",
             "|---|---|---|---|---|---|---|"]
    for label, row in sorted(xi.items()):
        op, prim = row["op"], row["prim"]
        all_pts = by_op.get(op, [])
        if not all_pts:
            continue

        def at(arm, pts=all_pts, prim=prim):
            # Fixed-T arms are held to the paper's global T=16 so the comparison
            # is timestep-matched. The fully dynamic arm has no single T -- it
            # solves T_j per bank, which is the thing being compared -- so it is
            # taken as built, and its ⟨T⟩ is in the §1 table.
            sel = [p for p in pts if p["backend"] == arm
                   and prim in p["spikes_by_prim"]
                   and (p["T"] == 16 or p["T"] is None)]
            if not sel:
                return None
            # the build on that arm's front with the best accuracy, so the
            # spike figure belongs to a build we would actually ship
            b = min(sel, key=lambda p: p[p["primary"]])
            return b["spikes_by_prim"][prim] / max(b["mult_by_prim"][prim], 1e-9)

        m, p = at("mbe"), at("pasn_rule") or at("pasn")
        ratio = f"{row['spikes'] / p:.2f}x" if p else "—"
        lines.append(
            f"| {label} | {row['spikes']:.2f} (rate {row['rate']}%, N={row['n']}) "
            f"| {op} | {prim} | {_fmt(m, '.2f')} | {_fmt(p, '.2f')} | {ratio} |")
    return "\n".join(lines)


def tablex_table(res) -> str:
    """Our MSE beside Table X, at the paper's T=16."""
    tx = res.get("paper", {}).get("table_x", {})
    by_op = {}
    for r in res["records"]:
        by_op.setdefault(r["op"], []).append(r)
    rows = [(f"activation:{f}", f) for f in tx
            if f"activation:{f}" in by_op]
    if not rows:
        return "_no Table X function was swept in this run_"
    lines = ["| function | N | paper MSE | our MBE MSE | our best PASN MSE "
             "(at any budget) |", "|---|---|---|---|---|"]
    for op, fname in sorted(rows):
        pts = [p for p in by_op[op] if p["T"] == 16]
        pasn_best = min((p["mse"] for p in pts if p["backend"] == "pasn"),
                        default=None)
        for n, pmse in sorted(tx[fname].items(), key=lambda kv: int(kv[0])):
            ours = next((p["mse"] for p in pts
                         if p["backend"] == "mbe" and p["budget"] == int(n)),
                        None)
            lines.append(f"| {fname} | {n} | {pmse:.1e} | {_fmt(ours, '.1e')} | "
                         f"{_fmt(pasn_best, '.1e')} |")
    return "\n".join(lines)


HEADER = """# Isolated-operator Pareto: global MBE vs PASN

Generated by `experiments/op_pareto_report.py` from `{json}`.
Sweep: `experiments/op_pareto.py`, {n} builds, {elapsed:.0f}s, torch {torch}, CPU.

Every number here is measured on the **whole operator**, not on its driver
primitive: `mbe.op_cost.SpikeMeter` wraps each primitive an op holds, runs the op,
and normalises the emitted spikes per *output element*. Multiplicities differ per
primitive (softmax's `1/x` runs once per row, its `2^x` once per element) and the
signed FP-multiply split differs *between the backends*, which is exactly why the
driver-primitive figure quoted in `results/pasn_ops_and_phase4.md` is not the
operator's cost.

**Accuracy axis** is `nrmse` (RMS error / RMS signal) for every op except
`fp_multiply`, which is judged on mean relative error because that is what a
product's operands are budgeted against. Both are scale-free, so the ops are
readable side by side.

**Four arms.** `mbe` is one global `(N, T)`. The three PASN arms differ only in
how much per-bank allocation they are allowed — `pasn` none (uniform banks, so
its front is *routing alone*), `pasn_rule_T` allocates `N_j` with `T` pinned to
the paper's global 16, `pasn_rule` allocates `N_j` **and** `T_j` and is the
method as it actually is. §3 chains them so each half of the allocation is
measured separately.

**Baseline caveat.** A Pareto needs a knob, and the paper reports single operating
points, so the swept MBE arm is *our* reimplementation — an internal ablation per
the 2026-08-01 decision, never a headline baseline. The paper's own published
numbers are compared separately in §4, at primitive level, where they are directly
comparable.
"""


def build(res, json_path) -> str:
    meta = res["meta"]
    return "\n".join([
        HEADER.format(json=json_path, n=len(res["records"]),
                      elapsed=meta.get("elapsed_s", 0.0),
                      torch=meta.get("torch", "?")),
        "\n## 1. Best accuracy reached per arm\n",
        "`⟨N⟩`/`⟨T⟩` are the mean per-bank budget and `×n` the max/min spread "
        "across banks. A spread of ×1 means every bank got the same budget, i.e. "
        "the rule degenerated to the uniform arm on that operator.\n",
        best_table(res),
        "\n\n## 2. Iso-accuracy cost vs global MBE (the headline)\n",
        "For each point on the MBE spike front, the cheapest **fully dynamic** "
        "PASN build (rule-allocated `N_j` *and* `T_j`) that is at least as "
        "accurate. Ratios are MBE / PASN: >1 means PASN is cheaper at the same "
        "accuracy, <1 means it costs more.\n",
        iso_table(res, "pasn_rule"),
        "\n\n## 3. What the per-bank allocation buys\n",
        "PASN's claim is not routing alone -- it is that each bank gets its own "
        "solved `(N_j, T_j)`. These rows separate the two, each measured at "
        "matched accuracy against the previous arm: uniform banks → allocate "
        "`N_j` → also allocate `T_j`. `T` is a linear factor in the spike count, "
        "so the last step is not free even where it is small.\n",
        alloc_table(res),
        "\n\n### 3b. Iso-accuracy for the intermediate arms\n",
        "Uniform banks (routing only), for reference:\n",
        iso_table(res, "pasn"),
        "\n\n## 4. Per-primitive spikes vs the paper's Table XI\n",
        "Table XI is measured during **ViT-M/16** inference; ours come from the "
        "isolated ops here, so this is cross-model. The two rows whose argument "
        "is an IEEE field rather than an activation (`2^x` sees "
        "`frac(x log2 e)`, `1/x` sees a mantissa) are close to "
        "architecture-independent and are the ones worth reading closely. "
        "Spikes/elem is `rate/100 * N * T`, per *invocation* of the primitive.\n",
        paper_table(res),
        "\n\n## 5. Function MSE vs the paper's Table X\n",
        "Same quantity, same functions, at the paper's `T=16`. The PASN column "
        "is its best over the swept budgets and is **not** aligned to the `N` "
        "column — a routed neuron has no single `N`. Read it as a level, and "
        "read cost-matched comparisons off §2.\n",
        tablex_table(res), "",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/op_pareto.json")
    ap.add_argument("--out", default="results/op_pareto.md")
    args = ap.parse_args()
    with open(args.json, encoding="utf-8") as f:
        res = json.load(f)
    if "summary" not in res:
        res["summary"] = OP.summarise(res["records"])
    md = build(res, args.json)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    # Not echoed to stdout: the tables carry em-dashes and angle brackets that a
    # cp949 console cannot encode, and crashing on the way out would lose the
    # file that had already been written.
    print(f"wrote {args.out}  ({len(md.splitlines())} lines, "
          f"{len(res['records'])} builds)")


if __name__ == "__main__":
    main()
