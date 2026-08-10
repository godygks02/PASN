"""Fold the tied re-measure back into the operator sweep, and restate the win.

`experiments/op_pareto.py` originally built every arm with `tied=False`, while
the conversion path defaults `pasn_id_tied=True`. E6 re-ran the three ops that
hold an identity (`fp_multiply`, `layernorm`, `attention`) with per-primitive
gating (`TIED_PRIMS={"identity"}`) into `results/op_pareto_tied.json`.

Tying is **not** spike- or accuracy-neutral at operator level -- E6 measured
per-cell swings of 0.19-1.20x on spikes and 0.72-6.94x on accuracy -- so the
headline sentence from 실험 12, *"PASN wins 10/10 operators on spikes,
1.61-6.41x"*, was computed on builds that are no longer the ones we ship. This
script recomputes it.

What it does:

1. **Checks the control first.** The `mbe` arm cannot be touched by a PASN-side
   flag, so its rows must be bit-identical across the two files. If they are
   not, the difference is environment drift rather than tying and the merge is
   refused -- that distinction is the whole point of the control (함정 13).
2. Replaces the three ops' records with the tied ones, leaving the other seven
   untouched, and re-summarises.
3. Prints old vs new iso-accuracy medians per operator, so the restated claim
   can be read off directly.

Usage::  python experiments/op_pareto_merge.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import op_pareto as OP  # noqa: E402

CONTROL_ARM = "mbe"
# Compared field-by-field rather than by dict equality: `budget_tables` carries
# nested floats whose repr can differ without the measurement differing.
CONTROL_FIELDS = ("spikes", "bytes", "params", "nrmse", "rel", "mse", "mae",
                  "max_abs", "calls")


def _key(r):
    return (r["op"], r["backend"], r["budget"], r["T"])


def check_control(old_rows, new_rows) -> tuple[int, list[str]]:
    """Verify the untouched arm reproduces. Returns (n_checked, complaints)."""
    o = {_key(r): r for r in old_rows if r["backend"] == CONTROL_ARM}
    n = {_key(r): r for r in new_rows if r["backend"] == CONTROL_ARM}
    bad = []
    if set(o) != set(n):
        bad.append(f"{CONTROL_ARM} arm covers different builds: "
                   f"{len(set(o) ^ set(n))} keys differ")
    for k in sorted(set(o) & set(n)):
        for f in CONTROL_FIELDS:
            a, b = o[k].get(f), n[k].get(f)
            if a != b:
                bad.append(f"{k}: {f} {a!r} -> {b!r}")
    return len(set(o) & set(n)), bad


def iso_medians(summary, arm="pasn_rule"):
    """Per-op median iso-accuracy ratios, plus how many MBE points matched."""
    out = {}
    for op, s in summary.items():
        rows = s["iso_accuracy"].get(arm, [])
        ok = [r for r in rows if r["matched"]]
        if not ok:
            out[op] = None
            continue
        out[op] = {
            "spikes": statistics.median(r["spikes_ratio"] for r in ok),
            "bytes": statistics.median(r["bytes_ratio"] for r in ok),
            "params": statistics.median(r["params_ratio"] for r in ok),
            "matched": len(ok),
            "total": len(rows),
        }
    return out


def _fmt(v):
    return "  --  " if v is None else f"{v:5.2f}x"


def report(before, after, tied_ops) -> str:
    lines = []
    lines.append("| operator | tied? | spikes MBE/PASN | bytes MBE/PASN | "
                 "matched |")
    lines.append("|---|---|---|---|---|")
    wins_b = wins_a = 0
    vals_a = []
    for op in sorted(after):
        b, a = before.get(op), after.get(op)
        touched = op in tied_ops
        if b and b["spikes"] > 1:
            wins_b += 1
        if a and a["spikes"] > 1:
            wins_a += 1
            vals_a.append(a["spikes"])
        if touched and b and a:
            sp = f"{b['spikes']:.2f}x -> **{a['spikes']:.2f}x**"
            by = f"{b['bytes']:.2f}x -> **{a['bytes']:.2f}x**"
        else:
            sp = _fmt(a and a["spikes"])
            by = _fmt(a and a["bytes"])
        m = f"{a['matched']}/{a['total']}" if a else "--"
        lines.append(f"| `{op}` | {'**yes**' if touched else '--'} | {sp} | "
                     f"{by} | {m} |")
    lines.append("")
    lines.append(f"spike wins: {wins_b}/{len(before)} (before) -> "
                 f"**{wins_a}/{len(after)} (after)**")
    if vals_a:
        lines.append(f"winning range: **{min(vals_a):.2f}x - {max(vals_a):.2f}x**"
                     f"  (median of the winners {statistics.median(vals_a):.2f}x)")
    byte_wins = sum(1 for v in after.values() if v and v["bytes"] > 1)
    lines.append(f"byte wins: **{byte_wins}/{len(after)}**")
    return "\n".join(lines)


def arm_ranges(base, out) -> str:
    """Both PASN arms, before and after, so a quoted range names its arm.

    The recorded sentence "10/10 operators, 1.61-6.41x" mixes them: 1.61x is
    `fp_multiply` on `pasn_rule`, 6.41x is `softmax` on `pasn`. `pasn_rule` is
    the method as it ships; `pasn` is routing alone, i.e. the ablation.
    """
    lines = ["| arm | what it is | before | after |", "|---|---|---|---|"]
    what = {"pasn": "routing alone (uniform banks) -- ablation",
            "pasn_rule_T": "+ allocate N_j, T pinned to 16",
            "pasn_rule": "+ allocate N_j and T_j -- **the method as it ships**"}
    for arm in ("pasn", "pasn_rule_T", "pasn_rule"):
        cells = []
        for res in (base, out):
            summ = res.get("summary") or OP.summarise(res["records"])
            m = iso_medians(summ, arm)
            vals = [v["spikes"] for v in m.values() if v]
            wins = [v for v in vals if v > 1]
            cells.append(f"{len(wins)}/{len(vals)} ops, "
                         f"{min(wins):.2f}x-{max(wins):.2f}x"
                         if wins else "--")
        lines.append(f"| `{arm}` | {what[arm]} | {cells[0]} | **{cells[1]}** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/op_pareto.json")
    ap.add_argument("--tied", default="results/op_pareto_tied.json")
    ap.add_argument("--out", default="results/op_pareto_merged.json")
    ap.add_argument("--force", action="store_true",
                    help="merge even if the control arm does not reproduce")
    args = ap.parse_args()

    with open(args.base, encoding="utf-8") as f:
        base = json.load(f)
    with open(args.tied, encoding="utf-8") as f:
        tied = json.load(f)

    tied_ops = sorted({r["op"] for r in tied["records"]})
    base_sub = [r for r in base["records"] if r["op"] in set(tied_ops)]

    n_ctrl, bad = check_control(base_sub, tied["records"])
    print(f"control: {CONTROL_ARM} arm, {n_ctrl} builds compared on "
          f"{len(CONTROL_FIELDS)} fields each")
    if bad:
        print(f"  FAILED -- {len(bad)} mismatches:")
        for line in bad[:10]:
            print(f"    {line}")
        if not args.force:
            raise SystemExit(
                "\nRefusing to merge. The untouched arm must reproduce, or the\n"
                "difference cannot be attributed to tying (trap 13). Re-run all\n"
                "four arms for these ops on one box, or pass --force knowingly.")
    else:
        print("  OK -- bit-identical, so the change is tying, not the box")

    before = iso_medians(base["summary"] if "summary" in base
                         else OP.summarise(base["records"]))

    merged = [r for r in base["records"] if r["op"] not in set(tied_ops)]
    merged += tied["records"]
    out = {
        "meta": {**base["meta"],
                 "merged_from": [args.base, args.tied],
                 "tied_ops": tied_ops,
                 "tied_prims": ["identity"],
                 "control_arm": CONTROL_ARM,
                 "control_builds_checked": n_ctrl,
                 "control_reproduced": not bad,
                 "json": args.out},
        "paper": base.get("paper", {}),
        "records": merged,
    }
    out["summary"] = OP.summarise(merged)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    after = iso_medians(out["summary"])
    print(f"\nmerged {len(merged)} builds "
          f"({len(tied['records'])} replaced across {tied_ops}) -> {args.out}\n")
    text = report(before, after, set(tied_ops))
    arms = arm_ranges(base, out)
    md = os.path.splitext(args.out)[0] + "_wins.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Operator wins, recomputed with identity tying on\n\n"
                f"From `{args.base}` + `{args.tied}` via "
                "`experiments/op_pareto_merge.py`.\n"
                f"Control: `{CONTROL_ARM}` arm, {n_ctrl} builds, "
                f"{'reproduced bit-identical' if not bad else 'DID NOT REPRODUCE'}"
                ".\n\n## Per operator (arm `pasn_rule`)\n\n"
                "Iso-accuracy medians; ratios are MBE / PASN "
                "(>1 = PASN cheaper).\n\n" + text +
                "\n\n## Per arm -- a quoted range must name one\n\n" + arms +
                "\n")
    print(text)
    print()
    print(arms)
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
