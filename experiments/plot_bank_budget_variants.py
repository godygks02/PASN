"""실험 2 그림 — 라우터 변종별 near-zero 오차 바닥 / iso-accuracy 절감.

사용법::

    python experiments/plot_bank_budget_variants.py out.png --dist gauss \
        "V1=results/bb_exp2_v1_single_e-2.json" "V2=..." ...
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = ["#9e9e9e", "#1f77b4", "#2ca02c", "#d62728"]


def near0_floor(fnres):
    worst = None
    for meta, cells in zip(fnres["banks"], fnres["grids"]):
        if not cells or not meta["kind"].startswith("near0"):
            continue
        m = min(c["mse"] for c in cells)
        worst = m if worst is None else max(worst, m)
    return worst


def iso(front, target):
    ok = [a for a in front if a["mse"] <= target]
    return min(ok, key=lambda a: a["spikes"])["spikes"] if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--dist", default="gauss")
    args = ap.parse_args()

    named = []
    for spec in args.variants:
        n, _, p = spec.partition("=")
        named.append((n, json.load(open(p, encoding="utf-8"))))
    fns = list(named[0][1]["functions"])
    x = np.arange(len(fns))
    w = 0.8 / len(named)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    for k, (n, res) in enumerate(named):
        vals = [near0_floor(res["functions"][f]) for f in fns]
        ax1.bar(x + k * w, vals, w, label=n, color=COLORS[k % len(COLORS)])
    ax1.set_yscale("log")
    # 로그축 기본 눈금은 mathtext(U+2212)라 한글 폰트에서 깨진다.
    ax1.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _p: f"1e{int(round(np.log10(v)))}" if v > 0 else ""))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_xticks(x + 0.4 - w / 2)
    ax1.set_xticklabels(fns)
    ax1.set_ylabel("near-zero 뱅크 최저 MSE  (낮을수록 좋음)")
    ax1.set_title("A. near-zero 뱅크의 오차 바닥")
    ax1.grid(axis="y", alpha=0.3, which="both", lw=0.4)
    ax1.legend(fontsize=8)

    for k, (n, res) in enumerate(named):
        vals = []
        for f in fns:
            b0 = next(b for b in named[0][1]["functions"][f]["by_dist"]
                      [args.dist]["baselines"] if "logspread" in b["label"])
            s = iso(res["functions"][f]["by_dist"][args.dist]["frontier"],
                    b0["mse"])
            vals.append(b0["spikes"] / s if s else 0.0)
        ax2.bar(x + k * w, vals, w, label=n, color=COLORS[k % len(COLORS)])
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")
    ax2.set_xticks(x + 0.4 - w / 2)
    ax2.set_xticklabels(fns)
    ax2.set_ylabel("고정 N=2,T=16 대비 스파이크 절감 (배)")
    ax2.set_title("B. 같은 정확도에서의 에너지 절감")
    ax2.grid(axis="y", alpha=0.3, lw=0.4)

    fig.suptitle(f"실험 2 — 라우터 변종 비교 (입력분포: {args.dist})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
