"""실험 1 그림 — 함수별 (스파이크, MSE) 프론티어 + 기준선.

사용법::

    python experiments/plot_bank_budget.py results/bank_budget.json out.png --dist gauss
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
# 로그축 눈금은 mathtext로 그려지는데 한글 폰트에는 U+2212(−)가 없어 깨진다.
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402


def _tick(v, _pos):
    if v <= 0:
        return ""
    e = int(round(__import__("math").log10(v)))
    if abs(v - 10.0 ** e) > 1e-9 * v:
        return ""
    return f"1e{e}" if e < 0 or e > 2 else f"{10 ** e:g}"


STYLE = {
    "고정 N=2, T=16 (logspread)": dict(marker="s", color="#d62728"),
    "고정 N=2, T=16 (uniform)": dict(marker="D", color="#ff7f0e"),
    "전역 MBE N=4, T=16": dict(marker="^", color="#7f7f7f"),
    "전역 MBE N=8, T=16": dict(marker="v", color="#7f7f7f"),
    "Signed MBE 4x2, T=16": dict(marker="*", color="#9467bd"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("out")
    ap.add_argument("--dist", default="gauss")
    args = ap.parse_args()
    res = json.load(open(args.json, encoding="utf-8"))
    fns = list(res["functions"])

    fig, axes = plt.subplots(1, len(fns), figsize=(4.0 * len(fns), 4.0),
                             sharey=False)
    if len(fns) == 1:
        axes = [axes]
    for ax, name in zip(axes, fns):
        d = res["functions"][name]["by_dist"][args.dist]
        fr = d["frontier"]
        ax.plot([a["spikes"] for a in fr], [a["mse"] for a in fr],
                "-o", ms=3, lw=1.6, color="#1f77b4",
                label="최적 배분 (뱅크별 N,T)")
        for b in d["baselines"]:
            st = STYLE.get(b["label"], dict(marker="x", color="k"))
            ax.plot(b["spikes"], b["mse"], ms=11, mew=1.4, ls="none",
                    label=b["label"], **st)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # 로그축 기본 눈금은 mathtext(U+2212)를 쓰는데 한글 폰트에 그 글자가 없다.
        ax.xaxis.set_major_formatter(FuncFormatter(_tick))
        ax.yaxis.set_major_formatter(FuncFormatter(_tick))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(name)
        ax.set_xlabel("스파이크 / 입력  (에너지)")
        ax.grid(alpha=0.3, which="both", lw=0.4)
    axes[0].set_ylabel("근사 MSE")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"뱅크별 예산 배분 프론티어 (입력분포: {args.dist})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
