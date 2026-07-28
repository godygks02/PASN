"""근사 곡선 그림 — 전체 도메인 / near-zero 확대 / 절대오차.

``approx_curves.py`` 가 만든 npz 들을 받아 함수당 세 줄로 그린다.

  1행 **전체 도메인** — 참값과 근사가 눈으로 구분되는지.
  2행 **near-zero 확대** — 곡률(과 ReLU의 꺾임)이 몰린 곳. 여기가 실제 병목이었다.
  3행 **절대오차** ``|f̂ − f|`` (로그) — 곡선이 겹쳐 보일 때 실제 차이는 여기서 보인다.

모든 변종은 **같은 스파이크 예산**에서 고른 동작점이라, 세로로 비교하면
"같은 전력에서 얼마나 더 정확한가"가 된다.

사용법::

    python experiments/plot_approx_curves.py out.png --npz curves_*.npz --zoom 0.3
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

STYLE = [
    dict(color="#d62728", ls="--", lw=1.6),    # 고정 예산
    dict(color="#ff9800", ls="-.", lw=1.6),    # 실험 1
    dict(color="#1f77b4", ls="-", lw=1.8),     # 실험 2
]


def _logtick(v, _p):
    return f"1e{int(round(np.log10(v)))}" if v > 0 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--zoom", type=float, default=0.3)
    ap.add_argument("--only", nargs="+", default=None,
                    help="그릴 변종 인덱스 (예: 0 1 -> 고정 + 실험1)")
    args = ap.parse_args()

    data = [np.load(p, allow_pickle=True) for p in args.npz]
    data.sort(key=lambda d: str(d["fn"]))
    keep = [int(i) for i in args.only] if args.only else None

    n = len(data)
    fig, axes = plt.subplots(3, n, figsize=(3.7 * n, 10.2), squeeze=False)

    for c, d in enumerate(data):
        x, y = d["x"], d["y"]
        labels = list(d["labels"])
        idxs = keep if keep is not None else list(range(len(labels)))
        zoom = np.abs(x) <= args.zoom

        ax = axes[0][c]
        ax.plot(x, y, color="k", lw=2.6, alpha=0.35, label="참값 f(x)")
        ax.text(0.02, 0.97, f"MSE {d['mse'][idxs[0]]:.1e} → "
                            f"{d['mse'][list(idxs)[-1]]:.1e}",
                transform=ax.transAxes, va="top", fontsize=7, color="#444")
        for k in idxs:
            ax.plot(x, d[f"yhat_{k}"], **STYLE[k % len(STYLE)],
                    label=f"{labels[k]}\n({d['spikes'][k]:.1f} spk)")
        ax.set_title(str(d["fn"]), fontsize=13)
        ax.grid(alpha=0.25, lw=0.4)
        if c == 0:
            ax.set_ylabel("전체 도메인")
        ax.legend(fontsize=6.5, loc="best")

        ax = axes[1][c]
        ax.plot(x[zoom], y[zoom], color="k", lw=2.6, alpha=0.35)
        for k in idxs:
            ax.plot(x[zoom], d[f"yhat_{k}"][zoom], **STYLE[k % len(STYLE)])
        ax.axvline(0.0, color="k", lw=0.6, ls=":")
        ax.grid(alpha=0.25, lw=0.4)
        if c == 0:
            ax.set_ylabel(f"near-zero 확대 (|x| ≤ {args.zoom})")

        ax = axes[2][c]
        for k in idxs:
            err = np.abs(d[f"yhat_{k}"] - y)
            ax.plot(x, np.maximum(err, 1e-12), **STYLE[k % len(STYLE)])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_logtick))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.axvline(0.0, color="k", lw=0.6, ls=":")
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.set_xlabel("x")
        if c == 0:
            ax.set_ylabel("절대오차 |근사 - 참값|")

    fig.suptitle("MBE-PASN 근사 곡선 — 같은 스파이크 예산에서의 비교", y=1.005,
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
