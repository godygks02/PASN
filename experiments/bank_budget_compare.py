"""실험 2 — 라우터 변종 비교 (near-zero 처리 × 라우터 깊이).

``bank_budget.py --mode analyse`` 로 만든 변종별 JSON 여러 개를 받아,

  1. **near-zero 뱅크의 오차 바닥** — 처방이 병목을 뚫었는가,
  2. **iso-accuracy 총 스파이크** — 같은 정확도를 몇 스파이크에,
  3. **메모리** — 뱅크가 늘어난 대가,
  4. **Signed MBE 기준선을 이기는가** — 실험 1의 유일한 패배(ReLU) 를 뒤집었는가

를 나란히 놓는다.

사용법::

    python experiments/bank_budget_compare.py \
        v1=results/bb_v1.json v2=results/bb_v2.json --dist gauss
"""
from __future__ import annotations

import argparse
import json


def iso_accuracy(front, target_mse):
    ok = [a for a in front if a["mse"] <= target_mse]
    return min(ok, key=lambda a: a["spikes"]) if ok else None


def near0_floor(fnres):
    """near-zero 뱅크(들)의 달성 가능 최저 MSE와 그때의 설정."""
    best = None
    for meta, cells in zip(fnres["banks"], fnres["grids"]):
        if not cells or not meta["kind"].startswith("near0"):
            continue
        c = min(cells, key=lambda c: c["mse"])
        if best is None or c["mse"] > best[0]:      # 가장 나쁜 near0 뱅크가 병목
            best = (c["mse"], c["N"], c["T"], meta["idx"])
    return best


def baseline(fnres, dist, label_sub):
    for b in fnres["by_dist"][dist]["baselines"]:
        if label_sub in b["label"]:
            return b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", help="이름=경로.json")
    ap.add_argument("--dist", default="gauss")
    args = ap.parse_args()

    named = []
    for spec in args.variants:
        name, _, path = spec.partition("=")
        named.append((name, json.load(open(path, encoding="utf-8"))))
    fns = list(named[0][1]["functions"])

    print(f"# 실험 2 — 라우터 변종 비교 (dist={args.dist})\n")

    print("## 1. near-zero 뱅크 오차 바닥 (가장 나쁜 near0 뱅크)\n")
    hdr = "| 함수 | " + " | ".join(n for n, _ in named) + " | 개선 |"
    print(hdr)
    print("|---" * (len(named) + 2) + "|")
    for fn in fns:
        vals, cells = [], []
        for _, res in named:
            f = near0_floor(res["functions"][fn])
            cells.append(f"{f[0]:.2e} (N={f[1]},T={f[2]})" if f else "—")
            vals.append(f[0] if f else float("nan"))
        gain = vals[0] / min(vals[1:]) if len(vals) > 1 and min(vals[1:]) > 0 else 0
        print(f"| {fn} | " + " | ".join(cells) + f" | **{gain:.0f}×** |")

    print("\n## 2. 뱅크 수 / 저장 파라미터 (최소 예산 기준)\n")
    print("| 함수 | " + " | ".join(n for n, _ in named) + " |")
    print("|---" * (len(named) + 1) + "|")
    for fn in fns:
        cells = []
        for _, res in named:
            r = res["functions"][fn]
            reach = sum(1 for g in r["grids"] if g)
            front = r["by_dist"][args.dist]["frontier"]
            cells.append(f"{reach}뱅크 / {front[0]['params']}p")
        print(f"| {fn} | " + " | ".join(cells) + " |")

    # 공통 타깃 — 첫 변종(=실험 1 구성)의 기준선 MSE 를 모든 변종에 그대로 쓴다.
    # 변종마다 자기 기준선을 쓰면 타깃이 달라져 사과 대 사과가 아니게 된다.
    for tag, sub in (("고정 N=2,T=16", "logspread"), ("Signed MBE", "Signed")):
        print(f"\n## 3. iso-accuracy — {tag} 의 MSE를 몇 스파이크에 "
              f"(타깃은 {named[0][0]} 기준선으로 통일)\n")
        print("| 함수 | 공통 타깃 MSE | 기준선 스파이크 | " +
              " | ".join(n for n, _ in named) + " |")
        print("|---" * (len(named) + 3) + "|")
        for fn in fns:
            b0 = baseline(named[0][1]["functions"][fn], args.dist, sub)
            if b0 is None:
                continue
            cells = []
            for _, res in named:
                r = res["functions"][fn]
                a = iso_accuracy(r["by_dist"][args.dist]["frontier"], b0["mse"])
                cells.append(f"{a['spikes']:.2f} ({b0['spikes']/a['spikes']:.1f}×)"
                             if a else "**도달 불가**")
            print(f"| {fn} | {b0['mse']:.2e} | {b0['spikes']:.2f} | " +
                  " | ".join(cells) + " |")

    print("\n## 4. 프론티어 최저 MSE (모델이 낼 수 있는 최고 정확도)\n")
    print("| 함수 | " + " | ".join(n for n, _ in named) + " |")
    print("|---" * (len(named) + 1) + "|")
    for fn in fns:
        cells = []
        for _, res in named:
            fr = res["functions"][fn]["by_dist"][args.dist]["frontier"]
            best = min(fr, key=lambda a: a["mse"])
            cells.append(f"{best['mse']:.2e} @{best['spikes']:.1f}spk")
        print(f"| {fn} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
