"""뱅크 예산의 **보편 규칙** 도출 및 검증.

``bank_budget.py`` 격자에서, 뱅크마다 목표 오차 ``eps`` 를 만족하는 최소-스파이크
설정 ``(N*, T*)`` 을 뽑고, 그것이 뱅크의 두 기술자로 예측되는지 본다.

  * ``b = log2(delta / sqrt(eps))`` — 그 구간에서 필요한 **유효 비트 수**.
    (스테어케이스 코드의 RMS 오차 ~ ``delta / levels`` 이므로 ``b`` 는 필요한
    해상도의 로그.)
  * ``kappa`` — 최적 직선 대비 잔차. 직선 하나로 안 되는 정도.

검증은 두 단계.

  1. **회귀** — ``T*`` 가 ``b`` 로 설명되는가, ``N*`` 가 ``kappa`` 로 설명되는가.
  2. **엔드투엔드** — 규칙이 예측한 배분을 실제로 적용했을 때, 탐색으로 얻은
     최적 배분에 얼마나 가까운가. (가까우면 빌드 때 탐색이 필요 없다.)

사용법::

    python experiments/bank_budget_rule.py results/bank_budget.json
"""
from __future__ import annotations

import argparse
import json
import math

TS = [2, 3, 4, 6, 8, 12, 16]
NS = [1, 2, 3, 4]


def min_spike_cell(cells, eps):
    ok = [c for c in cells if c["mse"] <= eps]
    return min(ok, key=lambda c: (c["spikes"], c["N"], c["T"])) if ok else None


def best_cell(cells, N, T):
    sub = [c for c in cells if c["N"] == N and c["T"] == T]
    return min(sub, key=lambda c: c["mse"]) if sub else None


def collect(res, epss):
    """(함수, 뱅크, eps) → 최소-스파이크 설정."""
    rows = []
    for name, fnres in res["functions"].items():
        for meta, cells in zip(fnres["banks"], fnres["grids"]):
            if not cells or meta.get("delta", 0.0) <= 0:
                continue
            for eps in epss:
                c = min_spike_cell(cells, eps)
                if c is None:
                    continue
                rows.append(dict(
                    fn=name, idx=meta["idx"], delta=meta["delta"],
                    kappa=meta["nonlin"], eps=eps,
                    b=math.log2(meta["delta"] / math.sqrt(eps)),
                    N=c["N"], T=c["T"], spikes=c["spikes"]))
    return rows


def snap_T(t):
    for T in TS:
        if T >= t:
            return T
    return TS[-1]


def rule_T(b, a, c):
    """T = snap(2^((b - a)/c)) — 데이터로 (a, c) 를 맞춘다."""
    if b <= a:
        return TS[0]
    return snap_T(2.0 ** ((b - a) / c))


def fit_rule(rows):
    best = None
    for a10 in range(-40, 61):
        a = a10 / 10.0
        for c10 in range(5, 61):
            c = c10 / 10.0
            err = sum(abs(math.log2(rule_T(r["b"], a, c)) - math.log2(r["T"]))
                      for r in rows)
            if best is None or err < best[0]:
                best = (err, a, c)
    err, a, c = best
    return a, c, err / len(rows)


def rule_N_kappa(kappa, b, a, c, thr):
    """가설 A — 비선형성 ``kappa`` 가 N을 정한다."""
    n = 2 if kappa >= thr else 1
    need = 2.0 ** ((b - a) / c) if b > a else 1.0
    if need > TS[-1]:
        n = max(n, min(4, 1 + int(math.ceil(math.log2(need / TS[-1])))))
    return n


def rule_N_b(b, cuts):
    """가설 B — 필요한 비트 수 ``b`` 하나가 N도 정한다."""
    n = 1
    for cut in cuts:
        if b >= cut:
            n += 1
    return min(n, NS[-1])


def fit_N_cuts(rows):
    grid = [i / 2.0 for i in range(0, 30)]
    best = None
    for i, c1 in enumerate(grid):
        for c2 in grid[i:]:
            for c3 in grid[grid.index(c2):]:
                cuts = (c1, c2, c3)
                acc = sum(1 for r in rows if rule_N_b(r["b"], cuts) == r["N"])
                if best is None or acc > best[0]:
                    best = (acc, cuts)
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--dist", default="gauss")
    args = ap.parse_args()
    res = json.load(open(args.json, encoding="utf-8"))

    epss = [10.0 ** (-k / 2.0) for k in range(4, 15)]     # 1e-2 .. 1e-7
    rows = collect(res, epss)
    print(f"표본 {len(rows)}개 (함수 {len(res['functions'])} × 뱅크 × eps {len(epss)})")

    a, c, mae = fit_rule(rows)
    print(f"\n## 규칙 적합:  T* = 2^((b - {a:.1f}) / {c:.1f})  로 스냅")
    print(f"   b = log2(delta / sqrt(eps));  평균 |log2 오차| = {mae:.3f} "
          f"(= {2**mae:.2f}× 배수 오차)")

    exact = sum(1 for r in rows if rule_T(r["b"], a, c) == r["T"])
    within = sum(1 for r in rows
                 if 0.5 <= rule_T(r["b"], a, c) / r["T"] <= 2.0)
    print(f"   정확 일치 {exact}/{len(rows)} ({100*exact/len(rows):.0f}%), "
          f"2배 이내 {within}/{len(rows)} ({100*within/len(rows):.0f}%)")

    print("\n## b 구간별 실제 T* 분포")
    print("| b 범위 | 표본 | 실제 T* 중앙값 | 규칙 T* | N*=1 비율 |")
    print("|---|---|---|---|---|")
    for lo in range(-2, 14, 2):
        sub = [r for r in rows if lo <= r["b"] < lo + 2]
        if not sub:
            continue
        ts = sorted(r["T"] for r in sub)
        med = ts[len(ts) // 2]
        n1 = sum(1 for r in sub if r["N"] == 1) / len(sub)
        print(f"| [{lo}, {lo+2}) | {len(sub)} | {med} | "
              f"{rule_T(lo + 1, a, c)} | {100*n1:.0f}% |")

    print("\n## kappa 구간별 N* 분포")
    print("| kappa 범위 | 표본 | N* 평균 | N*≥2 비율 |")
    print("|---|---|---|---|")
    bnds = [0, 1e-3, 1e-2, 3e-2, 1e-1, 1.0]
    for lo, hi in zip(bnds[:-1], bnds[1:]):
        sub = [r for r in rows if lo <= r["kappa"] < hi]
        if not sub:
            continue
        avg = sum(r["N"] for r in sub) / len(sub)
        ge2 = sum(1 for r in sub if r["N"] >= 2) / len(sub)
        print(f"| [{lo:.0e}, {hi:.0e}) | {len(sub)} | {avg:.2f} | "
              f"{100*ge2:.0f}% |")

    # 가설 A: kappa 가 N을 정한다
    best_thr, best_acc = None, -1
    for thr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        acc = sum(1 for r in rows
                  if rule_N_kappa(r["kappa"], r["b"], a, c, thr) == r["N"])
        if acc > best_acc:
            best_thr, best_acc = thr, acc
    print(f"\n   [가설 A] N ~ kappa (kappa_0 = {best_thr:.0e}) "
          f"→ 정확 일치 {best_acc}/{len(rows)} ({100*best_acc/len(rows):.0f}%)")

    # 가설 B: b 하나가 N도 정한다
    cuts, acc_b = fit_N_cuts(rows)
    print(f"   [가설 B] N ~ b (경계 b = {cuts}) "
          f"→ 정확 일치 {acc_b}/{len(rows)} ({100*acc_b/len(rows):.0f}%)")

    baseline = max(sum(1 for r in rows if r["N"] == n) for n in NS)
    print(f"   (참고: 항상 N=최빈값으로 찍으면 {baseline}/{len(rows)} "
          f"= {100*baseline/len(rows):.0f}%)")

    # ----------------------------------------------------------------
    print(f"\n## 엔드투엔드 (dist={args.dist}): 규칙 배분 vs 탐색 최적 배분\n")
    print("| 함수 | 목표 ε | 규칙: MSE / 스파이크 | 탐색 최적: 같은 MSE에서 스파이크 | 손해 |")
    print("|---|---|---|---|---|")
    for name, fnres in res["functions"].items():
        d = fnres["by_dist"][args.dist]
        p, front = d["p"], d["frontier"]
        for eps in (1e-4, 1e-5, 1e-6):
            tot_m = tot_s = 0.0
            ok = True
            for j, (meta, cells) in enumerate(zip(fnres["banks"],
                                                  fnres["grids"])):
                if not cells:
                    continue
                delta = meta.get("delta", 0.0)
                if delta <= 0:
                    cell = best_cell(cells, 1, 2)
                else:
                    b = math.log2(delta / math.sqrt(eps))
                    T = rule_T(b, a, c)
                    N = rule_N_b(b, cuts)
                    cell = best_cell(cells, N, T)
                if cell is None:
                    ok = False
                    break
                tot_m += p[j] * cell["mse"]
                tot_s += p[j] * cell["spikes"]
            if not ok:
                continue
            cand = [f for f in front if f["mse"] <= tot_m]
            opt = min(cand, key=lambda f: f["spikes"]) if cand else None
            loss = (tot_s / opt["spikes"]) if opt else float("nan")
            print(f"| {name} | {eps:.0e} | {tot_m:.2e} / {tot_s:.2f} | "
                  f"{opt['spikes']:.2f} | {loss:.2f}× |" if opt else
                  f"| {name} | {eps:.0e} | {tot_m:.2e} / {tot_s:.2f} | — | — |")


if __name__ == "__main__":
    main()
