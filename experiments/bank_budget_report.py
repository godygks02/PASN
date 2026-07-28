"""``bank_budget.py`` 결과 JSON → 분석 표 (연구일지용).

세 가지를 뽑는다.

1. **뱅크 난이도 지도** — 구간, 동적범위 ``delta``, 비선형성 ``nonlin``, 방문확률.
2. **배분 비교** — 고정 예산(현재 기본값) / 전역 MBE / SignedMBE 대비 최적 배분의
   iso-accuracy 스파이크, iso-energy MSE.
3. **보편 규칙 검증** — 뱅크가 T스텝에서 실제로 얻는 유효 비트
   ``bits = log2(delta / sqrt(mse))`` 가 ``T`` 에 비례하는지, 그리고 그 기울기/절편이
   ``N`` 과 ``nonlin`` 에 어떻게 의존하는지.

사용법::

    python experiments/bank_budget_report.py results/bank_budget.json --dist gauss
"""
from __future__ import annotations

import argparse
import json
import math


def best_cell(cells, **eq):
    sub = [c for c in cells if all(c[k] == v for k, v in eq.items())]
    return min(sub, key=lambda c: c["mse"]) if sub else None


def iso_accuracy(front, target_mse):
    """target 이하 MSE를 내는 프론티어 점 중 스파이크 최소."""
    ok = [a for a in front if a["mse"] <= target_mse]
    return min(ok, key=lambda a: a["spikes"]) if ok else None


def iso_energy(front, target_spikes):
    ok = [a for a in front if a["spikes"] <= target_spikes]
    return min(ok, key=lambda a: a["mse"]) if ok else None


def fmt(v, p=2):
    return "—" if v is None else f"{v:.{p}e}"


def section_banks(fnres, dist):
    p = fnres["by_dist"][dist]["p"]
    lines = ["| # | 구간 | 동적범위 Δ | 비선형성 κ | 방문확률 | 최저 MSE | 그때 (N,T) |",
             "|---|---|---|---|---|---|---|"]
    for meta, cells in zip(fnres["banks"], fnres["grids"]):
        i = meta["idx"]
        if not cells:
            lines.append(f"| {i} | (도달 불가) | — | — | {p[i]:.4f} | — | — |")
            continue
        b = min(cells, key=lambda c: c["mse"])
        lo, hi = meta["interval"]
        lines.append(
            f"| {i} | `[{lo:+.3f}, {hi:+.3f}]` | {meta['delta']:.3g} | "
            f"{meta['nonlin']:.2e} | {p[i]:.4f} | {b['mse']:.2e} | "
            f"N={b['N']}, T={b['T']} |")
    return "\n".join(lines)


def section_alloc(fnres, dist):
    d = fnres["by_dist"][dist]
    front = d["frontier"]
    rows = ["| 방식 | MSE | 스파이크/입력 | 저장 파라미터 |",
            "|---|---|---|---|"]
    for b in d["baselines"]:
        rows.append(f"| {b['label']} | {b['mse']:.2e} | {b['spikes']:.2f} | "
                    f"{b['params']} |")
    rows.append("| **최적 배분 (아래 표)** | | | |")

    iso = ["| 기준선 | 기준 MSE | 기준 스파이크 | 최적배분 스파이크 | **절감** | 최적배분 파라미터 |",
           "|---|---|---|---|---|---|"]
    for b in d["baselines"]:
        a = iso_accuracy(front, b["mse"])
        if a is None:
            iso.append(f"| {b['label']} | {b['mse']:.2e} | {b['spikes']:.2f} | "
                       f"도달 불가 | — | — |")
        else:
            iso.append(f"| {b['label']} | {b['mse']:.2e} | {b['spikes']:.2f} | "
                       f"{a['spikes']:.2f} | **{b['spikes']/max(a['spikes'],1e-9):.2f}×** | "
                       f"{a['params']} |")
    return "\n".join(rows), "\n".join(iso)


def section_frontier(fnres, dist, k=8):
    front = fnres["by_dist"][dist]["frontier"]
    step = max(1, len(front) // k)
    rows = ["| 스파이크/입력 | MSE | 파라미터 | 뱅크별 (N,T) 분포 |",
            "|---|---|---|---|"]
    for a in front[::step]:
        ch = [c for c in a["choice"] if c]
        hist = {}
        for N, T, _ in ch:
            hist[(N, T)] = hist.get((N, T), 0) + 1
        s = ", ".join(f"({N},{T})×{n}" for (N, T), n in sorted(hist.items()))
        rows.append(f"| {a['spikes']:.2f} | {a['mse']:.2e} | {a['params']} | {s} |")
    return "\n".join(rows)


def section_rule(fnres):
    """유효 비트 bits = log2(delta/sqrt(mse)) 가 T에 대해 어떻게 자라는지."""
    rows = ["| 뱅크 | Δ | κ | N | bits@T=2 | @4 | @8 | @16 | 기울기(bit/step) |",
            "|---|---|---|---|---|---|---|---|---|"]
    pts = []
    for meta, cells in zip(fnres["banks"], fnres["grids"]):
        if not cells:
            continue
        delta = meta["delta"]
        if delta <= 0:                      # 상수 구간(예: ReLU 음수측) — 비트 정의 불가
            continue
        for N in sorted({c["N"] for c in cells}):
            bits = {}
            for T in sorted({c["T"] for c in cells}):
                c = best_cell(cells, N=N, T=T)
                if c is None or c["mse"] <= 0:
                    continue
                bits[T] = math.log2(delta / math.sqrt(c["mse"]))
            if 2 in bits and 16 in bits:
                slope = (bits[16] - bits[2]) / 14.0
                pts.append(dict(idx=meta["idx"], delta=delta,
                                nonlin=meta["nonlin"], N=N, slope=slope,
                                bits=bits))
                if N in (1, 2):
                    rows.append(
                        f"| {meta['idx']} | {delta:.3g} | {meta['nonlin']:.1e} | "
                        f"{N} | " +
                        " | ".join(f"{bits.get(T, float('nan')):.1f}"
                                   for T in (2, 4, 8, 16)) +
                        f" | {slope:.2f} |")
    return "\n".join(rows), pts


def min_spike_config(cells, eps):
    ok = [c for c in cells if c["mse"] <= eps]
    return min(ok, key=lambda c: c["spikes"]) if ok else None


def section_epsrule(fnres, epss=(1e-3, 1e-4, 1e-5, 1e-6)):
    """목표 오차 eps별로 뱅크가 고르는 최소-스파이크 (N,T) — 규칙의 실사용 형태."""
    rows = ["| 뱅크 | Δ | κ | " + " | ".join(f"ε={e:.0e}" for e in epss) + " |",
            "|---|---|---|" + "---|" * len(epss)]
    data = []
    for meta, cells in zip(fnres["banks"], fnres["grids"]):
        if not cells:
            continue
        out = []
        for e in epss:
            c = min_spike_config(cells, e)
            out.append(f"N={c['N']},T={c['T']}" if c else "불가")
            if c:
                data.append(dict(idx=meta["idx"], delta=meta["delta"],
                                 nonlin=meta["nonlin"], eps=e, N=c["N"],
                                 T=c["T"], spikes=c["spikes"]))
        rows.append(f"| {meta['idx']} | {meta['delta']:.3g} | "
                    f"{meta['nonlin']:.1e} | " + " | ".join(out) + " |")
    return "\n".join(rows), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--dist", default="gauss")
    args = ap.parse_args()
    res = json.load(open(args.json, encoding="utf-8"))

    for name, fnres in res["functions"].items():
        print(f"\n{'='*78}\n## {name}  (dist={args.dist})\n{'='*78}")
        print("\n### 뱅크 난이도 지도\n")
        print(section_banks(fnres, args.dist))
        base, iso = section_alloc(fnres, args.dist)
        print("\n### 기준선\n")
        print(base)
        print("\n### iso-accuracy 비교\n")
        print(iso)
        print("\n### 최적 배분 프론티어\n")
        print(section_frontier(fnres, args.dist))
        rule, pts = section_rule(fnres)
        print("\n### 유효 비트 vs T\n")
        print(rule)
        eps_tbl, eps_data = section_epsrule(fnres)
        print("\n### 목표 오차별 최소-스파이크 설정\n")
        print(eps_tbl)


if __name__ == "__main__":
    main()
