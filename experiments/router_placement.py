"""실험 5 — 라우터 배치 ``(beta, gamma)``.

binade 격자는 0에 고정돼 있어 로그 조밀 구간이 항상 0 근처에 생긴다. 목표가 어려운
지점이 0이 아니면 그 해상도가 엉뚱한 곳에 쓰인다. 키 ``t = (x - beta)/2^g`` 로 라우팅하면
조밀점이 ``beta`` 로 옮겨간다.

가장 중요한 사례는 **좁은 도메인 프리미티브**다. ``1/S`` 의 인자는 이미 ``frexp`` 로 뽑은
가수라 ``[0.5, 1)`` — **binade 1개** 뿐이고 지수 라우터가 전역 뉴런으로 퇴화한다. 그런데
곡률 ``2/S^3`` 은 ``S = 0.5`` 에서 최대다. ``beta = 0.5`` 는 바로 그 끝을 여러 binade로
다시 펼친다.

``gamma`` 는 2의 거듭제곱으로 제한한다 — ``2^g`` 나누기는 지수 필드의 정수 뺄셈이라
라우터가 비트 연산으로 남는다. 또 gamma 는 도메인이 걸치는 binade **개수를 못 바꾼다**
(``[a,b] -> [ga,gb]``, 비율 동일). 범위를 늘리는 건 beta 다.

대조군이 핵심이다: 활성함수(gelu/relu/silu)는 어려운 점이 이미 0이므로 **beta=0 이
최적이어야 한다.** 활성함수까지 좋아지면 beta 가 "어려운 점을 찾는" 게 아니라 그냥
용량을 늘리는 것이므로 해석을 다시 해야 한다.

사용법::

    python experiments/router_placement.py --json results/router_placement.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402

# 목표: (도메인, beta 후보, 왜 그 후보인가)
SETUP = {
    "inv":     ((0.5, 1.0),   [0.0, 0.5, 1.0],  "곡률 2/S^3 최대 = 0.5 (좌단)"),
    "invsqrt": ((0.5, 2.0),   [0.0, 0.5, 2.0],  "곡률 최대 = 0.5 (좌단)"),
    "exp2":    ((0.0, 1.0),   [0.0, 1.0],       "곡률 최대 = 1 (우단)"),
    "gelu":    ((-8.0, 8.0),  [0.0, -8.0, 8.0], "대조군 — 굴곡이 0"),
    "relu":    ((-8.0, 8.0),  [0.0, -8.0, 8.0], "대조군 — 꺾임이 0"),
    "silu":    ((-8.0, 8.0),  [0.0, -8.0, 8.0], "대조군 — 굴곡이 0"),
}


def e_range(domain, beta, gamma_log2, n_binades):
    """키 공간에서 ``n_binades`` 개를 쓰도록 ``(e_min, e_max)`` 를 잡는다."""
    g = 2.0 ** gamma_log2
    t = [(domain[0] - beta) / g, (domain[1] - beta) / g]
    mag = max(abs(t[0]), abs(t[1]))
    if mag <= 0:
        return -1, 0
    e_max = int(math.ceil(math.log2(mag)))
    return e_max - n_binades, e_max


def evaluate(name, domain, beta, gamma_log2, n_binades, targets, epochs, seed,
             m=4001):
    fn, _ = functions.REGISTRY[name]
    x = torch.linspace(domain[0], domain[1], m)
    x = x[(x > domain[0]) & (x < domain[1])] if name in ("inv",) else x
    y = fn(x)
    e_min, e_max = e_range(domain, beta, gamma_log2, n_binades)
    out = []
    for tm in targets:
        try:
            mdl = build_mbe_pasn(name, domain, e_min=e_min, e_max=e_max,
                                 budget="rule", target_mse=tm, near0="signed",
                                 beta=beta, gamma_log2=gamma_log2,
                                 epochs=epochs, seed=seed, alpha_init="uniform")
            with torch.no_grad():
                mse = float((mdl(x) - y).pow(2).mean())
            out.append(dict(target=tm, mse=mse, spikes=mdl.mean_spikes(x),
                            params=mdl.num_learnable(),
                            banks=sum(1 for _ in mdl.bank_mods)))
        except Exception as exc:                       # noqa: BLE001
            out.append(dict(target=tm, error=str(exc)[:80]))
    return out


def best_at(points, mse_cap):
    ok = [p for p in points if "mse" in p and p["mse"] <= mse_cap]
    return min(ok, key=lambda p: p["spikes"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fns", nargs="+", default=list(SETUP))
    ap.add_argument("--n-binades", type=int, default=6)
    ap.add_argument("--gammas", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/router_placement.json")
    args = ap.parse_args()

    targets = [10.0 ** -k for k in (3, 4, 5, 6, 7)]
    res = {}
    for name in args.fns:
        domain, betas, why = SETUP[name]
        print(f"\n## {name}  도메인 {domain}  ({why})\n", flush=True)
        print("| beta | gamma | 뱅크 | 최저 MSE | 그때 스파이크 | 파라미터 |")
        print("|---|---|---|---|---|---|")
        res[name] = dict(domain=list(domain), why=why, runs=[])
        for beta in betas:
            for g in args.gammas:
                pts = evaluate(name, domain, beta, g, args.n_binades, targets,
                               args.epochs, args.seed)
                res[name]["runs"].append(dict(beta=beta, gamma_log2=g,
                                              points=pts))
                ok = [p for p in pts if "mse" in p]
                if not ok:
                    print(f"| {beta:g} | 2^{g} | — | 실패 | | |")
                    continue
                b = min(ok, key=lambda p: p["mse"])
                print(f"| {beta:g} | 2^{g} | {b['banks']} | {b['mse']:.2e} | "
                      f"{b['spikes']:.2f} | {b['params']} |", flush=True)

    print("\n\n# iso-accuracy 요약 — beta=0 대비\n")
    print("| 함수 | 타깃 MSE | beta=0 스파이크 | 최적 beta | 스파이크 | 절감 |")
    print("|---|---|---|---|---|---|")
    for name, r in res.items():
        base = next(x for x in r["runs"] if x["beta"] == 0.0
                    and x["gamma_log2"] == 0)
        ok = [p for p in base["points"] if "mse" in p]
        if not ok:
            continue
        cap = min(p["mse"] for p in ok) * 1.5
        b0 = best_at(base["points"], cap)
        best, bb = b0, 0.0
        for run in r["runs"]:
            c = best_at(run["points"], cap)
            if c and (best is None or c["spikes"] < best["spikes"]):
                best, bb = c, run["beta"]
        if b0 and best:
            print(f"| {name} | {cap:.1e} | {b0['spikes']:.2f} | {bb:g} | "
                  f"{best['spikes']:.2f} | **{b0['spikes']/best['spikes']:.2f}×** |")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
