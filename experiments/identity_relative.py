"""실험 4a — FP 곱셈의 항등원(`MBE_Id`)을 **상대오차** 기준으로 비교.

spike-driven FP 곱셈은 두 피연산자를 각각 ``MBE_Id`` 로 복원한 뒤 곱한다
(``spiking_ops.spiking_multiply``). 따라서 곱의 상대오차는 두 피연산자 복원의
상대오차 합에 지배된다. 그런데 지금까지 항등원은 **절대 MSE** 로 적합·평가돼 왔고,
전역 MBE 는 ``[0, hi]`` 를 균일 해상도로 덮으므로 **작은 피연산자에서 상대오차가
폭발**한다. (기록: "FP-mult 상대오차 ~1.9%, 피연산자 범위에 비례".)

라우팅된 항등원은 다르다. 뱅크 ``[2^e, 2^{e+1})`` 의 목표는 ``2^e (1+rho)`` 이고,
``2^e`` 는 라우터가 **스파이크 0개로 정확히** 준다. 필요한 해상도는

    b = log2( delta / sqrt(eps) ) = log2( 2^e / (r * 2^e) ) = log2(1/r)

로 **지수 e 가 소거된다** — 모든 크기 뱅크가 같은 예산을 쓰고, 상대오차가 크기와
무관해진다. 이 스크립트가 그 예측을 검증한다.

측정 항목: 십진 자릿수별 상대 RMSE, 전체 상대 RMSE, 스파이크/입력, 저장 파라미터.

사용법::

    python experiments/identity_relative.py --json results/identity_relative.json
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
from mbe.fit import fit_model  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402
from mbe.neuron import MBEConfig, MBENeuron  # noqa: E402


def log_grid(lo, hi, m):
    """로그 균등 표본 — 자릿수마다 같은 수의 점을 준다(상대오차의 올바른 측정면)."""
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), m))


def rel_rmse(pred, x):
    return float(((pred - x) / x).pow(2).mean().sqrt())


def per_decade(pred, x, e_lo, e_hi):
    """binade 별 상대 RMSE."""
    out = []
    for e in range(e_lo, e_hi):
        m = (x >= 2.0 ** e) & (x < 2.0 ** (e + 1))
        if bool(m.any()):
            out.append((e, rel_rmse(pred[m], x[m]), int(m.sum())))
    return out


def tied_identity(lo, hi, e_lo, e_hi, N, T, epochs, seed):
    """자기유사성을 쓴 항등원 — **한 번만** 적합하고 지수는 라우터가 준다.

    크기 뱅크 ``[2^e, 2^{e+1})`` 의 목표는 ``2^e (1+rho)`` 이고 뱅크는 ``rho`` 만
    본다. 즉 **모든 뱅크의 목표가 스케일만 다른 같은 함수** ``1+rho`` 다. 그래서
    ``(w0, b0)`` 를 한 번 풀고 뱅크 ``e`` 는 ``w = 2^e w0``, ``bias = 2^e b0`` 로 두면
    끝이다 — 그리고 그 ``2^e`` 는 라우터가 이미 공짜로 읽은 지수부다.

    결과: 저장은 모양 파라미터 **1벌**(뱅크 수와 무관), 스파이크는 크기와 무관.
    구현은 :func:`mbe.mbe_pasn.build_mbe_pasn` 의 ``tied=True``; 여기서는 규칙이 고른
    ``(N, T)`` 대신 프론티어를 보려고 예산을 직접 지정한다.
    """
    return build_mbe_pasn("identity", (lo, hi), e_min=e_lo, e_max=e_hi,
                          budget="fixed", n_local=N, n_steps=T, tied=True,
                          near0="single", epochs=epochs, seed=seed,
                          alpha_init="uniform")


def global_identity(lo, hi, N, T, epochs, seed):
    """지금 쓰이는 형태: 절대 MSE 로 적합한 전역 MBE_Id (bias 없음)."""
    x, y, _ = functions.sample("identity", m=4000, seed=seed, domain=(lo, hi))
    cfg = functions.make_config("identity", n_basis=N, n_steps=T,
                                domain=(lo, hi), use_bias=False)
    m = MBENeuron(cfg)
    fit_model(m, x, y, seed=seed, epochs=epochs)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-lo", type=int, default=-8, help="피연산자 최소 binade")
    ap.add_argument("--e-hi", type=int, default=8, help="피연산자 최대 binade")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/identity_relative.json")
    args = ap.parse_args()

    lo, hi = 2.0 ** args.e_lo, 2.0 ** args.e_hi
    x = log_grid(lo, hi * 0.999, 20001)
    print(f"# 항등원 상대오차 — 피연산자 [{lo:.3g}, {hi:.3g}] "
          f"({args.e_hi - args.e_lo} binade)\n")
    print("| 모델 | 상대 RMSE | 스파이크/입력 | 파라미터 |")
    print("|---|---|---|---|")

    rows = []
    for N in (4, 8):
        m = global_identity(lo, hi, N, 16, args.epochs, args.seed)
        with torch.no_grad():
            pred = m.reconstruct(x)
            spk = m.firing_rate(x) * N * 16
        rows.append(dict(label=f"전역 MBE_Id N={N},T=16", rel=rel_rmse(pred, x),
                         spikes=spk,
                         params=sum(p.numel() for p in m.parameters()
                                    if p.requires_grad),
                         decades=per_decade(pred, x, args.e_lo, args.e_hi)))

    for tgt in (1e-3, 1e-5, 1e-7):
        m = build_mbe_pasn("identity", (lo, hi), e_min=args.e_lo,
                           e_max=args.e_hi, budget="rule", target_mse=tgt,
                           near0="single", epochs=args.epochs, seed=args.seed,
                           alpha_init="uniform")
        with torch.no_grad():
            pred = m(x)
        rows.append(dict(label=f"MBE-PASN 절대 eps={tgt:.0e}",
                         rel=rel_rmse(pred, x), spikes=m.mean_spikes(x),
                         params=m.num_learnable(),
                         decades=per_decade(pred, x, args.e_lo, args.e_hi),
                         budget=[(b, k, n, t) for b, k, n, t
                                 in m.budget_table()]))

    for r in (1e-2, 1e-3, 1e-4):
        m = build_mbe_pasn("identity", (lo, hi), e_min=args.e_lo,
                           e_max=args.e_hi, budget="rule", target="relative",
                           target_rel=r, near0="single", epochs=args.epochs,
                           seed=args.seed, alpha_init="uniform")
        with torch.no_grad():
            pred = m(x)
        rows.append(dict(label=f"**MBE-PASN 상대 r={r:.0e}**",
                         rel=rel_rmse(pred, x), spikes=m.mean_spikes(x),
                         params=m.num_learnable(),
                         decades=per_decade(pred, x, args.e_lo, args.e_hi),
                         budget=[(b, k, n, t) for b, k, n, t
                                 in m.budget_table()]))

    for N, T in ((1, 8), (1, 16), (2, 16)):
        m = tied_identity(lo, hi, args.e_lo, args.e_hi, N, T, args.epochs,
                          args.seed)
        with torch.no_grad():
            pred = m(x)
        rows.append(dict(label=f"**연결(tied) N={N},T={T}**", rel=rel_rmse(pred, x),
                         spikes=m.mean_spikes(x), params=m.num_learnable(),
                         decades=per_decade(pred, x, args.e_lo, args.e_hi)))

    for r in rows:
        print(f"| {r['label']} | {r['rel']:.3e} | {r['spikes']:.2f} | "
              f"{r['params']} |")

    print("\n## binade 별 상대 RMSE (크기와 무관해지는가)\n")
    es = [e for e, _, _ in rows[0]["decades"]]
    print("| 모델 | " + " | ".join(f"2^{e}" for e in es[::2]) + " |")
    print("|---" * (len(es[::2]) + 1) + "|")
    for r in rows:
        d = {e: v for e, v, _ in r["decades"]}
        print(f"| {r['label']} | " +
              " | ".join(f"{d.get(e, float('nan')):.1e}" for e in es[::2]) + " |")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(dict(meta=vars(args), rows=rows), f, ensure_ascii=False,
                  indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
