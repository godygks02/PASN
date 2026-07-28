"""실험 6 — 디코더 차수와 커널 고정: 유효 비트 법칙을 바꿀 수 있는가.

실험 1이 측정한 법칙은 **유효 비트 ≈ 2 + 1.7·log₂T** 였다. 이론상 halving 커널
(``tau = dt/ln2``)은 basis 하나를 T비트 이진탐색(SAR)으로 만드는데, 실제로는 T=16에서
8.8비트뿐이다. 왜인가?

MBE의 디코드 ``f = sum_n w_n o_n`` 은 **특징에 선형**이다. basis 하나가 완벽한 SAR이면
``o = rho_hat`` 이므로 선형 readout은 **rho의 어파인 함수밖에** 못 만든다. 목표는 굽어
있으니, 최적화기는 커널을 halving에서 벗어나게 해 **계단 자체를 휘어** 곡률을 흉내내고,
그 대가로 해상도를 잃는다 -- 그래서 레벨 수가 T에 대해 로그로만 자란다.

가설: **곡률을 디코더가 담당하면 계단이 다시 효율적인 코드가 될 수 있다.**

  * ``readout_order``: ``[o, o^2, ...]`` -- 스파이크 0개 추가, 저장 계수만 증가.
  * ``learn_tau=False`` + ``tau = dt/ln2``: 커널을 이진탐색에 고정.

측정은 두 층이다.

  1. **비트 법칙** -- 실제 뱅크 목표들에 대해 ``bits = log2(delta/RMSE)`` 를 T에 대해
     회귀. 기울기 c가 1.7에서 올라가는가.
  2. **엔드투엔드** -- 같은 설정으로 MBE-PASN 전체를 지어 (MSE, 스파이크, 파라미터).

사용법::

    python experiments/readout_order.py --json results/readout_order.json
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
from mbe.mbe_pasn import _grid, _uniform_alpha, build_mbe_pasn  # noqa: E402
from mbe.fit import fit_model  # noqa: E402
from mbe.neuron import MBEConfig, MBENeuron  # noqa: E402

HALVING = 1.0 / math.log(2.0)          # tau with dt=1 -> every step halves
TS = (2, 3, 4, 6, 8, 12, 16)
DOMAIN = (-8.0, 8.0)
# 비트 법칙 측정용 뱅크 목표: (함수, 부호, binade e). 라우팅된 잔차 rho in [0,1) 에서
# g(rho) = f(sign * 2^e (1+rho)) 를 근사하는 문제 -- 실제 뱅크가 푸는 문제 그대로.
BANK_TARGETS = [(fn, s, e) for fn in ("gelu", "silu", "tanh", "sigmoid")
                for s, e in ((1.0, -1), (1.0, 1), (-1.0, 0))]


def bank_problem(fn_name, sign, e):
    fn, _ = functions.REGISTRY[fn_name]
    rho = _grid(0.0, 1.0, 1024, 0.0)
    rho_e = _grid(0.0, 1.0, 1024, 0.25)
    lo = 2.0 ** e
    return (rho, fn(sign * lo * (1.0 + rho)),
            rho_e, fn(sign * lo * (1.0 + rho_e)))


def fit_one(xf, yf, xe, ye, N, T, order, learn_tau, epochs, seed):
    tau = (HALVING, HALVING) if not learn_tau else (1.0, 8.0)
    cfg = MBEConfig(n_basis=N, n_steps=T, x_min=0.0, x_scale=1.0,
                    alpha_v=_uniform_alpha(N), use_bias=True,
                    readout_order=order, learn_tau=learn_tau,
                    tau_min=tau[0], tau_max=tau[1])
    m = MBENeuron(cfg)
    fit_model(m, xf, yf, seed=seed, epochs=epochs)
    with torch.no_grad():
        mse = float((m(xe) - ye).pow(2).mean())
        fr = m.firing_rate(xe)
    return mse, fr, sum(p.numel() for p in m.parameters() if p.requires_grad)


def slope(pairs):
    """bits = a + c*log2(T) 최소제곱 기울기 c."""
    xs = [math.log2(t) for t, _ in pairs]
    ys = [b for _, b in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return c, my - c * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--json", default="results/readout_order.json")
    args = ap.parse_args()

    res = dict(bits={}, e2e={})

    # ---------------- 1. 비트 법칙 ----------------
    print("# 1. 유효 비트 법칙  bits = a + c*log2(T)   (N=1, 뱅크 목표 12개)\n")
    print("| 디코더 차수 | 커널 | 기울기 c | 절편 a | bits@T=16 | vs 선형학습 |")
    print("|---|---|---|---|---|---|")
    base_bits16 = None
    for order in args.orders:
        for learn_tau in (True, False):
            per_T = {T: [] for T in TS}
            for fn_name, s, e in BANK_TARGETS:
                xf, yf, xe, ye = bank_problem(fn_name, s, e)
                delta = float(yf.max() - yf.min())
                if delta <= 0:
                    continue
                for T in TS:
                    mse, _, _ = fit_one(xf, yf, xe, ye, 1, T, order, learn_tau,
                                        args.epochs, args.seed)
                    if mse > 0:
                        per_T[T].append(math.log2(delta / math.sqrt(mse)))
            pairs = [(T, sum(v) / len(v)) for T, v in per_T.items() if v]
            c, a = slope(pairs)
            b16 = a + c * 4.0
            key = f"order{order}_{'learned' if learn_tau else 'halving'}"
            res["bits"][key] = dict(slope=c, intercept=a, bits16=b16,
                                    points=pairs)
            if base_bits16 is None:
                base_bits16 = b16
            print(f"| {order} | {'학습' if learn_tau else '이진탐색 고정'} | "
                  f"**{c:.2f}** | {a:.2f} | {b16:.1f} | "
                  f"{2 ** (b16 - base_bits16):.2f}× |", flush=True)

    # ---------------- 2. 엔드투엔드 ----------------
    if not args.skip_e2e:
        print("\n\n# 2. MBE-PASN 전체 (도메인 -8..8, 규칙 예산 eps=1e-5)\n")
        print("| 함수 | 차수 | 커널 | MSE | 스파이크 | 파라미터 |")
        print("|---|---|---|---|---|---|")
        x = torch.linspace(*DOMAIN, 4001)[1:-1]
        for name in ("gelu", "relu", "silu", "tanh", "sigmoid"):
            fn, _ = functions.REGISTRY[name]
            y = fn(x)
            res["e2e"][name] = []
            for order in args.orders:
                for learn_tau in (True, False):
                    kw = dict(readout_order=order, learn_tau=learn_tau)
                    if not learn_tau:
                        kw["tau_range"] = (HALVING, HALVING)
                    m = build_mbe_pasn(name, DOMAIN, e_min=-2, e_max=4,
                                       budget="rule", target_mse=1e-5,
                                       near0="signed", epochs=args.epochs,
                                       seed=args.seed, alpha_init="uniform",
                                       **kw)
                    with torch.no_grad():
                        mse = float((m(x) - y).pow(2).mean())
                    row = dict(order=order, learn_tau=learn_tau, mse=mse,
                               spikes=m.mean_spikes(x),
                               params=m.num_learnable())
                    res["e2e"][name].append(row)
                    print(f"| {name} | {order} | "
                          f"{'학습' if learn_tau else '고정'} | {mse:.2e} | "
                          f"{row['spikes']:.2f} | {row['params']} |", flush=True)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
