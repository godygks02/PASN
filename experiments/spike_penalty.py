"""실험 8 — 발화율을 손실항으로.

스파이크 `= 발화율 × N × T` 인데, 지금까지 우리는 `N·T` 만 배분했고 **발화율은 한 번도
최적화 대상이 아니었다.** 적합은 MSE만 최소화하고 발화율은 임계값이 어디 놓이느냐에
따라 나오는 부산물이었다 (실험 7 실측: 학습 커널 0.41, 이진탐색 커널 0.54).

    L = MSE + lambda * 발화율

readout `(w, bias)` 는 스파이크 동역학에 전혀 들어가지 않으므로, 이 항을 더해도
**닫힌형 readout solve 는 여전히 정확히 최적**이다. 런타임 비용도 0이다.

**대조가 핵심이다.** 스파이크는 `T` 를 줄여도 언제든 줄어든다. 그러니 이 실험이
의미가 있으려면 *"`T` 고정 + lambda 스윕"* 프론티어가 *"lambda=0 + `T` 스윕"*
프론티어보다 **바깥에 있어야** 한다. 안 그러면 그냥 T를 줄이면 되는 것이다.

사용법::

    python experiments/spike_penalty.py --json results/spike_penalty.json
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
from mbe.mbe_pasn import T_GRID, _grid, _uniform_alpha, build_mbe_pasn  # noqa: E402
from mbe.neuron import MBEConfig, MBENeuron  # noqa: E402

HALVING = 1.0 / math.log(2.0)
DOMAIN = (-8.0, 8.0)
BANK_TARGETS = [(fn, s, e) for fn in ("gelu", "silu", "tanh", "sigmoid")
                for s, e in ((1.0, -1), (1.0, 1), (-1.0, 0))]


def bank_problem(fn_name, sign, e):
    fn, _ = functions.REGISTRY[fn_name]
    rho, rho_e = _grid(0.0, 1.0, 1024, 0.0), _grid(0.0, 1.0, 1024, 0.25)
    lo = 2.0 ** e
    return (rho, fn(sign * lo * (1.0 + rho)),
            rho_e, fn(sign * lo * (1.0 + rho_e)))


def fit_one(xf, yf, xe, ye, N, T, order, learn_tau, lam, epochs, seed):
    tau = (1.0, 8.0) if learn_tau else (HALVING, HALVING)
    cfg = MBEConfig(n_basis=N, n_steps=T, x_min=0.0, x_scale=1.0,
                    alpha_v=_uniform_alpha(N), use_bias=True,
                    readout_order=order, learn_tau=learn_tau,
                    tau_min=tau[0], tau_max=tau[1])
    m = MBENeuron(cfg)
    fit_model(m, xf, yf, seed=seed, epochs=epochs, spike_lambda=lam)
    with torch.no_grad():
        mse = float((m(xe) - ye).pow(2).mean())
        fr = m.firing_rate(xe)
    return dict(N=N, T=T, lam=lam, mse=mse, fr=fr, spikes=fr * N * T)


def pareto(points):
    """(spikes, mse) 프론티어만 남긴다."""
    out = []
    for p in sorted(points, key=lambda p: p["spikes"]):
        if not out or p["mse"] < out[-1]["mse"]:
            out.append(p)
    return out


def geo(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kernels", nargs="+", choices=["learned", "halving"],
                    default=["learned"])
    ap.add_argument("--lams", nargs="+", type=float,
                    default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--lam-T", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--json", default="results/spike_penalty.json")
    args = ap.parse_args()

    print(f"# 실험 8/9 — 발화율 손실항 (차수 {args.order}, "
          f"커널 {', '.join(args.kernels)})\n")

    # ---- 뱅크 목표 × 커널마다 두 프론티어를 만든다 -----------------------
    rows = {"T_sweep": [], "lam_sweep": []}
    per_target = []                       # (fn, s, e, kernel, paretoA, paretoB)
    for kernel in args.kernels:
        lt = kernel == "learned"
        for fn_name, s, e in BANK_TARGETS:
            xf, yf, xe, ye = bank_problem(fn_name, s, e)
            # A = 지금까지의 방법(T 만 조절). B = A 를 **포함하는** (T, lambda) 격자.
            # B ⊇ A 이므로 B 가 A 보다 나쁠 수는 없다. 물어야 할 것은
            # "lambda>0 점이 공동 프론티어에 하나라도 오르는가" 다.
            A, B = [], []
            for T in T_GRID:
                for lam in args.lams:
                    r = fit_one(xf, yf, xe, ye, 1, T, args.order, lt, lam,
                                args.epochs, args.seed)
                    r.update(fn=fn_name, e=e, kernel=kernel)
                    B.append(r)
                    if lam == 0.0:
                        A.append(r)
            rows["T_sweep"] += A
            rows["lam_sweep"] += [r for r in B if r["lam"] > 0]
            per_target.append((fn_name, s, e, kernel, pareto(A), pareto(B)))
            print(f"  [{kernel}] {fn_name} s={s:+.0f} e={e} 완료", flush=True)

    # ---- 같은 스파이크에서 어느 쪽 MSE가 낮은가 --------------------------
    print("\n## 1. lambda 가 각 커널의 프론티어를 밀어내는가 "
          "(T만 조절 대비, 기하평균)\n")
    print("| 커널 | ≤1 스파이크 | ≤2 | ≤4 | ≤8 | 프론티어 위 lambda>0 비율 |")
    print("|---|---|---|---|---|---|")
    for kernel in args.kernels:
        sub = [t for t in per_target if t[3] == kernel]
        cells = []
        for budget in (1.0, 2.0, 4.0, 8.0):
            g = []
            for _, _, _, _, A, B in sub:
                a = [p for p in A if p["spikes"] <= budget]
                b = [p for p in B if p["spikes"] <= budget]
                if a and b:
                    g.append(min(p["mse"] for p in a) / min(p["mse"] for p in b))
            cells.append(f"**{geo(g):.2f}×**" if g else "도달 불가")
        n_lam = sum(sum(1 for p in B if p["lam"] > 0) for *_, B in sub)
        n_all = sum(len(B) for *_, B in sub)
        print(f"| {kernel} | " + " | ".join(cells) +
              f" | {n_lam}/{n_all} |")

    if len(args.kernels) > 1:
        print("\n## 2. 커널 정면 비교 — 같은 스파이크 예산에서의 MSE "
              "(각자 최선의 (T, lambda))\n")
        print("| 스파이크 | " + " | ".join(args.kernels) + " | 고정/학습 |")
        print("|---" * (len(args.kernels) + 2) + "|")
        for budget in (1.0, 2.0, 4.0, 8.0, 16.0):
            per_k = {}
            for kernel in args.kernels:
                g = []
                for _, _, _, k, _, B in per_target:
                    if k != kernel:
                        continue
                    b = [p for p in B if p["spikes"] <= budget]
                    if b:
                        g.append(min(p["mse"] for p in b))
                per_k[kernel] = geo(g) if g else None
            if any(v is None for v in per_k.values()):
                continue
            ratio = (per_k["learned"] / per_k["halving"]
                     if {"learned", "halving"} <= set(per_k) else float("nan"))
            print(f"| ≤{budget:g} | " +
                  " | ".join(f"{per_k[k]:.2e}" for k in args.kernels) +
                  f" | **{ratio:.2f}×** |")

    # ---- lambda 가 실제로 발화율을 낮추는가 ------------------------------
    print("\n## 3. lambda 가 발화율에 미치는 효과 (T=16)\n")
    print("| 커널 | lambda | 평균 발화율 | 평균 스파이크 | 기하평균 MSE |")
    print("|---|---|---|---|---|")
    allrows = rows["T_sweep"] + rows["lam_sweep"]
    for kernel in args.kernels:
        for lam in args.lams:
            sub = [r for r in allrows if r["lam"] == lam and r["T"] == 16
                   and r["kernel"] == kernel]
            if not sub:
                continue
            print(f"| {kernel} | {lam:g} | "
                  f"{sum(r['fr'] for r in sub) / len(sub):.3f} | "
                  f"{sum(r['spikes'] for r in sub) / len(sub):.2f} | "
                  f"{geo([r['mse'] for r in sub]):.2e} |")

    # ---- 엔드투엔드 ------------------------------------------------------
    if not args.skip_e2e:
        print("\n## 4. MBE-PASN 전체 (도메인 −8..8, 규칙 예산 ε=1e−5)\n")
        print("| 함수 | 커널 | lambda | MSE | 스파이크 | 파라미터 |")
        print("|---|---|---|---|---|---|")
        x = torch.linspace(*DOMAIN, 4001)[1:-1]
        for name in ("gelu", "relu", "tanh"):
            fn, _ = functions.REGISTRY[name]
            y = fn(x)
            for kernel in args.kernels:
                lt = kernel == "learned"
                for lam in (0.0, 1e-3, 1e-2):
                    kw = dict(readout_order=args.order, learn_tau=lt)
                    if not lt:
                        kw["tau_range"] = (HALVING, HALVING)
                    m = build_mbe_pasn(name, DOMAIN, e_min=-2, e_max=4,
                                       budget="rule", target_mse=1e-5,
                                       near0="signed", epochs=args.epochs,
                                       seed=args.seed, alpha_init="uniform",
                                       spike_lambda=lam, **kw)
                    with torch.no_grad():
                        mse = float((m(x) - y).pow(2).mean())
                    print(f"| {name} | {kernel} | {lam:g} | {mse:.2e} | "
                          f"{m.mean_spikes(x):.2f} | {m.num_learnable()} |",
                          flush=True)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
