"""실험 7 — 차수별 예산 규칙 재적합 + 발화율 기록.

실험 6이 남긴 두 구멍을 메운다.

1. **예산 규칙의 상수가 차수 1에서 적합된 것이다.** `T_j = 2^((b-2)/1.7)` 의 `(2, 1.7)`
   은 `readout_order=1` 에서 잰 값이라, 차수 ≥ 2 에서는 `T` 가 과잉 배정된다. 즉
   실험 6의 엔드투엔드 숫자는 차수 2를 **과소평가**한다.
2. **발화율을 기록하지 않았다.** 스파이크 = `발화율 × N × T` 인데 비트 법칙은 `T` 당
   비트였다. 이진탐색 커널은 구조적으로 절반쯤 발화하므로 `T` 당 해상도가 좋아도
   스파이크당으로는 다를 수 있다. **에너지 축의 법칙은 따로 재야 한다.**

그래서 두 법칙을 모두 적합한다.

    시간 법칙  bits = a_T + c_T * log2(T)        <- 예산 규칙이 쓰는 것
    에너지 법칙 bits = a_S + c_S * log2(spikes)   <- 실제로 최적화해야 하는 것

사용법 (설정별로 샤딩)::

    python experiments/budget_refit.py --mode measure --order 2 --kernel halving \
        --json shard.json
    python experiments/budget_refit.py --mode analyse --shards shard*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.fit import fit_model  # noqa: E402
from mbe.mbe_pasn import (T_GRID, _grid, _uniform_alpha,  # noqa: E402
                          build_mbe_pasn, rule_budget)
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


def measure(order, learn_tau, Ns, epochs, seed):
    tau = (1.0, 8.0) if learn_tau else (HALVING, HALVING)
    rows = []
    for fn_name, s, e in BANK_TARGETS:
        xf, yf, xe, ye = bank_problem(fn_name, s, e)
        delta = float(yf.max() - yf.min())
        if delta <= 0:
            continue
        for N in Ns:
            for T in T_GRID:
                cfg = MBEConfig(n_basis=N, n_steps=T, x_min=0.0, x_scale=1.0,
                                alpha_v=_uniform_alpha(N), use_bias=True,
                                readout_order=order, learn_tau=learn_tau,
                                tau_min=tau[0], tau_max=tau[1])
                m = MBENeuron(cfg)
                fit_model(m, xf, yf, seed=seed, epochs=epochs)
                with torch.no_grad():
                    mse = float((m(xe) - ye).pow(2).mean())
                    fr = m.firing_rate(xe)
                if mse <= 0:
                    continue
                rows.append(dict(
                    fn=fn_name, sign=s, e=e, N=N, T=T, mse=mse, fr=fr,
                    spikes=fr * N * T, delta=delta,
                    bits=math.log2(delta / math.sqrt(mse)),
                    params=sum(p.numel() for p in m.parameters()
                               if p.requires_grad)))
        print(f"  {fn_name} s={s:+.0f} e={e}", flush=True)
    return rows


def lsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    resid = [y - (my + c * (x - mx)) for x, y in zip(xs, ys)]
    rms = (sum(r * r for r in resid) / n) ** 0.5
    return c, my - c * mx, rms


def analyse(shards, epochs, seed, e2e=True):
    data = {}
    for p in shards:
        d = json.load(open(p, encoding="utf-8"))
        data[(d["order"], d["learn_tau"])] = d["rows"]

    print("# 1. 두 법칙 재적합\n")
    print("| 차수 | 커널 | 시간법칙 c_T | a_T | **에너지법칙 c_S** | a_S | "
          "평균 발화율 | 잔차(bits) |")
    print("|---|---|---|---|---|---|---|---|")
    laws = {}
    for (order, lt), rows in sorted(data.items()):
        one = [r for r in rows if r["N"] == 1]
        cT, aT, _ = lsq([math.log2(r["T"]) for r in one],
                        [r["bits"] for r in one])
        cS, aS, rs = lsq([math.log2(r["spikes"]) for r in rows if r["spikes"] > 0],
                         [r["bits"] for r in rows if r["spikes"] > 0])
        fr = sum(r["fr"] for r in rows) / len(rows)
        laws[(order, lt)] = dict(aT=aT, cT=cT, aS=aS, cS=cS, fr=fr)
        print(f"| {order} | {'학습' if lt else '이진탐색 고정'} | {cT:.2f} | "
              f"{aT:.2f} | **{cS:.2f}** | {aS:.2f} | {fr:.3f} | {rs:.2f} |")

    print("\n\n# 2. 같은 비트를 내는 데 드는 스파이크 (에너지법칙으로 환산)\n")
    print("| 차수 | 커널 | b=6 | b=8 | b=10 | 차수1/학습 대비 |")
    print("|---|---|---|---|---|---|")
    base = laws[(1, True)]

    def spk(law, b):
        return 2.0 ** ((b - law["aS"]) / law["cS"])

    for (order, lt), law in sorted(laws.items()):
        vals = [spk(law, b) for b in (6, 8, 10)]
        ref = spk(base, 8) / spk(law, 8)
        print(f"| {order} | {'학습' if lt else '고정'} | " +
              " | ".join(f"{v:.1f}" for v in vals) +
              f" | **{ref:.2f}×** |")

    print("\n\n# 3. 재적합한 규칙으로 다시 지은 MBE-PASN "
          "(도메인 −8..8, ε=1e−5)\n")
    if not e2e:
        return laws
    print("| 함수 | 차수 | 커널 | 규칙 | MSE | 스파이크 | 파라미터 |")
    print("|---|---|---|---|---|---|---|")
    x = torch.linspace(*DOMAIN, 4001)[1:-1]
    for name in ("gelu", "relu", "tanh"):
        fn, _ = functions.REGISTRY[name]
        y = fn(x)
        for (order, lt), law in sorted(laws.items()):
            for tag, (a, c) in (("옛(2,1.7)", (2.0, 1.7)),
                                ("재적합", (law["aT"], law["cT"]))):
                kw = dict(readout_order=order, learn_tau=lt)
                if not lt:
                    kw["tau_range"] = (HALVING, HALVING)
                m = build_mbe_pasn(name, DOMAIN, e_min=-2, e_max=4,
                                   budget="rule", target_mse=1e-5,
                                   near0="signed", epochs=epochs, seed=seed,
                                   alpha_init="uniform", rule_ac=(a, c), **kw)
                with torch.no_grad():
                    mse = float((m(x) - y).pow(2).mean())
                print(f"| {name} | {order} | {'학습' if lt else '고정'} | {tag} "
                      f"| {mse:.2e} | {m.mean_spikes(x):.2f} | "
                      f"{m.num_learnable()} |", flush=True)
    return laws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "analyse"], default="measure")
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--kernel", choices=["learned", "halving"], default="learned")
    ap.add_argument("--Ns", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", nargs="+", default=[])
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--json", default="results/budget_refit.json")
    args = ap.parse_args()

    if args.mode == "measure":
        lt = args.kernel == "learned"
        print(f"[measure] order={args.order} kernel={args.kernel}", flush=True)
        rows = measure(args.order, lt, args.Ns, args.epochs, args.seed)
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(order=args.order, learn_tau=lt, rows=rows), f,
                      ensure_ascii=False)
        print(f"wrote {args.json} ({len(rows)} rows)")
    else:
        shards = args.shards or sorted(glob.glob("results/_refit_*.json"))
        laws = analyse(shards, args.epochs, args.seed, e2e=not args.no_e2e)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({f"{k[0]}_{k[1]}": v for k, v in laws.items()}, f,
                      ensure_ascii=False, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
