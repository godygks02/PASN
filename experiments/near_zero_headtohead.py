"""near-zero 구간만 떼어낸 정면 비교 — MBE-PASN vs Signed MBE vs 전역 MBE.

실험 2는 **도메인 전체의 가중 MSE** 로 비교했다. 그런데 Signed MBE 도 정확히 0에서
극성 분할을 하므로, 전체에서 지면서 **0 근처에서는 더 정확할** 수 있다. 이 스크립트는
그 구간만 잘라서 직접 비교한다.

두 가지를 같이 본다.

* **절대 MSE** — 실험 1/2가 쓴 지표.
* **상대 오차** ``RMSE / Δ`` (Δ = 그 구간 안 f의 동적범위). 0 근처는 |f| 가 작아
  절대 오차가 작은 게 당연하므로, 원래 MBE 실패 서사(0 근처 상대 정확도)를 재려면
  이쪽이 맞다.

에너지도 같이 낸다 — 정확도만 비교하면 의미가 없다.

사용법::

    python experiments/near_zero_headtohead.py --fns gelu silu relu tanh sigmoid
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.fit import fit_model  # noqa: E402
from mbe.neuron import MBEConfig, MBENeuron  # noqa: E402
from bank_budget import fit_cell, grid  # noqa: E402


def stats(pred, y, delta, spikes):
    mse = float((pred - y).pow(2).mean())
    return dict(mse=mse, rel=float(mse ** 0.5) / delta, spikes=spikes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fns", nargs="+",
                    default=["gelu", "silu", "relu", "tanh", "sigmoid"])
    ap.add_argument("--domain", nargs=2, type=float, default=[-8.0, 8.0])
    ap.add_argument("--span", type=float, default=0.25,
                    help="near-zero 구간 반폭 (= 2^e_min)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lo, hi = args.domain
    S = args.span

    print(f"# near-zero 정면 비교 — 구간 (−{S}, +{S}), 도메인 ({lo}, {hi})\n")
    print("| 함수 | 모델 | 절대 MSE | 상대 RMSE/Δ | 스파이크/입력 |")
    print("|---|---|---|---|---|")

    for name in args.fns:
        fn, _ = functions.REGISTRY[name]
        xe = grid(-S, S, 2048, 0.25)
        ye = fn(xe)
        delta = float(ye.max() - ye.min())

        # --- 도메인 전체에 적합한 기준선들을, 이 구간에서만 평가 -------------
        xg, yg = grid(lo, hi, 4000, 0.0), fn(grid(lo, hi, 4000, 0.0))
        rows = []
        for label, model, n_eff in (
            ("전역 MBE N=8",
             MBENeuron(functions.make_config(name, 8, n_steps=16,
                                             domain=(lo, hi), use_bias=True)), 8),
            ("Signed MBE 4x2",
             functions.make_signed(name, 4, 4, pivot=0.0, n_steps=16,
                                   domain=(lo, hi)), 8),
        ):
            fit_model(model, xg, yg, epochs=args.epochs, seed=args.seed)
            with torch.no_grad():
                rows.append((label, stats(model(xe), ye, delta,
                                          model.firing_rate(xe) * n_eff * 16)))

        # --- MBE-PASN 의 near-zero 뱅크: 단일 vs 부호분할 --------------------
        d = dict(xf=grid(-S, S, 1024, 0.0), xe=grid(-S, S, 1024, 0.25),
                 x_min=-S, x_scale=2 * S, sign=None)
        d["yf"], d["ye"] = fn(d["xf"]), fn(d["xe"])
        cells = [fit_cell(fn, d, N, T, a, args.epochs, args.seed)
                 for N in (1, 2, 4) for T in (2, 4, 8, 16)
                 for a in ("uniform", "logspread")]
        c = min(cells, key=lambda c: c["mse"])
        rows.append((f"MBE-PASN 단일 (N={c['N']},T={c['T']})",
                     dict(mse=c["mse"], rel=c["mse"] ** 0.5 / delta,
                          spikes=c["spikes"])))

        halves = []
        for sgn in (-1.0, 1.0):
            dh = dict(xf=grid(0.0, S, 1024, 0.0), xe=grid(0.0, S, 1024, 0.25),
                      x_min=0.0, x_scale=S, sign=sgn)
            dh["yf"], dh["ye"] = fn(sgn * dh["xf"]), fn(sgn * dh["xe"])
            hs = [fit_cell(fn, dh, N, T, a, args.epochs, args.seed)
                  for N in (1, 2, 4) for T in (2, 4, 8, 16)
                  for a in ("uniform", "logspread")]
            halves.append(min(hs, key=lambda c: c["mse"]))
        mse2 = sum(h["mse"] for h in halves) / 2
        spk2 = sum(h["spikes"] for h in halves) / 2
        rows.append((f"**MBE-PASN 부호분할** "
                     f"(N={halves[0]['N']}/{halves[1]['N']},"
                     f"T={halves[0]['T']}/{halves[1]['T']})",
                     dict(mse=mse2, rel=mse2 ** 0.5 / delta, spikes=spk2)))

        # 같은 스파이크 예산에서의 부호분할도 (정확도만 비교하면 불공정)
        budget = rows[1][1]["spikes"]
        cheap = []
        for sgn, hs_all in zip((-1.0, 1.0), (None, None)):
            dh = dict(xf=grid(0.0, S, 1024, 0.0), xe=grid(0.0, S, 1024, 0.25),
                      x_min=0.0, x_scale=S, sign=sgn)
            dh["yf"], dh["ye"] = fn(sgn * dh["xf"]), fn(sgn * dh["xe"])
            hs = [fit_cell(fn, dh, N, T, a, args.epochs, args.seed)
                  for N in (1, 2, 4) for T in (2, 4, 8, 16)
                  for a in ("uniform", "logspread")]
            ok = [h for h in hs if h["spikes"] <= budget]
            cheap.append(min(ok or hs, key=lambda h: h["mse"]))
        m3 = sum(h["mse"] for h in cheap) / 2
        s3 = sum(h["spikes"] for h in cheap) / 2
        rows.append((f"MBE-PASN 부호분할 @Signed예산 "
                     f"(N={cheap[0]['N']}/{cheap[1]['N']},"
                     f"T={cheap[0]['T']}/{cheap[1]['T']})",
                     dict(mse=m3, rel=m3 ** 0.5 / delta, spikes=s3)))

        for i, (label, s) in enumerate(rows):
            head = f"| {name} " if i == 0 else "| "
            print(f"{head}| {label} | {s['mse']:.2e} | {s['rel']:.2e} | "
                  f"{s['spikes']:.2f} |")


if __name__ == "__main__":
    main()
