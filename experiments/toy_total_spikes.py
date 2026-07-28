"""실험 11 — 토이 Transformer 총 스파이크 재측정.

실험 1–10에서 바뀐 것을 전부 켠 뒤, **네트워크 전체**의 스파이크를 다시 잰다.
활성함수는 예산의 8–17%뿐이고 **항등원(FP곱)이 60–70%** 이므로, 실험 4·10의 tied +
상대타깃 항등원이 여기서 전체 숫자를 바꿔야 한다.

비교 설정:

  * ``mbe``            -- 전역 MBE 뉴런 (논문 기준선)
  * ``mbe_pasn (구)``  -- 실험 1 이전의 라우팅: 고정 예산 N=2/T=16, near0 단일,
                          절대타깃 항등원, 디코더 차수 1
  * ``mbe_pasn (신)``  -- 실험 1–10 전부: 규칙 예산, near0 부호분할, 차수 2 디코더,
                          **tied + 상대타깃 항등원**

``spiking_cost_report`` 가 forward 를 계측해 **모든** 스파이킹 프리미티브를 센다
(손으로 유도한 배수가 아니라 실측).

사용법::

    python experiments/toy_total_spikes.py --json results/toy_total_spikes.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.metrics import neuron_params  # noqa: E402
from mbe.toy import make_inputs, make_toy  # noqa: E402


def rel_err(out, ref):
    return float((out - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6))


def total_params(model) -> int:
    return sum(neuron_params(n) for _, _, n in cv._spiking_primitives(model))


def id_params(model) -> int:
    return sum(neuron_params(n) for name, _, n in cv._spiking_primitives(model)
               if name.endswith((".idn", ".idn2", ".id_dev", ".id_istd")))


SETUPS = {
    "mbe (전역, 기준선)": dict(backend="mbe"),
    "mbe_pasn (구: 실험1 이전)": dict(
        backend="mbe_pasn", pasn_budget="fixed", pasn_near0="single",
        pasn_n_local=2, pasn_readout_order=1, pasn_id_tied=False),
    # 항등원이 예산의 60-70% 이므로 그 정확도 타깃 하나가 총계를 지배한다.
    # 한 점이 아니라 프론티어로 보여야 한다.
    **{f"mbe_pasn (신, 항등원 r={r:.0e})": dict(
        backend="mbe_pasn", pasn_budget="rule", pasn_near0="signed",
        pasn_readout_order=2, pasn_id_tied=True, pasn_id_target_rel=r)
       for r in (1e-1, 3e-2, 1e-2, 3e-3)},
}


def run(label, kw, model, calib, test, ann, epochs):
    m = copy.deepcopy(model)
    rec = cv.calibrate(m, calib)
    cfg = cv.ConvertConfig(epochs=epochs, spike_mult=True, pasn_e_min=-3,
                           pasn_id_e_min=-6, **kw)
    t0 = time.perf_counter()
    cv.convert(m, rec, cfg=cfg)
    secs = time.perf_counter() - t0
    with torch.no_grad():
        out = m(test)
    rep = cv.spiking_cost_report(m, test)
    # by_kind 는 절대 스파이크 -- 입력 원소 수로 나눠 total 과 같은 단위로 맞춘다.
    n_in = max(rep["input_elements"], 1)
    return dict(label=label, err=rel_err(out, ann),
                total=rep["spikes_per_input"],
                by_kind={k: v / n_in for k, v in rep["by_kind"].items()},
                params=total_params(m), id_params=id_params(m), build_s=secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/toy_total_spikes.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = 16
    model = make_toy(seed=args.seed, d_model=d, n_heads=2, n_layers=2)
    batches = make_inputs(3, batch=4, seq=8, d_model=d, seed=args.seed)
    calib, test = batches[:2], batches[2]
    with torch.no_grad():
        ann = model(test)

    rows = []
    for label, kw in SETUPS.items():
        print(f"[{label}] 변환 중...", flush=True)
        rows.append(run(label, kw, model, calib, test, ann, args.epochs))
        r = rows[-1]
        print(f"  err={r['err']:.2e} total={r['total']:.0f} "
              f"params={r['params']} ({r['build_s']:.0f}s)", flush=True)

    base = rows[0]
    print("\n## 총 스파이크 (입력 원소당, forward 계측)\n")
    print("| 설정 | forward 상대오차 | **총 스파이크** | vs 전역 MBE | "
          "저장 파라미터 | 항등원 파라미터 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['err']:.2e} | **{r['total']:.0f}** | "
              f"{base['total'] / r['total']:.2f}× | {r['params']} | "
              f"{r['id_params']} |")

    print("\n## 연산 종류별 분해 (%)\n")
    kinds = sorted({k for r in rows for k in r["by_kind"]})
    print("| 설정 | " + " | ".join(kinds) + " |")
    print("|---" * (len(kinds) + 1) + "|")
    for r in rows:
        tot = max(r["total"], 1e-9)
        print(f"| {r['label']} | " +
              " | ".join(f"{100 * r['by_kind'].get(k, 0.0) / tot:.1f}"
                         for k in kinds) + " |")

    print("\n## 연산 종류별 절대 스파이크\n")
    print("| 설정 | " + " | ".join(kinds) + " |")
    print("|---" * (len(kinds) + 1) + "|")
    for r in rows:
        print(f"| {r['label']} | " +
              " | ".join(f"{r['by_kind'].get(k, 0.0):.0f}" for k in kinds) + " |")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
