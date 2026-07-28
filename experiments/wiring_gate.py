"""P0.2 게이트 — β 와 signed=False 가 **실제 변환 경로에서** 값을 하는가.

실험 5의 β(`inv` 13.6×, `invsqrt` 8.3×)와 실험 10의 `signed=False`(호출 절반)는
`build_mbe_pasn` API 에만 있었고 `convert.py` 경로에서는 쓰이지 않았다. 배선한 뒤
**실제 프리미티브의 실제 인자 분포에서** 그 이득이 재현되는지 확인한다.

재현이 안 되면 그것도 발견이다 — 1-D 합성 분포와 네트워크 안의 실제 인자 분포가
다르다는 뜻이므로.

게이트:

  A. β on/off × primitive  -- 토이 전체 forward 오차 / 총 스파이크 / 파라미터
  B. signed on/off         -- 출력이 유지되면서 **뉴런 호출 수**가 절반이 되는가
  C. 계측 정합성           -- primitive 별 calls 가 소스 호출 횟수와 맞는가

사용법::

    python experiments/wiring_gate.py --json results/wiring_gate.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.metrics import neuron_params  # noqa: E402
from mbe.toy import make_inputs, make_toy  # noqa: E402


def rel_err(out, ref):
    return float((out - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6))


def build(kw, model, calib, epochs):
    m = copy.deepcopy(model)
    rec = cv.calibrate(m, calib)
    cfg = cv.ConvertConfig(epochs=epochs, spike_mult=True, backend="mbe_pasn",
                           pasn_e_min=-3, pasn_id_e_min=-6, pasn_budget="rule",
                           pasn_near0="signed", pasn_readout_order=2,
                           pasn_id_tied=True, pasn_id_target_rel=1e-2, **kw)
    cv.convert(m, rec, cfg=cfg)
    return m


def measure(m, test, ann):
    with torch.no_grad():
        out = m(test)
    rep = cv.spiking_cost_report(m, test)
    n_in = max(rep["input_elements"], 1)
    calls = {k: sum(p["calls"] for p in rep["primitives"].values()
                    if p["kind"] == k)
             for k in rep["by_kind"]}
    return dict(err=rel_err(out, ann), total=rep["spikes_per_input"],
                by_kind={k: v / n_in for k, v in rep["by_kind"].items()},
                calls=calls, total_calls=sum(calls.values()),
                params=sum(neuron_params(n)
                           for _, _, n in cv._spiking_primitives(m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/wiring_gate.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = 16
    model = make_toy(seed=args.seed, d_model=d, n_heads=2, n_layers=2)
    batches = make_inputs(3, batch=4, seq=8, d_model=d, seed=args.seed)
    calib, test = batches[:2], batches[2]
    with torch.no_grad():
        ann = model(test)

    SETUPS = {
        "기준 (β 없음, signed=True)": dict(pasn_beta={}, pasn_id_signed=False),
        "β 만": dict(pasn_id_signed=False),
        "signed 만": dict(pasn_beta={}),
        "β + signed (기본값)": dict(),
    }
    rows = {}
    for label, kw in SETUPS.items():
        print(f"[{label}] 변환 중...", flush=True)
        rows[label] = measure(build(kw, model, calib, args.epochs), test, ann)
        r = rows[label]
        print(f"  err={r['err']:.3e} spikes={r['total']:.0f} "
              f"calls={r['total_calls']} params={r['params']}", flush=True)

    base = rows["기준 (β 없음, signed=True)"]
    print("\n## A/B — 토이 전체\n")
    print("| 설정 | forward 오차 | 총 스파이크 | vs 기준 | 뉴런 호출 | "
          "vs 기준 | 파라미터 |")
    print("|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        print(f"| {label} | {r['err']:.3e} | {r['total']:.0f} | "
              f"{base['total'] / r['total']:.2f}× | {r['total_calls']} | "
              f"{base['total_calls'] / max(r['total_calls'], 1):.2f}× | "
              f"{r['params']} |")

    print("\n## 연산 종류별 스파이크\n")
    kinds = sorted(base["by_kind"])
    print("| 설정 | " + " | ".join(kinds) + " |")
    print("|---" * (len(kinds) + 1) + "|")
    for label, r in rows.items():
        print(f"| {label} | " +
              " | ".join(f"{r['by_kind'].get(k, 0.0):.0f}" for k in kinds) + " |")

    print("\n## 연산 종류별 뉴런 호출 수 (signed 의 효과가 나타나야 할 곳)\n")
    print("| 설정 | " + " | ".join(kinds) + " |")
    print("|---" * (len(kinds) + 1) + "|")
    for label, r in rows.items():
        print(f"| {label} | " +
              " | ".join(str(r["calls"].get(k, 0)) for k in kinds) + " |")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
