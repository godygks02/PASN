"""P0.3 — 공정한 matched Pareto + 워터폴 ablation.

**이 실험이 논문의 주장을 결정한다.** 실험 11의 14.2× 는 7가지가 섞인 값이다 —
라우팅, 뱅크별 예산, signed near-zero, order-2 디코더, 상대오차 항등원, tied 항등원,
그리고 고른 동작점 `r`. "MoE 로 구간별 적은 basis" 라는 메인 기여가 실제로 몇 배인지
아직 분리되지 않았다.

두 가지를 고친다.

1. **matched 기준선.** 전역 MBE 에도 같은 것을 준다 — `(N, T)` 스윕, order-2 readout,
   항등원의 로그균등 캘리브레이션(전역 뉴런이 낼 수 있는 상대오차 대응물). 한 점이
   아니라 **프론티어**를 만들고 그 프론티어와 비교한다.
2. **워터폴.** 라우팅 쪽을 0단계부터 하나씩 켜서 각 단계의 기여를 분리한다.

성공 기준:
  (a) 전체가 best matched global 대비 iso-forward-error 에서 **≥2×**
  (b) **tied 항등원을 빼고도** 라우팅+예산(1–4단계)이 **≥20%** 기여
  (c) seed 별 순위 유지

(b) 가 실패하면 논문 중심을 "routed homogeneous identity" 로 옮겨야 한다.

사용법::

    python experiments/waterfall.py --seed 0 --json results/_wf_s0.json
    python experiments/waterfall.py --mode report --shards results/_wf_s*.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.metrics import neuron_params  # noqa: E402
from mbe.toy import make_inputs, make_toy  # noqa: E402

# 전역 MBE 프론티어: 기준선도 자기 (N,T)·디코더·항등원 표본을 고를 수 있어야 공정하다.
GLOBAL_GRID = [dict(n_basis_act=na, n_basis_ln=na, n_basis_sm=na, n_basis_mm=na,
                    n_steps=16, mbe_readout_order=o, mbe_id_logsample=ls)
               for na in (4, 8) for o in (1, 2) for ls in (False, True)]

# 워터폴: 0단계는 라우팅 없음, 이후 한 번에 하나씩만 켠다.
#
# 각 단계는 **자기 단계에서 실제로 동작하는** 손잡이로 프론티어를 만들어야 한다.
# 처음엔 모든 단계에 `pasn_id_target_rel` 을 훑었는데, 1-4 단계는 항등원이 절대
# 타깃이라 그 값이 아무 효과가 없어 **같은 설정 3개**가 나왔다. 그러면 1-4 단계만
# 동작점이 1개인 채로 비교돼, "라우팅 단독 기여" 기준이 엉뚱한 이유로 실패한다.
_ROUTED = dict(backend="mbe_pasn")
WATERFALL = [
    ("1. 라우팅만", dict(_ROUTED, pasn_budget="fixed", pasn_near0="single",
                       pasn_readout_order=1, pasn_id_tied=False,
                       pasn_id_target="absolute")),
    ("2. + 뱅크별 (N,T)", dict(_ROUTED, pasn_budget="rule", pasn_near0="single",
                             pasn_readout_order=1, pasn_id_tied=False,
                             pasn_id_target="absolute")),
    ("3. + signed near-zero", dict(_ROUTED, pasn_budget="rule",
                                   pasn_near0="signed", pasn_readout_order=1,
                                   pasn_id_tied=False,
                                   pasn_id_target="absolute")),
    ("4. + order-2 디코더", dict(_ROUTED, pasn_budget="rule",
                               pasn_near0="signed", pasn_readout_order=2,
                               pasn_id_tied=False,
                               pasn_id_target="absolute")),
    ("5. + 상대오차 항등원", dict(_ROUTED, pasn_budget="rule",
                              pasn_near0="signed", pasn_readout_order=2,
                              pasn_id_tied=False)),
    ("6. + tied 항등원", dict(_ROUTED, pasn_budget="rule", pasn_near0="signed",
                            pasn_readout_order=2, pasn_id_tied=True)),
    ("7. + β", dict(_ROUTED, pasn_budget="rule", pasn_near0="signed",
                    pasn_readout_order=2, pasn_id_tied=True,
                    pasn_beta=cv.BETA_CURVATURE_PEAK)),
]
#: 단계별 동작점 손잡이: (설정 키, 값들). 고정 예산 단계는 basis 수가, 규칙+절대
#: 타깃 단계는 절대 오차 목표가, 상대 타깃 단계는 상대 오차 목표가 손잡이다.
STAGE_KNOB = {
    "1. 라우팅만": ("pasn_n_local", (1, 2, 3)),
    "2. + 뱅크별 (N,T)": ("pasn_target_mse", (1e-3, 1e-5, 1e-7)),
    "3. + signed near-zero": ("pasn_target_mse", (1e-3, 1e-5, 1e-7)),
    "4. + order-2 디코더": ("pasn_target_mse", (1e-3, 1e-5, 1e-7)),
    "5. + 상대오차 항등원": ("pasn_id_target_rel", (1e-1, 3e-2, 1e-2)),
    "6. + tied 항등원": ("pasn_id_target_rel", (1e-1, 3e-2, 1e-2)),
    "7. + β": ("pasn_id_target_rel", (1e-1, 3e-2, 1e-2)),
}


def rel_err(out, ref):
    return float((out - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6))


def run_one(kw, model, calib, test, ann, epochs, seed):
    m = copy.deepcopy(model)
    rec = cv.calibrate(m, calib)
    # alpha_init="uniform" throughout: the default "auto" fits both placements
    # and doubles the build, and an ablation only needs the *same* choice at every
    # stage. It is a fixed cost, not a stage.
    base = dict(epochs=epochs, seed=seed, spike_mult=True, pasn_e_min=-3,
                pasn_id_e_min=-6, pasn_id_signed=False,
                pasn_alpha_init="uniform")
    cfg = cv.ConvertConfig(**{**base, **kw})
    cv.convert(m, rec, cfg=cfg)
    with torch.no_grad():
        out = m(test)
    rep = cv.spiking_cost_report(m, test)
    return dict(err=rel_err(out, ann), spikes=rep["spikes_per_input"],
                params=sum(neuron_params(n)
                           for _, _, n in cv._spiking_primitives(m)))


def measure(seed, epochs):
    torch.manual_seed(seed)
    d = 16
    model = make_toy(seed=seed, d_model=d, n_heads=2, n_layers=2)
    batches = make_inputs(3, batch=4, seq=8, d_model=d, seed=seed)
    calib, test = batches[:2], batches[2]
    with torch.no_grad():
        ann = model(test)

    out = {"global": [], "waterfall": []}
    for i, g in enumerate(GLOBAL_GRID):
        r = run_one(dict(g, backend="mbe"), model, calib, test, ann, epochs, seed)
        r["label"] = (f"global N={g['n_basis_act']} T={g['n_steps']} "
                      f"o={g['mbe_readout_order']} "
                      f"{'log' if g['mbe_id_logsample'] else 'unif'}")
        out["global"].append(r)
        print(f"  [{seed}] global {i+1}/{len(GLOBAL_GRID)} "
              f"err={r['err']:.2e} spk={r['spikes']:.0f}", flush=True)

    for label, kw in WATERFALL:
        knob, values = STAGE_KNOB[label]
        for v in values:
            r = run_one(dict(kw, **{knob: v}), model, calib, test, ann,
                        epochs, seed)
            r["label"], r["knob"], r["value"] = label, knob, v
            out["waterfall"].append(r)
        print(f"  [{seed}] {label} 완료 ({knob})", flush=True)
    return out


def pareto(pts):
    keep = []
    for p in sorted(pts, key=lambda p: p["spikes"]):
        if not keep or p["err"] < keep[-1]["err"]:
            keep.append(p)
    return keep


def iso(pts, target_err):
    ok = [p for p in pts if p["err"] <= target_err]
    return min(ok, key=lambda p: p["spikes"]) if ok else None


def geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def report(shards):
    runs = [json.load(open(p, encoding="utf-8")) for p in shards]
    print(f"# P0.3 워터폴 — seed {len(runs)}개\n")

    print("## 0. matched 전역 MBE 프론티어 (seed 0)\n")
    print("| 설정 | forward 오차 | 스파이크 | 파라미터 |")
    print("|---|---|---|---|")
    for p in pareto(runs[0]["global"]):
        print(f"| {p['label']} | {p['err']:.2e} | {p['spikes']:.0f} | "
              f"{p['params']} |")

    # 기준: 전역 프론티어에서 "가장 정확한 점" 의 오차를 타깃으로 잡는다.
    print("\n## 1. 워터폴 — 전역 최고정확도와 같은 오차를 몇 스파이크에\n")
    print("| 단계 | " + " | ".join(f"seed{i}" for i in range(len(runs))) +
          " | 기하평균 | 직전 대비 | 파라미터 |")
    print("|---" * (len(runs) + 4) + "|")
    prev = None
    stage_geo, stage_par = {}, {}
    for label, _ in WATERFALL:
        cells, vals, pars = [], [], []
        for run in runs:
            gf = pareto(run["global"])
            tgt = min(p["err"] for p in gf)
            g_iso = iso(gf, tgt)
            w = pareto([p for p in run["waterfall"] if p["label"] == label])
            w_iso = iso(w, tgt)
            if g_iso and w_iso:
                v = g_iso["spikes"] / w_iso["spikes"]
                vals.append(v)
                pars.append(w_iso["params"])
                cells.append(f"{v:.2f}×")
            else:
                cells.append("도달 불가")
        g = geo(vals) if vals else float("nan")
        stage_geo[label] = g
        stage_par[label] = geo(pars) if pars else float("nan")
        step = f"{g / prev:.2f}×" if prev and vals else "—"
        prev = g if vals else prev
        print(f"| {label} | " + " | ".join(cells) +
              f" | **{g:.2f}×** | {step} | {stage_par[label]:.0f} |")

    print("\n## 2. 성공 기준\n")
    full = stage_geo[WATERFALL[-1][0]]
    no_tie = stage_geo["5. + 상대오차 항등원"]
    routing_only = stage_geo["4. + order-2 디코더"]
    print(f"| 기준 | 값 | 판정 |")
    print("|---|---|---|")
    print(f"| (a) 전체가 matched global 대비 ≥2× | **{full:.2f}×** | "
          f"{'✅' if full >= 2 else '❌'} |")
    print(f"| (b) tied 없이 라우팅+예산(1–4)이 ≥20% 기여 | "
          f"**{routing_only:.2f}×** (전체의 "
          f"{100 * math.log(max(routing_only, 1.001)) / math.log(max(full, 1.001)):.0f}%) | "
          f"{'✅' if routing_only >= 1.2 else '❌'} |")
    print(f"| tied 항등원 단독 기여 (5→6), 스파이크 | "
          f"**{stage_geo['6. + tied 항등원'] / no_tie:.2f}×** | |")
    print(f"| tied 항등원 단독 기여 (5→6), **파라미터** | "
          f"**{stage_par['5. + 상대오차 항등원'] / stage_par['6. + tied 항등원']:.2f}× 감소** | |")
    gpar = geo([min(p["params"] for p in pareto(r["global"])) for r in runs])
    print(f"| 전역 MBE 파라미터 (프론티어 최소) | {gpar:.0f} | |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "report"], default="measure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--shards", nargs="+", default=[])
    ap.add_argument("--json", default="results/_wf_s0.json")
    args = ap.parse_args()

    if args.mode == "measure":
        out = measure(args.seed, args.epochs)
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"wrote {args.json}")
    else:
        report(args.shards or sorted(glob.glob("results/_wf_s*.json")))


if __name__ == "__main__":
    main()
