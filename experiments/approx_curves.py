"""근사 곡선 재구성 — 스윕에서 고른 예산 배분을 실제 모델로 다시 지어 곡선을 뽑는다.

``bank_budget.py`` 스윕은 뱅크 × (N,T,alpha) 의 **지표만** 저장하고 모델은 버린다.
곡선을 그리려면 그 배분대로 뱅크를 다시 적합해 :class:`MBEPASNNeuron` 을 조립해야
한다. 이 스크립트가 그 일을 하고 ``(x, f, f_hat)`` 을 npz 로 떨군다(함수 하나씩,
병렬 실행용).

동작점 선택은 **같은 에너지**에서 한다 — 고정 `N=2,T=16` 이 쓰는 스파이크 이하에서
가장 정확한 프론티어 점. "같은 전력으로 얼마나 더 잘 맞추나"가 눈으로 보인다.

사용법::

    python experiments/approx_curves.py --fn gelu --out curves_gelu.npz \
        --v1 results/bb_exp2_v1_single_e-2.json --v2 results/bb_exp2_v2_signed_e-2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.mbe_pasn import (MBEPASNNeuron, PrefixRouter,  # noqa: E402
                          fill_unreachable)
from bank_budget import bank_data, fit_cell_model  # noqa: E402


def build_from_choice(name, domain, e_min, e_max, near0, choice, epochs, seed,
                      m=1024):
    """뱅크별 ``(N, T, alpha)`` 리스트대로 MBE-PASN 을 조립한다."""
    fn, _ = functions.REGISTRY[name]
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max,
                          near0=near0)
    banks = []
    for bi, spec in enumerate(router.banks):
        span = router.reachable(bi, domain[0], domain[1])
        ch = choice[bi] if bi < len(choice) else None
        if span is None or ch is None:
            banks.append(None)          # 아래에서 최근접 뱅크로 채운다
            continue
        N, T, akind = ch
        d = bank_data(fn, spec, span, m)
        bank, _ = fit_cell_model(fn, d, int(N), int(T), akind, epochs, seed)
        banks.append(bank)
    return MBEPASNNeuron(router, fill_unreachable(router, banks))


def iso_energy(front, budget):
    """예산 이하에서 가장 정확한 프론티어 점."""
    ok = [a for a in front if a["spikes"] <= budget]
    return min(ok, key=lambda a: a["mse"]) if ok else min(
        front, key=lambda a: a["spikes"])


def fixed_choice(res, name, N=2, T=16, akind="logspread"):
    grids = res["functions"][name]["grids"]
    return [None if not g else (N, T, akind) for g in grids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1", required=True, help="실험1 구성 JSON (단일 near0)")
    ap.add_argument("--v2", required=True, help="실험2 구성 JSON (부호분할)")
    ap.add_argument("--domain", nargs=2, type=float, default=[-8.0, 8.0])
    ap.add_argument("--e-min", type=int, default=-2)
    ap.add_argument("--e-max", type=int, default=4)
    ap.add_argument("--dist", default="gauss")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    name, dom = args.fn, tuple(args.domain)
    fn, _ = functions.REGISTRY[name]
    r1 = json.load(open(args.v1, encoding="utf-8"))
    r2 = json.load(open(args.v2, encoding="utf-8"))

    d1 = r1["functions"][name]["by_dist"][args.dist]
    base = next(b for b in d1["baselines"] if "logspread" in b["label"])
    budget = base["spikes"]

    specs = [
        ("고정 N=2,T=16", "single", fixed_choice(r1, name), base["mse"],
         base["spikes"]),
    ]
    for tag, res, near0 in (("실험1 최적배분", r1, "single"),
                            ("실험2 최적배분(부호분할)", r2, "signed")):
        d = res["functions"][name]["by_dist"][args.dist]
        pt = iso_energy(d["frontier"], budget)
        specs.append((tag, near0, pt["choice"], pt["mse"], pt["spikes"]))

    x = torch.linspace(dom[0], dom[1], 4001)
    out = dict(x=x.numpy(), y=fn(x).numpy(), labels=[], mse=[], spikes=[])
    for tag, near0, choice, mse, spk in specs:
        model = build_from_choice(name, dom, args.e_min, args.e_max, near0,
                                  choice, args.epochs, args.seed)
        with torch.no_grad():
            yh = model(x).numpy()
        # 보고용 MSE 는 스윕이 고른 값(가중 MSE)과 다를 수 있으므로 둘 다 남긴다
        out[f"yhat_{len(out['labels'])}"] = yh
        out["labels"].append(tag)
        out["mse"].append(mse)
        out["spikes"].append(spk)
        print(f"  {tag:26s} MSE(스윕)={mse:.2e}  스파이크={spk:.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, fn=name, domain=np.array(dom),
             labels=np.array(out["labels"], dtype=object),
             mse=np.array(out["mse"]), spikes=np.array(out["spikes"]),
             **{k: v for k, v in out.items()
                if k.startswith("yhat_") or k in ("x", "y")})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
