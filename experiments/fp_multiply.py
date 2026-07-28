"""실험 10 — 실제 FP 곱셈. 항등원 1-D가 아니라 곱 그 자체를 잰다.

변환된 Transformer 안의 곱셈은 **두 종류이고 성질이 완전히 다르다.**

* **가중치 × 활성 (`Wx`)** — 근사가 **아니다.** 이진 스파이크 × 저장 가중치는 gated
  accumulate 이고, ``x_i = sum_{n,t} a_{n,t} s_{i,n,t}`` 이면
  ``(Wx)_j = sum_{n,t} a_{n,t} (sum_i W_ji s_{i,n,t})`` 라는 **정확한 재배열**이다.
  현재 구현은 실수로 디코드한 뒤 ``nn.Linear`` 를 부르므로 수학적으로 동일하다.
  다만 accumulate-only 형태를 실제로 쓰려면 **활성값을 다시 스파이크로 인코딩**해야
  하고, 그 인코딩이 곧 ``MBE_Id`` 다 -- 그 대가는 아무도 안 쟀다. 여기서 잰다.
* **활성 × 활성 (실수 곱)** — 두 피연산자를 각각 ``MBE_Id`` 로 복원해 곱한다
  (``spiking_ops.spiking_multiply`` / ``spiking_matmul``). 여기가 근사가 들어가는
  유일한 지점이고, 오차는 **두 복원의 상대오차**가 지배한다.

측정:

  A. 원소별 실수 곱 -- 전역 MBE_Id vs 라우팅/tied 항등원 (상대오차 + 스파이크)
  B. 부호 처리 -- 라우팅 항등원은 라우터가 부호를 읽으므로 ``signed=False`` 로
     복원 횟수를 **절반**으로 줄일 수 있는가
  C. 행렬곱 (활성 x 활성)
  D. 가중치 경로 -- 정확성 확인 + accumulate-only 형태의 인코딩 비용

사용법::

    python experiments/fp_multiply.py --json results/fp_multiply.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402
from mbe.spiking_ops import (calibrate_identity, spiking_matmul,  # noqa: E402
                             spiking_multiply)


# --------------------------------------------------------------------------
# 스파이크 계산 -- spiking_multiply 가 실제로 먹이는 텐서에 대해 센다
# --------------------------------------------------------------------------

def recon_spikes(idn, x) -> float:
    """``idn.reconstruct(x)`` 한 번이 내는 총 스파이크 수."""
    if hasattr(idn, "mean_spikes"):                     # MBE-PASN
        return idn.mean_spikes(x) * x.numel()
    cfg = idn.cfg
    return idn.firing_rate(x) * x.numel() * cfg.n_basis * cfg.n_steps


def multiply_spikes(idn, x1, x2, signed: bool) -> float:
    """곱 하나당 평균 스파이크 (양쪽 피연산자 합)."""
    if not signed:
        tot = recon_spikes(idn, x1) + recon_spikes(idn, x2)
    else:                                               # relu 분할 -> 4회 복원
        tot = sum(recon_spikes(idn, t) for t in
                  (torch.relu(x1), torch.relu(-x1),
                   torch.relu(x2), torch.relu(-x2)))
    return tot / x1.numel()


def rel_err(pred, exact, eps=1e-12):
    """원소별 상대오차 RMS. 곱 ``x1*x2`` 처럼 **결과가 작으면 원인도 작은** 양에만
    의미가 있다(상대 타깃 항등원이면 오차도 같이 작아진다)."""
    return float(((pred - exact) / (exact.abs() + eps)).pow(2).mean().sqrt())


def norm_err(pred, exact):
    """``||err|| / ||exact||``. ``Wx`` 처럼 **상쇄로 0을 지나는** 양은 이쪽으로 재야
    한다 -- 큰 항들이 상쇄돼 결과가 0에 가까워도 절대 오차는 안 줄어들므로,
    원소별 상대오차는 발산한다."""
    return float((pred - exact).norm() / exact.norm())


# --------------------------------------------------------------------------

def operands(kind, m, seed, lo, hi):
    g = torch.Generator().manual_seed(seed)
    if kind == "gauss":                                 # 활성값에 가까움
        x = torch.randn(m, generator=g)
    elif kind == "logunif":                             # 자릿수 균등 (상대오차 스트레스)
        s = torch.where(torch.rand(m, generator=g) < 0.5, -1.0, 1.0)
        x = s * torch.exp(torch.rand(m, generator=g) *
                          (math.log(hi) - math.log(lo)) + math.log(lo))
    else:
        raise ValueError(kind)
    return x.clamp(-hi, hi)


def build_identities(hi, epochs, seed, e_lo):
    """비교할 항등원들: 현재 기본(전역) vs 라우팅(tied, 상대 타깃)."""
    out = {}
    out["전역 MBE_Id N=8,T=16"] = dict(
        idn=calibrate_identity(0.0, hi, n_basis=8, n_steps=16, epochs=epochs,
                               seed=seed, use_cache=False),
        signed=True)
    for r in (1e-2, 1e-3):
        out[f"MBE-PASN tied r={r:.0e}"] = dict(
            idn=build_mbe_pasn("identity", (-hi, hi), e_min=e_lo, e_max=None,
                               budget="rule", target="relative", target_rel=r,
                               tied=True, near0="signed", epochs=epochs,
                               seed=seed, alpha_init="uniform"),
            signed=False)                               # 라우터가 부호를 읽는다
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=20000)
    ap.add_argument("--hi", type=float, default=64.0)
    ap.add_argument("--lo", type=float, default=1.0 / 64)
    ap.add_argument("--e-lo", type=int, default=-6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/fp_multiply.json")
    args = ap.parse_args()

    res = {}
    ids = build_identities(args.hi, args.epochs, args.seed, args.e_lo)

    # ---------------- A. 원소별 실수 곱 ----------------
    print(f"# A. 원소별 실수 곱 (활성 × 활성), |x| ∈ [{args.lo:.3g}, {args.hi:g}]\n")
    print("| 항등원 | 분포 | 곱 상대 RMSE | 스파이크/곱 | 저장 파라미터 |")
    print("|---|---|---|---|---|")
    res["elementwise"] = []
    for kind in ("gauss", "logunif"):
        x1 = operands(kind, args.m, args.seed, args.lo, args.hi)
        x2 = operands(kind, args.m, args.seed + 1, args.lo, args.hi)
        exact = x1 * x2
        # convert.py 는 전역 항등원을 **실측 피연산자 범위**로 캘리브레이션한다.
        # 고정된 넓은 범위로만 비교하면 전역 기준선에 불리하므로 이 행을 추가한다.
        mx = float(max(x1.abs().max(), x2.abs().max()))
        local = dict(ids)
        local[f"전역 MBE_Id (실측범위 0..{mx:.1f})"] = dict(
            idn=calibrate_identity(0.0, mx, n_basis=8, n_steps=16,
                                   epochs=args.epochs, seed=args.seed,
                                   use_cache=False),
            signed=True)
        for label, spec in local.items():
            idn, sg = spec["idn"], spec["signed"]
            pred = spiking_multiply(idn, x1, x2, signed=sg)
            e = rel_err(pred, exact)
            spk = multiply_spikes(idn, x1, x2, sg)
            npar = sum(p.numel() for p in idn.parameters() if p.requires_grad)
            res["elementwise"].append(dict(label=label, dist=kind, rel=e,
                                           spikes=spk, params=npar))
            print(f"| {label} | {kind} | {e:.3e} | {spk:.2f} | {npar} |",
                  flush=True)

    # ---------------- B. 부호 처리 ----------------
    print("\n\n# B. 부호 처리 — 라우팅 항등원에 relu 4분할이 필요한가\n")
    print("| 항등원 | signed | 곱 상대 RMSE | 스파이크/곱 | 뉴런 호출/곱 |")
    print("|---|---|---|---|---|")
    res["signed"] = []
    x1 = operands("gauss", args.m, args.seed, args.lo, args.hi)
    x2 = operands("gauss", args.m, args.seed + 1, args.lo, args.hi)
    exact = x1 * x2
    for label, spec in ids.items():
        if not label.startswith("MBE-PASN"):
            continue
        for sg in (True, False):
            pred = spiking_multiply(spec["idn"], x1, x2, signed=sg)
            e, spk = rel_err(pred, exact), multiply_spikes(spec["idn"], x1, x2, sg)
            calls = 4 if sg else 2
            res["signed"].append(dict(label=label, signed=sg, rel=e, spikes=spk,
                                      calls=calls))
            print(f"| {label} | {sg} | {e:.3e} | {spk:.2f} | {calls} |",
                  flush=True)

    # ---------------- C. 행렬곱 ----------------
    print("\n\n# C. 행렬곱 (활성 × 활성), 64×64 @ 64×64\n")
    print("| 항등원 | 상대 Frobenius 오차 | 스파이크/원소 |")
    print("|---|---|---|")
    res["matmul"] = []
    g = torch.Generator().manual_seed(args.seed)
    A = torch.randn(64, 64, generator=g)
    B = torch.randn(64, 64, generator=g)
    exactC = A @ B
    for label, spec in ids.items():
        idn, sg = spec["idn"], spec["signed"]
        C = spiking_matmul(idn, A, B, signed=sg)
        e = float((C - exactC).norm() / exactC.norm())
        spk = (multiply_spikes(idn, A.reshape(-1), B.reshape(-1), sg))
        res["matmul"].append(dict(label=label, rel=e, spikes=spk))
        print(f"| {label} | {e:.3e} | {spk:.2f} |", flush=True)

    # ---------------- D. 가중치 곱 ----------------
    print("\n\n# D. 가중치 × 활성 — 근사가 아니다\n")
    lin = nn.Linear(64, 64, bias=False)
    xa = torch.randn(256, 64, generator=g)
    exact_wx = lin(xa)
    print(f"현재 구현(실수 디코드 후 nn.Linear)의 오차: "
          f"**{float((lin(xa) - exact_wx).abs().max()):.1e}** — 정확히 0. "
          f"가중치 곱에는 근사가 들어가지 않는다.\n")
    print("accumulate-only 형태를 쓰려면 활성값을 다시 스파이크로 인코딩해야 하고, "
          "그 인코딩이 곧 MBE_Id 다. 그 대가:\n")
    print("| 항등원 | 인코딩 자체 오차 | **인코딩 후 Wx 오차** | 증폭 | "
          "인코딩 스파이크/활성 |")
    print("|---|---|---|---|---|")
    res["weight"] = []
    for label, spec in ids.items():
        idn, sg = spec["idn"], spec["signed"]
        if sg:
            rec = (idn.reconstruct(torch.relu(xa))
                   - idn.reconstruct(torch.relu(-xa)))
            spk = sum(recon_spikes(idn, t)
                      for t in (torch.relu(xa), torch.relu(-xa))) / xa.numel()
        else:
            rec = idn.reconstruct(xa)
            spk = recon_spikes(idn, xa) / xa.numel()
        e_in = norm_err(rec, xa)
        e_out = norm_err(lin(rec), exact_wx)
        res["weight"].append(dict(label=label, enc=e_in, wx=e_out, spikes=spk,
                                  amp=e_out / max(e_in, 1e-30)))
        print(f"| {label} | {e_in:.3e} | **{e_out:.3e}** | "
              f"{e_out / max(e_in, 1e-30):.2f}× | {spk:.2f} |", flush=True)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
