"""실험 1 — 뱅크별 (N, T) 예산 배분.

MBE-PASN의 각 prefix 뱅크는 서로 난이도가 전혀 다른 구간을 맡는데, 현재 빌더는
모든 뱅크에 같은 예산(`n_local=2`, `n_steps=16`)을 준다. 이 스크립트는

  1. 뱅크마다 ``(N, T, alpha_init)`` 격자를 전부 학습시켜 **예산-오차 지도**를 만들고,
  2. 실제 입력 분포에서 각 뱅크의 **방문확률** ``p_j`` 를 재고,
  3. 라그랑주 스윕으로 ``min sum_j p_j*MSE_j  s.t.  sum_j p_j*spikes_j`` 의
     **최적 배분 프론티어**를 뽑은 뒤,
  4. 고정 예산(현재 기본값) · 전역 MBE · SignedMBE 와 **iso-accuracy** 비교한다.

라그랑주 배분: 각 뱅크가 독립적으로 ``argmin_c MSE_j(c) + lam * spikes_j(c)`` 를
고르면 그 조합이 전체 (스파이크, 오차) 프론티어의 볼록껍질 위 점이 된다. ``p_j`` 는
두 항에 공통으로 곱해지므로 **뱅크별 선택에는 영향을 주지 않는다** (분포는 어느
``lam`` 이 어느 동작점에 대응하는지만 바꾼다). 이 성질 자체가 결과의 일부다.

사용법::

    python experiments/bank_budget.py --fns gelu tanh silu sigmoid relu \
        --json results/bank_budget.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe.fit import fit_model  # noqa: E402
from mbe.mbe_pasn import PrefixRouter  # noqa: E402
from mbe.mbe_pasn_s import uniform_alpha  # noqa: E402
from mbe.neuron import MBEConfig, MBENeuron  # noqa: E402


# --------------------------------------------------------------------------
# 결정적 격자 (적합용 midpoint / 평가용 offset)
# --------------------------------------------------------------------------

def grid(lo: float, hi: float, m: int, offset: float = 0.0) -> torch.Tensor:
    """``[lo, hi]`` 의 midpoint 격자. ``offset`` 은 셀 단위 이동(평가용 0.25)."""
    i = torch.arange(m, dtype=torch.float32)
    u = (i + 0.5 + offset) / m
    return lo + u.clamp(0.0, 1.0) * (hi - lo)


# --------------------------------------------------------------------------
# 뱅크 기술자 — "이 구간이 얼마나 어려운가"
# --------------------------------------------------------------------------

def descriptors(y: torch.Tensor, xfeed: torch.Tensor) -> dict:
    """뱅크가 맡은 목표 ``g`` 의 두 가지 성질.

    * ``delta``  — 구간 안 **동적 범위** ``max g - min g``. 스테어케이스 코드가
      해상해야 할 진폭이므로 필요한 **T**(비트 수)를 좌우한다고 예상.
    * ``nonlin`` — 최적 **직선**을 뺀 잔차의 RMS를 ``delta`` 로 정규화한 값.
      순수 선형이면 0. 직선 하나로 안 되는 정도이므로 필요한 **N**(basis 수)를
      좌우한다고 예상.
    """
    delta = float(y.max() - y.min())
    A = torch.stack([torch.ones_like(xfeed), xfeed], dim=1)
    coef = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
    resid = y - (A @ coef).squeeze(1)
    nonlin = float(resid.pow(2).mean().sqrt() / max(delta, 1e-12))
    return dict(delta=delta, nonlin=nonlin)


# --------------------------------------------------------------------------
# 뱅크 데이터 준비
# --------------------------------------------------------------------------

def bank_data(fn, spec, span, m: int):
    """한 뱅크의 (적합 입력, 적합 목표, 평가 입력, 평가 목표, 평가 x, 정규화 파라미터).

    크기 뱅크는 뉴런에 ``|x|`` 를 먹이고 목표는 ``fn(sign*|x|)``; near-zero 뱅크는
    부호 있는 ``x`` 를 그대로 먹인다.
    """
    a, b = span
    if spec["kind"] == "near0":
        sign = None
        flo, fhi = a, b
    else:
        sign = spec["sign"]
        flo, fhi = min(abs(a), abs(b)), max(abs(a), abs(b))
    xf = grid(flo, fhi, m, 0.0)
    xe = grid(flo, fhi, m, 0.25)
    if sign is None:
        yf, ye, xe_real = fn(xf), fn(xe), xe
    else:
        yf, ye, xe_real = fn(sign * xf), fn(sign * xe), sign * xe
    return dict(xf=xf, yf=yf, xe=xe, ye=ye, xe_real=xe_real,
                x_min=float(flo), x_scale=float(fhi - flo), sign=sign)


# --------------------------------------------------------------------------
# 한 뱅크 × 한 (N, T, alpha) 설정 적합
# --------------------------------------------------------------------------

def alpha_init(kind: str, fn, sign, flo, fhi, N):
    if kind == "uniform":
        return uniform_alpha(N)
    tgt = (lambda t: fn(sign * t)) if sign is not None else fn
    return functions.curvature_alpha(tgt, flo, fhi, N)


def fit_cell_model(fn, d, N, T, akind, epochs, seed):
    """한 설정을 적합하고 ``(적합된 뱅크, 지표 dict)`` 를 반환.

    MSE·발화율은 **적합에 쓰이지 않은 offset 격자**에서 측정한다(적합 격자에서 재면
    스테어케이스가 격자점만 맞추고 사이에서 흘러가는 것을 못 잡는다).
    """
    a = alpha_init(akind, fn, d["sign"], d["x_min"], d["x_min"] + d["x_scale"], N)
    cfg = MBEConfig(n_basis=N, n_steps=T, x_min=d["x_min"], x_scale=d["x_scale"],
                    alpha_v=a, use_bias=True)
    bank = MBENeuron(cfg)
    fit_model(bank, d["xf"], d["yf"], epochs=epochs, seed=seed)
    with torch.no_grad():
        mse = float((bank(d["xe"]) - d["ye"]).pow(2).mean())
        fr = bank.firing_rate(d["xe"])
    return bank, dict(N=N, T=T, alpha=akind, mse=mse, fr=fr,
                      spikes=fr * N * T, params=bank.num_learnable())


def fit_cell(fn, d, N, T, akind, epochs, seed):
    """:func:`fit_cell_model` 의 지표만 (스윕에서 모델은 버린다)."""
    return fit_cell_model(fn, d, N, T, akind, epochs, seed)[1]


# --------------------------------------------------------------------------
# 방문확률
# --------------------------------------------------------------------------

def visit_probs(router, domain, dist: str, sigma: float, m: int = 200_000,
                seed: int = 0):
    """입력 분포에서 각 뱅크로 가는 비율."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = domain
    if dist == "uniform":
        x = torch.rand(m, generator=g) * (hi - lo) + lo
    elif dist == "gauss":
        x = torch.randn(m, generator=g) * sigma
        x = x.clamp(lo, hi)
    else:
        raise ValueError(dist)
    idx = router.route(x)
    counts = torch.bincount(idx, minlength=router.n_banks).float()
    return (counts / counts.sum()).tolist()


# --------------------------------------------------------------------------
# 라그랑주 배분 → 프론티어
# --------------------------------------------------------------------------

def allocate(bank_grids, p, lam):
    """각 뱅크가 ``argmin MSE + lam*spikes``. 전체 (mse, spikes, params, 선택)."""
    tot_mse = tot_spk = 0.0
    tot_par = 0
    choice = []
    for j, cells in enumerate(bank_grids):
        if not cells:                     # 도달 불가 뱅크: 비용도 오차도 없음
            choice.append(None)
            continue
        best = min(cells, key=lambda c: c["mse"] + lam * c["spikes"])
        tot_mse += p[j] * best["mse"]
        tot_spk += p[j] * best["spikes"]
        tot_par += best["params"]
        choice.append((best["N"], best["T"], best["alpha"]))
    return dict(lam=lam, mse=tot_mse, spikes=tot_spk, params=tot_par,
                choice=choice)


def frontier(bank_grids, p, lams):
    pts, seen = [], set()
    for lam in lams:
        a = allocate(bank_grids, p, lam)
        key = tuple(str(c) for c in a["choice"])
        if key in seen:
            continue
        seen.add(key)
        pts.append(a)
    return sorted(pts, key=lambda a: a["spikes"])


def fixed_budget(bank_grids, p, N, T, akind):
    """모든 뱅크에 같은 예산을 준 경우(= 현재 기본값)의 총계."""
    tot_mse = tot_spk = 0.0
    tot_par = 0
    for j, cells in enumerate(bank_grids):
        if not cells:
            continue
        c = next((c for c in cells
                  if c["N"] == N and c["T"] == T and c["alpha"] == akind), None)
        if c is None:
            return None
        tot_mse += p[j] * c["mse"]
        tot_spk += p[j] * c["spikes"]
        tot_par += c["params"]
    return dict(mse=tot_mse, spikes=tot_spk, params=tot_par,
                label=f"고정 N={N}, T={T} ({akind})")


# --------------------------------------------------------------------------
# 전역 MBE 베이스라인 (같은 가중치로 평가)
# --------------------------------------------------------------------------

def global_baseline(fn, name, domain, N, T, datas, p, epochs, seed, signed=False):
    """전역 MBE(또는 SignedMBE)를 도메인 전체에 적합하고 뱅크 가중으로 평가."""
    lo, hi = domain
    x = grid(lo, hi, 4000, 0.0)
    y = fn(x)
    if signed:
        model = functions.make_signed(name, N, N, pivot=0.0, n_steps=T,
                                      domain=domain)
        n_eff = 2 * N
    else:
        cfg = functions.make_config(name, N, n_steps=T, domain=domain,
                                    use_bias=True)
        model = MBENeuron(cfg)
        n_eff = N
    fit_model(model, x, y, epochs=epochs, seed=seed)
    tot_mse = tot_spk = 0.0
    with torch.no_grad():
        for j, d in enumerate(datas):
            if d is None or p[j] == 0.0:
                continue
            mse = float((model(d["xe_real"]) - d["ye"]).pow(2).mean())
            fr = model.firing_rate(d["xe_real"])
            tot_mse += p[j] * mse
            tot_spk += p[j] * fr * n_eff * T
    label = f"Signed MBE {N}x2" if signed else f"전역 MBE N={N}"
    return dict(mse=tot_mse, spikes=tot_spk,
                params=sum(q.numel() for q in model.parameters()
                           if q.requires_grad),
                label=f"{label}, T={T}")


# --------------------------------------------------------------------------
# 함수 하나 처리
# --------------------------------------------------------------------------

def sweep_function(name, domain, e_min, e_max, Ns, Ts, alphas, epochs, seed,
                   m, near0="single", verbose=True):
    """뱅크 × (N,T,alpha) 격자만 계산한다(샤딩 가능한 무거운 부분)."""
    fn, _ = functions.REGISTRY[name]
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max,
                          near0=near0)

    bank_grids, bank_meta = [], []
    t0 = time.time()
    for bi, spec in enumerate(router.banks):
        span = router.reachable(bi, domain[0], domain[1])
        if span is None:
            bank_grids.append([])
            bank_meta.append(dict(idx=bi, kind="unreachable",
                                  interval=list(router.bank_interval(bi))))
            continue
        d = bank_data(fn, spec, span, m)
        desc = descriptors(d["yf"], d["xf"])
        cells = []
        for N in Ns:
            for T in Ts:
                for ak in alphas:
                    cells.append(fit_cell(fn, d, N, T, ak, epochs, seed))
        bank_grids.append(cells)
        if spec["kind"] == "near0":
            kind = "near0"
        elif spec["kind"] == "near0s":
            kind = f"near0 s={spec['sign']:+.0f}"
        else:
            kind = f"mag s={spec['sign']:+.0f} e={spec['e']}"
        bank_meta.append(dict(idx=bi, kind=kind, interval=list(span), **desc))
        if verbose:
            best = min(cells, key=lambda c: c["mse"])
            print(f"    bank {bi:2d} {kind:16s} "
                  f"[{span[0]:+.3f},{span[1]:+.3f}] "
                  f"delta={desc['delta']:.3g} nonlin={desc['nonlin']:.2e} "
                  f"best mse={best['mse']:.2e} @N={best['N']},T={best['T']}",
                  flush=True)

    return dict(name=name, domain=list(domain), e_min=router.e_min,
                e_max=router.e_max, banks=bank_meta, grids=bank_grids,
                fit_seconds=time.time() - t0)


def analyse_function(fnres, domain, e_min, e_max, dists, sigma, epochs, seed,
                     m, near0="single", baselines=True):
    """격자(여러 샤드를 합친 것)에 방문확률·프론티어·기준선을 붙인다."""
    name = fnres["name"]
    fn, _ = functions.REGISTRY[name]
    router = PrefixRouter(domain[0], domain[1], e_min=e_min, e_max=e_max,
                          near0=near0)
    bank_grids = fnres["grids"]

    datas = []
    for bi, spec in enumerate(router.banks):
        span = router.reachable(bi, domain[0], domain[1])
        datas.append(None if span is None else bank_data(fn, spec, span, m))

    lams = [0.0] + [10.0 ** k for k in torch.linspace(-9, 1, 120).tolist()]
    fnres["by_dist"] = {}
    for dist in dists:
        p = visit_probs(router, domain, dist, sigma)
        base = []
        if baselines:
            base = [fixed_budget(bank_grids, p, 2, 16, "logspread"),
                    fixed_budget(bank_grids, p, 2, 16, "uniform")]
            base = [b for b in base if b]
            for N in (4, 8):
                base.append(global_baseline(fn, name, domain, N, 16, datas, p,
                                            epochs, seed))
            if domain[0] < 0:
                base.append(global_baseline(fn, name, domain, 4, 16, datas, p,
                                            epochs, seed, signed=True))
        fnres["by_dist"][dist] = dict(
            p=p, frontier=frontier(bank_grids, p, lams), baselines=base)
    return fnres


def merge_shards(shards):
    """같은 함수의 여러 샤드 격자를 뱅크별로 이어 붙인다."""
    merged = {}
    for sh in shards:
        for name, fnres in sh["functions"].items():
            if name not in merged:
                merged[name] = json.loads(json.dumps(fnres))
                continue
            tgt = merged[name]
            for j, cells in enumerate(fnres["grids"]):
                tgt["grids"][j].extend(cells)
            tgt["fit_seconds"] += fnres["fit_seconds"]
    return merged


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "analyse"], default="sweep")
    ap.add_argument("--fns", nargs="+",
                    default=["gelu", "tanh", "silu", "sigmoid", "relu"])
    ap.add_argument("--domain", nargs=2, type=float, default=[-8.0, 8.0])
    ap.add_argument("--e-min", type=int, default=-2)
    ap.add_argument("--e-max", type=int, default=4)
    ap.add_argument("--near0", choices=["single", "signed"], default="single",
                    help="near-zero 영역을 부호로 나눌지 (실험 2)")
    ap.add_argument("--Ns", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--Ts", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--alphas", nargs="+", default=["uniform", "logspread"])
    ap.add_argument("--dists", nargs="+", default=["uniform", "gauss"])
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", nargs="+", default=[],
                    help="analyse 모드: 합칠 sweep JSON들")
    ap.add_argument("--json", default="results/bank_budget.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)

    if args.mode == "sweep":
        res = dict(meta=vars(args), functions={})
        for name in args.fns:
            print(f"[{name}] domain={tuple(args.domain)} "
                  f"e=[{args.e_min},{args.e_max}) alphas={args.alphas}",
                  flush=True)
            res["functions"][name] = sweep_function(
                name, tuple(args.domain), args.e_min, args.e_max, args.Ns,
                args.Ts, args.alphas, args.epochs, args.seed, args.m,
                near0=args.near0)
            print(f"[{name}] {res['functions'][name]['fit_seconds']:.0f}s",
                  flush=True)
    else:
        shards = [json.load(open(p, encoding="utf-8")) for p in args.shards]
        merged = merge_shards(shards)
        res = dict(meta=vars(args), functions={})
        for name, fnres in merged.items():
            print(f"[analyse] {name} "
                  f"({sum(len(g) for g in fnres['grids'])} cells)", flush=True)
            res["functions"][name] = analyse_function(
                fnres, tuple(args.domain), args.e_min, args.e_max, args.dists,
                args.sigma, args.epochs, args.seed, args.m, near0=args.near0)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
