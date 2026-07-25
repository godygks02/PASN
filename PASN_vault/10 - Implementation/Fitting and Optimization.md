---
tags: [implementation, optimization]
updated: 2026-07-25
---

# Fitting and Optimization

**코드**: `src/mbe/fit.py` (`fit_function`, `fit_model`, `solve_readout`),
`src/mbe/functions.py` (`make_config`, `make_signed`, `curvature_alpha`, `sample`).

## 학습 레시피 (`fit_model`)
1. **닫힌형 readout 초기해** — `solve_readout`로 시작.
2. **Adam** (shape 파라미터만; lr=0.03, ExponentialLR γ=0.999) + `solve_every=50`마다
   readout 재-solve.
3. **LBFGS 정련** (surrogate α를 6→12→20 annealing, strong_wolfe) + 각 라운드 후
   readout solve, 나빠지면 롤백.
4. 마지막 readout solve.

## 핵심: 닫힌형 readout (`solve_readout`)
`f̂ = features(x)·w + bias`는 나머지(shape) 파라미터가 고정되면 `(w,bias)`에 대해
**선형** → SVD 최소제곱(`torch.linalg.lstsq` gelsd, ridge rows)으로 정확해.
- shape gradient step 사이에 교대로 풀면 surrogate 훈련이 **표현 상한까지 도달**.
- **모든 함수 MSE 약 1자릿수 개선**, GELU 재현의 결정적 요인.
- decay-off(공선형 특징)에서 singular 방지 위해 SVD + ridge 사용.

## 곡률 기반 임계값 배치 (`curvature_alpha`)
basis별 선두 임계값 `α_v,n`을 **|f′|의 분위수**에 배치 (Algorithm 1 step 3 "collect
intervals R" 정신). 평탄부 대신 고곡률 영역에 해상도 집중. GELU 같은 넓은 도메인에 필수.

## 도메인 처리 (`make_config`)
입력 `[lo,hi]`를 `[0,1]`로 아핀 정규화(`x_min=lo, x_scale=hi−lo`). bias가
`f(lo)≠0` baseline 공급. 부호/양수 도메인 통일 처리.

## 파라미터 유의
- 기본 `learn_alpha_v=True, use_bias=True` → 6N+1.
- 논문 정확 재현(5N)은 `learn_alpha_v=False, use_bias=False`.

배경/함정: [[Key Insights and Gotchas]], 결정: [[Design Decisions]].
