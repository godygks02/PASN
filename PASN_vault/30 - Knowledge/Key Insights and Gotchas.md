---
tags: [knowledge, insights, gotchas]
updated: 2026-07-25
---

# Key Insights and Gotchas

이 세션에서 실제로 부딪히고 해결한 것들. **미래 세션이 같은 삽질을 반복하지 않도록.**
(여정 원문: `MBE_Implementation_Notes.md`)

## 재현의 5가지 열쇠 (순서대로)
순진한 Eq.4–8 구현은 ~1e-2에서 막힌다. 논문 수치(~5e-5)엔 아래가 모두 필요:
1. **이진탐색 초기화** — 커널 `τ=Δt/ln2`면 매 스텝 절반(ADC). 단일 basis floor ~1e-3
   = 논문 N=1과 일치.
2. **basis별 `α_v`** — 공유 α면 basis가 nested/공선형으로 붕괴(특징행렬 singular,
   basis 평균 스파이크 `[14.6,6.8,...]`로 중복). 서로 다른 선두 임계값이 핵심.
3. **곡률 기반 배치 + 입력 정규화 + DC bias** — GELU 넓은 평탄 도메인 대응,
   `f(x_min)≠0` baseline.
4. **닫힌형 readout solve** — `(w,bias)`가 선형 → SVD 최소제곱. 모든 함수 ~1자릿수 개선.
5. **signed 극성 분할** — GELU/SiLU 0 근처 음의 딥. 표현 상한 0.5→3e-5.

## Gotchas (함정)
- **`x_min` 시프트 함정**: `u[0]=x−x_min`로 시프트하면 최소 입력이 `u=0`→발화 0→출력 0.
  `f(x_min)≠0` 함수(1/x,2^x)는 baseline 상실. → bias로 해결(정규화+bias 조합).
- **공유 α = basis 붕괴**: N을 늘려도 N=1과 동일. 반드시 basis별 α.
- **LBFGS overflow**: 130 스케일 도메인(GELU)에서 폭주 → 입력 정규화 + 라운드 롤백.
- **decay-off singular**: 커널 상수→특징 공선형→`solve` singular. SVD lstsq + ridge 사용.
- **GELU N=1이 "너무 좋았던" 이전 버전**: 논문 N=1(1e-3)보다 낮아 곡선 모양이 달랐음.
  닫힌형 readout 도입 후 N=1이 논문 floor와 일치 → **더 충실**해짐.

## PASN에의 함의 (중요)
전역 MBE 뱅크가 GELU에서 실패한 지점 = **곡률 몰린 좁은 범위에 해상도 집중 불가**.
→ 이것이 [[PASN Method]] 범위 특화 뱅크의 직접적 동기. 논문화 시 강력한 motivation.

## 진단 도구 (재사용)
- **선형 최소제곱 상한(LS ceiling)**: 임계값 고정 후 최적 강도로 도달 가능한 MSE.
  "표현력 vs 최적화" 문제 구분에 유용. (실제로 GELU 진단에 결정적이었음)

관련: [[Design Decisions]], [[MBE Neuron Core]], [[Fitting and Optimization]].
