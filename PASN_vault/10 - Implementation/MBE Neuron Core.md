---
tags: [implementation, mbe]
updated: 2026-07-25
---

# MBE Neuron Core

**코드**: `src/mbe/neuron.py` (`MBENeuron`, `MBEConfig`),
`src/mbe/surrogate.py` (surrogate gradients).

## 동역학 (논문 Eq.4–8 그대로)
per basis `n`, `t∈[0,T−1]`:
- 커널(Eq.4): `d_n[t]=exp(−tΔt/τ_d)` (w로 스케일), `r_n[t]=α_v·exp(−tΔt/τ_r)`,
  `Vth_n[t]=α_v·exp(−tΔt/τ_Vth)`
- `u[0]=x_norm`; `s=H(u−Vth)`; `u←u−s·r`; `o←o+s·d`
- 출력: `f̂=Σ_n w_n·o_n(T) (+bias)`

## 파라미터
- 학습: `τ_d, τ_r, τ_Vth`(log-space), `Δt`(log), `w`, (+ `α_v` 학습 시) — 기본 **6N+1**.
- `learn_alpha_v=False, use_bias=False` → **논문의 정확한 5N** (테스트로 검증).
- 커널은 log-param으로 양수 보장. `decay=False`면 커널이 시간상 상수(=ablation).

## 핵심 메서드
- `forward(x)` = `features(x) @ w (+ bias)` — 임의 shape에 element-wise.
- `features(x)` → `(M,N)` per-basis 누적출력 `o_n(T)` (닫힌형 readout용). → [[Fitting and Optimization]]
- `spike_train(x)` → `(M,N,T)` 이진 스파이크 (FP-mult용). → [[Phase 3 - Spiking Ops]]
- `intensities()` → `(N,T)` = `w_n·d_n[t]` (사전계산 강도 D).
- `reconstruct(x)` = `Σ a_{n,t} s_n[t] (+bias)` ≈ f(x).
- `firing_rate(x)`, `num_learnable()`.

## 벡터화
M 샘플 × N basis는 텐서 연산, T(=16)만 파이썬 루프 → CPU에서 빠름.

## 왜 "성공적으로 작동"하는가 (통찰)
임계값/reset 커널은 입력을 **이진탐색(successive approximation, ADC)**으로 분해
(`τ=Δt/ln2`면 매 스텝 절반). 단일 basis의 근사 floor(~1e-3)가 논문 N=1과 일치.
다중 basis 이득의 열쇠는 **basis별 서로 다른 α_v**. 상세: [[Key Insights and Gotchas]],
결정 근거: [[Design Decisions]].

## 관련
[[Signed MBE Neuron]] (부호 확장), [[Fitting and Optimization]] (학습),
[[Table X Reproduction]] (결과).
