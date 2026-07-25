---
tags: [implementation, phase3, spiking]
updated: 2026-07-25
---

# Phase 3 — Spiking Ops (FP-mult / Softmax / LayerNorm)

**코드**: `src/mbe/spiking_ops.py`. 검증: `experiments/verify_phase3.py`.
결과: [[Phase 3 Verification]]. 원문 정리: `PHASE3_RESULTS.md`.

MBE 뉴런으로 Transformer 비선형 3종을 스파이크 기반으로 조립. 각 비선형은 캘리브레이션된
MBE 프리미티브(`MBE_Id`, `MBE_exp`, `MBE_inv`, `MBE_invsqrt`)로 근사, 정수 2의 거듭제곱은
정확한 비트연산.

## 1. FP 곱 (Eq.9–12, 22–27)
- `MBE_Id`(bias 없는 항등)로 피연산자 → 스파이크. `reconstruct(x)=Σa_{n,t}s_n[t]≈x`.
- 강도 외적 `D=a⊗a'`(사전계산) + 스파이크 외적 `S=b(x1)⊗b(x2)`(이진) → `Σ(D⊙S)`.
- 분리 가능 → `recon(x1)·recon(x2)`와 동일(대량계산엔 이 형태). `multiply_outer`가
  명시적 D⊙S로 등가 확인.
- **부호**: `x=relu(x)−relu(−x)` 분해(4항). → [[Signed MBE Neuron]] 원리 재사용.
- API: `spiking_multiply(idn, x1, x2, signed=True, idn2=None)`.

## 2. Softmax (Eq.13, Table VIII) — `SpikingSoftmax`
- `e^x=2^⌊x·log2e⌋·2^frac`; 정수부=비트시프트, `2^frac`=`MBE_exp`([0,1]).
- 합 S(정확 reduce) → `1/S`: `frexp`로 `S=m·2^e`(m∈[0.5,1)) → `MBE_inv`([0.5,1])·2^{−e}.
- 최종 `e^x·(1/S)` = 스파이크 FP곱. 안정화 위해 row-max 차감.

## 3. LayerNorm (Fig.5c) — `SpikingLayerNorm`
- 평균 μ(정확) → `(x−μ)²/n` = FP곱(1/n을 강도에 흡수) → 분산.
- `1/√var`: `frexp` + **지수 홀짝 보정**(E 짝수로) → `MBE_invsqrt`([0.5,2])·2^{−E/2}.
- 정규화 `(x−μ)·invstd` = FP곱(signed) → affine(γ,β).

## 빌더 (범위 자동 캘리브레이션)
- `build_softmax(sample_logits)`, `build_layernorm(sample_x)` — 측정 범위로 프리미티브 학습.
- `spike_mult=False`로 곱을 정확연산 대체 → 다른 프리미티브(exp/inv/invsqrt) 오차만 격리.

## 주의
- 주 오차원 = `MBE_Id` 항등 복원(피연산자 범위 비례). Softmax/LayerNorm은 피연산자가
  좁은 범위([0,1], mantissa[0.5,2])라 오차가 한 자릿수 작다.
- **아직 실제 Transformer 블록에 배선/실 가중치 변환은 안 함** → Phase 4. [[Roadmap and Next Steps]]
