---
tags: [results, phase3]
updated: 2026-07-25
---

# Phase 3 Verification (스파이크 연산 vs torch)

**스크립트**: `experiments/verify_phase3.py`. 구현: [[Phase 3 - Spiking Ops]].
`spike-mult`=모든 곱을 스파이크 외적; `exact-mult`=곱만 정확연산(다른 프리미티브 오차 격리).

| 연산 | mean\|err\| | max\|err\| | 비고 |
|---|---|---|---|
| FP-mult (비음수, [0,8]) | 5.1e-2 | 3.8e-1 | 상대 ~1.9% |
| FP-mult (부호, [−6,6]) | 2.9e-2 | 2.3e-1 | |
| outer-product D⊙S | — | — | **분리형과 정확히 일치** (16.831=16.831) |
| Softmax (spike-mult) | 6.2e-4 | 5.5e-3 | 행합 ≈0.997 |
| Softmax (exact-mult) | 3.2e-4 | 5.0e-3 | exp+역수만 |
| LayerNorm (spike-mult) | 4.5e-3 | 3.4e-2 | γ,β 포함 |
| LayerNorm (exact-mult) | 1.4e-3 | 3.3e-2 | invsqrt만 |

## 판독
- Softmax·LayerNorm이 정확연산을 **~1e-3(mean) 수준으로 재현**, Softmax 정규화(Σ≈1)도
  스파이크+IEEE754 분해만으로 성립 → Table VIII/Fig.5c 분해가 올바르게 조립됨.
- 명시적 D⊙S 외적 = 분리형 recon곱 정확히 일치 → 논문의 Hadamard 형식으로 구현됨을 확인.
- 주 오차원 = `MBE_Id` 항등복원(범위 비례). Softmax/LayerNorm은 좁은 범위라 오차 작음.

## 한계
단일 연산 수치검증. **실제 Transformer 블록 배선·실 가중치 end-to-end는 미착수** → Phase 4.
[[Roadmap and Next Steps]], [[Open Issues and Caveats]].
