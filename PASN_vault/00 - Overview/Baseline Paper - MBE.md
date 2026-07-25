---
tags: [overview, paper, mbe]
updated: 2026-07-25
---

# Baseline Paper — MBE 뉴런

**"Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformers"**
(AAAI-26). PDF: `paper/Training-Free ANN to SNN conversion with appendix.pdf`
(appendix 10–15p는 스캔 이미지 → `paper/appendix_p*.png`로 추출해 판독함).

## 핵심 아이디어
- **FS 뉴런의 한계**: EDI(초기화 과의존) + GSO(전역 최적성 부족) → 특히 0 근처
  고곡률 영역에서 근사 실패.
- **MBE 뉴런**: 지수 감쇠(exponential decay) 파라미터 갱신 + 다중 basis 인코딩으로
  다해상도 근사. 5N 파라미터(basis당 5개)로 3T의 FS보다 적음.

## 핵심 수식 (구현 대상)
- **Eq.4** 파라미터 갱신: `Para(τ_n,t) = α·exp(−t·Δt/τ_n)`, `τ_n∈{τ_d,τ_r,τ_Vth}`
- **Eq.5–7** basis 동역학: `u[0]=x`; `s=H(u−Vth)`; `u←u−s·r`; `o←o+s·d`
- **Eq.8** 출력: `f̂(x)=Σ_n w_n·o_n(T)`
- **Eq.9–12,22–27** FP 곱: 두 MBE_Id 스파이크의 외적 D,S → `Σ(D⊙S)`
- **Eq.13,Table VIII** Softmax: `e^x=2^⌊·⌋·2^frac`, IEEE754 역수
- **Fig.5c** LayerNorm: 제곱합 + 역제곱근 분해

우리 구현 매핑: [[MBE Neuron Core]], [[Phase 3 - Spiking Ops]]

## 재현 대상 표 (우선순위: CPU 가능한 것부터)
| 표/그림 | 내용 | 우리 상태 |
|---|---|---|
| **Table X** | N=1..8, decay on/off 별 GELU/invsqrt/inv/2^x MSE | ✅ [[Table X Reproduction]] |
| **Table VII** | SiLU 구간별 FS vs MBE | ✅ [[Table VII and SiLU]] |
| Table VI | 파라미터 수 (5N vs 3T) | ✅ 산술 검증 |
| Table 1 | ImageNet ViT/CNN acc | ⬜ GPU |
| Table 2 | NLU RoBERTa | ⬜ GPU |
| Table 3 | NLG GPT-2 perplexity | ⬜ GPU |
| Table 4 | T=8/10/12/16 | ⬜ GPU |
| Table 11 | firing rate 에너지 | ⬜ |

## 논문 설정값 (appendix G.1)
- GELU/Tanh: **N=4, T=16**. FP-Id/exp/inv/invsqrt: **N=8, T=16**.
- 도메인: GELU (−120,10); 2^x [0,1]; 1/x [0.5,1]; 1/√x [0.5,2]; M=10,000.
- 학습: lr=0.01, 200 epoch, "exponential optimizer decay 0.99".
- **미명시(우리가 결정)**: surrogate, 초기화("fixed as [20]"), α 규약, optimizer 세부.
  → [[Design Decisions]]

## Table X 논문 수치 (재현 기준)
| Func | N=1 | N=2 | N=4 | N=6 | N=8 | N=8(no decay) |
|---|---|---|---|---|---|---|
| GELU | 7.1e-3 | 4.1e-3 | 2.3e-4 | 1.7e-4 | 1.0e-4 | 2.8e-1 |
| invsqrt | 1.4e-3 | 1.2e-3 | 3.1e-4 | 1.0e-4 | 4.9e-5 | 2.8e-3 |
| inv | 8.6e-3 | 2.1e-4 | 1.1e-3 | 2.2e-3 | 4.4e-5 | 4.5e-3 |
| 2^x | 8.9e-4 | 4.5e-4 | 4.0e-4 | 2.4e-4 | 5.3e-5 | 9.5e-3 |
