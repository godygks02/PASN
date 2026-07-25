---
tags: [overview, pasn]
updated: 2026-07-25
---

# Project Overview

## 목표
1. 기존 **MBE 뉴런** 논문(training-free ANN→SNN 변환, AAAI-26)을 **재현·검증**한다.
2. 그 위에 개선 프레임워크 **[[PASN Method|PASN]]**(Prefix-Adaptive Spiking Neuron)을
   고안·구현·검증한다.
3. 최종적으로 **논문을 작성**한다.

## 왜 하는가 (연구 동기)
- SNN은 event-driven·spike 기반이라 에너지 효율이 높지만, Transformer의 비선형
  연산(GELU, Softmax, LayerNorm, FP 곱)을 스파이크로 변환하기 어렵다.
- 기존 A2S(ANN→SNN)는 재학습이 필요하거나 긴 timestep을 요구한다.
- MBE 뉴런은 **재학습 없이(training-free)** 짧은 timestep으로 near-lossless 변환을
  주장한다 → 우리의 baseline.
- **PASN**은 MBE의 "하나의 전역 basis 뱅크" 한계를 **입력 범위별 특화 뱅크 + 결정적
  prefix 라우터**로 개선하려는 시도. → [[PASN Method]]

## 범위 (Phase)
| Phase | 내용 | 상태 |
|---|---|---|
| 1 | MBE 뉴런 코어 + surrogate | ✅ |
| 2 | 함수근사 검증 (Table X/VII/Fig.8) | ✅ 재현·능가 |
| 3 | FP-mult/Softmax/LayerNorm 스파이크 조립 | ✅ CPU 검증 |
| 4 | 변환 프레임워크(Algorithm 1) + 소규모 sanity check | ⬜ GPU |
| 5 | 전체 벤치마크 (ImageNet/NLU/NLG × T) | ⬜ GPU/vast.ai |
| 6 | 에너지 추정 + 논문 표/그림 정리 | ⬜ |

상세 상태: [[Project Status]] · 계획: [[Roadmap and Next Steps]]

## 산출물 위치
코드·문서는 프로젝트 루트(`C:\Users\cm120\Project\PASN`)에 있고, 이 vault는 그
지식을 정리한 것. → [[Environment and Repo Map]]

## 제약
- 로컬 GPU 없음 (CUDA 불가). 실제 학습/추론은 **vast.ai** 예정.
- 이 세션까지의 모든 검증은 **CPU 전용**.
