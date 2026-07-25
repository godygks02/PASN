---
tags: [reference, glossary]
updated: 2026-07-25
---

# Glossary

- **A2S (ANN→SNN)**: 사전학습 ANN을 스파이킹 신경망으로 변환.
- **MBE 뉴런**: Multi-Basis Exponential Decay. baseline 뉴런. → [[Baseline Paper - MBE]]
- **PASN**: Prefix-Adaptive Spiking Neuron. 우리 개선안. → [[PASN Method]]
- **FS 뉴런**: Few-Spikes. MBE의 전신, 초기화 과의존(EDI)·전역최적성부족(GSO) 문제.
- **basis**: MBE 뉴런의 시간 basis 성분. N개. 각 5 파라미터.
- **α_v (alpha_v)**: basis의 선두(t=0) 임계값 스케일. **basis별로 다르게** 두는 것이 핵심.
- **surrogate gradient**: Heaviside 발화의 역전파용 매끄러운 대체 미분.
- **닫힌형 readout**: `(w,bias)`를 선형 최소제곱으로 정확히 푸는 최적화 기법. → [[Fitting and Optimization]]
- **signed 분할**: 입력을 x=0에서 relu(x)/relu(−x)로 나눠 각 뱅크가 처리. → [[Signed MBE Neuron]]
- **LS ceiling**: 임계값 고정 후 최적 강도로 도달 가능한 MSE 상한(진단 도구). → [[Key Insights and Gotchas]]
- **MBE_Id / MBE_exp / MBE_inv / MBE_invsqrt**: 각각 항등/2^x/1/x/1/√x 근사 프리미티브 뉴런.
- **decay ablation (N=8\*)**: 지수 감쇠 제거 시 성능 → 감쇠의 중요성 입증용.
- **T / N / M**: timestep 수 / basis 수 / 학습 샘플 수. 논문 기본 T=16.
- **firing rate η**: 스파이크 발화 비율(에너지 추정에 사용).
- **SOP / MAC**: Synaptic OP / Multiply-Accumulate. SNN/ANN 에너지 비교 단위.
