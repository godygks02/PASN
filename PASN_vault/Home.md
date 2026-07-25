---
tags: [moc, pasn]
updated: 2026-07-25
status: active
---

# 🏠 PASN Vault — Home (Start Here)

> **미래 세션 진입점.** 이 노트만 읽으면 프로젝트가 무엇인지, 지금까지 무엇을
> 했는지, 다음에 무엇이 필요한지 바로 파악할 수 있다. 세부는 링크를 따라간다.

## 한 줄 요약
기존 **MBE 뉴런**(training-free ANN→SNN 변환, AAAI-26) 논문을 재현·검증하고, 그
위에 개선 프레임워크 **PASN**(Prefix-Adaptive Spiking Neuron)을 고안·구현·검증해
**최종적으로 논문을 작성**하는 연구 프로젝트.

## 지금 상태 (2026-07-25 기준)
- ✅ **Phase 1–2 완료** — MBE 뉴런 구현 + 함수근사 검증(Table X/VII **재현·능가**)
- ✅ **Phase 3 완료** — 스파이크 기반 FP-mult / Softmax / LayerNorm 조립·검증(CPU)
- ⬜ **Phase 4–5 미착수** — 실제 Transformer 블록 배선 + ImageNet/NLU/NLG end-to-end (GPU/vast.ai)
- ⬜ **PASN 본체 미구현** — 라우터·뱅크는 R=1 등가성만 테스트됨

자세히: [[Project Status]] · [[Roadmap and Next Steps]]

## 🗺️ Map of Content

### 개요 (무엇인가)
- [[Project Overview]] — 목표, 범위, 왜 하는가
- [[Baseline Paper - MBE]] — 재현 대상 논문 핵심 수식·표
- [[PASN Method]] — 우리가 만들 개선안(prefix 라우터 + range 뱅크)

### 구현 (어떻게 만들었나)
- [[MBE Neuron Core]] — Eq.4–8 뉴런
- [[Signed MBE Neuron]] — 부호/비단조(GELU) 해결
- [[Fitting and Optimization]] — 학습 레시피 + 닫힌형 readout
- [[Phase 3 - Spiking Ops]] — FP-mult/Softmax/LayerNorm 조립

### 결과 (얼마나 잘 됐나)
- [[Table X Reproduction]] — 함수근사 MSE vs N
- [[Table VII and SiLU]] — SiLU 구간별
- [[Phase 3 Verification]] — 3종 연산 오차

### 지식 (놓치면 안 되는 것)
- [[Design Decisions]] — 논문 미명시 → 우리가 결정한 것들
- [[Key Insights and Gotchas]] — 함정과 핵심 통찰
- [[Open Issues and Caveats]] — 아직 못 한 것 / 조심할 것

### 계획·레퍼런스
- [[Roadmap and Next Steps]] — 다음에 할 일
- [[Session Log]] — 세션별 진행 기록
- [[Environment and Repo Map]] — SNN 환경, 파일 위치, 실행법
- [[Glossary]] — 용어

## 🔑 미래 세션이 가장 먼저 알아야 할 3가지
1. **환경**: 반드시 `SNN` conda env 사용 (py3.10, torch 2.11 CPU). 시스템 기본
   py3.14 아님. → [[Environment and Repo Map]]
2. **핵심 통찰**: 논문 수치 재현엔 basis별 α + 곡률배치 + bias + **닫힌형 readout**
   + **signed 분할**이 필요했다(논문 미명시). → [[Key Insights and Gotchas]]
3. **정직한 범위**: 함수근사·단일연산은 재현/능가했지만, **전체 변환 파이프라인과
   다운스트림 벤치마크(Table 1–4)는 미착수**. → [[Open Issues and Caveats]]
