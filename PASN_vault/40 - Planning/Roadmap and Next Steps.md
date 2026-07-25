---
tags: [planning, roadmap]
updated: 2026-07-25
---

# Roadmap and Next Steps

## 지금 바로 할 수 있는 것 (우선순위 후보)
1. **[검증] 논문 exact 5N 형태 정밀 검증** — `learn_alpha_v=False, use_bias=False`로
   Table X 재현이 되는지 확인. 안 되면 논문 누락 vs 우리 해석 오류 판정. **GPU 불필요,
   가장 값싸고 결정적.** → [[Open Issues and Caveats]] #1
2. **[구현] PASN 라우터·뱅크** — order-preserving key + prefix 라우터 + R개 뱅크.
   R=1이 MBE와 bit-identical인지부터. 그다음 GELU에 R개 뱅크로 near-zero 특화가 이득인지
   측정(PASN 가설의 첫 실증). → [[PASN Method]]
3. **[구현] Phase 4 — Transformer 블록 배선** — [[Phase 3 - Spiking Ops]]의 3연산을
   attention+MLP+LN에 배선, 작은 사전학습 ViT/RoBERTa 1개로 sanity forward.

## Phase 4 진행 상황 (2026-07-25 갱신)
- ✅ **4a–4c (CPU 완결)**: 변환 프레임워크(calibrate→build→replace) + 토이 Transformer
  배선·forward 등가성 검증. spike-mult≈exact-mult로 **배선 정확**, 오차원=GELU 근사.
  → [[Phase 4 - Conversion Framework]].
- ⬜ **4d (vast.ai)**: **GPT-2 × WikiText-2** 실 모델 sanity, T=16, perplexity 낙폭 <1%.
  HF functional softmax/QK^T 배선 처리가 관건.
- ⬜ **4e**: 발화율 계측 → Phase 6 에너지.

## Phase 4–5 (GPU/vast.ai 필요)
- **Phase 4(잔여)**: 위 4d/4e. 첫 타깃 **GPT-2×WikiText-2** (라이선스·fine-tuning 불필요로
  가장 값쌈). ANN 대비 낙폭 <1% 확인.
- **Phase 5**: 전체 벤치마크 — ImageNet(ViT-B/16, ViT-M/16, VGG16, ResNet34), NLU
  (RoBERTa-B/L × SST-2/SST-5/MR/Subj), NLG(GPT-2 × WikiText-2/103), T={8,10,12,16}.
- **Phase 6**: 에너지 추정(Table 11) + 논문 표/그림 + PASN vs MBE 비교.

## vast.ai 워크플로우 (로컬 GPU 없음)
로컬(이 세션)은 코드·스크립트·분석 담당, 사용자가 vast.ai에서 실행 후 로그/결과를
프로젝트 폴더로 동기화 → 다시 분석. git repo화 권장. (원문: `MBE_Implementation_Plan.md` §0)

## 데이터/모델 리스크
- ImageNet val(5만장) 라이선스 다운로드 필요.
- RoBERTa fine-tuning 레시피 논문 미상 → source ANN 직접 학습 가능성.
- GPT-2 정확 변종(346M) 확인 필요.

## PASN 논문화 각도
"전역 MBE가 GELU near-zero에서 실패 → 범위 특화 뱅크로 해결"이 핵심 스토리.
→ [[Key Insights and Gotchas]], [[PASN Method]].
