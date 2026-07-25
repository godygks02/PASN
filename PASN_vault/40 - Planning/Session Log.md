---
tags: [planning, log]
updated: 2026-07-25
---

# Session Log

세션별 진행 기록. 최신이 위.

## 2026-07-24 ~ 07-25 — Phase 1–3 구축
**한 일:**
- baseline 논문 정독 (appendix 스캔 이미지 OCR 판독 포함).
- [[MBE Neuron Core]] 구현 (Eq.4–8, surrogate, 5N/6N 파라미터, decay ablation).
- [[Fitting and Optimization]] — Adam+LBFGS+**닫힌형 readout**, 곡률배치, 정규화, bias.
- [[Table X Reproduction]] / [[Table VII and SiLU]] — 4함수 + SiLU 논문 재현·능가.
- [[Signed MBE Neuron]] — GELU/SiLU 비단조 굴곡 해결 (사용자 아이디어 = 극성 분할).
- [[Phase 3 - Spiking Ops]] — FP-mult/Softmax/LayerNorm 조립 + [[Phase 3 Verification]].
- 유닛 테스트 11개, 문서(`MBE_RESULTS.md`, `MBE_Implementation_Notes.md`, `PHASE3_RESULTS.md`).
- 이 vault 생성.

**핵심 발견:** basis별 α + 닫힌형 readout + signed 분할이 논문 수치 재현의 열쇠
(논문 미명시). GELU 실패 지점이 PASN 동기. → [[Key Insights and Gotchas]]

**결정 사항:** 환경은 `SNN` conda env 고정. → [[Environment and Repo Map]]

**미해결로 넘긴 것:** 논문 exact 5N 검증, PASN 본체, Phase 4–5. → [[Open Issues and Caveats]]

---
<!-- 새 세션은 여기 위에 항목 추가 -->
