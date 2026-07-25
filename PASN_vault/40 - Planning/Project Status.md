---
tags: [planning, status]
updated: 2026-07-25
---

# Project Status (스냅샷)

> 이 노트는 "지금 어디까지 됐나"의 단일 진실. 새 세션 시작 시 여기부터 확인.

## Phase 진행도
- ✅ **Phase 1** — MBE 뉴런 코어 + surrogate ([[MBE Neuron Core]])
- ✅ **Phase 2** — 함수근사 검증, Table X/VII **재현·능가** ([[Table X Reproduction]], [[Table VII and SiLU]])
- ✅ **Phase 3** — 스파이크 FP-mult/Softmax/LayerNorm 조립·검증 ([[Phase 3 Verification]])
- 🟡 **Phase 4** — 변환 프레임워크(calibrate→build→replace) + 토이 Transformer 배선
  **CPU 완결·검증** ([[Phase 4 - Conversion Framework]]). 남은 것: 4d 실 모델
  sanity(GPT-2×WikiText-2, vast.ai) + 4e 발화율 계측.
- ⬜ **Phase 5** — 전체 벤치마크 (GPU/vast.ai)
- ⬜ **Phase 6** — 에너지 추정 + 논문 정리

## 부가 상태
- 🟢 **PASN 본체 코어 구현·검증 + 전방위 비교 완료** — `src/mbe/pasn.py` (prefix 라우터 +
  binade 뱅크 + build_pasn + 적응 N_j). v2 novelty(적응예산+공짜 coarse code).
  - **함수근사**(`results/pasn_vs_mbe.md`): GELU/SiLU 프론티어 1~2자릿수 지배; 좁은
    1/x·2^x·1/√x는 정확도 동급이나 **3~11× 적은 스파이크**. 적응 N_j가 저장·스파이크 절감.
  - **조립 op**(`results/pasn_ops_and_phase4.md`): FP곱 ~8× 정확·절반 스파이크;
    softmax 동급·~4× 적은 스파이크; layernorm 더 정확·~12× 적은 스파이크.
  - **Phase 4 변환**: PASN-변환이 forward 오차 ~7×↓(2.5e-2→3.4e-3) + GELU 스파이크 ~2.7×↓.
  → [[PASN Method]]. 남은 것: faithful 도메인(−120,10)·다층·SiLU 모델·논문 표.
- ⬜ **논문 exact 5N 형태 검증** — 미착수 ([[Open Issues and Caveats]] #1)

## 테스트/검증 현황
- `tests/test_mbe_neuron.py` — **16/16 통과** (뉴런, 5N, signed, readout, Phase 3 3종,
  Phase 4 3종, **PASN 3종**: 라우터 binade·R=1 bit-identical·dispatch).
- `experiments/reproduce_table10.py`, `verify_phase3.py`, `verify_phase4.py`,
  **`compare_pasn_mbe.py`** — 재현 OK.
- **Phase 4 핵심 결과**(토이 1층 d=16): full spike-mult rel|err|=4.1e-2 ≈ exact-mult 4.0e-2
  → **배선 버그 없음**. op별: activation(GELU) 4.0e-2 지배, layernorm 7.4e-3, softmax 1.7e-3,
  matmul 6.9e-4. 오차원=GELU 근사 품질(n_basis·signed-split이 레버) → PASN 동기와 직결.

## 다음 결정 필요 (사용자에게 물을 것)
결정됨: Phase 4→5 진행(GPT-2×WikiText-2 첫 타깃). 4d(vast.ai) 착수 시 git 동기화 루프 확정.
→ [[Roadmap and Next Steps]]

## 최근 세션
[[Session Log]] 참조 (2026-07-24~25: Phase 1–3 + signed + 닫힌형 readout).
