---
tags: [knowledge, decisions]
updated: 2026-07-25
---

# Design Decisions (논문 미명시 → 우리가 결정)

논문이 명시하지 않아 우리가 결정한 사항들. **재현 주장 시 반드시 함께 밝혀야 함.**
(원문: `MBE_Implementation_Notes.md`, `MBE_RESULTS.md` §2)

| 항목 | 논문 | 본 구현 | 이유 |
|---|---|---|---|
| Surrogate gradient | 미명시 | ATan(기본) + LBFGS서 sharpening | 역전파에 필요 |
| 입력 스케일 | 암묵 | `[0,1]` 정규화(캘리브레이션) | 수치안정, 도메인 통일 |
| 선두 임계값 `α_v` | 단일 하이퍼파라미터 | **basis별, 곡률배치, 학습가능** | 다해상도 표현(핵심) |
| 출력 bias | 없음(LN β만) | 학습가능 DC offset 1개 | `f(x_min)≠0` baseline |
| Readout 최적화 | lr0.01/200ep/exp0.99 | Adam+LBFGS+**닫힌형 (w,bias) 해** | 표현 상한 도달(핵심) |
| 부호 처리 | 미명시 | **극성 분할 signed 뉴런** | 비단조 음의 딥 |
| 파라미터 수 | 5N | 기본 6N+1, 옵션으로 5N | α_v 학습화로 +N |

## 5N 정확 복원 스위치
`MBEConfig(learn_alpha_v=False, use_bias=False)` → 논문의 정확한 5N 형태.
(단, 이 형태가 논문 수치를 내는지는 **미검증** → [[Open Issues and Caveats]])

## 왜 이 결정들이 정당한가 (요지)
- **basis별 α_v**: 공유 α는 basis를 nested(공선형) 코드로 붕괴시킴을 실험으로 확인
  (특징행렬 singular). 다해상도는 서로 다른 선두 임계값을 요구.
- **곡률배치**: Algorithm 1 step 3 "collect intervals R"의 정신과 부합.
- **닫힌형 readout**: readout이 선형이라는 사실의 활용, 근사가 아님.
- **signed 분할**: PASN §6이 명시적으로 요구하는 signed extension.

상세 근거·함정: [[Key Insights and Gotchas]].
