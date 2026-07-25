---
tags: [knowledge, open-issues, caveats]
updated: 2026-07-25
---

# Open Issues and Caveats

**"거의 완벽 재현"이라 말할 수 있나?** → CPU 함수근사·단일연산 계층에 한해 **재현/능가**.
그러나 아래 유보가 남아있다. 과장 금지.

## 아직 검증 안 된 것
1. **논문의 exact 5N·고정 α 형태가 논문 수치를 내는지 미검증.** 본 구현은 MBE
   *동역학*엔 충실하나, 논문 미명시 메커니즘(basis별 α, 닫힌형 readout, signed)을
   **보강**했다. 논문에 쓰인 그대로의 뉴런이 그 수치를 내는지는 별도 실험 필요.
   → 재현 주장의 최우선 관문. [[Roadmap and Next Steps]]
2. **전체 변환 프레임워크 미배선** — Phase 3는 단일 연산 검증. 실제 attention+MLP+LN
   블록에 배선해 실 가중치로 forward한 적 없음.
3. **다운스트림 벤치마크 전무** — Table 1–4(ImageNet/NLU/NLG)는 손도 안 댐. 이게
   논문의 헤드라인 결과. GPU/vast.ai 필요.
4. **PASN 본체 미구현** — 라우터/뱅크 클래스 없음. R=1 등가성만 테스트.

## 알려진 오차/약점
- **FP-mult 상대오차 ~1.9%** — `MBE_Id` 항등복원이 주 오차원, 피연산자 범위에 비례.
  넓은 범위 곱에서 커짐. (Softmax/LayerNorm은 좁은 범위라 작음)
- **GELU 학습 편차** — basis를 늘려도 단조 개선 아님. 시드/최적화 편차 존재.

## 논문이 미명시해 추측한 것
surrogate, 초기화("fixed as [20]"), α 규약, optimizer 세부, calibration 배치 크기,
GPT-2 정확 변종, RoBERTa fine-tuning 레시피. → [[Design Decisions]]

## 정직한 한 줄
> "MBE 뉴런 **동역학**을 충실히 구현하고, CPU에서 **함수근사(Table X/VII)와 스파이크
> 단일연산(Phase 3)을 논문 수준으로 재현/능가**했다. 단, 논문 미명시 메커니즘을
> **보강**한 형태이며, 전체 파이프라인·벤치마크와 PASN 본체는 **미착수**다."
