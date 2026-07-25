---
tags: [overview, pasn, method]
updated: 2026-07-25
---

# PASN Method (우리가 만들 개선안)

원문: `PASN_method.md` (프로젝트 루트).

## 아이디어
MBE는 전체 입력 도메인에 **하나의 전역 basis 뱅크**를 쓴다. PASN은 같은 MBE 동역학을
유지하되 basis를 **범위별 특화 뱅크(range-specialized banks)** 여러 개로 나누고,
**결정적 prefix 라우터**가 입력마다 뱅크 하나를 선택한다.

```
MBE : x → (전역 basis 뱅크) → 출력
PASN: x → (FP prefix 라우터) → 선택된 뱅크 → 출력
```

## 구성요소
1. **Order-preserving key**: FP32 비트를 부호 정렬 가능한 unsigned 키로 변환
   (`x<0`이면 `~b(x)`, 아니면 `b(x)⊕0x80000000`).
2. **Prefix 라우터**: 공통 상위비트 무시 후 다음 k비트를 prefix로 → 뱅크 인덱스
   `z(x)∈{0..R−1}`, `R≤2^k`. **학습 파라미터 없음**(결정적).
3. **범위별 뱅크** `B_j`: 각 뱅크는 완전한 MBE basis 집합. 입력당 **한 뱅크만** 활성.

## 핵심 성질
| 성질 | MBE | PASN |
|---|---|---|
| basis 동역학 | 지수 감쇠 | 동일 |
| 뱅크 수 | 1 | R (다수) |
| 뱅크 선택 | 없음 | 결정적 prefix |
| 입력당 활성 뱅크 | 1 | 1 |
| 학습 라우팅 | 없음 | 없음 |
| 저장 basis | N | Σ_r N_r |

- **R=1이면 PASN = MBE** (구현 검증 포인트: 1-뱅크 PASN은 MBE와 bit-identical).
  → 현재 [[MBE Neuron Core]]의 `test_r1_reduces_to_single_bank`로만 확인됨.
- **부호 규칙(§6)**: prefix는 부호 정렬만 담당. 음수 표현 능력은 **MBE baseline의
  signed 메커니즘**이 제공해야 하고, PASN 모든 뱅크가 이를 동일 적용해야 함. → 우리가
  구현한 [[Signed MBE Neuron]]이 바로 그 메커니즘.

## 왜 PASN이 유리할 수 있나 (이 세션에서 얻은 근거)
전역 MBE 뱅크가 GELU (−120,10)에서 실패한 이유는 곡률이 몰린 좁은 범위에 해상도를
집중 못 해서다. **범위 특화 뱅크는 임계 범위(near-zero)에 전용 해상도를 배정**할 수
있다 → PASN의 직접적 동기. (단, 실제 이득은 별도 검증 필요.) → [[Key Insights and Gotchas]]

## v2 구체화 — Adaptive-Budget Prefix Routing (novelty 확정)
"MBE 유지 + 범위 비닝"만으론 novelty 약함(§8 자인). 헤드라인 novelty를 다음으로 고정
(원문 `PASN_method.md` §9–13):
1. **FP 지수 prefix = 공짜·정확 coarse code** — sign+exponent 비트로 binade
   `|x|∈[2^e,2^{e+1})` 라우팅. **0 근처 로그밀도** → GELU 곡률 집중부에 자동 전용 뱅크.
2. **뱅크는 binade 내 잔차만 근사** — `x=σ·2^e·(1+ρ)`, 뱅크는 `g(ρ)=f(σ2^e(1+ρ))`,
   `ρ∈[0,1)`만. 지수 `2^e`는 라우팅이 **스파이크 0으로 제공**.
3. **뱅크별 이질 예산 `(N_j,T_j)`** — 평탄 꼬리 N=1, 곡률부 큰 N. compute를 범위별 배정.
4. **부호 흡수** — σ가 최상위 비트 → 크기 뱅크는 단일부호만 봄, near-zero 뱅크 1개만 부호 교차.

## 구현 상태 (2026-07-25)
✅ **코어 구현 + 전방위 비교 완료**: `src/mbe/pasn.py` (`PrefixRouter`/`PASNNeuron`/
`build_pasn` + 적응 N_j). 테스트 — 라우터 binade, **R=1 = MBE bit-identical**, dispatch.
- **함수근사**(`results/pasn_vs_mbe.md`): GELU/SiLU 프론티어 **1~2자릿수 지배**; 좁은
  1/x·2^x·1/√x는 정확도 동급·**3~11× 적은 스파이크**.
- **조립 op**(`results/pasn_ops_and_phase4.md`): FP곱 ~8× 정확·½ 스파이크, softmax 동급·
  ~4×↓ 스파이크, layernorm 더 정확·~12×↓ 스파이크.
- **Phase 4 변환**: PASN-변환 forward 오차 ~7×↓ + GELU 스파이크 ~2.7×↓.

정직한 비용: 넓은 도메인서 저장 파라미터↑ + prefix/뱅크 주소지정 오버헤드.
다음: faithful 도메인(−120,10)·다층·논문 표/그림. → [[Roadmap and Next Steps]]
