---
tags: [implementation, mbe, signed]
updated: 2026-07-25
---

# Signed MBE Neuron

**코드**: `src/mbe/signed.py` (`SignedMBENeuron`), 빌더 `functions.make_signed(...)`.

## 왜 필요한가
비음수 막전위 뉴런은 **GELU/SiLU의 0 근처 음의 딥**(비단조)을 표현 못 한다.
GELU (−120,10)의 선형-readout 표현 상한이 **~0.5**(사실상 불가능). → [[Open Issues and Caveats|초기 미해결 이슈]]였다가 이 세션에서 해결.

## 메커니즘 (signed-magnitude 분할)
입력을 피벗 `c`(기본 0)에서 분할:
```
f̂(x) = [ pos.features(relu(x−c)) , neg.features(relu(c−x)) ] · w + bias
```
- `pos` 뱅크: 양수부 `relu(x−c)` 인코딩. `neg` 뱅크: 음수부 크기 `relu(c−x)` 인코딩.
- 각 뱅크는 독립 `MBENeuron`(feature 전용, 내부 w 동결), **하나의 공유 readout**.
- x=0을 피벗으로 → GELU 고곡률점에 정렬 + 음수 영역 전용 해상도.

## 효과 (검증)
- GELU (−120,10) 표현 상한: **0.5 → 2.9e-5** (paper 수준 아래).
- 학습 결과 GELU N=8: **1.7e-4** (paper 1.0e-4). 과거 ~1e-1 → 해결.
- SiLU [−2,5](굴곡 포함): 7.2e-3 → **7.6e-4** (10× 개선).
- 상세: [[Table X Reproduction]], [[Table VII and SiLU]].

## PASN와의 관계
이것이 [[PASN Method]] §6이 요구한 "signed extension"의 구현. **모든 PASN 뱅크가
동일 적용**해야 하는 baseline 성질. prefix 라우터는 부호 정렬만 담당하고 음수 표현은
이 메커니즘이 제공.

## 사용
```python
sm = functions.make_signed("gelu", n_pos=4, n_neg=4, pivot=0.0)
fit_model(sm, x, y)   # 두 뱅크 곡률배치 자동 + 닫힌형 readout
```
비단조 함수(GELU)엔 signed, 단조 함수(1/x 등)엔 일반 [[MBE Neuron Core]] 사용.
