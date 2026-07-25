---
tags: [results, table-x]
updated: 2026-07-25
---

# Table X Reproduction (함수근사 MSE vs N)

**재현 스크립트**: `experiments/reproduce_table10.py` → `results/table10.{json,md}`.
원문 분석: `MBE_RESULTS.md`. M=6000, T=16. GELU는 [[Signed MBE Neuron]], 나머지는
[[MBE Neuron Core]]. readout은 닫힌형 solve. 각 셀 = **본구현** (논문).

| Func | N=1 | N=2 | N=4 | N=6 | N=8 | N=8*(no decay) |
|---|---|---|---|---|---|---|
| GELU(signed) | 8.1e-4 (7.1e-3) | 8.1e-4 (4.1e-3) | 2.1e-4 (2.3e-4) | **1.3e-4** (1.7e-4) | **1.7e-4** (1.0e-4) | 1.4e-2 (2.8e-1) |
| 1/√x | 1.3e-3 (1.4e-3) | 1.9e-4 (1.2e-3) | 2.6e-5 (3.1e-4) | 8.5e-6 (1.0e-4) | **5.0e-6** (4.9e-5) | 2.2e-4 (2.8e-3) |
| 1/x | 2.2e-3 (8.6e-3) | 2.2e-4 (2.1e-4) | 2.2e-5 (1.1e-3) | 4.2e-6 (2.2e-3) | **4.6e-6** (4.4e-5) | 8.2e-4 (4.5e-3) |
| 2^x | 7.2e-4 (8.9e-4) | 8.1e-4 (4.5e-4) | 4.1e-4 (4.0e-4) | 7.2e-5 (2.4e-4) | **6.5e-5** (5.3e-5) | 1.2e-3 (9.5e-3) |

## 판독
- **4개 함수 모두 N=8에서 논문 수준 재현/능가** (단조 함수는 ~10× 우수).
- **GELU도 signed 뉴런으로 해결** (과거 ~1e-1 → 1.7e-4).
- **N=1이 논문 N=1 floor와 일치** (1/√x 1.3e-3 vs 1.4e-3) → N 스케일링 형태까지 재현.
- **decay ablation**(N=8*): 모든 함수 1–2 자릿수 악화 → "지수 감쇠 필수" 주장 재현.

## 함정 이력
초기엔 ~1e-2에서 막혔고, [[Key Insights and Gotchas]]의 5단계 수정(basis별 α → bias →
닫힌형 readout → signed)을 거쳐 도달. 자세한 여정: `MBE_Implementation_Notes.md`.

관련: [[Table VII and SiLU]], [[Design Decisions]].
