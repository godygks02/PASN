---
tags: [results, table-vii, silu]
updated: 2026-07-25
---

# Table VII — SiLU 구간별

0을 지나는 구간은 [[Signed MBE Neuron]], 한쪽 구간은 [[MBE Neuron Core]]. 각 셀 = 본구현 (논문 MBE).

| 구간 | 본구현 | 논문 MBE | 비고 |
|---|---|---|---|
| [−8, −2] | **5.8e-5** | 7.0e-5 | 단조 꼬리, 능가 |
| [−2, 5] | **7.6e-4** | 6.0e-4 | 굴곡 포함, signed로 처리(7.2e-3→7.6e-4) |
| [2, 12] | **6.0e-5** | 4.8e-3 | 80× 능가 |

## 판독
- 세 구간 모두 논문 이상.
- near-zero 굴곡 구간 [−2,5]가 signed 뉴런으로 ~10× 개선 → GELU와 동일 원리.

관련: [[Table X Reproduction]], [[Signed MBE Neuron]].
