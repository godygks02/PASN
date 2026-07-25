---
tags: [reference, environment]
updated: 2026-07-25
---

# Environment and Repo Map

## ⚠️ 환경 (반드시 확인)
- **`SNN` conda env 사용**: `C:\Users\cm120\miniconda3\envs\SNN`
  (Python 3.10.0, torch 2.11.0+cpu, numpy 2.2, matplotlib 3.10).
- Git Bash에서 호출: `~/miniconda3/envs/SNN/python.exe`
- **시스템 기본 py3.14 쓰지 말 것** (torch 2.13, ecosystem 호환 나쁨).
- 로컬 GPU 없음(CUDA 불가). 실제 학습/추론은 vast.ai.

## 실행 명령
```bash
~/miniconda3/envs/SNN/python.exe tests/test_mbe_neuron.py          # 11/11
~/miniconda3/envs/SNN/python.exe experiments/reproduce_table10.py  # Table X
~/miniconda3/envs/SNN/python.exe experiments/verify_phase3.py      # Phase 3
~/miniconda3/envs/SNN/python.exe experiments/plot_approximation.py # 그림
```

## 프로젝트 루트: `C:\Users\cm120\Project\PASN`
```
src/mbe/
  neuron.py        MBENeuron, MBEConfig (Eq.4-8)          → [[MBE Neuron Core]]
  signed.py        SignedMBENeuron                        → [[Signed MBE Neuron]]
  surrogate.py     ATan/sigmoid/triangle surrogate
  functions.py     타깃함수, make_config/make_signed, curvature_alpha, sample
  fit.py           fit_function, fit_model, solve_readout → [[Fitting and Optimization]]
  spiking_ops.py   FP-mult/Softmax/LayerNorm              → [[Phase 3 - Spiking Ops]]
experiments/
  reproduce_table10.py   Table X 재현
  plot_approximation.py  근사 그림
  signed_test.py         signed 비교(초기 실험)
  verify_phase3.py       Phase 3 검증
tests/test_mbe_neuron.py 11 tests
results/                 table10.{json,md}, approx_*.png
paper/                   PDF + appendix_p*.png (스캔 추출) + extracted.txt
```

## 루트 문서 (vault 밖 원문)
- `MBE_Implementation_Plan.md` — 사전 계획(Phase, 리스크, vast.ai 워크플로우)
- `PASN_method.md` — PASN 방법 정의                       → [[PASN Method]]
- `MBE_RESULTS.md` — 결과·설계 근거
- `MBE_Implementation_Notes.md` — 구현 여정(그대로/문제/수정)
- `PHASE3_RESULTS.md` — Phase 3 상세
- `README.md` — 저장소 개요

## 메모리 (자동 로드)
`C:\Users\cm120\.claude\projects\C--Users-cm120-Project-PASN\memory\`
(snn-conda-env, mbe-reproduction-key-insight, phase3-spiking-ops) — 이 vault와 중복 요약.
