---
tags: [implementation, phase4, conversion]
updated: 2026-07-25
---

# Phase 4 — Conversion Framework (ANN→SNN, Algorithm 1)

**코드**: `src/mbe/convert.py`, `src/mbe/toy.py`. 검증: `experiments/verify_phase4.py`,
`tests/test_mbe_neuron.py`(Phase 4 3종). Phase 3 프리미티브([[Phase 3 - Spiking Ops]])를
실제 Transformer 블록에 배선한 단계.

## 파이프라인 (Algorithm 1)
1. **Calibrate** — `CalibrationRecorder`가 forward pre-hook으로 각 비선형 op의 입력 실측
   범위/샘플 기록. `calibrate(model, batches)`.
2. **Build** — 측정 범위로 MBE 프리미티브 fit (`build_activation`/`build_softmax`/
   `build_layernorm`/`calibrate_identity`).
3. **Replace** — `convert(model, recorder, cfg, only=...)`가 변환 포인트를 스파이킹
   모듈로 **in-place 교체**. `only={kind}`로 op별 격리 가능.
4. **Evaluate** — 호출자가 지표로 forward.

## 변환 포인트 (op 모듈)
`convert.py`가 마킹 모듈로 스왑 지점을 노출 (functional softmax/matmul은 훅 불가하므로):
- `nn.LayerNorm` → `_SpikingLayerNormModule` (weight/bias 보존)
- `Activation(kind)` → `_SpikingActModule` (GELU/SiLU/Tanh, 단일 MBE 뉴런)
- `Softmax(dim)` → `_SpikingSoftmaxModule`
- `MatMulAA` (activation×activation, 예: QK^T·attn·V) → `_SpikingMatMulModule`
- **`nn.Linear`(activation×weight)는 불변** — native accumulation, training-free 핵심.

## 핵심 설계
- **`spiking_matmul`** (spiking_ops): `A@B = recon(A)@recon(B)`. 분리형 FP곱을 matmul로
  리프트, signed는 relu 4항 분해. activation×activation만 스파이크화.
- **fit 캐시** (`spiking_ops._FIT_CACHE`): 고정 도메인 프리미티브(exp2[0,1], inv[0.5,1],
  invsqrt[0.5,2])와 반복되는 identity 범위를 재사용 → 격리·전체 변환 6회가 재fit 없이 돎.
- **토이 Transformer** (`toy.py`): pre-norm 블록을 op 모듈로 구성해 calibrate→build→replace
  루프를 CPU에서 end-to-end 검증. (임베딩 생략, `(B,S,D)` 연속입력.)

## 검증 결과 (토이 1층, d=16)
| 지표 | rel\|err\| |
|---|---|
| full **spike-mult** | 4.1e-2 |
| full **exact-mult** (프리미티브 오차만) | 4.0e-2 |
| op 격리: activation(GELU) | **4.0e-2 (지배)** |
| op 격리: layernorm | 7.4e-3 |
| op 격리: softmax | 1.7e-3 |
| op 격리: matmul | 6.9e-4 |

- **spike-mult ≈ exact-mult** → 스파이크 FP-mult/matmul/softmax 재구성 경로가 오차를
  거의 안 더함 = **배선 정확**. 오차원은 순수 GELU 근사 품질.
- GELU가 지배 → [[Key Insights and Gotchas]]의 "전역 MBE가 GELU에서 실패"와 일치,
  n_basis↑·signed-split([[Signed MBE Neuron]])이 레버. **PASN 동기와 직결** → [[PASN Method]].

## 남은 것 (vast.ai)
- **4d**: 실 사전학습 모델 sanity — **GPT-2 × WikiText-2**, T=16, ANN 대비 perplexity
  낙폭 <1% 확인. HF의 functional softmax/QK^T 배선(모듈 스왑 불가 지점) 처리가 관건.
- **4e**: 발화율 훅(`MBENeuron.firing_rate`) → Phase 6 에너지(Table 11).
→ [[Roadmap and Next Steps]] · [[Project Status]]
