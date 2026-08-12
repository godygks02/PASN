# CV 축 — ViT × ImageNet-1k

**E8 (ViT-B/16) + E8-M (ViT-M/16).** 2026-08-11 … 08-12, vast.ai RTX 5060 Ti.
논문 Table 1 의 ViT 두 행을 **둘 다** 채웠다.

> **한 줄**: 논문의 두 ViT 셀에서 **−0.020%** 와 **−0.002%**, 저쪽은 −0.527% 와
> −0.745%. **C10(3모달리티) 성립.** 다만 절대 top-1 은 두 셀 모두 비교 불가이고,
> 그 이유가 체크포인트가 아니라 **우리 평가 파이프라인**이라는 것이 이번에 드러났다.

---

## 1. 결과

| 행 | | ANN | SNN | Δ 절대 | **Δ 상대** | 뒤집힌 예측 |
|---|---|---:|---:|---:|---:|---:|
| **ViT-B/16** | **우리** | 80.316 | 80.300 | −0.016 pp | **−0.020%** | **8 / 50,000** |
| | 논문 | 83.44 | 83.00 | −0.44 pp | −0.527% | ~220 / 50,000 |
| **ViT-M/16** | **우리** | 84.922 | 84.920 | −0.002 pp | **−0.0024%** | **1 / 50,000** |
| | 논문 | 85.95 | 85.31 | −0.64 pp | −0.745% | ~320 / 50,000 |

전부 **전체 5만장 validation**, Stage 2(활성함수·LayerNorm·softmax·활성×활성 matmul),
논문과 같은 전역 `T=16`, 프리즈 빌드 설정(relative 항등원 1e-2, epochs 300).

### ⚠️ 정밀도 — 4자리로 쓰지 말 것

**ViT-M 의 −0.0024% 는 5만장 중 1장이다.** 측정 해상도가 1장 = 0.002 pp 이므로 이
값은 **"몇 장 수준"** 이지 소수 넷째 자리가 아니다. 95% 구간으로 보면 참값은 대략
**0 – 5장**, 즉 **|Δ| ≲ 0.01%** 다. **쓸 문장은 "5만장 중 한 자릿수 장이 바뀐다"**
이고, 어느 쪽이든 논문의 0.745%(≈320장)와는 자릿수가 다르다.

---

## 2. 🔴 절대 top-1 은 두 셀 모두 비교 불가 — 그리고 원인이 우리 쪽이다

ViT-M/16 은 **논문이 쓴 바로 그 모델**(timm `vit_medium_patch16_reg4_gap_256`,
Darcet 2023)이라 절대값도 볼 수 있으리라 기대했는데 **아니었다.**

| 모델 | 프레임워크 | 공개 top-1 | **우리 (5만장)** | 차 |
|---|---|---:|---:|---:|
| ViT-B/16 | HF | 81.1 | 80.32 | **−0.78 pp** |
| ViT-M/16 | timm | ~85.9–86.0 | 84.92 | **−1.03 pp** |
| ViT-M/16 (논문 보고) | timm | **85.95** | 84.92 | **−1.03 pp** |

**두 모델·두 프레임워크에서 일관되게 0.8–1.0 pp 낮다.** 모델 속성이 아니라
**평가 파이프라인의 속성**이다. 유력한 원인은 HF `imagenet-1k` parquet 의 JPEG
재인코딩, 또는 리사이즈 보간·crop 세부다. **규명하지 않았다.**

**결론에는 영향 없다** — ANN 과 SNN 양쪽에 똑같이 걸리므로 **상대 손실은 온전하다.**
그러나 *"논문 모델을 쓰니 절대값도 같은 선상"* 이라는 기대는 **접어야 하고**, 그건
이 문서를 쓰는 과정에서 내가 한 번 잘못 말한 것이기도 하다.

---

## 3. timm 배선 — 없었으면 ViT-M 셀은 불가능했다

`hf_convert` 가 기대는 두 훅이 timm 에는 **둘 다 없다.**

| | HF | timm |
|---|---|---|
| 활성함수 | `GELUActivation` 등 래퍼 클래스 | 평범한 **`torch.nn.GELU`** |
| 어텐션 | `ALL_ATTENTION_FUNCTIONS` 레지스트리 | **인라인 계산**, 레지스트리 없음 |
| 마킹 결과 | 12 / 12 | **0 / 0** |

**0/0 은 에러보다 나쁘다** — 아무것도 변환되지 않고, "SNN" 이 ANN 이고, 손실이
**0.00% 로 인쇄된다.** `src/mbe/timm_convert.py` 가 두 훅을 공급하고,
`vit_imagenet.py` 는 **마커가 0 이면 하드 실패**한다.

**세 가지가 핵심이다:**
1. **`nn.GELU` 는 두 개의 다른 함수다** — `approximate` 플래그가 정확 erf 와 tanh 를
   가른다. 안 읽고 매핑하면 RoBERTa 에서 낸 버그를 그대로 반복한다. 인스턴스마다 읽는다.
2. **`fused_attn` 을 강제로 끈다** — 기본 `True` 면 SDPA 한 방으로 가서 마커가 들어갈
   틈이 없다. timm 자신의 unfused 분기를 그대로 복제하고 그 자리에 마커를 넣는다.
3. **무음성을 unfused 기준으로 검사한다** — fused→unfused 전환 자체가 수치를 움직일
   수 있으므로 그것과 마킹을 섞으면 안 된다.

**검증**(`tests/test_timm_convert.py`, 4개):
`torch.equal(unfused, marked) == True`, `torch.equal(fused, act_marked) == True`,
fused/unfused 차이는 **상한만**(`< 1e-5`) — 테스트를 쓰다가 그 차이가 **정확히 0 일
수도** 있음을 발견했다(CPU 에서 같은 커널로 디스패치). "작지만 0 이 아니다" 는 내가
검증 없이 쓴 단언이었다.

📌 **랜덤 초기화 스모크는 `tied=True needs at least one reachable magnitude bank` 로
죽는다** — 랜덤 가중치 + `randn` 입력의 산물이지 timm 문제가 아니다. 실가중치·실이미지
에서는 안 난다. **이걸 모르면 항등원 빌더를 뒤지느라 하루를 쓴다.**

---

## 4. op 예산에 모델 축이 생겼다

| 운영점 | matmul | softmax | LN | 활성함수 | 어텐션 계 |
|---|---:|---:|---:|---:|---:|
| GPT-2, per-bank `T_j` | 48.6% | 37.9% | 10.2% | **3.3%** | **86.5%** |
| GPT-2, 전역 `T=16` | 48.3% | 33.5% | 10.0% | 8.1% | **81.8%** |
| ViT-B/16, 전역 `T=16` | 34.1% | 33.7% | 13.3% | 18.9% | **67.8%** |
| **ViT-M/16, 전역 `T=16`** | 33.8% | 36.3% | 18.9% | **11.0%** | **70.1%** |

**"어텐션이 예산의 86.5%" 는 GPT-2 의 긴 시퀀스(1024) 이야기다.** ViT 는 seq 197–260
이라 `S²` 항이 작고, 어텐션이 68–70% 로 내려간다. **융합(§3)의 값어치도 모델 의존이다.**

⚠️ **줄인 epochs 로 이 표를 만들지 말 것** — ViT-B 스모크(epochs 50)는 LN 27.9% ·
어텐션 55.9% 로 본 런과 크게 달랐다.

---

## 5. 📌 저장은 모델 **폭**에 무관하다 — 깊이에만 비례한다

| 모델 | depth | width | primitives | params | bytes |
|---|---:|---:|---:|---:|---:|
| GPT-2-medium | 24 | 1024 | 339 | 13,472 | 53,888 |
| ViT-B/16 | 12 | 768 | 171 | 6,500 | 26,000 |
| **ViT-M/16** | 12 | **512** | **171** | **6,404** | **25,616** |

ViT-M 은 폭이 ViT-B 의 **2/3** 인데 저장은 **1.5%** 만 작고 프리미티브 수는 같다.
GPT-2 대비 비율(2.10×)은 **깊이 비 24/12 = 2** 와 맞는다.

> **변환의 저장 비용은 `O(사이트 수)` 이고 사이트는 깊이에 비례한다 — 폭이 아니라.**
> *"파라미터를 추가하는 방법 아니냐"* 는 반론에 대해 **모델이 넓어질수록 상대
> 오버헤드가 줄어든다**는 스케일링 논거가 된다. E5(파라미터 축 3중 2 패)를 부분적으로
> 상쇄하는 방향이다.

---

## 6. 📌 스모크는 결과의 예고편이 아니다 — 두 번 확인됐다

| | ViT-B/16 | ViT-M/16 |
|---|---|---|
| epochs 50 (스모크) | −0.96% | −0.46% |
| **epochs 300 (본 런)** | **−0.020%** (48×) | **−0.0024%** (192×) |
| 저장 | **완전 동일** | **완전 동일** |

**적합 epochs 를 6× 늘리면 변환 손실이 50–200× 좋아진다.** 나는 ViT-B 스모크의
−0.96% 를 근거로 *"질 수도 있다"* 고 적었는데, **그 판단의 근거 자체가 인용해선 안
되는 종류의 숫자**였다.

**반대로 저장은 두 빌드가 바이트 동일하다** — 예산 규칙 출력이 **적합 epochs 의 함수가
아니다.** E12(캘리브레이션 draw 불변) · E3(코퍼스 불변)에 이어 **세 번째·네 번째 축**
에서 같은 불변성이 확인됐다.

---

## 7. 남은 것 / 한계

* **CNN 행(VGG16 / ResNet34)은 안 했다.** timm 배선은 됐지만 CNN 은 어텐션이 없어
  변환 범위가 다르다 — 별도 확인 필요.
* **절대 top-1 격차 0.8–1.0 pp 의 원인을 규명하지 않았다**(§2).
* **각 셀이 한 점이다** — 다른 해상도·다른 체크포인트에서의 거동은 모른다.
* **전처리가 단일 스레드**라 평가가 데이터에 묶인다(배치당 ≈ 5.9 s 가 디코딩).
  ViT-B 본 런 11h19m, ViT-M 11h13m 중 상당 부분이 그것이다.
* **`<unk>` 같은 다른 축의 함정은 없다** — 이미지 파이프라인은 GPT-2 의 토크나이저
  문제에 대응하는 것이 없다.

---

## 8. 재현

```bash
# 조달 (라이선스 수락은 계정 행위). ⚠ load_dataset 을 쓰지 말 것 --
# split 인자와 무관하게 레포 전체 155 GiB 를 받고, arrow 재인코딩이 디스크를 채운다.
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('imagenet-1k', repo_type='dataset', \
                    allow_patterns=['data/validation-*'])"

# 게이트 -- 공개값 근처가 안 나오면 그 뒤는 의미 없다
python experiments/vit_imagenet.py --backend none --limit 2000 --batch-size 64

# ViT-B/16 (HF)
python experiments/vit_imagenet.py --backend mbe_pasn --convert-ops all \
    --model google/vit-base-patch16-224 --epochs 300 --batch-size 32 \
    --pasn-t-fixed 16 --pasn-id-target relative --pasn-id-target-rel 1e-2 \
    --paper-row ViT-B/16 --json results/e8_vit_imagenet.json --tag e8-vitb16-t16

# ViT-M/16 (timm -- 논문의 그 모델)
python experiments/vit_imagenet.py --framework timm \
    --model vit_medium_patch16_reg4_gap_256 --backend mbe_pasn --convert-ops all \
    --epochs 300 --batch-size 32 --pasn-t-fixed 16 \
    --pasn-id-target relative --pasn-id-target-rel 1e-2 \
    --paper-row ViT-M/16 --json results/e8m_vit_medium.json --tag e8m-vitm16-t16
```

**GPU 실측**: ViT-B **11h19m**, ViT-M **11h13m** (각각 ANN ~2.5–3 h + 빌드 45.8분 +
SNN ~8 h). 코드: `experiments/vit_imagenet.py`, `src/mbe/timm_convert.py`.
결과: `results/e8_vit_imagenet.json`, `results/e8m_vit_medium.json`,
`results/vit_gate.json`(체크포인트 스캔), `results/vit_smoke.json`,
`results/vit_timm_smoke.json`. 테스트 **37 + 9 + 4**.

---

관련: `results/RESULTS_2026-08_cycle.md`(사이클 전체) · `PAPER_PLAN.md` §E8 ·
`RELATED_WORK.md`(3모달리티) · `PASN_vault/60 - 연구일지/E8 - ViT x ImageNet (CV 축).md`.
