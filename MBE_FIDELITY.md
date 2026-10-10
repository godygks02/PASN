# MBE 재현 충실도 감사 — "우리 MBE로 논문 MBE를 대표할 수 있는가"

작성 2026-08-15. 코드와 논문 원문(`paper/extracted.txt` Eq. 4–8, Appendix G.1) 대조.
이 문서는 **논문 작성 시 우리 MBE 재현을 어디까지 근거로 쓸 수 있는지**를 정한다.
E15(MBE 백엔드 네트워크 층위 실행, `FEEDBACK_2026-08-14.md` §5)의 선행 문서다.

관련: [`MBE_Implementation_Notes.md`](MBE_Implementation_Notes.md) (구현 경위) ·
[`MBE_RESULTS.md`](MBE_RESULTS.md) (수치) · [`PAPER_PLAN.md`](PAPER_PLAN.md) §2 E2 (재현 간극)

---

## 0. 판정 — 층위마다 답이 다르다

| 층위 | 「논문과 같다」고 주장 가능한가 | 근거 |
|---|---|---|
| **동역학** (Eq. 4–8) | ✅ **가능** — 축자 이식 | §1 |
| **뉴런** (함수 근사) | ⚠️ **"같다"가 아니라 "논문보다 강하다"** | §2. 논문 미명시 7곳을 우리가 채웠고 방향이 전부 MBE에 유리. Table X 4/4 재현·능가 |
| **하이퍼파라미터** (G.1) | ⚠️ **4곳 어긋남 — 고칠 수 있다** | §3 |
| **네트워크** | ❌ **불가** | §4. 우리 MBE는 발산(ppl 611–589287, ANN 21.71), 논문은 near-lossless |

**한 줄**: *"우리가 baseline을 약하게 만들어서 이겼다"는 반론은 함수·op 층위에서 방어된다.
그러나 "우리 MBE = 논문 MBE"는 네트워크 층위에서 주장할 수 없다.* 두 층위의 어긋남
**방향이 서로 반대**여서 하나의 서술로 접히지 않는 것이 이 감사의 핵심 결과다.

---

## 1. 축자 일치 — 여기는 깨끗하다

| 논문 | 구현 | |
|---|---|---|
| Eq. 4 `α·exp(−tΔt/τ_n)`, `τ_n ∈ {τ_d, τ_r, τ_Vth}` | [`neuron.py:179-191`](src/mbe/neuron.py#L179) | ✅ |
| Eq. 5–7 (`u` 갱신 / Heaviside / `o` 누적), `u_n[0] = x` | [`neuron.py:216-219`](src/mbe/neuron.py#L216) | ✅ |
| Eq. 8 선형 readout `Σ w_n·o_n(T)` | [`neuron.py:236`](src/mbe/neuron.py#L236), `mbe_readout_order=1` 기본 | ✅ |
| Eq. 11 사전계산 intensity 행렬 `D = w_n·d_n[t]` | [`neuron.py:283-298`](src/mbe/neuron.py#L283) | ✅ |
| Appendix E.1 초기값(`d[0], r[0], Vth[0]`을 `α, Δt, τ`로 결정) | `_kernels()` 가 `t=0`에서 `r=vth=α_v` | ✅ |
| G.1 `T = 16` · 항등원·`2ˣ`·`1/x`·`1/√x` **N = 8** | [`convert.py:332-335`](src/mbe/convert.py#L332) | ✅ |
| Table VI `5N` 파라미터 폼 | `learn_alpha_v=False, use_bias=False` 로 복원, 유닛테스트 고정 [`test_mbe_neuron.py:33-44`](tests/test_mbe_neuron.py#L33) | ✅ 뉴런 층위만 (§3 참조) |
| decay 제거 ablation | `MBEConfig.decay=False` | ✅ |

**Table X 재현** (N=8, 우리 / 논문):

| 함수 | 우리 | 논문 |
|---|---:|---:|
| GELU (signed) | **1.7e-4** | 1.0e-4 |
| `1/√x` | **5.0e-6** | 4.9e-5 |
| `1/x` | **4.6e-6** | 4.4e-5 |
| `2ˣ` | **6.5e-5** | 5.3e-5 |

`N=1` 바닥도 일치(`1/√x` 1.3e-3 vs 논문 1.4e-3)하므로 **N 스케일링까지 재현**됐고,
decay 제거 시 1–2 자릿수 악화라는 논문 주장도 재현된다.

---

## 2. 논문이 정하지 않은 것을 우리가 채웠다 — 7곳, 방향은 전부 MBE에 유리

| # | 논문 | 우리 구현 | 이게 없으면 |
|---|---|---|---|
| 1 | `α` = **단일 하이퍼파라미터** (원문: 타깃 함수의 최대값으로 두는 것이 보통) | **basis별 · 학습 가능 · 곡률 배치** `alpha_v_n` ([`neuron.py:114-132`](src/mbe/neuron.py#L114)) | **multi-basis가 붕괴한다.** 공유 α는 basis를 nested 이진코드로 만들어 특징행렬이 특이해지고, N=8이 N=1과 같은 ~1e-3에서 멈춘다 |
| 2 | 출력 bias 없음 (Eq. 8) | 학습 DC offset 1개 | `f(x_min) ≠ 0`인 `1/x`·`2ˣ`의 baseline 표현 불가 |
| 3 | 최적화 = lr 0.01 / 200 epoch / decay 0.99 **만** 명시 | **닫힌형 `(w, bias)` 해** (ridge normal equations, [`fit.py:36-74`](src/mbe/fit.py#L36)) | 전 함수 MSE 약 1자릿수 악화 |
| 4 | surrogate 미명시 | ATan + **LBFGS sharpening** ([`fit.py:158-187`](src/mbe/fit.py#L158)) | ~1e-4 바닥 도달 불가 |
| 5 | 부호 처리 미명시 | **극성 분할 signed 뉴런** ([`signed.py`](src/mbe/signed.py)) | GELU 표현 상한 **0.5** — 사실상 표현 불가. 변환하면 발산 |
| 6 | 입력 스케일 (암묵) | `[0,1]` 아핀 정규화 | GELU 130 스케일에서 LBFGS overflow |
| 7 | — | ATan surrogate 기본 | 역전파 불가 |

**1번과 5번이 결정적이다.** 이 둘 없이는 논문 Table X 자체가 재현되지 않는다.
그래서 **"논문 수식만으로는 논문 수치가 나오지 않는다"**가 우리가 가진 가장 단단한 사실이고,
논문의 재현성 관련 서술로 그대로 쓸 수 있다.

### 대가 — 파라미터 축은 반대 방향으로 움직인다

확장 뉴런은 `5N` 이 아니라 **`6N+1`**(signed 는 **`12N+1`**)이다.
[`mbe_param_model.py:67,98`](experiments/mbe_param_model.py#L67)이 MBE 쪽을 **우리 확장 폼으로**
세고 있으므로, E5의 `MBE 5,929 vs PASN 13,472 = 0.44×`는 **MBE에 관대한** 계산이다.
논문 원문 `5N`으로 세면 우리가 더 크게 진다. §1.2 표에 각주 필수.

### 논문 자체의 모호성 하나 (limitation 절 근거)

`u[0] = x` 이므로 임계값은 **입력 단위**인데, `α`는 **출력(타깃 함수) 최대값**으로 두라고
한다. 두 스케일이 다르면 이 규정은 성립하지 않는다. 우리 `[0,1]` 정규화는 그 모호성의
한 해석이다 — **재현자가 임의로 정할 수밖에 없는 자리**이고, 그렇게 적어야 한다.

---

## 3. 실제로 G.1과 어긋난 것 — E15 전에 고쳐야 한다

| 항목 | 논문 G.1 | 현재 코드 | 위치 |
|---|---|---|---|
| GELU/Tanh basis 수 | **N = 4** | **N = 6** | [`convert.py:331`](src/mbe/convert.py#L331) |
| 학습률 / epoch / decay | 0.01 / 200 / 0.99 | **0.03 / 1000 / 0.999** + LBFGS | [`fit.py:95-100`](src/mbe/fit.py#L95) (변환 경로는 `epochs=300`) |
| 캘리브레이션 표본 | **M = 10,000** | **4,000** | [`spiking_ops.py:78`](src/mbe/spiking_ops.py#L78), [`:102`](src/mbe/spiking_ops.py#L102) — `functions.sample` 기본값은 10,000인데 변환 경로가 덮는다 |
| GELU 정의역 | 고정 `(−120, 10)` | 실측 범위 ± margin | [`spiking_ops.py:268-271`](src/mbe/spiking_ops.py#L268) |
| **`5N` 폼 · unsigned 폼** | 논문의 뉴런 | **변환 경로에서 도달 불가** — `ConvertConfig`에 `learn_alpha_v`/`use_bias`/`signed` 손잡이가 없다 | [`convert.py:330-360`](src/mbe/convert.py#L330) |

마지막 줄이 핵심이다. **지금 `--backend mbe`로 돌려서 나오는 것은 논문 뉴런이 아니라 우리
확장 뉴런이다.** E15 계획서는 "설정은 논문 G.1 그대로"를 전제하는데, **그 arm은 현재 빌드
자체가 불가능하다.**

---

## 4. 네트워크 층위 — 여기서 대표성이 깨진다

* **E2** (`results/e2_mbe_profile.json`, `e2_global_fit_sweep{,_e3000}.json`):
  같은 정의역에서 전역 MBE는 **6e-2 – 5e-1**, PASN은 **5e-3** (10–55×). 24층 변환은
  **모든 용량 설정에서 발산**(ppl 611–589287, ANN 21.71). 원인은 표현력이 아니라
  **적합 탐색의 실패** — `N ∈ {1,2,4,6,8,12}` 스윕에서 여섯 지표 조합 전부 비단조이고
  (`ln_id`는 N=2에서 137× 악화), epochs 10×로도 안 고쳐진다.
* **op 층위**(`op_pareto_merged.json`): 우리 MBE는 **`T=8`에서 무너지지 않는다.**
  `gelu`·`exp2`·`inv`는 `T=16`과 사실상 동일하고, 무너지는 건 `attention`(5×)과
  `fp_multiply`(20×)뿐. **논문 Table 4의 네트워크 붕괴를 우리 재현이 뒷받침하지 않는다.**

⇒ 우리 MBE는 논문 MBE의 **상한 대리도, 하한 대리도 아니다.** 함수 층위에선 논문보다 좋고
네트워크 층위에선 논문보다 훨씬 나쁘다.

---

## 5. 논문에 쓸 수 있는 문장 / 못 쓰는 문장

**쓸 수 있다**

* "operator·function 층위 비교는 code-vs-code다" — `op_pareto_merged.json`
  (4 arm, control 45 빌드 bit-identical). 그리고 **우리 MBE가 Table X를 4/4 재현·능가**하므로
  *baseline을 약화시켜 이겼다*는 반론이 수치로 방어된다. 랩미팅 피드백 1에 대한 가장 강한 답.
* "논문 수식(Eq. 4–8)만으로는 논문 수치에 도달하지 않는다. per-basis `α`와 극성 분할이
  필요하며 둘 다 논문에 없다" — 재현 노력의 증거이자 그 자체로 기여.
* 네트워크 층위 baseline은 **논문 published 표**를 쓰고, 재현 격차를 각주로 정직 보고.

**못 쓴다**

* "우리 코드로 MBE의 `T=8` 붕괴를 재현했다" — 현재 우리 op 런이 반증한다.
* "우리 MBE = 논문 MBE."
* 우리 MBE의 발산을 근거로 한 메커니즘 주장 (C9 — 이미 삭제됨, PAPER_PLAN §2 E2).

---

## 6. E15 착수 전 필수 작업 3개

1. **`ConvertConfig`에 `mbe_learn_alpha_v` / `mbe_use_bias` / `mbe_signed` 추가.**
   없으면 G.1 축자 arm이 빌드조차 안 된다. 기본값은 현행 유지(프리즈 빌드 무영향),
   37 필드 직렬화에 자동 편입.
2. **E15를 arm 2개로 돌린다.**
   **(a) 논문 축자** — `N=4` · `5N` · unsigned · lr 0.01 / 200 epoch / 0.99 · `M=10,000` ·
   고정 정의역. **(b) 우리 최선** — 현행 확장 + E2가 정한 규칙대로 **스윕 후 최선**.
   (a)만 돌리면 *"구현을 못 했다"*, (b)만 돌리면 *"그건 논문 뉴런이 아니다"* — 양쪽 반론을
   하나로는 못 막는다. **(a)가 발산해도 그것이 결과다**: 논문 미명시 부분이 결과를 지배한다는
   직접 증거가 된다.
3. **모든 MBE 수치를 3중 병기**: 논문-published / 우리-축자 / 우리-최선.
