# MBE 뉴런 (Training-Free ANN-to-SNN, AAAI-26) 재구현 계획

이 문서는 코드 구현 전 단계의 분석/계획서입니다. 목표는 논문 "Training-Free ANN-to-SNN
Conversion for High-Performance Spiking Transformers" (MBE neuron)의 Table 1~4, 6, 9~11 및
Fig. 6, 8을 최대한 동일한 방법으로 재현하는 것이며, 이후 PASN 확장의 baseline이 됩니다.

---

## 0. vast.ai SSH 직접 사용 가능 여부 (결론부터)

**직접 사용 불가능합니다.** 확인해본 결과 이 세션의 shell 샌드박스는 임의의 외부 호스트로
아웃바운드 소켓 연결이 막혀 있습니다 (`curl vast.ai` → 응답 없음, 임의 포트로의 TCP 연결
→ `Network is unreachable`). `ssh` 클라이언트 자체는 설치되어 있지만, vast.ai 인스턴스가
쓰는 임의의 IP:포트로 나가는 연결이 허용되지 않아 SSH 세션을 열 수 없습니다. 즉 저는
vast.ai 터미널을 원격으로 직접 조작할 수 없습니다.

**대안 워크플로우 (권장):**
1. 제가 이 프로젝트 폴더(`PASN`)에 학습/평가 코드, 실행 스크립트, requirements를 전부
   작성합니다.
2. 사용자가 vast.ai 인스턴스를 대여하고, `git`/`scp`/`rsync`로 코드를 인스턴스에 올리거나,
   vast.ai의 Jupyter/웹 터미널에서 `git clone`으로 받습니다 (프로젝트를 GitHub/사설 저장소에
   올려두면 가장 편함).
3. 사용자가 vast.ai 터미널에서 정해진 명령을 실행하고, 로그/에러/결과 파일을 이 폴더로
   복사해오면 제가 그걸 읽고 디버깅·분석·다음 단계를 이어갑니다.
4. 실험 결과(체크포인트 요약, 로그, CSV/JSON 결과)를 다시 로컬 프로젝트 폴더에 동기화하면
   제가 표/그래프로 정리합니다.

즉 저는 "코드 작성 + 실행 명령/스크립트 준비 + 결과 분석"을 담당하고, "vast.ai 상에서의 실제
명령 실행"은 사용자가 수행하는 반복 루프가 됩니다. (참고: 이 샌드박스 자체에는 GPU가 없고,
설령 네트워크가 열려도 로컬 GPU 연산은 불가능합니다 — 어차피 실제 학습/평가는 vast.ai에서
돌아가야 합니다.)

---

## 1. 재현 대상 정리

### 1.1 핵심 수식 (구현해야 하는 것)
- MBE 뉴런 파라미터 갱신식 (Eq.4): `Para(τ_n, t) = α · exp(-tΔt/τ_n)`, `τ_n ∈ {τ_d, τ_r, τ_Vth}`
- Basis 동역학 (Eq.5-7): `u[t+1] = u[t] - s[t]r[t]`, `s[t] = H(u[t]-Vth[t])`,
  `o[t+1] = o[t] + s[t]d[t]`
- 뉴런 출력 (Eq.8): `f̂(x) = Σ_n w^(n) · o^(n)(T)`
- FP 곱셈 근사 (Eq.9-12, 22-27): 두 MBE 뉴런의 spike/intensity를 외적하여 D, S 행렬 구성,
  Hadamard product + sum
- Softmax 분해 (Eq.13, Table VIII): `e^x = 2^floor(x·log2e) · 2^frac`, 정수부는 하드웨어
  덧셈, 소수부는 MBE_exp, 역수는 IEEE754 지수/가수 분해 + MBE_inv
- LayerNorm 분해: 제곱합(FP-mult 재사용, D를 1/n로 스케일), 역제곱근은 지수 E의 홀짝 보정
  + MBE_invsqrt
- 학습 파라미터 수: basis 당 5개 (τ_d, τ_r, τ_Vth, Δt, w) → MBE는 `5N`개, FS는 `3T`개
  (Table VI 재현은 단순 산술 검증)

### 1.2 재현해야 하는 표
| 표 | 내용 | 필요 자원 |
|---|---|---|
| Table 1 | ImageNet: ViT-B/16, ViT-M/16, VGG16, ResNet34 정확도 | ImageNet-1k val (5만장), 4개 pretrained ANN |
| Table 2 | NLU: RoBERTa-Base/Large × SST-2/SST-5/MR/Subj | GLUE류 데이터셋 + fine-tuned RoBERTa 체크포인트 (직접 학습 필요 가능성 높음) |
| Table 3 | NLG: GPT-2(345M) × WikiText-2/103 perplexity | GPT-2-medium pretrained (HF 공개), WikiText 데이터셋 |
| Table 4 | T=8/10/12/16 에 따른 정확도/perplexity | 위 모델들 재사용, T만 바꿔 반복 평가 |
| Table 6 | 파라미터 수 비교 (FS 3T vs MBE 5N) | 산술 계산, GPU 불필요 |
| Table 9 | VGG16/ResNet34 A2S 비교 | Table 1과 동일 자원 |
| Table 10 | N=1,2,4,6,8 (±decay) 에서 GELU/invsqrt/inv/2^x MSE | 1차원 함수 피팅만 필요, **GPU 거의 불필요** |
| Table 11 / Fig.6b | 발화율 기반 에너지 추정 | ViT-M/16 실제 추론 중 firing rate 측정 필요 |
| Fig. 8, Table VII | FS vs MBE 함수 근사 비교 (SiLU, 1/x, e^x, GELU 등) | 1차원 함수 피팅만 필요, **GPU 거의 불필요** |

---

## 2. 구현 아키텍처

### 2.1 MBE 뉴런 코어 모듈
- `MBEBasis`: 단일 basis의 (τ_d, τ_r, τ_Vth, Δt, w) 파라미터와 T-step 시뮬레이션 (Eq.4-7)
  을 PyTorch `nn.Module`로 구현. Heaviside `H(·)`는 순전파에서 이진 spike, 역전파는
  surrogate gradient 필요 (논문은 surrogate 함수의 정확한 형태를 명시하지 않음 — **오픈
  이슈, 아래 5절 참조**).
- `MBENeuron`: N개의 `MBEBasis`를 병렬로 갖고 Eq.8로 합산.
- 벡터화: T×N을 텐서 연산으로 처리해 임의의 활성화 텐서(배치×채널×시퀀스)에 브로드캐스트
  적용 가능하게 설계 (Transformer 전체에 적용해야 하므로 element-wise 적용이 빨라야 함).

### 2.2 프리미티브 함수별 MBE 인스턴스 (Appendix G.1 하이퍼파라미터)
| 프리미티브 | 용도 | N | T | 정의역 | 샘플 수 M |
|---|---|---|---|---|---|
| MBE_Id | FP 곱셈용 항등함수 | 8 (파라미터 고정, "[20]" 기재 — 의미 확인 필요) | 16 | 활성화 실측 범위 (calibration) | 10,000 |
| MBE_GELU / MBE_Tanh | 활성화 함수 | 4 | 16 | GELU: (-120, 10) | 10,000 |
| MBE_exp (2^x) | Softmax 지수부 | 8 | 16 | [0, 1] | 10,000 |
| MBE_inv (1/x) | Softmax 역수 | 8 | 16 | [0.5, 1] | 10,000 |
| MBE_invsqrt (1/√x) | LayerNorm 역제곱근 | 8 | 16 | [0.5, 2] | 10,000 |

학습: lr=0.01, 200 epoch, "decay rate 0.99의 exponential optimizer" (Adam+ExponentialLR로
해석, 아래 5절에서 검증 필요).

### 2.3 FP 곱셈 spike 근사
`MBE_Id`로 두 피연산자를 spike-train화 → intensity 외적 D, spike 외적 S → `Σ(D⊙S)`.
D는 파라미터만으로 결정되므로 사전 계산/캐싱 가능 (Appendix F.1).

### 2.4 Softmax / LayerNorm 조립 (Algorithm 1, Table VIII)
- Softmax: `e^x` (정수부 비트시프트 + MBE_exp 소수부) → 합산 → IEEE754 분해 후 MBE_inv →
  FP-mult 조합으로 최종 출력.
- LayerNorm: 평균 계산(비-spike, 텐서 reduce) → 편차 제곱합(FP-mult 재사용) → 분산의
  지수/가수 분해(홀짝 보정) → MBE_invsqrt → FP-mult로 정규화 → affine(γ,β).

### 2.5 변환 프레임워크 (Algorithm 1)
1. **Calibration**: 미니배치를 pretrained ANN에 통과시켜 각 비선형 연산의 입력 실측 범위
   기록 (LayerNorm 입력, GELU 입력, attention score 등 — Table 4 설명에 나온 "ViT-M/16
   identity mapping range [0,62]"가 이 단계의 산출물로 추정됨).
2. **Build**: 위 범위로 프리미티브 MBE 뉴런들을 학습(오프라인, ANN 가중치와 무관).
3. **Replace**: 모델의 각 모듈(행렬곱, 활성화함수, Softmax, LayerNorm)을 spiking 버전으로
   교체. ANN 가중치는 전혀 수정하지 않음 (training-free의 핵심).
4. **Evaluate**: 변환된 SNN을 표준 평가 파이프라인(top-1 acc / perplexity)으로 측정.

---

## 3. 모델·데이터셋별 준비 계획

### 3.1 CV (Table 1, 9)
- ViT-B/16, ViT-M/16(Reg4-Gap-256): `timm`에서 ImageNet pretrained 가중치 로드 가능.
- VGG16, ResNet34: `torchvision.models` pretrained.
- 데이터: ImageNet-1k validation (5만 장). **주의**: 공식 배포는 라이선스 등록 및 대용량
  다운로드(devkit+val ≈ 6.3GB)가 필요 — 사용자 계정으로 직접 다운로드해야 하며, 제가
  자동으로 받을 수 없음 (인증 필요). Kaggle/HuggingFace 미러 등 대체 경로도 확인 필요.

### 3.2 NLU (Table 2)
- RoBERTa-Base(125M)/Large(355M): HuggingFace pretrained.
- 데이터셋: SST-2, SST-5, MR, Subj — 표준 벤치마크지만 SST-5/MR/Subj는 GLUE에 없어
  별도 소스 필요(예: 기존 SpikeBERT/SpikeZIP-TF repo가 쓰는 데이터 스플릿을 그대로 따라야
  숫자가 맞음).
- **가장 큰 리스크**: 논문은 "adapt RoBERTa"라고만 하고 fine-tuning 레시피를 안 줌 → 우리가
  직접 4개 태스크에 대해 RoBERTa-Base/Large를 fine-tuning해서 source ANN을 만들어야 함.
  ANN 정확도(Table 2의 "ANN" 행: 94.49/96.22 등)가 논문 수치와 비슷하게 나오는지가 1차
  검증 포인트.

### 3.3 NLG (Table 3)
- GPT-2 (Param=346M → GPT-2-medium/large 계열 중 345M인 GPT-2-medium으로 추정, 확인 필요):
  HuggingFace에서 바로 로드 가능.
- WikiText-2, WikiText-103: HuggingFace `datasets`로 바로 확보 가능 (가장 준비가 쉬움).

---

## 4. 검증 전략 (GPU 예산을 아끼기 위한 순서)

값비싼 전체 파이프라인을 돌리기 전에, **GPU가 거의 필요 없는 저비용 검증부터** 논문 수치와
맞춰봅니다:

1. **함수 피팅 단위 검증** (CPU로 가능, vast.ai 불필요): Table 10 (N=1,2,4,6,8, decay on/off,
   GELU/invsqrt/inv/2^x MSE), Table VII/Fig.8 (SiLU 등 FS vs MBE 비교), Table V (FS 실패
   재현). 이게 맞아야 MBE 코어 구현이 논문과 동일하다는 확신을 가질 수 있음. **가장 먼저
   할 일.**
2. **소규모 변환 sanity check**: 작은 사전학습 ViT나 RoBERTa-base 한 개 태스크에 대해
   변환 파이프라인을 T=16으로 돌려 ANN 대비 낙폭이 논문 수준(<1%)인지 확인 (vast.ai 저사양
   GPU 몇 시간이면 충분).
3. **Table 11 에너지 추정**: 3번 결과에서 firing rate만 뽑으면 계산 가능 (추가 학습 불필요).
4. **전체 표 재현**: 검증된 파이프라인으로 Table 1/2/3/4/9 전체 모델·데이터셋 조합 실행
   (가장 GPU 시간이 많이 드는 단계, vast.ai 다중/장시간 대여 필요).

---

## 5. 논문에 명시되지 않아 구현 시 직접 결정해야 하는 사항 (오픈 이슈)

이 항목들은 "논문에 없어서 추측으로 채워야 하는 부분"입니다. 임의로 정하고 넘어가지 않고,
각 항목을 구현 착수 시점에 사용자와 함께 확정하거나, 여러 후보로 실험해 논문 표 10/VII의
기존 수치와 가장 가깝게 맞는 쪽을 선택하는 방식으로 진행하는 게 안전합니다.

- **Surrogate gradient 형태**: Heaviside `H`의 역전파 함수가 본문에 없음 (Appendix E는
  파라미터 개수만 다룸). SNN 표준 surrogate(예: sigmoid 미분, arctan, triangular) 중 선택
  필요.
- **Optimizer 세부**: "학습률 0.01, 200 epoch, decay rate 0.99의 exponential optimizer"
  문구 — Adam/SGD 여부, ExponentialLR(γ=0.99)로 해석할지 확인 필요.
- **τ, Δt, w 초기화**: FS의 binary init(`2^(T-t)`)에 대응하는 MBE 초기화가 명시적으로 안 나옴
  ("[20]으로 고정"이라는 문구는 각주/부록 번호 인용으로 추정, 원문 확인 필요).
- **RoBERTa fine-tuning 레시피** (Table 2 ANN 베이스라인 재현용): 논문은 You et al. 2024 /
  Zhu et al. 2023 설정을 따른다고만 언급 — 해당 두 논문(SpikeZIP-TF, SpikeGPT)의 실험
  세팅을 별도로 조사해야 함.
- **Calibration 미니배치 크기/샘플링 방법** (Algorithm 1 Step 1): "Sample minibatch data
  from D"라고만 되어 있고 배치 크기 불명.
- **GPT-2 정확한 변종**: Param=346M이 GPT-2-medium(345M)인지 다른 커스텀 설정인지 확인
  필요.

---

## 6. 단계별 실행 계획 (Phase)

| Phase | 내용 | GPU 필요 여부 |
|---|---|---|
| 0 | 개발 환경 구성 (requirements.txt, Docker 이미지, vast.ai 인스턴스 셋업 스크립트) | 불필요 (준비만) |
| 1 | MBE 뉴런 코어 + surrogate gradient 구현, 단위 테스트 | 불필요 (CPU) |
| 2 | 프리미티브 함수 피팅 (Id/GELU/Tanh/exp/inv/invsqrt) → Table 10/VII/V/Fig.8 재현·비교 | 불필요~저사양 |
| 3 | FP-mult, Softmax, LayerNorm spike 조립 + 단위 정확성 검증 | 불필요~저사양 |
| 4 | 변환 프레임워크(Algorithm 1) 구현 + 소규모 sanity check (RoBERTa-base 1개 태스크 등) | vast.ai 저사양 GPU |
| 5 | 전체 벤치마크 (ImageNet 4모델, NLU 8조합, NLG 2데이터셋) × T={8,10,12,16} | vast.ai 중~고사양 GPU, 다수 실행 |
| 6 | 에너지 추정(Table 11), 최종 표/그래프 정리, 논문 수치와 diff 리포트 | 불필요 |

Phase 5가 가장 비용이 큼 (ImageNet 5만장 × 여러 모델 × 여러 T, RoBERTa fine-tuning 포함).
vast.ai 예산에 맞춰 Phase 5의 실행 순서(어떤 표부터 재현할지)를 사용자와 우선순위 조율하는
것을 권장합니다.

---

## 7. 다음 단계 제안

계획에 동의하시면:
1. Phase 0-1 (환경/코어 모듈)부터 이 세션에서 코드 작성 시작 (CPU만으로 검증 가능한 부분).
2. Phase 2 결과(Table 10/VII 재현치)를 논문과 비교해 코어 구현 정합성 확인.
3. vast.ai 인스턴스 준비되면, 이 프로젝트 폴더를 git 저장소로 만들어 동기화 방식 확정
   (예: GitHub private repo, 또는 사용자가 매번 폴더를 zip으로 옮기는 방식) 후 Phase 3~4 진행.
