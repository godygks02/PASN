# 레코드 스키마

`results/*.json`의 계약. 이게 흔들리면 검증이 매번 다른 걸 읽게 되고 자동화가 무너진다.

**기존 레코드는 이미 이 형태다.** 이 문서는 새 규약을 도입하는 게 아니라, 이미 지켜지고
있던 것을 명시해서 기계가 검사할 수 있게 만든 것이다.

## 형태

레코드 파일은 **레코드 객체의 리스트**다(단일 객체도 허용). 한 스윕의 각 arm이 한 원소다.

```json
[ { "tag": "freeze-t16-unif", "...": "..." } ]
```

## 필수 — 출처 필드

이게 없으면 그 수치는 **인용할 수 없다.** `check_build.py`가 검사한다.

| 필드 | 뜻 |
|---|---|
| `tag` | 이 arm의 이름. 레코드 안에서 유일해야 한다 |
| `backend` | `mbe_pasn` / `mbe` / `none`(ANN 기준런) 등 |
| `model` | `gpt2-medium`, `vit_medium_...` — 지문 대조 범위를 정한다 |
| `convert_cfg` | **`ConvertConfig` 37필드 전부.** 빌드를 소급 판정하는 유일한 근거 |
| `pasn_beta` | `{"inv":0.5}`(프리즈) 또는 `{}`(no-beta) |
| `stored_bytes` / `stored_params` / `n_primitives` | 빌드 지문 |

> **왜 `convert_cfg`가 전부여야 하나**: `pasn_beta=0.5` 기본값이 모든 GPT-2 레코드보다
> **뒤에** 들어와서, 헤드라인 레코드가 어느 빌드인지 소급 판정할 수 없게 된 적이 있다.
> E0(빌드 프리즈) 전체가 그걸 복구하느라 생겼다. 08-09 이후 레코드는 37필드를 다 싣는다.

## 결과 필드

정확도 축과 비용 축을 **같이** 적는다. 한쪽만 있으면 등정확도 비교를 할 수 없다.

| 필드 | 뜻 |
|---|---|
| `ppl_ann` / `ppl_snn` / `delta_pct` | 정확도. `delta_pct`가 헤드라인 축 |
| `spikes_per_token` / `total_spikes` / `by_kind` | 스파이크. `by_kind`는 op별 분해 |
| `ops_per_input` / `energy_pj_per_input` | 에너지 (G.4 통화) |
| `stored_by_role` / `stored_shared_factor` | 메모리 분해 |
| `n_steps` / `ctx` / `stride` / `eval_mode` | **운영점.** 이게 없으면 수치가 무의미 |
| `started` / `build_s` / `eval_snn_s` | 언제, 얼마나 걸렸나 |

## 실패한 런

**실패도 레코드를 남긴다.** 실패가 안 남으면 탐색 공간이 왜곡되고, 같은 벽에 두 번 부딪친다.

```json
{ "tag": "e15-attempt-1", "status": "failed",
  "error": "CUDA OOM at block 512", "convert_cfg": { }, "started": "..." }
```

## 검사

```bash
python research/tools/check_build.py                     # results/ 전체
python research/tools/check_build.py results/새레코드.json
python research/tools/check_build.py --strict            # 프리즈 아닌 것도 실패로
```

판정은 넷 중 하나다:

| 표시 | 뜻 |
|---|---|
| `OK` | 프리즈 빌드. 그대로 인용 가능 |
| `NAME` | 다른 빌드(no-beta). **인용 시 빌드 이름을 반드시 붙인다** |
| `??` | 지문이 어느 빌드와도 안 맞는 변종. 헤드라인으로 쓸 수 없다 |
| `--` | 대조 대상 아님 (다른 모델이거나 ANN 기준런) |

## 인용 전 마지막 관문

수치를 발표·원고에 올리기 전에 교차모델 검증을 통과시킨다:

```bash
python research/tools/verify.py --claim "<주장 한 문장>" results/<레코드>.json
```

Codex가 레코드 원문만 보고 판정한다(`supported` / `refuted` / `unverifiable`).
빌드가 섞인 주장은 여기서 걸린다 — 실제로 `−0.14%`와 `53,888 B`를 한 문장에 넣으면
`refuted`가 나온다.
