# CLAUDE.md — PASN

Training-free ANN→SNN 변환 뉴런 **PASN**. IEEE-754 지수 비트 라우터가 입력을 binade
구간으로 나누고, 각 구간이 자기 multi-basis 스파이킹 뱅크와 예산 `(N_j, T_j)`를 갖는다.
방법 명세는 [`PASN_method.md`](PASN_method.md).

**베이스라인은 MBE 논문의 published 표다** — 우리 재현이 아니다(결정 2026-08-01).

## 프리즈 빌드 — 수치 인용 전에 확인

`freeze-t16-unif` / [`results/freeze_e1.json`](results/freeze_e1.json) ·
`pasn_beta={"inv":0.5}` · **53,888 B / 13,472 p / 339 prims** ·
ΔPPL **−0.19%** @ stride 1024 · 운영점 **T=8**

`−0.14%` · `49,952 B` · `12,488 p`는 **`--no-pasn-beta` 빌드 전용 과거 수치다.**
그 빌드를 인용할 땐 반드시 빌드 이름을 붙인다.

```bash
python research/tools/check_build.py <레코드>.json   # 인용 전 기계 대조
```

## 환경

```bash
C:/Users/cm120/miniconda3/envs/SNN/python.exe    # py3.10, torch 2.11 CPU
```

기본 python(3.14)이 아니다. GPU 런은 vast.ai — **반드시 tmux/supervisor 아래에서.**
SSH nohup으로 띄우면 드라이버가 죽어도 큐가 조용히 멈춘다.

## 더 읽을 것 (필요할 때만 — 크다)

| 무엇 | 어디 |
|---|---|
| 도구 사용법 | [`docs/HOWTO.md`](docs/HOWTO.md) |
| 콜드스타트 브리핑 | [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md) (8k 토큰) |
| 실험 우선순위 · 계측 규약 · 하지 말 것 | [`PAPER_PLAN.md`](PAPER_PLAN.md) §2·§4·§5 (28k 토큰) |
| 레코드 스키마 | [`research/RECORD_SCHEMA.md`](research/RECORD_SCHEMA.md) |

## 하지 말 것

- **`PASN_vault/`를 쓰거나 지우지 않는다.** Obsidian 기록 공간, git 밖. 읽기만.
- **`results/`·`logs/`의 기존 레코드를 수정하지 않는다.** 새 레코드는 새 파일로.
- **런 없이 수치를 추정해 적지 않는다.** 없으면 `N/A (레코드 없음)`.
- **한 운영점에서 잰 값을 다른 운영점에 인용하지 않는다.** arm과 T를 항상 명시.
- **실패한 런도 레코드를 남긴다** (`status: failed`).
- **결과 JSON은 원격 런이 끝날 때마다 커밋한다.** 닫힌 vast.ai 박스가 Block C와
  Stage 2 원본을 통째로 가져간 적이 있다.
- **승인 없이 `PAPER_PLAN.md`·`NEXT_EXPERIMENTS.md`를 재작성하지 않는다.**
