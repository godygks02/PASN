# CLAUDE.md — PASN

Training-free ANN→SNN 변환 뉴런 **PASN** (Prefix-Adaptive Spiking Neuron) 연구.
IEEE-754 지수 비트 라우터가 입력을 binade 구간으로 나누고, 각 구간이 자기 multi-basis
스파이킹 뱅크와 풀어낸 예산 `(N_j, T_j)`를 갖는다. 방법 명세는 [`PASN_method.md`](PASN_method.md).

**베이스라인은 MBE 논문의 published 표다** — 우리 MBE 재현이 아니다(결정 2026-08-01).
우리 재현은 내부 ablation으로만 남는다.

## 이 파일의 역할

이 파일은 **규칙을 새로 만들지 않는다.** 이 프로젝트의 규칙은 이미 아래에 있고,
충돌하면 항상 아래가 이긴다.

| 무엇 | 어디 |
|---|---|
| **실험 우선순위의 단일 출처** | [`PAPER_PLAN.md`](PAPER_PLAN.md) §2 (E0–E14, 의존 순서) |
| **계측 규약** — 어기면 숫자가 무의미 | [`PAPER_PLAN.md`](PAPER_PLAN.md) §4 |
| **하지 말 것** | [`PAPER_PLAN.md`](PAPER_PLAN.md) §5 + [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md) §4 |
| **콜드스타트 브리핑** (이거 하나로 착수 가능) | [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md) |
| **역할별 쓰기 경계** | [`AGENTS.md`](AGENTS.md) |
| **레코드 스키마 / 프리즈 지문** | [`research/RECORD_SCHEMA.md`](research/RECORD_SCHEMA.md) |
| **연구일지** (Obsidian, git 밖) | `PASN_vault/60 - 연구일지/00 - 연구일지 인덱스.md` |

새 세션은 **`NEXT_EXPERIMENTS.md` → `PAPER_PLAN.md` §2 순으로 읽고 시작한다.**

## 환경

```bash
C:/Users/cm120/miniconda3/envs/SNN/python.exe    # py3.10, torch 2.11 CPU
```

기본 python(3.14)이 아니다. GPU 런은 vast.ai. 원격 스윕은 **반드시 tmux/supervisor
아래에서** 돌린다 — SSH에서 nohup으로 띄우면 드라이버가 죽어도 큐가 조용히 멈춘다.

```bash
python tests/test_mbe_neuron.py          # 단위 테스트 16
python experiments/gpt2_wikitext.py      # 헤드라인 런 (GPU)
python research/tools/check_build.py     # 레코드 지문 대조 ← 수치 인용 전에
```

## 프리즈 빌드 — 인용 전에 반드시 확인

**빌드는 2026-08-10에 프리즈됐고, `pasn_beta={"inv":0.5}` 빌드다.**

| | 프리즈 빌드 |
|---|---|
| tag / 파일 | `freeze-t16-unif` / [`results/freeze_e1.json`](results/freeze_e1.json) |
| bytes / params / prims | **53,888 / 13,472 / 339** |
| ΔPPL @ stride 1024 | **−0.19%** (마진 1.76 pp) |
| 운영점 | **T=8** (T=16은 논문 축 정합용) |

`−0.14%` · `49,952 B` · `12,488 p`는 **`--no-pasn-beta` 빌드에만 속하는 과거 수치다.**
그 빌드 결과를 인용해야 할 때는 반드시 빌드 이름을 붙인다.

기계적 대조: `python research/tools/check_build.py <레코드.json>`

## 이 리포에서 하지 말 것

`PAPER_PLAN.md` §5가 본체다. 에이전트 작업에 한정해 덧붙이면:

- **`PASN_vault/`를 쓰거나 지우지 않는다.** Obsidian 기록 공간이고 git 밖이다. 읽기만.
- **`results/`와 `logs/`의 기존 레코드를 수정하지 않는다.** 새 레코드는 새 파일로.
- **런을 돌리지 않고 수치를 추정해 적지 않는다.** 없으면 `N/A (레코드 없음)`.
- **검증되지 않은 인용을 추가하지 않는다.** DOI 또는 arXiv ID 없으면 "미확인 후보"로 격리.
- **`PAPER_PLAN.md`·`NEXT_EXPERIMENTS.md`를 승인 없이 재작성하지 않는다.** 실험이
  끝나 상태가 바뀌는 것은 append/표시 변경으로.
- **한 운영점에서 잰 값을 다른 운영점에 인용하지 않는다.** arm과 T를 항상 명시.
- **실패한 런도 레코드를 남긴다** (`status: failed`). 실패가 안 남으면 탐색 공간이 왜곡된다.
- **결과 JSON은 원격 런이 끝날 때마다 커밋한다.** 닫힌 vast.ai 박스가 Block C와 Stage 2
  원본을 통째로 가져간 적이 있다.

## 레이아웃

```
src/mbe/          MBE + PASN 패키지 (neuron, signed, pasn, spiking_ops, convert, toy, fit)
experiments/      재현 + 비교 스크립트 (플랫 스크립트, 디렉터리 단위 아님)
tests/            단위 테스트 16
results/          생성된 JSON 레코드 + 마크다운 표 + 그림  ← 수정 금지, 추가만
logs/             원본 실행 로그 (vastai/ 포함)          ← 수정 금지
research/         에이전트 레이어 (스키마 · 가설 스테이징 · 검증 도구 · 판정)
paper/            베이스라인 논문 PDF (gitignored, 재배포 안 함)
PASN_vault/       Obsidian 연구일지 (gitignored, 로컬 전용)  ← 건드리지 않음
```

## 일정

| 언제 | 무엇 |
|---|---|
| ~8/30 | NeurIPS 2026 워크숍 제출 (현재 결과 + E0 + E1 + E10) |
| 1/29/2027 | **ICML 2027 (주 타깃)** |
| 5/16/2027 | (리젝 시) NeurIPS 2027 |

ICLR 2027(9/26)은 권하지 않는다 — CV 미완 + 스케일 격차인 채로 A*에 들어간다.
