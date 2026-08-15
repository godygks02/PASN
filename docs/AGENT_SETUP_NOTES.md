# 에이전트 세팅 — 무엇을 만들었고 왜 되돌렸나

2026-08-14. 원본 지시서는 [`RESEARCH_AGENT_SETUP.md`](RESEARCH_AGENT_SETUP.md)에 그대로 보존.

> ## ⛔ 롤백됨 — 서브에이전트와 스킬은 제거했다 (같은 날)
>
> 에이전트 6개(planner·scout·runner·analyst·verifier·writer)와 스킬 4개를 만들었고,
> 실제로 한 사이클을 돌려봤다. **토큰이 실험 예산을 잡아먹어서 뺐다.**
>
> 원인은 설계 자체에 있었다. 모든 에이전트 정의에 *"`PAPER_PLAN.md` §2를 읽고
> 시작하라"*고 써놨는데 그 파일이 **28k 토큰**이고, `NEXT_EXPERIMENTS.md`가 8k다.
> 에이전트는 콜드 스타트라 매번 처음부터 다시 읽는다:
>
> | | 토큰 |
> |---|---|
> | 에이전트 1개 오리엔테이션 | ~36k |
> | 사이클 한 바퀴 (에이전트 5개) | **~180k** — 실제 작업 전에 |
> | `check_build.py`로 레코드 74건 판정 | **0** |
>
> **교훈: 맥락이 큰 프로젝트에서 멀티에이전트는 맥락 비용을 에이전트 수만큼 곱한다.**
> 역할 분리의 논리(실행과 해석을 나눈다, 검증은 다른 모델로)는 여전히 옳지만,
> 그걸 사는 값이 이 리포에서는 너무 비쌌다.
>
> **남긴 것** — 전부 Claude 토큰 0:
> - `research/tools/check_build.py` — 빌드 지문 판정 (변종 13건을 실제로 찾아냈다)
> - `research/tools/verify.py` — Codex 교차검증 (**ChatGPT 할당량**이라 실험과 별개 예산)
> - `research/tools/cycle.py` — 사이클 장부 + vast.ai 커밋-전-종료 확인
> - `research/builds.json` · `RECORD_SCHEMA.md` · `ideas/TEMPLATE.md`
>
> **제거한 것**: `.claude/agents/` · `.claude/skills/` · `AGENTS.md`.
> 되살리려면 `git revert` 대상은 이 파일 아래 커밋 목록이 아니라
> `git log --diff-filter=D -- .claude/agents` 로 찾는다.

가이드는 **빈 리포를 전제로 쓰였다.** 이 리포는 이미 E0–E14 실험 체계, 빌드 프리즈,
사전등록 게이트, 계측 규약을 갖고 있어서, 가이드의 스켈레톤을 그대로 씌우면 기존
규율과 충돌하고 진실원이 둘로 갈라진다. 아래가 그 조정 내역이다.

## 따른 것

- **파일시스템이 유일한 진실원** — 에이전트는 결과를 대화로 넘기지 않는다
- **권한 경계로 역할 분리** — [`AGENTS.md`](../AGENTS.md)의 쓰기 경로 표
- **검증은 다른 모델로** — 아래 참조
- **novelty·인용 최종 책임은 사람** — scout은 후보와 근거까지만
- 각 역할의 "실패하는 방식"을 명시 — 다만 일반론이 아니라 **이 프로젝트에서 실제로
  일어난 사고**로 채웠다

## 바꾼 것

| 가이드 | 이 리포 | 왜 |
|---|---|---|
| `ideas/` 를 가설의 진실원으로 | `research/ideas/` 는 **스테이징**, `PAPER_PLAN.md` §2가 단일 출처 | §2가 이미 "실험 우선순위의 단일 출처"라고 못박혀 있다. 카드가 두 번째 출처가 되면 어느 쪽이 최신인지 아무도 모른다 |
| `experiments/<id>/config.yaml + run.sh + raw/` | 기존 플랫 스크립트 + `results/*.json` 유지 | 60개 스크립트를 디렉터리로 재배치하면 모든 경로 참조가 깨진다. 사용자가 구조 변경을 명시적으로 금지했다 |
| `metrics.json` 스키마 신설 | 기존 레코드 형태를 **문서화**만 ([`RECORD_SCHEMA.md`](../research/RECORD_SCHEMA.md)) | 이미 `convert_cfg` 37필드를 싣는 규약이 있다. 새 스키마는 기존 74개 레코드를 무효화한다 |
| `paper/` 에 원고 | `research/manuscript/` | `paper/` 는 베이스라인 논문 PDF 자리이고 gitignored다(저작권). 원고를 두면 git에 안 올라간다 |
| `analysis/` 신설 | `results/*.md` + `research/verdicts/` | 해석은 이미 `results/` 에 있다. `verdicts/` 는 교차모델 판정 전용 |
| 스킬: 문헌 서베이 | 스킬: **claim-audit** | 문헌 규율은 `scout` 에이전트가 이미 강제한다. 대신 이 프로젝트가 실제로 겪은 실패(빌드 혼합, 아는 약점 미발표)를 잡는 스킬로 대체 |
| Phase 0 승인 게이트에서 정지 | 리포에서 답을 직접 확인 | 6개 질문 중 5개(연구 내용, 환경, 벤뉴, 마감, 원고 형식)는 `PAPER_PLAN.md`·`NEXT_EXPERIMENTS.md`에 이미 있었다 |

## 검증기 — 가장 크게 벗어난 지점

가이드는 "다른 모델로 검증하라"고만 한다. 이 리포는 **Codex CLI**로 나간다.

```bash
python research/tools/verify.py --claim "<주장>" results/<레코드>.json
```

이유:

1. **다른 모델 계열이다.** Opus→Sonnet은 같은 계열이라 사각지대를 공유할 수 있다.
2. **경계가 강제된다.** 검증자는 레코드 원문만 프롬프트로 받고 파일시스템에 접근하지
   않는다. "쓰지 마라"는 약속이 아니라 구조적으로 쓸 수 없다.
3. **판정이 스키마로 강제된다.** `--output-schema`가 `supported|refuted|unverifiable`과
   근거 배열을 요구해서, 그럴듯한 산문으로 얼버무릴 수 없다.

**폴백을 막아뒀다.** Codex가 없으면 스크립트가 멈춘다. 조용히 Claude로 넘어가면
자기가 자기를 채점하는 상태가 되고, 그걸 아무도 모르게 된다.

### 검증된 동작

세팅 시점에 실제로 확인했다:

| 던진 주장 | 판정 | 비고 |
|---|---|---|
| "ΔPPL −0.14% 를 53,888 B / 13,472 p 로 달성" | **refuted** (high) | 빌드 혼합을 정확히 짚음: −0.14%는 no-beta, 53,888은 beta. 레코드의 실제값 −0.187288 인용 |
| "프리즈 빌드 ΔPPL −0.187%, 53,888 B" | **supported** (high) | 문제를 억지로 만들지 않음 |
| (파일을 못 읽던 초기 배선) | **unverifiable** (low) | 값을 지어내지 않고 못 읽었다고 보고 |

세 번째가 중요하다 — 근거가 없을 때 채우지 않는 것이 이 역할의 핵심이고,
그게 실제로 확인됐다.

## Windows 샌드박스 — 원인과 수동 수정 (2026-08-14 해결)

`codex exec -s read-only`가 셸을 못 띄우고 죽었다:
`orchestrator_helper_launch_failed: helper=codex-windows-sandbox-setup.exe, error=program not found`.

**헬퍼가 없어서가 아니라 codex가 못 찾아서였다.** standalone Windows 빌드는 헬퍼를
*실행 파일 옆*에서 찾는데, PATH에 걸린 `codex.exe`는 shim이고 헬퍼는 릴리스 패키지의
형제 디렉터리에 있다. 로그가 그대로 말해준다:

```
helper copy failed for command-runner: helper not found next to current executable
  or under codex-resources: ...\Programs\OpenAI\Codex\bin\codex.exe
```

`codex update`(0.144.1 → 0.147.0)로는 안 고쳐졌다. 헬퍼 두 개를 shim 옆에 복사해서 해결:

```bash
cp ~/.codex/packages/standalone/releases/0.147.0-x86_64-pc-windows-msvc/codex-resources/{codex-windows-sandbox-setup.exe,codex-command-runner.exe} \
   "$LOCALAPPDATA/Programs/OpenAI/Codex/bin/"
```

> ⚠️ **codex를 업데이트하면 다시 해야 할 수 있다.** 버전이 올라가면 릴리스 경로가
> 바뀌고, shim 옆의 헬퍼는 옛 버전으로 남는다. 증상이 다시 나오면 위 명령을 새 버전
> 경로로 다시 돌린다. 확인은 `codex sandbox cmd /c echo hello`.

## 알려진 한계

- **셸로 읽으면 한글이 깨진다** (Windows 콘솔 cp949). 이 리포는 문서 상당수가 한글이라
  실제 위험이다. 그래서 `verify.py`의 **기본값은 여전히 embed 모드** — 파이썬이 UTF-8로
  읽어 프롬프트에 싣는다. 탐색이 필요할 때만 `--explore`를 쓰고, 그 모드의 프롬프트는
  "깨진 텍스트를 근거로 삼지 말고 problems에 적으라"고 지시한다.
- `--explore`는 느리고 토큰을 더 쓴다. 헤드라인 주장에만 쓰는 게 좋다.
- 레코드 총량이 240K자를 넘으면 뒤쪽이 잘린다. 잘리면 그 사실을 프롬프트에 명시한다.
- `check_build.py`의 bytes/params 지문은 **`gpt2-medium` 전용**이다. ViT/RoBERTa
  레코드는 `n/a`로 빠진다 — 모델이 다르면 지문이 다른 게 정상이기 때문이다.

## 아직 안 한 것

가이드 Phase 7(MCP)은 **제안만 하고 설치하지 않았다.** 검토 없이 MCP 서버를 붙이면
프롬프트 인젝션 경로가 된다.

- **인용 검증 MCP (Crossref/arXiv)** — 선택이 아니라 필수로 권한다. 원고의 모든 인용을
  제목·연도·식별자까지 대조하는 단계 없이 투고하지 않는다. 지금은 scout이 수동으로 한다.
- **논문 검색 MCP (arXiv/Semantic Scholar/OpenAlex)** — scout의 웹검색을 대체
- Zotero 연동 — 사용자가 Zotero를 쓰는 경우에만
