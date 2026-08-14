# 사용법 — PASN 연구 자동화

이 리포에 뭘 시킬 수 있고, 결과를 어디서 보는지.

> **먼저**: 스킬과 에이전트는 **세션 시작할 때 로드된다.** 지금 세션에서 만들었다면
> Claude Code를 **한 번 껐다 켜야** `/claim-audit` 같은 게 잡힌다.
> **스크립트 2개는 지금 당장 된다** — 재시작 필요 없음.

---

## 1. 한 장 요약

세 가지 방식으로 쓴다. 아래로 갈수록 자동화 정도가 높고, 위로 갈수록 확실하다.

| 방식 | 예 | 언제 |
|---|---|---|
| **스크립트** 직접 | `python research/tools/check_build.py` | 확실하고 빠름. 재시작 불필요 |
| **스킬** (`/이름`) | `/claim-audit 발표자료 점검` | 절차가 정해진 반복 작업 |
| **에이전트** (말로) | "verifier로 이 주장 검증해줘" | 여러 단계를 맡길 때 |

무엇을 쓰든 **결과는 항상 파일로 남는다.** 대화창의 요약은 참고용이고, 근거는 파일이다.

---

## 2. 스크립트 2개 — 이것만 알아도 절반

### `check_build.py` — 이 수치, 인용해도 되나?

```bash
python research/tools/check_build.py
```

`results/` 전체를 감사한다. 인자를 주면 그 파일만:

```bash
python research/tools/check_build.py results/freeze_e1.json
```

**읽는 법:**

```
OK   results/freeze_e1.json [0] freeze-t16-unif   build=beta
NAME results/stride_sens.json [2] stride-sens-s256 build=no-beta
       └ 인용 시 빌드 이름을 반드시 붙일 것
??   results/e4_pasn_s.json  [0] e4-pasn-s-t16    build=unknown

레코드 74건 — frozen=23, n/a=31, off-freeze=7, unknown=13
```

| 표시 | 뜻 | 할 일 |
|---|---|---|
| `OK` | 프리즈 빌드 | 그대로 인용 가능 |
| `NAME` | 다른 빌드(no-beta) | **빌드 이름을 붙여서** 인용 |
| `??` | 어느 빌드와도 안 맞는 변종 | 헤드라인으로 쓰지 말 것 |
| `--` | 다른 모델(ViT/RoBERTa)이거나 ANN 기준런 | 대조 대상 아님 (정상) |

옵션: `--json`(기계 판독용) · `--strict`(프리즈 아닌 것도 실패로 셈)

종료 코드 `0`이면 전부 판정됨, `1`이면 `??`가 있음.

### `verify.py` — 이 주장, 근거 있나?

**다른 모델(Codex)이 판정한다.** 자기가 자기를 채점하지 않으려고 이렇게 했다.

```bash
python research/tools/verify.py --claim "프리즈 빌드의 ΔPPL은 −0.187%다" results/freeze_e1.json
```

출력:

```
검증자: codex (codex-cli 0.147.0) · 레코드 원문만
주장: 프리즈 빌드의 ΔPPL은 −0.187%다
레코드: 1건 — 판정 중...

판정: SUPPORTED  (confidence=high)

판정 파일: research/verdicts/20260814-213714-....json
```

**판정 3종:**

| 판정 | 뜻 | 종료 코드 |
|---|---|---|
| `SUPPORTED` | 레코드가 뒷받침함 | 0 |
| `REFUTED` | 레코드와 어긋남 | 1 |
| `UNVERIFIABLE` | **근거가 없음** (틀린 게 아니라 모름) | 2 |
| (실행 실패) | codex 없음/인증 안 됨 | 3 |

`UNVERIFIABLE`을 `REFUTED`로 읽지 말 것. 근거가 없는 것과 틀린 것은 다르다.

**주장은 한 문장씩 던진다.** "−0.19%를 53,888 B로 달성하고 스파이크 5.27× 줄인다"는
세 주장이라, 합쳐 던지면 하나만 틀려도 전체가 refuted가 되고 어디가 틀렸는지 안 남는다.

**빌드가 섞였는지 보려면 관련 레코드를 다 넘긴다:**

```bash
python research/tools/verify.py --claim "<주장>" results/freeze_e1.json results/stride_sens.json
```

---

## 3. `--explore` — 검증자가 직접 뒤지게 하기

```bash
python research/tools/verify.py --explore --claim "<주장>" results/freeze_e1.json
```

기본 모드는 **넘겨준 레코드만** 본다. `--explore`를 붙이면 저장소를 읽기 전용으로
뒤져서, 주지 않은 파일에서도 반증을 찾는다.

실제로 `freeze_e1.json` 하나만 주고 "5.27× 절감을 헤드라인으로 실을 수 있다"를 던졌더니,
주지도 않은 파일에서 `strict_pj`를 찾아내 범위 비대칭을 지적하고 refuted를 냈다.

**언제 쓰나**: 논문·발표에 올라갈 헤드라인 주장에만. 느리고 토큰을 더 쓴다.

**주의**: 셸로 읽으면 한글이 깨진다(Windows cp949). 그래서 기본값은 embed 모드다.
`--explore` 프롬프트는 "깨진 텍스트는 근거로 쓰지 말고 problems에 적어라"라고 지시한다.

---

## 4. 스킬 3개 — `/이름`으로 부른다

세션 재시작 후 쓸 수 있다. 슬래시로 부르거나, 그냥 말로 해도 라우팅된다.

### `/claim-audit` — 발표 전 수치 점검

```
/claim-audit PASN_lab_meeting_2026-08-14_spike_only_v2.pptx 점검해줘
```

문서의 수치를 전부 뽑아서 레코드로 역추적하고, 빌드 혼합·운영점 불일치·근거 없는 수치를
찾는다. 결과는 `research/verdicts/<날짜>-<문서명>-audit.md`에 표로 남는다.

**이게 가장 자주 쓸 스킬이다.** 08-14 피드백 4개 중 3개가 이미 리포에 🔴로 적혀 있었는데
발표에 안 올라간 것이었다.

### `/hypothesis-card` — 새 실험 제안

```
/hypothesis-card e_min을 -6에서 -8로 낮추면 어떨까
```

반증 조건이 **숫자로** 없으면 카드를 완성하지 않는다. GPU 쓰기 전 설계 점검
체크리스트(seed 스윕 금지, 균일 그리드 채점 금지 등)를 같이 돌린다.
결과는 `research/ideas/idea-NNN.md`.

### `/experiment-report` — 런 끝나고 정리

```
/experiment-report results/e15_새실험.json 정리해줘
```

레코드와 로그만 읽어서 `results/<실험>.md`를 쓴다. 로그에 없는 수치는 절대 안 쓰고
`N/A (레코드에 없음)`으로 남긴다. 헤드라인 필드만 보지 않고 `by_kind`·`ops_per_input`까지
훑어서 같은 파일 안의 모순을 찾는다.

---

## 5. 에이전트 5개 — 말로 시킨다

이름을 부르거나 그냥 일을 설명하면 된다.

| 부를 때 | 하는 일 | 결과가 남는 곳 |
|---|---|---|
| "planner로 다음 실험 정해줘" | 우선순위 판단, 가설 카드 | `research/ideas/` |
| "scout으로 관련 논문 찾아줘" | 문헌 검색, novelty 후보 | `research/refs/` |
| "runner로 이 실험 돌려줘" | 실험 작성·실행 | `experiments/`, `results/` |
| "verifier로 이 주장 검증해줘" | Codex 검증 실행·정리 | `research/verdicts/` |
| "writer로 결과 절 초안 써줘" | 원고 초안 | `research/manuscript/` |

**각 에이전트는 자기 경로에만 쓴다.** verifier는 `experiments/`를 못 고치고, writer는
수치를 창작할 수 없다. 경계는 `AGENTS.md`에 표로 있다.

---

## 6. 실전 시나리오

### 랩미팅 발표 전

```bash
python research/tools/check_build.py
```

먼저 이걸 돌려서 `??`가 있는지 본다. 그다음:

```
/claim-audit 이번 발표자료 수치 전부 점검해줘
```

헤드라인 주장 2~3개는 따로 `--explore`로 한 번 더:

```bash
python research/tools/verify.py --explore --claim "<헤드라인>" results/freeze_e1.json
```

### 실험이 끝났을 때

```bash
python research/tools/check_build.py results/새레코드.json   # 빌드 먼저
```

```
/experiment-report results/새레코드.json
```

그다음 헤드라인이 될 문장만 `verify.py`에 넘긴다.

### 논문 문장을 쓸 때

writer에게 시키되, 수치마다 출처 주석이 달려 나온다:

```markdown
ΔPPL은 −0.19%다 <!-- results/freeze_e1.json[0] freeze-t16-unif, beta 빌드 -->
```

주석 없는 수치가 있으면 그게 위험 신호다.

---

## 7. 결과가 어디에 남나

```
research/
├── ideas/       가설 카드          ← planner, /hypothesis-card
├── refs/        문헌 비교표         ← scout
├── verdicts/    검증 판정 JSON/MD   ← verify.py, /claim-audit
└── manuscript/  원고 초안          ← writer

results/         레코드 JSON + 해석 MD  ← runner, /experiment-report
```

`research/verdicts/`의 JSON에는 판정·근거·문제·"무엇이 나오면 뒤집히나"가 들어 있다.
`_meta.mode`가 `embed`인지 `explore`인지도 기록된다.

---

## 8. 막힐 때

**`교차모델 검증 불가 — codex 실행 파일을 찾을 수 없다`**

```bash
codex login
```

로그인 상태 확인은 `python research/tools/verify.py --check-only`.

**`windows sandbox failed: ... program not found`** (`--explore` 쓸 때)

codex 업데이트 후 재발할 수 있다. 헬퍼를 shim 옆에 다시 복사한다
(경로의 버전 번호를 현재 버전으로 바꿀 것):

```bash
cp ~/.codex/packages/standalone/releases/0.147.0-x86_64-pc-windows-msvc/codex-resources/{codex-windows-sandbox-setup.exe,codex-command-runner.exe} "$LOCALAPPDATA/Programs/OpenAI/Codex/bin/"
```

확인: `codex sandbox cmd /c echo hello`. 자세한 경위는
[`AGENT_SETUP_NOTES.md`](AGENT_SETUP_NOTES.md).

**`/claim-audit`이 없다고 나온다** → Claude Code를 재시작한다. 스킬은 세션 시작에 로드된다.

**한글이 깨져 보인다** → `--explore` 모드에서 셸로 읽을 때 생긴다. 기본(embed) 모드를 쓰면 안 생긴다.

**스크립트에 conda가 필요한가?** → 아니다. `check_build.py`와 `verify.py`는 표준 라이브러리만
써서 기본 python으로도 돈다. `SNN` 환경은 **실험 실행**에만 필요하다.

---

## 9. 이 자동화가 하지 않는 것

경계를 알아야 믿고 쓸 수 있다.

- **novelty 판단을 대신하지 않는다.** scout은 후보와 근거까지만.
- **인용을 확정하지 않는다.** DOI/arXiv 대조는 아직 수동이다.
- **`PAPER_PLAN.md`를 고치지 않는다.** §2가 실험 우선순위의 단일 출처이고,
  가설 카드는 스테이징일 뿐이다. 승격은 사람이 한다.
- **`PASN_vault/`를 건드리지 않는다.** 읽기만. 기록 보존 공간이다.
- **기존 레코드를 덮어쓰지 않는다.** 새 결과는 새 파일.
- **런 없이 수치를 채우지 않는다.** 없으면 `N/A (레코드 없음)`.

투고 전에는 사람이 직접 확인해야 하는 체크리스트가 [`../AGENTS.md`](../AGENTS.md) 맨 아래에 있다.

---

## 10. 더 볼 곳

| 무엇 | 어디 |
|---|---|
| 프로젝트 컨텍스트 (세션마다 자동 로드) | [`../CLAUDE.md`](../CLAUDE.md) |
| 역할별 쓰기 경계 + 실패 방식 | [`../AGENTS.md`](../AGENTS.md) |
| 레코드 스키마 | [`../research/RECORD_SCHEMA.md`](../research/RECORD_SCHEMA.md) |
| 빌드 지문 대장 | [`../research/builds.json`](../research/builds.json) |
| 왜 이렇게 만들었나 (가이드 대비 변경점) | [`AGENT_SETUP_NOTES.md`](AGENT_SETUP_NOTES.md) |
| 실험 우선순위 | [`../PAPER_PLAN.md`](../PAPER_PLAN.md) §2 |
