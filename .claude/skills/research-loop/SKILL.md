---
name: research-loop
description: 가설부터 검증까지 한 사이클을 돌린다. planner→scout→runner→analyst→verifier를 순서대로 진행하고, 문제가 생기면 올바른 앞 단계로 되돌린다. "실험 사이클 돌려줘", "이 가설 끝까지 진행해줘", 새 연구 주제 착수, 루프 재개에 사용한다.
---

# research-loop — 한 사이클 돌리기

```
planner ──> scout ──> [승인] ──> runner ──> analyst ──> verifier ──> 종료
   ^          ^                     ^          ^            │
   └──────────┴─────────────────────┴──────────┴────────────┘
                        문제가 생기면 해당 단계로
```

## 먼저 알 것 — 이건 무인 루프가 아니다

에이전트를 자동 실행하지 않는다. **상태를 기록하고 게이트를 강제한다.**
GPU 시간과 사전등록 게이트가 걸린 프로젝트에서 무인 루프는 비용을 태우고 규율을
무너뜨린다. 사람이 게이트에 서 있고, 스킬은 그 사이의 반복을 맡는다.

한 단계가 끝날 때마다 **사용자에게 결과를 보고하고 다음 단계로 갈지 확인한다.**
특히 `runner` 앞에서는 반드시 멈춘다 — GPU를 쓰기 직전이다.

## 상태는 파일에 있다

```bash
python research/tools/cycle.py status     # 지금 어디인가
python research/tools/cycle.py list       # 모든 사이클
python research/tools/cycle.py routing    # 피드백을 어디로 보낼까
```

대화 맥락이 아니라 `research/cycles/<id>/state.json`이 진실원이다.
세션이 끊겨도 `status`로 이어서 하면 된다.

## 절차

### 시작

가설 카드가 먼저 있어야 한다. 없으면 `/hypothesis-card`로 만든다.

```bash
python research/tools/cycle.py new --idea research/ideas/idea-001.md --title "저온 붕괴"
```

### 각 단계

단계마다 해당 에이전트를 부르고, **산출물 파일이 실제로 생긴 뒤에** 전진시킨다.
대화 요약은 산출물이 아니다 — `advance`가 파일 존재를 확인하고, 없으면 거부한다.

| 단계 | 에이전트 | 산출물 |
|---|---|---|
| planner | planner | `research/ideas/idea-NNN.md` |
| scout | scout | `research/refs/<주제>.md` |
| runner | runner | `results/<실험>.json` |
| analyst | analyst | `results/<실험>.md` |
| verifier | verifier | `research/verdicts/*.json` |

```bash
python research/tools/cycle.py advance --artifact results/e15.json
```

### 승인 게이트 — `runner` 앞

```bash
python research/tools/cycle.py approve hypothesis --note "반증 조건 확인함"
```

**사람만 승인한다.** 에이전트가 대신 누르지 않는다. 승인 전에 확인할 것:

- 반증 조건이 **숫자로** 적혀 있는가
- 프리즈 빌드로 돌리는가
- 운영점(T, arm, stride)을 정했는가

### 자동 검사 — `runner` 다음

`advance`가 `check_build.py`를 돌려서 레코드 지문을 확인한다. 변종 빌드(`??`)가
섞여 있으면 **해석 단계로 못 넘어간다.** 인용할 수 없는 수치로 문서를 쓰는 것을
여기서 막는다. 의도한 변종이면 `--force`.

## 피드백 — 어디로 되돌릴 것인가

**이게 이 루프의 핵심이고, 가장 많이 틀리는 부분이다.**

```bash
python research/tools/cycle.py feedback --to analyst --reason "5.27x가 arm 미명시"
```

| 증상 | 어디로 | 왜 |
|---|---|---|
| 주장이 레코드보다 과하다 / 빌드를 섞었다 | **analyst** | 데이터는 멀쩡하다. 해석만 고치면 된다 |
| 레코드에 출처 필드가 없다 / 지문이 변종 | **runner** | 레코드 자체를 다시 만들어야 한다 |
| `unverifiable` — 근거 자체가 없다 | **runner** | 측정이 빠졌다. 해석으로 못 메운다 |
| 운영점이 기록 안 됨 | **runner** | 나중에 인용 불가능해진다 |
| 선행연구가 전제를 무너뜨린다 | **planner** | 실험 설계부터 다시 |
| 반증 조건이 검증 불가능한 형태였다 | **planner** | 카드가 잘못 만들어졌다 |
| **가설이 반증됐다** | **되돌리지 않는다** | 이건 결과다 → `close --conclusion refuted` |

**전부 planner로 되돌리지 마라.** 레코드 하나 고치면 될 일에 사이클 전체를 다시 도는
것이 이 루프가 망가지는 방식이다. 되돌린 뒤 고쳤으면 닫는다:

```bash
python research/tools/cycle.py resolve --note "arm 명시하고 주장 축소"
```

열린 피드백이 있으면 전진이 막힌다 — 고치지 않고 넘어가는 것을 방지한다.

## 반증은 실패가 아니다

가설이 틀렸다고 나오면 그건 **사이클이 성공한 것이다.** 반증 조건을 사전에 숫자로
적어둔 이유가 그것이다.

```bash
python research/tools/cycle.py close --conclusion refuted --note "T=4에서 +2.1%, 밴드 초과"
```

**밴드를 사후에 넓혀서 supported로 만들지 마라.** 이 프로젝트는 게이트를 한 번
놓쳤고(0.187% > 0.14%), 넓히지 않기로 결정했고, 결론은 흔들리지 않았다.

`supported`로 닫으려면 verifier 판정이 있어야 한다 — `close`가 확인한다.

## 종료 후

사이클은 자동으로 `PAPER_PLAN.md`를 고치지 않는다. 결과를 §2에 반영할지는 **사람이**
정한다. 카드의 `status`와 `promoted_to`를 갱신하고 사용자에게 무엇을 반영할지 묻는다.

## 보고 형식

각 단계가 끝날 때 짧게:

```
[C001 · runner 완료 · 2회차]
산출: results/e15.json (레코드 검사 통과, 프리즈 빌드)
다음: analyst — 레코드만 보고 해석
```

사이클이 끝나면 무엇이 반증/지지됐고, 되돌린 지점이 어디였는지 요약한다.
**되돌린 이력이 가장 유용한 정보다** — 다음 사이클에서 같은 곳에서 안 막히게 한다.
