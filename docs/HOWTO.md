# 사용법 — 검증 도구

> **2026-08-14 롤백**: 서브에이전트 6개와 스킬 4개를 제거했다. 에이전트 하나가 착수할
> 때마다 `PAPER_PLAN.md`(28k) + `NEXT_EXPERIMENTS.md`(8k)를 콜드 스타트로 다시 읽어,
> 사이클 한 바퀴에 ~180k 토큰이 오리엔테이션에만 들어갔다. 실험 예산을 잡아먹어서
> 뺐다. 경위는 [`AGENT_SETUP_NOTES.md`](AGENT_SETUP_NOTES.md).
>
> **남은 것은 스크립트 3개뿐이고, 이건 Claude 토큰을 하나도 안 쓴다** (순수 파이썬).
> `verify.py`만 Codex 할당량을 쓰는데, 그건 별도 예산이라 실험과 경쟁하지 않는다.

conda 불필요 — 표준 라이브러리만 쓴다. 기본 python으로 돌아간다.

---

## 1. `check_build.py` — 이 수치, 인용해도 되나

```bash
python research/tools/check_build.py
```

`results/` 전체를 감사한다. 파일을 지정하면 그것만:

```bash
python research/tools/check_build.py results/freeze_e1.json
```

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

옵션: `--json` · `--strict`(프리즈 아닌 것도 실패로 셈)
종료 코드 `0` 전부 판정됨 · `1` 변종 있음

**언제 쓰나**: 발표·원고에 수치를 올리기 전. 새 레코드가 나온 직후.

---

## 2. `verify.py` — 이 주장, 근거 있나

**Codex(다른 모델)가 판정한다.** 자기가 자기를 채점하지 않으려는 것이고,
Claude 토큰이 아니라 ChatGPT 할당량을 쓴다.

```bash
python research/tools/verify.py --claim "프리즈 빌드의 ΔPPL은 −0.187%다" results/freeze_e1.json
```

```
판정: SUPPORTED  (confidence=high)
판정 파일: research/verdicts/20260814-213714-....json
```

| 판정 | 뜻 | 종료 코드 |
|---|---|---|
| `SUPPORTED` | 레코드가 뒷받침함 | 0 |
| `REFUTED` | 레코드와 어긋남 | 1 |
| `UNVERIFIABLE` | **근거가 없음** (틀린 게 아니라 모름) | 2 |
| (실행 실패) | codex 없음/인증 안 됨 | 3 |

**주장은 한 문장씩.** "−0.19%를 53,888 B로 달성하고 5.27× 줄인다"는 세 주장이라,
합쳐 던지면 하나만 틀려도 어디가 틀렸는지 안 남는다.

**빌드 혼합을 보려면 관련 레코드를 다 넘긴다:**

```bash
python research/tools/verify.py --claim "<주장>" results/freeze_e1.json results/stride_sens.json
```

### `--explore` — 넘긴 레코드 밖까지 뒤지기

```bash
python research/tools/verify.py --explore --claim "<주장>" results/freeze_e1.json
```

기본은 넘겨준 레코드만 본다. `--explore`는 저장소를 읽기 전용으로 뒤져 반증을 찾는다.
실제로 `freeze_e1.json` 하나만 주고 "5.27×를 헤드라인으로 실을 수 있다"를 던졌더니,
주지도 않은 파일에서 `strict_pj`를 찾아내 범위 비대칭을 지적하고 refuted를 냈다.

느리고 Codex 할당량을 더 쓴다. **헤드라인 주장에만.**
셸로 읽으면 한글이 깨지므로(cp949) 기본은 embed 모드다.

---

## 3. `cycle.py` — 실험 사이클 장부 (선택)

에이전트 없이 **사람이 직접 모는 원장**이다. 상태가 파일에 남아서 세션이 끊겨도 이어진다.

```bash
python research/tools/cycle.py status      # 지금 어디인가
python research/tools/cycle.py list        # 모든 사이클
python research/tools/cycle.py routing     # 문제가 생기면 어느 단계로 되돌릴지
```

```bash
python research/tools/cycle.py new --idea research/ideas/idea-001.md --title "E16 에너지 계상"
python research/tools/cycle.py advance --artifact results/e16.json
python research/tools/cycle.py feedback --to runner --reason "운영점 미기록"
python research/tools/cycle.py close --conclusion refuted
```

단계: `planner → scout → runner → analyst → verifier`.
에이전트가 없어도 **누가 하든 그 순서로 일한다** — 해석은 실행과 분리하는 게 낫다.

### vast.ai 안전장치 — 이것 때문에 남겼다

```bash
python research/tools/cycle.py approve gpu --note "4090 열었음"   # 박스 열고 나서
python research/tools/cycle.py gpu-off                            # 끄기 전에 확인
```

`gpu-off`는 **runner 레코드가 전부 커밋됐는지 확인하고, 안 됐으면 거부한다:**

```
🔴 아직 끄면 안 된다 — 커밋 안 된 레코드:
     results/e16.json
```

닫힌 vast.ai 박스가 P0.4 Block C와 Stage 2 원본을 통째로 가져간 적이 있다.
**커밋이 먼저, 종료가 나중.** 통과하면:

```
✅ 레코드 1건 전부 커밋됨 — vast.ai 인스턴스를 꺼도 된다.
```

---

## 4. 결과가 어디에 남나

```
research/
├── builds.json    빌드 지문 대장 (사람만 수정)
├── ideas/         가설 카드 (TEMPLATE.md 참고)
├── verdicts/      verify.py 판정 JSON
├── cycles/        cycle.py 상태
└── refs/          문헌 비교표
```

---

## 5. 막힐 때

**`교차모델 검증 불가 — codex 실행 파일을 찾을 수 없다`**

```bash
codex login
```

확인: `python research/tools/verify.py --check-only`

**`windows sandbox failed: ... program not found`** (`--explore` 쓸 때)

codex 업데이트 후 재발한다. 헬퍼를 shim 옆에 다시 복사한다 (버전 번호를 현재 것으로):

```bash
cp ~/.codex/packages/standalone/releases/0.147.0-x86_64-pc-windows-msvc/codex-resources/{codex-windows-sandbox-setup.exe,codex-command-runner.exe} "$LOCALAPPDATA/Programs/OpenAI/Codex/bin/"
```

확인: `codex sandbox cmd /c echo hello` · 경위는 [`AGENT_SETUP_NOTES.md`](AGENT_SETUP_NOTES.md)

---

## 6. 토큰을 아끼려면

에이전트를 뺀 뒤로 비용은 **회원님이 무엇을 읽히느냐**로 결정된다.

- `PAPER_PLAN.md`는 **28k 토큰**이다. 통째로 읽히지 말고 필요한 절(§2/§4/§5)만 지정한다.
- `NEXT_EXPERIMENTS.md`는 8k. 콜드스타트가 아니면 다시 읽을 필요 없다.
- 수치 확인은 **스크립트로** 한다 — `check_build.py`는 74개 레코드를 0 토큰에 판정한다.
- 주장 검증은 `verify.py`로 넘긴다 — Codex 할당량이라 실험 예산과 별개다.
