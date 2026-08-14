# research/ — 에이전트 레이어

기존 구조를 대체하지 않는다. `experiments/`·`results/`·`PAPER_PLAN.md`는 그대로고,
여기는 **에이전트가 읽고 쓰는 얇은 층**만 얹는다.

```
research/
├── builds.json          빌드 지문 대장 — 사람만 수정
├── RECORD_SCHEMA.md     results/*.json 계약
├── ideas/               가설 카드 (스테이징. 승인되면 사람이 PAPER_PLAN §2로 승격)
├── refs/                문헌 비교표 (식별자 없으면 "미확인 후보"로 격리)
├── verdicts/            교차모델 검증 판정 (codex 출력, 자동 생성)
├── manuscript/          원고 초안 — paper/ 는 베이스라인 PDF 자리라 못 쓴다
└── tools/
    ├── check_build.py       레코드가 어느 빌드인지 기계 판정
    ├── verify.py            주장을 Codex로 교차검증
    └── verdict.schema.json  판정 출력 스키마
```

## 왜 `paper/`가 아니라 `research/manuscript/`인가

`paper/`는 **베이스라인 논문 PDF**가 있는 자리이고 gitignored다(저작권, 재배포 안 함).
원고를 거기 두면 git에 안 올라간다. 원고는 `research/manuscript/`로 간다.

## 왜 가설 카드가 `PAPER_PLAN.md`를 대체하지 않는가

`PAPER_PLAN.md` §2가 **실험 우선순위의 단일 출처다.** `research/ideas/`는 아직 승인되지
않은 제안의 스테이징일 뿐이다. 카드가 두 번째 진실원이 되면 어느 쪽이 최신인지 아무도
모르게 되고, 그건 이 프로젝트가 이미 한 번 겪은 실패 모드다.

흐름: `ideas/` 카드 작성 → 사람이 반증 조건 승인 → **사람이** §2로 승격 → 실행.

## 두 도구

```bash
# 이 레코드 인용해도 되나?
python research/tools/check_build.py results/freeze_e1.json

# 이 주장이 레코드로 뒷받침되나? (Codex — 실행한 모델과 다른 모델)
python research/tools/verify.py --claim "<주장>" results/freeze_e1.json
```

`verify.py`는 레코드 **원문을 프롬프트에 실어** 보낸다. 파일 경로만 주면 검증자가 셸과
샌드박스에 의존하게 되는데, 이 환경에서 그게 실제로 깨졌다(Windows 샌드박스 헬퍼 부재).
내용을 직접 실으면 검증자는 파일시스템에 손댈 일이 없고, 쓰기 경계가 절대적이 된다.

Codex가 없거나 인증이 안 되어 있으면 **조용히 Claude로 폴백하지 않는다** — 그러면
자기가 자기를 채점하는 것이고, 이 단계의 목적이 없어진다. 스크립트가 그 사실을 말하고 멈춘다.
