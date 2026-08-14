#!/usr/bin/env python
"""연구 사이클 원장 — planner → scout → runner → analyst → verifier.

에이전트를 자동으로 실행하지 않는다. **상태를 기록하고 게이트를 강제한다.**
이 구분이 중요하다: GPU 시간과 사전등록 게이트가 걸린 프로젝트에서 무인 루프는
비용을 태우고 규율을 무너뜨린다. 사람이 게이트에 서 있고, 이 스크립트는 그 사이의
추적을 맡는다.

    python research/tools/cycle.py new --idea research/ideas/idea-001.md
    python research/tools/cycle.py status
    python research/tools/cycle.py advance --artifact research/refs/lowT.md
    python research/tools/cycle.py approve hypothesis
    python research/tools/cycle.py feedback --to runner --reason "레코드에 pasn_beta 누락"
    python research/tools/cycle.py close --conclusion refuted

상태는 `research/cycles/<id>/state.json`. 대화가 아니라 이 파일이 진실원이다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CYCLES = REPO / "research" / "cycles"
CHECK_BUILD = REPO / "research" / "tools" / "check_build.py"

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

STAGES = ["planner", "scout", "runner", "analyst", "verifier", "closed"]

STAGE_INFO = {
    "planner":  ("가설 카드",     "research/ideas/"),
    "scout":    ("문헌 대조",     "research/refs/"),
    "runner":   ("실험 실행",     "results/ (레코드 JSON)"),
    "analyst":  ("결과 해석",     "results/ (분석 MD)"),
    "verifier": ("교차모델 검증", "research/verdicts/"),
    "closed":   ("종료",          "—"),
}

# 게이트: 이 단계로 넘어가려면 사람이 승인해야 한다.
GATES = {
    "runner": [
        ("hypothesis", "반증 조건이 숫자로 적혀 있고 사람이 승인했는가 "
                       "(GPU를 쓰기 전 마지막 지점)"),
        ("gpu", "vast.ai 인스턴스가 떠 있고 tmux 안에서 돌릴 준비가 됐는가 "
                "— 사람이 박스를 열어줘야 시작할 수 있다"),
    ],
}

# 피드백이 어디로 가야 하는가. 이걸 안 정해두면 전부 planner로 돌아가고,
# 그러면 레코드 하나 고치면 될 일에 사이클 전체를 다시 돈다.
ROUTING = [
    ("주장이 레코드보다 과하다 / 빌드를 섞었다",        "analyst"),
    ("레코드에 출처 필드가 없다 / 지문이 변종이다",      "runner"),
    ("측정이 빠졌다 — 근거 자체가 없다 (unverifiable)",  "runner"),
    ("운영점이 기록되지 않았다",                        "runner"),
    ("선행연구가 이 실험의 전제를 무너뜨린다",           "planner"),
    ("반증 조건이 애초에 검증 불가능한 형태였다",         "planner"),
    ("가설이 반증됐다",                                 "(피드백 아님 — 결과다. close --conclusion refuted)"),
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def cycle_dir(cid: str) -> Path:
    return CYCLES / cid


def load(cid: str) -> dict:
    p = cycle_dir(cid) / "state.json"
    if not p.exists():
        sys.exit(f"사이클 없음: {cid} ({p})")
    return json.loads(p.read_text(encoding="utf-8"))


def save(state: dict) -> None:
    d = cycle_dir(state["cycle_id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def active_cycle() -> str:
    """열려 있는 사이클. 여럿이면 가장 최근."""
    if not CYCLES.exists():
        sys.exit("사이클이 없다. `cycle.py new --idea <카드>` 로 시작한다.")
    open_ = []
    for d in sorted(CYCLES.iterdir()):
        f = d / "state.json"
        if f.exists():
            s = json.loads(f.read_text(encoding="utf-8"))
            if s.get("status") == "active":
                open_.append(s)
    if not open_:
        sys.exit("열린 사이클이 없다. `cycle.py new` 로 시작하거나 `cycle.py list`.")
    return sorted(open_, key=lambda s: s["created"])[-1]["cycle_id"]


def next_id() -> str:
    CYCLES.mkdir(parents=True, exist_ok=True)
    n = sum(1 for d in CYCLES.iterdir() if (d / "state.json").exists())
    return f"C{n + 1:03d}"


def log(state: dict, event: str, **kw) -> None:
    state.setdefault("history", []).append({"at": now(), "event": event, **kw})


def rel(p: Path) -> str:
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


# --------------------------------------------------------------------------- #

def cmd_new(args) -> int:
    idea = Path(args.idea)
    idea = idea if idea.is_absolute() else REPO / idea
    if not idea.exists():
        sys.exit(f"가설 카드가 없다: {idea}\n"
                 f"`/hypothesis-card` 로 먼저 만든다.")

    cid = next_id()
    state = {
        "cycle_id": cid,
        "title": args.title or idea.stem,
        "idea": rel(idea),
        "created": now(),
        "status": "active",
        "stage": "planner",
        "iteration": 1,
        "gates": {},
        "artifacts": {s: [] for s in STAGES[:-1]},
        "open_feedback": [],
        "history": [],
    }
    log(state, "new", idea=state["idea"])
    save(state)
    print(f"사이클 {cid} 시작 — {state['title']}")
    print(f"  가설 카드: {state['idea']}")
    print(f"  현재 단계: planner")
    print(f"\n다음: 카드를 완성하고 `cycle.py advance --artifact {state['idea']}`")
    return 0


def cmd_status(args) -> int:
    cid = args.cycle or active_cycle()
    s = load(cid)
    stage = s["stage"]
    label, out = STAGE_INFO[stage]

    print(f"사이클 {cid} — {s['title']}   [{s['status']}]  반복 {s['iteration']}회차")
    print(f"가설 카드: {s['idea']}")
    print()

    for st in STAGES[:-1]:
        mark = "▶" if st == stage else ("✓" if STAGES.index(st) < STAGES.index(stage) else " ")
        arts = s["artifacts"].get(st, [])
        print(f" {mark} {st:<9} {STAGE_INFO[st][0]:<10} " +
              (f"({len(arts)}건)" if arts else ""))
        for a in arts:
            print(f"       └ {a}")

    if s["gates"]:
        print("\n게이트:")
        for g, v in s["gates"].items():
            print(f"  {'✓' if v['approved'] else '✗'} {g} — {v.get('note','')}")

    if s["open_feedback"]:
        print("\n🔴 열린 피드백:")
        for f in s["open_feedback"]:
            print(f"  {f['from']} → {f['to']}: {f['reason']}")

    if s["status"] == "active":
        print(f"\n현재: {stage} ({label}) · 산출물은 {out}")
        pending = [(g, why) for g, why in GATES.get(_next_stage(stage), [])
                   if not s["gates"].get(g, {}).get("approved")]
        for g, why in pending:
            print(f"⚠ 승인 필요: `cycle.py approve {g}` — {why}")

        if stage == "runner":
            dirty = _uncommitted(s["artifacts"].get("runner", []))
            if dirty:
                print(f"🔴 커밋 안 된 레코드 {len(dirty)}건 — 박스를 끄면 사라진다")
            elif s["artifacts"].get("runner"):
                print("✅ 레코드 커밋됨 — vast.ai 인스턴스를 꺼도 된다")
    else:
        print(f"\n종료됨 — 결론: {s.get('conclusion')}")
    return 0


def _next_stage(stage: str) -> str:
    return STAGES[min(STAGES.index(stage) + 1, len(STAGES) - 1)]


def _uncommitted(paths: list[str]) -> list[str]:
    """아직 git에 안 들어간 레코드. 박스를 끄기 전에 반드시 비어야 한다.

    닫힌 vast.ai 박스가 P0.4 Block C와 Stage 2 원본 레코드를 통째로 가져간 적이
    있다. 원격 박스는 언제든 사라지고, 커밋 안 된 레코드는 같이 사라진다.
    """
    if not paths:
        return []
    proc = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                          cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return [ln[3:].strip() for ln in proc.stdout.splitlines() if ln.strip()]


def _check_records(paths: list[str]) -> tuple[bool, str]:
    """runner 산출 레코드의 빌드 지문을 기계로 확인한다."""
    jsons = [p for p in paths if p.endswith(".json")]
    if not jsons:
        return True, "확인할 레코드 JSON이 없음"
    proc = subprocess.run(
        [sys.executable, str(CHECK_BUILD), *jsons, "--json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return True, "check_build 판독 실패 (건너뜀)"
    unknown = [r for r in data["records"] if r["status"] == "unknown"]
    missing = [r for r in data["records"] if r["missing_provenance"]]
    notes = []
    if unknown:
        notes.append(f"변종 빌드 {len(unknown)}건")
    if missing:
        notes.append(f"출처 필드 누락 {len(missing)}건")
    return (not unknown), "; ".join(notes) or "전부 판정됨"


def cmd_advance(args) -> int:
    cid = args.cycle or active_cycle()
    s = load(cid)
    if s["status"] != "active":
        sys.exit(f"{cid}은 이미 종료됐다 ({s['status']}).")

    stage = s["stage"]
    nxt = _next_stage(stage)

    if s["open_feedback"] and not args.force:
        print("🔴 열린 피드백이 있다. 해결하고 `cycle.py resolve` 하거나 --force:")
        for f in s["open_feedback"]:
            print(f"  {f['from']} → {f['to']}: {f['reason']}")
        return 1

    arts = []
    for a in args.artifact or []:
        p = Path(a) if Path(a).is_absolute() else REPO / a
        if not p.exists():
            sys.exit(f"산출물이 없다: {a}\n"
                     f"단계를 넘기려면 파일이 실제로 있어야 한다 — "
                     f"대화 요약은 산출물이 아니다.")
        arts.append(rel(p))
    if not arts and not args.allow_empty:
        sys.exit(f"{stage} 단계의 산출물을 --artifact 로 지정해야 한다.\n"
                 f"({STAGE_INFO[stage][1]} 에 떨어진 파일)\n"
                 f"산출물이 정말 없으면 --allow-empty.")

    # runner를 떠날 때는 레코드 지문을 기계로 확인한다.
    if stage == "runner" and arts:
        ok, note = _check_records(arts)
        print(f"레코드 검사: {note}")
        if not ok and not args.force:
            print("\n🔴 변종 빌드가 섞여 있다. 이대로 해석하면 인용할 수 없는 수치가 나온다.")
            print("   레코드를 고치거나, 의도한 변종이면 --force.")
            return 1

    pending = [(g, why) for g, why in GATES.get(nxt, [])
               if not s["gates"].get(g, {}).get("approved")]
    if pending:
        print(f"⚠ {nxt} 로 가려면 승인이 필요하다:")
        for g, why in pending:
            print(f"   `cycle.py approve {g}`  — {why}")
        return 1

    # 산출물은 전진이 확정된 뒤에만 기록한다. 막힌 시도에서 기록하면
    # 재시도할 때 같은 파일이 두 번 쌓인다.
    for a in arts:
        if a not in s["artifacts"][stage]:
            s["artifacts"][stage].append(a)

    s["stage"] = nxt
    log(s, "advance", **{"from": stage, "to": nxt, "artifacts": arts})
    save(s)

    print(f"{cid}: {stage} → {nxt}")

    # runner를 떠났다 = GPU 작업이 끝났다. 박스를 꺼도 되는지 여기서 판정한다.
    if stage == "runner":
        dirty = _uncommitted(arts)
        print()
        if dirty:
            print("🔴 vast.ai 인스턴스를 아직 끄지 마라 — 커밋 안 된 레코드가 있다:")
            for d in dirty:
                print(f"     {d}")
            print("   닫힌 박스가 Block C와 Stage 2 원본을 가져간 적이 있다.")
            print("   커밋·푸시한 뒤 `cycle.py gpu-off` 로 다시 확인한다.")
        else:
            print("✅ 레코드가 전부 커밋됐다 — vast.ai 인스턴스를 꺼도 된다.")
            _release_gpu(s)

    if nxt == "closed":
        print("  `cycle.py close --conclusion supported|refuted|abandoned` 로 마무리")
    else:
        print(f"  다음 산출물: {STAGE_INFO[nxt][1]}")
    save(s)
    return 0


def _release_gpu(state: dict) -> None:
    """GPU 게이트를 내린다. 다음 런은 박스를 다시 열고 다시 승인받아야 한다."""
    if state["gates"].get("gpu", {}).get("approved"):
        state["gates"]["gpu"] = {"approved": False, "at": now(),
                                 "note": "런 종료 후 자동 해제 (박스 꺼도 됨)"}
        log(state, "gpu_released")


def cmd_gpu_off(args) -> int:
    """박스를 꺼도 되는지 확인한다. runner 산출 레코드가 전부 커밋됐는가."""
    cid = args.cycle or active_cycle()
    s = load(cid)
    arts = s["artifacts"].get("runner", [])
    if not arts:
        print("runner 산출 레코드가 없다. 끌 게 있는지 사람이 판단한다.")
        return 0

    dirty = _uncommitted(arts)
    if dirty:
        print("🔴 아직 끄면 안 된다 — 커밋 안 된 레코드:")
        for d in dirty:
            print(f"     {d}")
        return 1

    print(f"✅ 레코드 {len(arts)}건 전부 커밋됨 — vast.ai 인스턴스를 꺼도 된다.")
    for a in arts:
        print(f"     {a}")
    _release_gpu(s)
    save(s)
    return 0


def cmd_feedback(args) -> int:
    cid = args.cycle or active_cycle()
    s = load(cid)
    if args.to not in STAGES[:-1]:
        sys.exit(f"돌아갈 단계가 잘못됐다: {args.to} (가능: {', '.join(STAGES[:-1])})")

    frm = args.from_stage or s["stage"]
    if STAGES.index(args.to) > STAGES.index(frm):
        sys.exit(f"피드백은 앞 단계로만 간다 ({frm} → {args.to} 은 전진이다).")

    fb = {"at": now(), "from": frm, "to": args.to, "reason": args.reason}
    s["open_feedback"].append(fb)
    s["stage"] = args.to
    s["iteration"] += 1
    log(s, "feedback", **{"from": frm, "to": args.to, "reason": args.reason})
    save(s)

    print(f"{cid}: {frm} → {args.to} 로 되돌림 (반복 {s['iteration']}회차)")
    print(f"  사유: {args.reason}")
    print(f"  고친 뒤 `cycle.py resolve` 로 피드백을 닫는다.")
    return 0


def cmd_resolve(args) -> int:
    cid = args.cycle or active_cycle()
    s = load(cid)
    if not s["open_feedback"]:
        print("열린 피드백이 없다.")
        return 0
    closed = s["open_feedback"]
    s.setdefault("resolved_feedback", []).extend(
        [{**f, "resolved_at": now(), "note": args.note or ""} for f in closed])
    s["open_feedback"] = []
    log(s, "resolve", count=len(closed), note=args.note or "")
    save(s)
    print(f"{cid}: 피드백 {len(closed)}건 닫음")
    return 0


def cmd_approve(args) -> int:
    cid = args.cycle or active_cycle()
    s = load(cid)
    s["gates"][args.gate] = {"approved": True, "at": now(), "note": args.note or ""}
    log(s, "approve", gate=args.gate, note=args.note or "")
    save(s)
    print(f"{cid}: 게이트 '{args.gate}' 승인됨")
    return 0


def cmd_close(args) -> int:
    cid = args.cycle or active_cycle()
    s = load(cid)

    if args.conclusion == "supported" and not s["artifacts"].get("verifier"):
        sys.exit("verifier 산출물 없이 supported로 닫을 수 없다.\n"
                 "판정을 받거나(`verify.py`), refuted/abandoned 로 닫는다.")

    s["status"] = "closed"
    s["stage"] = "closed"
    s["conclusion"] = args.conclusion
    s["closed_at"] = now()
    s["closing_note"] = args.note or ""
    log(s, "close", conclusion=args.conclusion, note=args.note or "")
    save(s)

    print(f"{cid} 종료 — {args.conclusion}")
    if args.conclusion == "refuted":
        print("  반증은 실패가 아니라 결과다. 카드 상태를 refuted로 바꾸고,")
        print("  무엇이 반증됐는지 PAPER_PLAN.md 에 반영할지 사람이 정한다.")
    return 0


def cmd_list(args) -> int:
    states = sorted(CYCLES.glob("*/state.json")) if CYCLES.exists() else []
    if not states:
        print("사이클 없음. `cycle.py new --idea <카드>` 로 시작한다.")
        return 0
    for f in states:
        s = json.loads(f.read_text(encoding="utf-8"))
        flag = "🔴" if s["open_feedback"] else ("·" if s["status"] == "active" else "✓")
        concl = f" → {s.get('conclusion')}" if s["status"] == "closed" else ""
        print(f" {flag} {s['cycle_id']}  {s['stage']:<9} 반복{s['iteration']}  "
              f"{s['title']}{concl}")
    return 0


def cmd_routing(args) -> int:
    print("피드백을 어디로 보낼 것인가\n")
    w = max(len(a) for a, _ in ROUTING)
    for symptom, target in ROUTING:
        print(f"  {symptom:<{w}}  →  {target}")
    print("\n전부 planner로 되돌리지 마라. 레코드 하나 고치면 될 일에")
    print("사이클 전체를 다시 도는 것이 이 루프가 망가지는 방식이다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycle", help="사이클 ID (기본: 열려 있는 최신)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="사이클 시작")
    p.add_argument("--idea", required=True, help="가설 카드 경로")
    p.add_argument("--title")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("status", help="현재 상태")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("advance", help="다음 단계로")
    p.add_argument("--artifact", nargs="*", help="이 단계가 만든 파일")
    p.add_argument("--allow-empty", action="store_true")
    p.add_argument("--force", action="store_true", help="레코드 경고/피드백 무시")
    p.set_defaults(fn=cmd_advance)

    p = sub.add_parser("feedback", help="앞 단계로 되돌리기")
    p.add_argument("--to", required=True)
    p.add_argument("--from", dest="from_stage")
    p.add_argument("--reason", required=True)
    p.set_defaults(fn=cmd_feedback)

    p = sub.add_parser("resolve", help="열린 피드백 닫기")
    p.add_argument("--note")
    p.set_defaults(fn=cmd_resolve)

    p = sub.add_parser("approve", help="게이트 승인 (사람)")
    p.add_argument("gate", choices=sorted({g for gs in GATES.values() for g, _ in gs}))
    p.add_argument("--note")
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("gpu-off", help="vast.ai 박스를 꺼도 되는지 확인")
    p.set_defaults(fn=cmd_gpu_off)

    p = sub.add_parser("close", help="사이클 종료")
    p.add_argument("--conclusion", required=True,
                   choices=["supported", "refuted", "abandoned"])
    p.add_argument("--note")
    p.set_defaults(fn=cmd_close)

    p = sub.add_parser("list", help="모든 사이클")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("routing", help="피드백 라우팅 표")
    p.set_defaults(fn=cmd_routing)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
