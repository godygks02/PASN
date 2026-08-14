#!/usr/bin/env python
"""레코드가 어느 빌드에서 나왔는지 판정한다.

이 프로젝트는 한 번 데였다: `pasn_beta=0.5` 기본값이 모든 GPT-2 레코드보다 뒤에
들어와서, 헤드라인 레코드의 빌드를 소급 판정할 수 없게 됐다(E0가 그걸 고치느라
생겼다). 그 뒤로 규칙은 "빌드를 바꾸는 손잡이는 전부 레코드에 적는다"이지만,
규칙은 사람이 잊는다. 이 스크립트는 그 대조를 기계에 맡긴다.

    python research/tools/check_build.py                    # results/ 전체 감사
    python research/tools/check_build.py results/freeze_e1.json
    python research/tools/check_build.py --json             # 기계 판독용

종료 코드: 0 = 전부 판정됨, 1 = 미상 빌드 또는 지문 불일치가 있음.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDS = REPO / "research" / "builds.json"

# Windows 콘솔 기본이 cp949라 한글/em-dash에서 죽는다. 출력만 UTF-8로 돌린다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def load_builds() -> dict:
    with BUILDS.open(encoding="utf-8") as fh:
        return json.load(fh)


def iter_records(payload):
    """레코드 JSON은 리스트이거나 단일 dict다. 둘 다 받는다."""
    if isinstance(payload, list):
        for i, rec in enumerate(payload):
            if isinstance(rec, dict):
                yield i, rec
    elif isinstance(payload, dict):
        yield 0, payload


def classify(rec: dict, builds: dict) -> tuple[str, list[str]]:
    """(빌드명, 불일치 사유) — 빌드명이 'unknown'이면 지문이 부족하거나 안 맞는다."""
    notes: list[str] = []
    for name, spec in builds["builds"].items():
        fp = spec["fingerprint"]
        present = {k: rec.get(k) for k in fp if rec.get(k) is not None}
        if not present:
            continue
        if all(rec.get(k) == v for k, v in fp.items() if rec.get(k) is not None):
            # 지문은 맞다. beta 손잡이가 레코드에 있으면 그것도 대조한다.
            if "pasn_beta" in rec and rec["pasn_beta"] != spec.get("pasn_beta"):
                notes.append(
                    f"지문은 {name}인데 pasn_beta={rec['pasn_beta']!r} "
                    f"(기대 {spec.get('pasn_beta')!r})"
                )
                return "unknown", notes
            return name, notes
    return "unknown", notes


def audit_record(path: Path, idx: int, rec: dict, builds: dict) -> dict:
    missing = [f for f in builds["provenance_required_fields"] if f not in rec]
    scope_model = builds.get("fingerprint_scope_model")

    # bytes/params 지문은 모델 구조에서 나온다. 다른 모델 레코드에 들이대면
    # 전부 불일치로 뜨고, 그렇게 무뎌진 경보는 아무도 안 본다.
    if scope_model and rec.get("model") not in (None, scope_model):
        return {
            "file": str(path.relative_to(REPO)).replace("\\", "/"),
            "index": idx,
            "tag": rec.get("tag"),
            "build": f"n/a ({rec.get('model')})",
            "status": "n/a",
            "citable": True,
            "must_name_build": False,
            "missing_provenance": missing,
            "notes": [],
            "delta_pct": rec.get("delta_pct"),
        }

    # ANN 기준런(backend='none')은 SNN 빌드가 아예 없다. 대조 대상이 아니다.
    has_fp = any(rec.get(k) is not None
                 for spec in builds["builds"].values()
                 for k in spec["fingerprint"])
    if not has_fp:
        return {
            "file": str(path.relative_to(REPO)).replace("\\", "/"),
            "index": idx,
            "tag": rec.get("tag"),
            "build": f"n/a (backend={rec.get('backend')})",
            "status": "n/a",
            "citable": True,
            "must_name_build": False,
            "missing_provenance": [],
            "notes": [],
            "delta_pct": rec.get("delta_pct"),
        }

    build, notes = classify(rec, builds)
    spec = builds["builds"].get(build, {})
    citable = bool(spec.get("citable"))
    if build == builds["frozen"]:
        status = "frozen"
    elif build == "unknown":
        status = "unknown"
    else:
        status = "off-freeze"

    return {
        "file": str(path.relative_to(REPO)).replace("\\", "/"),
        "index": idx,
        "tag": rec.get("tag"),
        "build": build,
        "status": status,
        "citable": citable,
        "must_name_build": bool(spec.get("must_name_build")),
        "missing_provenance": missing,
        "notes": notes,
        "delta_pct": rec.get("delta_pct"),
    }


def is_record_file(path: Path) -> bool:
    """스윕 레코드로 보이는 JSON만 고른다 (지문 필드를 하나라도 가진 것)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False
    for _, rec in iter_records(payload):
        if any(k in rec for k in ("stored_bytes", "stored_params", "convert_cfg")):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path,
                    help="레코드 JSON. 생략하면 results/ 전체를 감사한다.")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="판정을 JSON으로 출력 (에이전트/CI 용)")
    ap.add_argument("--strict", action="store_true",
                    help="프리즈 빌드가 아닌 레코드도 실패로 센다")
    args = ap.parse_args()

    builds = load_builds()

    if args.paths:
        targets = [p if p.is_absolute() else REPO / p for p in args.paths]
    else:
        targets = sorted(p for p in (REPO / "results").glob("*.json")
                         if is_record_file(p))

    rows: list[dict] = []
    for path in targets:
        if not path.exists():
            print(f"없는 파일: {path}", file=sys.stderr)
            return 2
        payload = json.loads(path.read_text(encoding="utf-8"))
        for idx, rec in iter_records(payload):
            rows.append(audit_record(path, idx, rec, builds))

    if args.as_json:
        print(json.dumps({"frozen": builds["frozen"], "records": rows},
                         ensure_ascii=False, indent=1))
    else:
        by_status: dict[str, int] = {}
        for r in rows:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1

        width = max((len(r["file"]) for r in rows), default=20)
        mark = {"frozen": "OK  ", "off-freeze": "NAME",
                "unknown": "??  ", "n/a": "--  "}
        for r in rows:
            tag = r["tag"] or "-"
            print(f"{mark[r['status']]} {r['file']:<{width}} [{r['index']}] "
                  f"{tag:<24} build={r['build']}")
            if r["must_name_build"]:
                print("       └ 인용 시 빌드 이름을 반드시 붙일 것")
            for n in r["notes"]:
                print(f"       └ {n}")
            if r["missing_provenance"]:
                print(f"       └ 출처 필드 누락: {', '.join(r['missing_provenance'])}")

        print()
        print(f"레코드 {len(rows)}건 — " +
              ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
        print(f"프리즈 빌드 = {builds['frozen']}  "
              f"({builds['builds'][builds['frozen']]['anchor_record']})")

    bad = sum(1 for r in rows if r["status"] == "unknown")
    if args.strict:
        bad += sum(1 for r in rows if r["status"] == "off-freeze")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
