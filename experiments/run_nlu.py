"""Driver for the NLU sweep -- resumable, skips whatever is already recorded.

The first version of this was a shell loop that redid every cell on restart, which
matters when one cell is 50-80 minutes. This reads the result JSON, works out
which (task, size, arm) pairs are missing, and runs only those. Safe to re-run
after an interruption, and safe to re-run when it has nothing to do.

Each cell is two runs: the ANN baseline, then the ``T=16`` arm -- the one matched
to the paper's global timestep, which is the only arm comparable to Table 2.

    python experiments/run_nlu.py            # continue
    python experiments/run_nlu.py --dry-run  # just say what is missing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON = os.path.join(ROOT, "results", "roberta_nlu.json")
RUNNER = os.path.join(HERE, "roberta_sst2.py")

#: Ordered by checkpoint provenance -- solid public fine-tunes first, so an
#: interrupted sweep still leaves the most defensible cells done.
CELLS = [("mr", "base"), ("sst2", "large"), ("subj", "large"), ("sst5", "large")]


def recorded(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {r.get("tag") for r in json.load(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=JSON)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    have = recorded(a.json)
    # sst2/base lives in the other file from before the sweep existed
    other = os.path.join(ROOT, "results", "roberta_sst2.json")
    print(f"already recorded: {sorted(have) or '(none)'}")
    if os.path.exists(other):
        print(f"(sst2/base is in {os.path.basename(other)}: "
              f"{sorted(recorded(other))})")

    todo = []
    for task, size in CELLS:
        for arm in ("ann", "t16"):
            tag = f"{task}-{size}-{arm}"
            if tag not in have:
                todo.append((task, size, arm, tag))
    if not todo:
        print("nothing to do -- every cell is recorded")
        return
    print(f"\nto run ({len(todo)}):")
    for _, _, _, tag in todo:
        print(f"  {tag}")
    if a.dry_run:
        return

    for task, size, arm, tag in todo:
        cmd = [sys.executable, RUNNER, "--task", task, "--size", size,
               "--json", a.json, "--tag", tag]
        if arm == "ann":
            cmd += ["--backend", "none"]
        else:
            cmd += ["--backend", "mbe_pasn", "--convert-ops", "all",
                    "--epochs", str(a.epochs), "--pasn-id-target", "relative",
                    "--pasn-id-target-rel", "1e-2", "--pasn-t-fixed", "16"]
        print(f"\n##### {tag}  {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            # A missing checkpoint should not take the rest of the sweep with it.
            print(f"!! {tag} exited {rc} -- skipping the rest of this cell",
                  flush=True)
            if arm == "ann":
                break
    print(f"\nNLU SWEEP DONE {time.strftime('%Y-%m-%dT%H:%M:%S')}")


if __name__ == "__main__":
    main()
