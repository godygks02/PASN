"""Tables for the P0.4 GPT-2 x WikiText-2 runs, read off ``results/gpt2_p04.json``.

The runs append one record each from a shell loop, so the deliverable table is a
separate step. This is that step.

Beyond formatting it checks the two things P0.4 got caught by:

* **Duplicate arms** (trap 9). ``pasn_id_target`` was silently ignored under
  ``pasn_id_tied=True`` and the two headline arms came back bit-identical --
  same ppl, spikes, params and bytes -- which reads as a clean null result rather
  than a dead knob. Any two rows agreeing on all four are flagged here.
* **Mixed evaluation recipes.** ``ann-full`` predates the stride fix and is scored
  on stride 512, which is ~17% low and not comparable to anything else. Rows are
  grouped by stride and a warning is printed if more than one appears.

    python experiments/p04_report.py
    python experiments/p04_report.py --json results/gpt2_p04.json --sort spikes
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

DEFAULT_JSON = os.path.join(os.path.dirname(__file__), "..", "results",
                            "gpt2_p04.json")

#: The MBE paper's published NLG numbers (Table 3), the external baseline.
#:
#: **Unit trap.** The table prints ``22.69 (+0.35)`` and the prose calls it "0.35%
#: conversion loss", but 0.35 is the *absolute* perplexity difference. Relative,
#: 22.69/22.34 is **+1.57%** -- 4.5x the quoted figure. Our delta_pct is relative,
#: so it must be compared against the relative column, never against "0.35%".
PAPER_TABLE3 = {
    "WikiText-2": dict(ann=22.34, snn=22.69, T=16),
    "WikiText-103": dict(ann=22.65, snn=23.41, T=16),
}


def paper_rel_pct(name: str) -> float:
    p = PAPER_TABLE3[name]
    return 100.0 * (p["snn"] - p["ann"]) / p["ann"]


def _fmt(v, spec: str, dash: str = "--") -> str:
    return dash if v is None else format(v, spec)


def _stage(r: dict) -> int | None:
    """Conversion stage, inferred for records written before the field existed.

    Those all predate the attention wiring, so a converting row without a
    ``stage`` is necessarily Stage 1. Inferring rather than returning ``None``
    keeps the Table-3 guard airtight: an unlabelled row must never read as
    "maybe Stage 2".
    """
    if r.get("stage") is not None:
        return r["stage"]
    return 1 if r.get("backend") not in (None, "none") else None


def _fingerprint(r: dict) -> tuple:
    """What a row would look like if it were a duplicate of another arm."""
    return (r.get("ppl_snn"), r.get("spikes_per_token"),
            r.get("stored_params"), r.get("stored_bytes"))


def _knobs(r: dict) -> str:
    """Only the knobs that reach this backend -- the rest are recorded as None."""
    bits = []
    for key, label in (("r", "r"), ("id_target", "id"),
                       ("pasn_id_tied", "tied"), ("pasn_id_e_min", "id_emin"),
                       ("n_basis_act", "Nact"), ("n_basis_ln", "Nln"),
                       ("mbe_readout_order", "ord")):
        v = r.get(key)
        if v is not None:
            bits.append(f"{label}={v}")
    return " ".join(bits)


def table(rows: list, title: str) -> None:
    if not rows:
        return
    print(f"\n## {title}")
    print(f"{'tag':<28} {'scope':<10} {'st':>2} {'ppl':>11} {'d%':>10} "
          f"{'spikes/tok':>11} {'params':>7} {'bytes':>7}  knobs")
    print("-" * 122)
    for r in rows:
        print(f"{r.get('tag', '?'):<28} {str(r.get('scope', '?')):<10} "
              f"{_fmt(_stage(r), '2d', ' ?')} "
              f"{_fmt(r.get('ppl_snn'), '11.3f')} "
              f"{_fmt(r.get('delta_pct'), '+10.2f')} "
              f"{_fmt(r.get('spikes_per_token'), '11.4g')} "
              f"{_fmt(r.get('stored_params'), '7d')} "
              f"{_fmt(r.get('stored_bytes'), '7d')}  {_knobs(r)}")


def _paper_section(recs: list) -> None:
    """Place only Stage-2 rows next to the paper's published Table 3.

    Stage 1 leaves attention exact, so its delta-perplexity comes from converting
    roughly half the network's nonlinear work while the paper converted all of
    it. Relative perplexity fixes the *different ANN* problem (ours 21.71, theirs
    22.34, a detokenisation difference); it does **not** fix the different-scope
    problem. So the comparison is gated on the stage, not left to the reader.
    """
    ref = paper_rel_pct("WikiText-2")
    print(f"\n## vs the paper's Table 3 (WikiText-2)")
    print(f"published: ANN {PAPER_TABLE3['WikiText-2']['ann']} -> SNN "
          f"{PAPER_TABLE3['WikiText-2']['snn']} at T=16  =  "
          f"**{ref:+.2f}% relative**")
    print('  (the paper writes "(+0.35)" / "0.35% conversion loss"; 0.35 is the '
          "absolute ppl difference, not a percentage)")

    stage2 = [r for r in recs if _stage(r) == 2
              and r.get("delta_pct") is not None]
    if not stage2:
        print("\n  no Stage-2 rows yet -- nothing here is comparable to Table 3.")
        n1 = sum(1 for r in recs if _stage(r) == 1)
        if n1:
            print(f"  ({n1} Stage-1 row(s) present: attention still exact FP, so "
                  "they convert less than the paper does and must not be quoted "
                  "against it. Use --convert-ops all.)")
        return

    print(f"\n{'tag':<28} {'ours (rel)':>12} {'paper (rel)':>12} {'margin':>10}")
    print("-" * 66)
    for r in sorted(stage2, key=lambda r: r.get("delta_pct")):
        d = r["delta_pct"]
        print(f"{r.get('tag', '?'):<28} {d:+11.2f}% {ref:+11.2f}% "
              f"{ref - d:+9.2f}pp")
    print("\n  Both columns are each method's own ANN baseline, so the differing "
          "absolute perplexities do not enter.")
    print("  Still not matched on T: the paper uses a global T=16, our banks use "
          "per-bank T_j from the budget rule. State it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--sort", choices=["order", "spikes", "ppl"],
                    default="order")
    a = ap.parse_args()

    with open(a.json, encoding="utf-8") as fh:
        recs = json.load(fh)

    key = {"order": lambda r: 0,
           "spikes": lambda r: r.get("spikes_per_token") or 0.0,
           "ppl": lambda r: r.get("ppl_snn") or 0.0}[a.sort]
    if a.sort != "order":
        recs = sorted(recs, key=key)

    ann = [r.get("ppl_ann") for r in recs if r.get("ppl_ann") is not None]
    print(f"# P0.4 -- {len(recs)} runs from {os.path.basename(a.json)}")
    if ann:
        print(f"ANN reference: {min(ann):.4f}"
              + ("" if max(ann) - min(ann) < 1e-6
                 else f" .. {max(ann):.4f}  (differing ANN baselines!)"))

    strides = {r.get("stride") for r in recs}
    if len(strides) > 1:
        print(f"\n!! rows span strides {sorted(s for s in strides if s)} -- "
              "perplexities across different strides are NOT comparable "
              "(stride 512 reads ~17% low; see P0.4 section 1)")

    routed = [r for r in recs if r.get("backend") == "mbe_pasn"]
    glob = [r for r in recs if r.get("backend") == "mbe"]
    none = [r for r in recs if r.get("backend") == "none"]
    table(routed, "MBE-PASN (routed)")
    table(glob, "global MBE (baseline)")
    table(none, "ANN")

    # -- trap 9: two arms that differ only in a knob the code ignored ------
    seen = defaultdict(list)
    for r in recs:
        fp = _fingerprint(r)
        if any(v is None for v in fp):
            continue
        seen[fp].append(r.get("tag", "?"))
    dups = {fp: tags for fp, tags in seen.items() if len(tags) > 1}
    if dups:
        print("\n!! IDENTICAL ARMS -- a knob that was accepted and then ignored "
              "looks exactly like this (trap 9). Build both and diff the state "
              "dicts before reading any of these as a null result:")
        for fp, tags in dups.items():
            print(f"   {tags}  ->  ppl={fp[0]}, spikes={fp[1]}, "
                  f"params={fp[2]}, bytes={fp[3]}")

    # -- pre-P0.5 rows carry the un-de-duplicated byte count -----------------
    # ``stored_bytes_naive`` only exists on records written after the storage fix.
    # Its absence is the marker that a row's ``bytes`` is the old sum, which
    # charged a shared prototype once per bank and so inflated the routed
    # backend alone. Those numbers are not comparable to the corrected ones.
    stale = [r.get("tag") for r in recs
             if r.get("stored_bytes") is not None
             and "stored_bytes_naive" not in r]
    if stale:
        print("\n!! PRE-P0.5 BYTES -- these rows' `bytes` double-count tensors "
              "shared between banks, which inflates mbe_pasn and leaves mbe "
              "untouched. Do not compare them:")
        print(f"   {stale}")
        print("   corrected values: results/p05_bytes.json "
              "(re-measure with --build-only)")

    _paper_section(recs)

    incomplete = [r.get("tag") for r in recs if r.get("ppl_snn") is None
                  and r.get("backend") != "none"]
    if incomplete:
        print(f"\n(build-only or unfinished, no perplexity: {incomplete})")


if __name__ == "__main__":
    main()
