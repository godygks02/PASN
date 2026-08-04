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


#: The paper prices energy off the spike count alone:
#:
#:     E_MBE = T * eta * N * C * N_h * E_AC        (appendix G.4)
#:
#: with ``E_AC = 0.9 pJ``, which is the same constant ``mbe.metrics.OP_ENERGY_PJ``
#: uses. So "spikes x E_AC" is not a shortcut we invented -- it is the baseline's
#: own currency, and quoting it makes the two methods commensurable.
#:
#: Our own accounting (P0.5, the ``energy_pj_per_input`` field) is **stricter**: it
#: additionally charges the threshold compare that happens every timestep whether
#: or not the neuron fires, the readout MACs, the order-2 powers, and the router.
#: Reporting both is the point -- a saving that survives the harsher accounting is
#: not an artefact of the metric, and the paper's convention is the one that
#: flatters us.
E_AC_PJ = 0.9

#: Conversion loss for the Stage-2 arms, keyed by the configuration that produced
#: it: ``(pasn_t_fixed, pasn_n_fixed)`` at ``convert-ops all``, relative identity,
#: ``r=1e-2``.
#:
#: **Why this is a constant and not read off a record.** The perplexity runs and
#: the post-softmax-fix cost re-measures are separate runs, and the raw records of
#: the former went down with a closed vast.ai box -- they survive only because
#: they were transcribed into the research log. Every cost row for these arms is
#: therefore ``--build-only`` and carries no ``delta_pct``, which would leave the
#: one comparison worth making -- the solved budget against the paper's own N=4,
#: at equal accuracy -- impossible to state.
#:
#: **Why borrowing it across runs is sound here, when borrowing spikes is not.**
#: Conversion loss reproduced exactly on new hardware (-0.140% against a recorded
#: -0.14%) while spike totals moved 0.945x. Accuracy travels between boxes;
#: absolute spike counts do not. Rows filled from this table are marked ``*``.
MEASURED_DELTA = {
    (None, None): -0.07,          # per-bank T_j and N_j, the budget rule
    (16, None): -0.14,            # timestep-matched to the paper's global T=16
    (None, 4): -0.09,             # basis-matched to the paper's own N=4
}


def measured_delta(r: dict) -> tuple[float | None, bool]:
    """``(conversion loss, borrowed)`` -- borrowed means it came from the table."""
    if r.get("delta_pct") is not None:
        return r["delta_pct"], False
    if (r.get("scope") == "all" and r.get("backend") == "mbe_pasn"
            and r.get("id_target") == "relative" and r.get("r") == 1e-2):
        key = (r.get("pasn_t_fixed"), r.get("pasn_n_fixed"))
        if key in MEASURED_DELTA:
            return MEASURED_DELTA[key], True
    return None, False


def paper_rel_pct(name: str) -> float:
    p = PAPER_TABLE3[name]
    return 100.0 * (p["snn"] - p["ann"]) / p["ann"]


def paper_energy_pj(r: dict) -> float | None:
    """Energy per input element in the paper's currency: spikes x E_AC."""
    s = r.get("spikes_per_token")
    return None if s is None else s * E_AC_PJ


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


def _energy_section(recs: list, tol: float) -> None:
    """Energy in both currencies, and the iso-accuracy ratios that follow.

    Two things this exists to stop.

    *Quoting a ratio against a baseline that fell over.* The 5.45x figure was
    measured against our own global-MBE reimplementation at a setting where it
    diverged (ppl 611 against an ANN 21.71), so it is a ratio between a working
    model and a broken one, not an energy result. Only arms whose conversion loss
    actually agrees may be divided, which is what the iso-accuracy grouping below
    enforces.

    *Quoting a ratio built from arms measured in different places.* Absolute spike
    totals moved 0.945x on new hardware -- softmax alone 0.855x, bytes and
    delta-perplexity identical -- so a ratio is only meaningful when both of its
    arms were built together. That cannot be detected from the records, hence the
    warning rather than a check.
    """
    priced = [r for r in recs if r.get("spikes_per_token") is not None]
    if not priced:
        return

    print("\n## energy, two currencies")
    print(f"paper (G.4): spikes x E_AC, E_AC = {E_AC_PJ} pJ -- accumulate only")
    print("ours (P0.5): + threshold compares, readout MACs, order-2 powers, router")
    print(f"\n{'tag':<28} {'d%':>9} {'spikes/tok':>11} {'paper pJ':>11} "
          f"{'ours pJ':>11} {'ours/paper':>10}")
    print("-" * 85)
    borrowed_any = False
    for r in priced:
        pe, oe = paper_energy_pj(r), r.get("energy_pj_per_input")
        ratio = None if not (pe and oe) else oe / pe
        d, borrowed = measured_delta(r)
        borrowed_any |= borrowed
        dcol = ("       --" if d is None
                else f"{d:+8.2f}{'*' if borrowed else ' '}")
        print(f"{r.get('tag', '?'):<28} {dcol} "
              f"{r['spikes_per_token']:11.4g} {_fmt(pe, '11.4g')} "
              f"{_fmt(oe, '11.4g')} {_fmt(ratio, '10.2f')}")
    if borrowed_any:
        print("\n  * conversion loss taken from the perplexity run of the same "
              "configuration (MEASURED_DELTA): these rows are --build-only, and "
              "accuracy -- unlike the spike count -- reproduces across boxes.")

    # -- iso-accuracy groups: only these may be divided ----------------------
    cmp_rows = []
    for r in priced:
        d, _ = measured_delta(r)
        if d is not None:
            cmp_rows.append((d, r))
    groups: list[list] = []
    for d, r in sorted(cmp_rows, key=lambda t: t[0]):
        r = dict(r, delta_pct=d)
        if groups and abs(d - groups[-1][0]["delta_pct"]) <= tol:
            groups[-1].append(r)
        else:
            groups.append([r])

    usable = [g for g in groups if len(g) > 1]
    if usable:
        print(f"\n### iso-accuracy energy ratios (conversion loss within {tol} pp)")
        for g in usable:
            base = min(g, key=lambda r: r["spikes_per_token"])
            print(f"\n  baseline: {base.get('tag', '?')}  "
                  f"(d {base['delta_pct']:+.2f}%, {base['spikes_per_token']:.4g} "
                  "spikes/tok)")
            for r in sorted(g, key=lambda r: -r["spikes_per_token"]):
                if r is base:
                    continue
                ps = r["spikes_per_token"] / base["spikes_per_token"]
                oe, be = r.get("energy_pj_per_input"), base.get(
                    "energy_pj_per_input")
                os_ = None if not (oe and be) else oe / be
                by = (None if not (r.get("stored_bytes")
                                   and base.get("stored_bytes"))
                      else r["stored_bytes"] / base["stored_bytes"])
                def x(v):                    # "2.259x", or "--" when unpriced
                    return "--" if v is None else f"{v:.3f}x"
                print(f"    vs {r.get('tag', '?'):<24} "
                      f"d {r['delta_pct']:+.2f}%  "
                      f"paper {ps:.3f}x  ours {x(os_):<8} bytes {x(by)}")
        print("\n  'paper' is the spikes-only ratio, 'ours' the priced one. Ours "
              "is usually the *smaller* number: the router costs the same per "
              "element whatever N is, so it is a fixed floor that dilutes the "
              "cheaper arm. Reporting the smaller figure is the honest move.")
    else:
        print("\n(no two rows agree on conversion loss within "
              f"{tol} pp -- nothing here may be divided into an energy ratio)")

    print("\n  !! Ratios are only valid between arms BUILT TOGETHER. Absolute "
          "spike totals are environment-dependent (0.945x on new hardware, "
          "softmax 0.855x, bytes and delta-ppl unchanged), and even a same-box "
          "ratio only survives when the shift hits both arms alike -- it did for "
          "T_j (1.065 -> 1.072) and did not for N_j (2.392 -> 2.551).")
    print("  !! Memory access and data movement are in neither currency. The "
          "paper does not count them either, so the comparison is fair, but "
          "routing plausibly has worse locality than a global neuron, so the "
          "missing term may run against us. Say so rather than implying a "
          "complete energy model.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", nargs="+", default=[DEFAULT_JSON],
                    help="one or more result files. Several are read as one "
                         "table, which is how the same-box decomposition arms "
                         "(each written to its own file) get compared")
    ap.add_argument("--sort", choices=["order", "spikes", "ppl"],
                    default="order")
    ap.add_argument("--iso-tol", type=float, default=0.10, metavar="PP",
                    help="conversion losses within this many pp count as equal "
                         "for the energy ratios. The default is half the 0.197 pp "
                         "band the same model spans across evaluation strides, so "
                         "it is inside the measured noise of the quantity itself")
    a = ap.parse_args()

    recs = []
    for path in a.json:
        with open(path, encoding="utf-8") as fh:
            recs.extend(json.load(fh))

    key = {"order": lambda r: 0,
           "spikes": lambda r: r.get("spikes_per_token") or 0.0,
           "ppl": lambda r: r.get("ppl_snn") or 0.0}[a.sort]
    if a.sort != "order":
        recs = sorted(recs, key=key)

    ann = [r.get("ppl_ann") for r in recs if r.get("ppl_ann") is not None]
    print(f"# P0.4 -- {len(recs)} runs from "
          f"{', '.join(os.path.basename(p) for p in a.json)}")
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
    _energy_section(recs, a.iso_tol)

    incomplete = [r.get("tag") for r in recs if r.get("ppl_snn") is None
                  and r.get("backend") != "none"]
    if incomplete:
        print(f"\n(build-only or unfinished, no perplexity: {incomplete})")


if __name__ == "__main__":
    main()
