"""E16 -- price both methods with the same rule, and report the multiple as a bracket.

idea-001 is the card. This is the experiment the last lab meeting was promised.

**What is wrong with 5.27x.** `FEEDBACK_2026-08-14.md` §2.1 dissected the headline
energy multiple as **all of MBE divided by part of PASN**: their number is spikes
only on every operator, ours excludes the classes our own accounting charges.
Worse, 93.5% of the margin comes from one term -- the FP multiply -- whose
compare count is *not derivable*, because the paper never says how the two
operands are reconstructed (`E_FP = T^2 * eta_1 * eta_2 * MO` is a coincidence
model, not a circuit). And the two halves move in **opposite** directions: on the
nonlinear operators alone, the as-printed reading has us losing.

**So there is no single honest multiple.** There are two, and the card fixed both
before any arithmetic so that "which assumption was it" cannot be re-litigated:

    E_MBE_nl   MBE, nonlinear 6 ops, all six of OUR classes, per-basis reading
    E_MBE_fp   MBE, FP multiply, the paper's own E_FP. No cmp, no mac -- not derivable
    E_PASN_all PASN, all 7 ops, all six classes, strict, from the record

    bracket A = (E_MBE_nl + E_MBE_fp) / E_PASN_all
    bracket B =  E_MBE_nl             / E_PASN_all      <-- a LOWER BOUND, not a point

`A` puts the same 7 operators on both sides. It mixes a term favourable to us
(the huge coincidence model survives) with one unfavourable (that path's MBE
compares are set to 0). **A is not an upper bound** -- reading "not derivable" as
"zero" is a choice, and if MBE's FP compares are ever derived A goes *up*.

`B` charges MBE for 6 operators and PASN for 7. That asymmetry is deliberate and
points against us. Since `E_PASN_nl <= E_PASN_all`, `B <= E_MBE_nl / E_PASN_nl`,
so the symmetric nonlinear-only truth is **at least** B. Quoting B as a point
estimate would repeat the exact sin 5.27x committed (mismatched scope); quoting
it as a lower bound does not. **The record labels it as a bound.**

`B* = E_MBE_nl / E_PASN_nl` is the symmetric nonlinear-only truth and is **not
computable today**: no record carries PASN's class-by-class ops for the nonlinear
subset. `by_kind` is spikes per op-kind, `ops_per_input` is classes summed over
all ops, and there is no cross-tab. `spiking_cost_report()`'s `primitives` already
holds both, so it is a serialisation job plus one forward -- but it needs the
ImageNet calibration, so it is remote work (Stage 2, R2-gated).

Do **not** print A or B next to 5.27x: that figure is the as-printed reading, and
mixing readings is two incompatible physical descriptions of one neuron.

Usage::  python experiments/e16_symmetric_energy.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mbe.metrics import OP_ENERGY_PJ, op_energy_pj                   # noqa: E402
import energy_ratio_vitm as ER                                       # noqa: E402

#: G1 -- the input record must be the full ViT-M/16 run, not a smoke build. A
#: reduced-epoch build allocates a different budget, so its energy split is a
#: different experiment (idea-001 §4.3).
G1_REQUIRED = dict(tag="e8m-vitm16-t16", epochs=300, n_eval=50000, smoke=False)
G1_N_STEPS = 16


def check_g1(rec: dict) -> dict:
    got = {k: rec.get(k) for k in G1_REQUIRED}
    n_steps = (rec.get("convert_cfg") or {}).get("n_steps")
    fails = [k for k, v in G1_REQUIRED.items() if got[k] != v]
    if n_steps != G1_N_STEPS:
        fails.append(f"convert_cfg.n_steps={n_steps}")
    return dict(required=G1_REQUIRED, required_n_steps=G1_N_STEPS,
                got=got, got_n_steps=n_steps, failures=fails,
                passed=not fails)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=os.path.join(ROOT, "results", "e8m_vit_medium.json"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "e16_symmetric_energy.json"))
    a = ap.parse_args()

    raw = json.load(open(a.record, encoding="utf-8"))
    rec = (raw if isinstance(raw, list) else raw.get("records", [raw]))[0]

    g1 = check_g1(rec)
    if not g1["passed"]:
        print("G1 FAIL -- refusing to judge. " + ", ".join(g1["failures"]))
        print("  (idea-001 G1: a smoke build's budget split is a different experiment)")
        sys.exit(2)

    g = ER.geometry()
    macs = ER.ann_macs(g)
    elems = ER.converted_elements(g)
    e_ann_pj = macs["total"] * ER.E_MAC

    # ---- our side, per image. Both unit conversions are G2's business ------
    # `ops_per_input` / `energy_pj_per_input` are per element of the *model input
    # tensor* -> scale by one image's pixels. `by_kind` is a batch total for one
    # forward -> recover the batch from the record's own two figures.
    px = 3 * g["img"] * g["img"]
    total_spikes = sum(rec["by_kind"].values())
    n_images = round(total_spikes / rec["spikes_per_input"] / px)
    ours_spikes = total_spikes / n_images
    e_pasn_all = rec["energy_pj_per_input"] * px

    # ---- their side, split into the two halves the card defined -------------
    ma = ER.mbe_all_op(elems)                       # nonlinear 6 ops, per-basis
    tot = {k: sum(v[k] for v in ma.values())
           for k in ("cmp", "ac", "mac", "poly", "router", "bitop")}
    e_mbe_nl = op_energy_pj(tot)

    # The FP multiply, the paper's own model. `paper_spikes` deliberately does not
    # apply the per-basis factor here (squaring B across two operands would be our
    # invention), and there is no cmp/mac term because the operand reconstruction
    # is unspecified. Both choices make MBE *smaller*, i.e. they cost us.
    sp_as_printed = ER.paper_spikes(elems, per_basis=False)
    e_mbe_fp = sp_as_printed["attention_score"] * ER.E_AC

    bracket_a = (e_mbe_nl + e_mbe_fp) / e_pasn_all
    bracket_b = e_mbe_nl / e_pasn_all

    # ---- gates, decided in the order the card lists them --------------------
    verdict, note = None, None
    if bracket_a < 1.0:
        verdict, note = "R3_refuted_withdraw", (
            "both brackets below 1.0x. Take energy off the headline and move to "
            "low-T robustness + iso-accuracy operator efficiency (FEEDBACK §7).")
    elif bracket_a >= 3.0:
        verdict, note = "R1s_strong_refuted", (
            "more than half the presented multiple survives symmetric accounting "
            "-> FEEDBACK §2.1's diagnosis is itself wrong and must be rewritten.")
    elif bracket_a >= 2.0:
        verdict, note = "R1_refuted_upper", (
            "the shrink is smaller than claimed; the headline multiple partly "
            "survives.")
    elif bracket_b < 1.0:
        verdict, note = "R2_refuted_lower", (
            "B is a LOWER BOUND, so this alone does NOT establish that the sign "
            "flipped. It triggers Stage 2 (B*), which needs the remote box.")
    else:
        verdict, note = "S_supported", (
            "quote the multiple as an interval only. Never fold it to one number.")

    w = 46
    print("=" * 78)
    print(f"E16 -- symmetric all-op energy, ViT-M/16, global T={ER.T}")
    print("=" * 78)
    print(f"  build          {rec['tag']}  ({g['source']} geometry)")
    print(f"  price table    OP_ENERGY_PJ (45 nm): ac={ER.E_AC} mac={ER.E_MAC} pJ")
    print(f"  unit           per image ({g['tokens']} tokens, d={g['dim']}, "
          f"{g['depth']} blocks)")
    print(f"  E_ANN          {e_ann_pj/1e9:.3f} mJ/image\n")

    print("## The four quantities (same price table, same unit)")
    print(f"{'':<{w}}{'mJ/image':>12}")
    print(f"{'E_MBE_nl   6 nonlin ops, 6 classes, PER-BASIS':<{w}}{e_mbe_nl/1e9:>12.3f}")
    print(f"{'E_MBE_fp   FP multiply, PAPER E_FP, no cmp/mac':<{w}}{e_mbe_fp/1e9:>12.3f}")
    print(f"{'E_PASN_all 7 ops, 6 classes, strict, measured':<{w}}{e_pasn_all/1e9:>12.3f}")
    print(f"{'E_PASN_nl  6 nonlin ops only':<{w}}{'N/A':>12}   <- Stage 2, no record")

    print("\n## The brackets")
    print(f"  bracket A = (E_MBE_nl + E_MBE_fp) / E_PASN_all = {bracket_a:>6.2f}x")
    print("              same 7 ops both sides. NOT an upper bound: MBE's FP")
    print("              compares are set to 0 because they are not derivable.")
    print(f"  bracket B =  E_MBE_nl            / E_PASN_all = {bracket_b:>6.2f}x")
    print("              LOWER BOUND. MBE billed for 6 ops, PASN for 7.")
    print(f"  B* (symmetric nonlinear-only truth)            =    N/A   <- Stage 2")

    print(f"\n## Verdict: {verdict}")
    print(f"  {note}")
    print(f"  interval to quote: {min(bracket_a, bracket_b):.2f}x .. "
          f"{max(bracket_a, bracket_b):.2f}x  (A > B always, since E_MBE_fp > 0)")

    print("\n## Conversion loss, both sides -- this is NOT an iso-accuracy comparison")
    print(f"  PASN  delta_pct      = {rec['delta_pct']:+.4f}%   "
          f"(acc {rec.get('acc_ann')} -> {rec.get('acc_snn')})")
    print(f"  paper paper_rel_pct  = {rec['paper_rel_pct']:+.4f}%")
    print("  ^ idea-001 §(e): the brackets are whole-network totals at different")
    print("    conversion losses. Print these beside every multiple or the")
    print("    'you bought accuracy with spikes' objection cannot be answered.")

    print("\n## Do not do these")
    print("  - do NOT print A or B beside 5.27x (that is the as-printed reading)")
    print("  - do NOT quote B as a point estimate; it is a bound")
    print("  - do NOT move these numbers to GPT-2, T=8, or the per-bank arm")

    rec_out = [dict(
        tag="e16-symmetric-energy", experiment="E16",
        idea="research/ideas/idea-001.md",
        backend="analysis", model=rec["model"],
        convert_cfg=rec.get("convert_cfg"), pasn_beta=rec.get("pasn_beta"),
        stored_bytes=rec.get("stored_bytes"), stored_params=rec.get("stored_params"),
        n_primitives=rec.get("n_primitives"),
        status="ok",
        started=_dt.datetime.now().isoformat(timespec="seconds"),
        source_record=dict(path=os.path.relpath(a.record, ROOT), tag=rec["tag"]),
        operating_point=dict(model="vit_medium_patch16_reg4_gap_256",
                             T=ER.T, arm="pasn_rule (conversion path)",
                             scope="Stage 2 -- activation, LayerNorm, softmax, act x act matmul",
                             geometry_source=g["source"],
                             note="ViT-M/16 -- NOT the GPT-2 frozen build; do not "
                                  "place beside 53,888 B / 13,472 p / 339 prims"),
        # ---- G3: the field names say what was counted, per the card ----------
        energy_mJ_per_image=dict(
            E_MBE_nl__6nonlinear_ops__6classes__per_basis_reading=e_mbe_nl / 1e9,
            E_MBE_fp__fp_multiply__paper_E_FP__no_cmp_no_mac=e_mbe_fp / 1e9,
            E_PASN_all__7ops__6classes__strict_measured=e_pasn_all / 1e9,
            E_PASN_nl__6nonlinear_ops_only=None,
            E_ANN=e_ann_pj / 1e9),
        brackets=dict(
            A__same_7_ops_both_sides=bracket_a,
            A_is_not_an_upper_bound=("MBE's FP-path compares are set to 0 because "
                                     "the paper does not define the operand "
                                     "reconstruction. If they are ever derived, A rises."),
            B__lower_bound_only=bracket_b,
            B_is_a_bound_because=("E_PASN_nl <= E_PASN_all, so "
                                  "B <= E_MBE_nl / E_PASN_nl. Quoting B as a point "
                                  "estimate repeats 5.27x's scope mismatch."),
            B_star__symmetric_nonlinear_only=None,
            B_star_blocked_by=("no record cross-tabulates PASN's op classes with "
                               "op kind. Needs spiking_cost_report primitives "
                               "serialised + one forward, which needs the ImageNet "
                               "calibration -> remote. Stage 2, R2-gated."),
            interval=[min(bracket_a, bracket_b), max(bracket_a, bracket_b)]),
        mbe_nonlinear_ops_by_class=tot,
        mbe_nonlinear_by_op={k: dict(v) for k, v in ma.items()},
        pasn_ops_by_class_per_image={k: v * px for k, v in rec["ops_per_input"].items()},
        pasn_spikes_per_image=ours_spikes,
        unit_conversion=dict(pixels_per_image=px, batch_recovered=n_images,
                             note="G2: both sides pass op_energy_pj and are per image"),
        conversion_loss_pct=dict(pasn=rec["delta_pct"], paper=rec["paper_rel_pct"],
                                 caveat="not an iso-accuracy comparison (idea-001 §e)"),
        gates=dict(G1=g1,
                   G2_same_price_table_and_unit=True,
                   G3_readings_named_in_fields=True,
                   G4_brackets_serialised=True,
                   G_paper_transcription=dict(
                       source="paper/appendix_p15.png via energy_ratio_vitm.PAPER_FIRING/PAPER_BASIS",
                       cross_checked_against_vault=True,
                       vault_note="PASN_vault/60 - 연구일지/Table XI 발화율 대조.md",
                       matched="7/7 firing rates identical (8.31 / 46.94 / 3.74 / "
                               "25.27 / 27.61 / 38.46 / 38.22 %)",
                       limitation=("both are transcriptions of the same appendix "
                                   "image, so this catches a typo, NOT a misreading "
                                   "of the source. The whole bracket still rests on "
                                   "those 7 numbers."))),
        verdict=verdict, verdict_note=note,
        do_not=["do not print A or B beside the 5.27x spike-only figure "
                "(different reading of the same paper)",
                "do not quote B as a point estimate",
                "do not carry these to GPT-2, T=8, or the per-bank T_j arm"],
    )]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec_out, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
