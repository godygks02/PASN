#!/usr/bin/env python
"""E21 -- prototype sharing by *reference* vs by ``deepcopy``: the build-identity gate.

``ConvertConfig.share_fits`` already fits one activation prototype and one LayerNorm
prototype per network and reuses them at every other site -- but it inserts them with
``copy.deepcopy``, so 24 activation sites and 49 LayerNorm modules each store a
byte-identical copy. ``storage_bytes`` de-duplicates by storage, so references are
charged once and copies are charged every time (measured: one neuron 100 B, two
deepcopies 200 B, the same reference twice 100 B). By the E11 audit's arithmetic
**1,143 of the frozen build's 1,832 stored bases are duplicates.**

``ConvertConfig.share_fit_refs`` inserts the prototype by reference instead. The two
arms are then *mathematically the same network*, so this script is not a sweep: it is
a gate that asks whether the only thing that moved is the stored footprint.

Pre-registered gates (``research/ideas/idea-002.md``; do not widen them afterwards)::

    R1  stored_params(ref) <= 5500          # 689 bases x 7.3537 ~ 5067, +8% headroom
    R2  spikes_per_input / total_spikes     # EXACTLY equal, not approximately
    R3  by_kind                             # max relative move <= 1e-4 (0.01%)
    R5  per-site parameters, model logits   # max |delta| == 0
    R4  delta-PPL within 0.0064 pp          # NOT here -- needs the GPU eval

R5 is this script's own addition and is the strongest of the four: reference sharing
reads the same tensors, so any difference at all is a wiring bug rather than an error.

Two things are *expected* to move and are therefore not failures:

* ``n_primitives`` drops (339 -> ~172 on gpt2-medium). ``_spiking_primitives``
  de-duplicates by object id, so shared neurons are listed once. **This changes one of
  the three frozen-fingerprint fields**, which is why ``share_fit_refs`` defaults off
  and why promoting this build is a human decision.
* Per-site spike attribution collapses: one probe now accumulates 24 sites. Totals
  survive (R2 checks exactly that), the per-layer split does not.

Usage::

    python experiments/shared_refs_gate.py --smoke                 # wiring, seconds
    python experiments/shared_refs_gate.py --json results/e21_shared_refs.json

Local CPU, no GPU: both arms are ``--build-only`` equivalents. If R1/R2/R3/R5 fail
here, do not spend the GPU on R4.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import (make_spikable, make_attention_spikable,  # noqa: E402
                              convert_gpt2)
from gpt2_wikitext import (build_smoke, load_wikitext_ids,  # noqa: E402
                           storage_report, write_record)

_DEF = cv.ConvertConfig()

#: Wrapper types the conversion inserts. Walking these gives one entry per *site*
#: even in the reference arm, where ``named_modules`` would collapse the shared
#: prototypes -- the wrappers themselves are always distinct objects.
_WRAPPERS = (cv._SpikingActModule, cv._SpikingLayerNormModule,
             cv._SpikingSoftmaxModule, cv._SpikingMatMulModule)


def cfg_snapshot(cfg) -> dict:
    """Every ``ConvertConfig`` field, JSON-coerced (same rule as gpt2_wikitext)."""
    out = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, (str, int, float, bool, type(None))):
            out[f.name] = v
        elif isinstance(v, dict):
            out[f.name] = {str(k): (x if isinstance(
                x, (str, int, float, bool, type(None))) else str(x))
                for k, x in v.items()}
        elif isinstance(v, (list, tuple)):
            out[f.name] = [x if isinstance(
                x, (str, int, float, bool, type(None))) else str(x) for x in v]
        else:
            out[f.name] = str(v)
    return out


def load_inputs(args, device):
    """``(model, calib, ids, block)`` -- the freeze build's calibration draw."""
    if args.smoke:
        model, ids, calib, block = build_smoke()
        return model.to(device).eval(), calib, ids, block
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    ids = load_wikitext_ids(tok, "test", config=args.dataset)
    start = args.calib_offset * args.block
    need = start + 8 * args.block
    cids = load_wikitext_ids(tok, "train", config=args.dataset)[
        : max(64 * args.block, need)]
    if cids.numel() < need:
        raise SystemExit(f"--calib-offset {args.calib_offset} needs {need} tokens")
    calib = [cids[i:i + args.block].unsqueeze(0)
             for i in range(start, need, args.block)]
    return model, calib, ids, args.block


def build_arm(share_refs: bool, args, device):
    """Convert a fresh model with ``share_fit_refs`` on or off; measure everything."""
    label = "ref" if share_refs else "copy"
    _SCOPE_KINDS = {"all": None, "both": {"activation", "layernorm"},
                    "activation": {"activation"}, "layernorm": {"layernorm"}}
    only = _SCOPE_KINDS[args.convert_ops]
    model, calib, ids, block = load_inputs(args, device)
    make_spikable(model)
    if args.convert_ops == "all":
        make_attention_spikable(model)
    cfg = cv.ConvertConfig(
        epochs=args.epochs, backend=args.backend, n_steps=args.n_steps,
        spike_mult=True, n_basis_act=args.n_basis_act,
        mbe_readout_order=args.mbe_readout_order,
        pasn_t_fixed=args.pasn_t_fixed, pasn_id_target=args.pasn_id_target,
        pasn_id_target_rel=args.pasn_id_target_rel,
        pasn_id_tied=args.pasn_id_tied,
        share_fit_refs=share_refs, verbose_fits=args.verbose_fits,
    )
    print(f"\n=== arm [{label}] share_fit_refs={share_refs} ===", flush=True)
    t0 = time.perf_counter()
    convert_gpt2(model, calib, cfg=cfg, only=only, verbose=False)
    build_s = time.perf_counter() - t0
    print(f"[{label}] conversion took {build_s / 60:.1f} min", flush=True)

    batch = ids[:block].unsqueeze(0).to(device)
    store = storage_report(model)
    spikes = cv.spiking_cost_report(model, batch)
    with torch.no_grad():
        logits = model(batch).logits.detach().float().cpu()

    # Site *paths*, not module objects: ``named_modules`` de-duplicates shared
    # instances, so the reference arm would report 1 activation site instead of 24.
    # Both arms have the same module tree, so the copy arm's paths are the index.
    site_names = [name for name, mod in model.named_modules()
                  if isinstance(mod, _WRAPPERS)]
    return dict(label=label, model=model, cfg=cfg, store=store, spikes=spikes,
                logits=logits, site_names=site_names, build_s=build_s, batch=batch)


def unique_neuron_ids(arm: dict, site_names) -> dict:
    """Distinct object ids per shared-prototype family -- did sharing survive ``.to``?

    ``Module.to`` is a no-op when device and dtype already match (it returns the same
    tensor), so a shared prototype stays shared. If a dtype cast ever crept in, each
    site would silently get its own converted tensors and the whole saving would
    evaporate *without* any numeric difference -- this is the check that catches it.

    Resolved by path, not by walking ``named_modules``: that walk de-duplicates
    repeated modules, so in the reference arm it would visit each shared prototype
    once and undercount the sites it is asked about.
    """
    act, ln, sites = set(), set(), 0
    for name in site_names:
        mod = arm["model"].get_submodule(name)
        sites += 1
        if isinstance(mod, cv._SpikingActModule):
            act.add(id(mod.act.neuron))
        elif isinstance(mod, cv._SpikingLayerNormModule):
            ln.add(id(mod.ln))
    return dict(sites=sites, activation_objects=len(act),
                layernorm_objects=len(ln))


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-for-byte tensor equality, NaN and -0.0 included.

    ``torch.equal`` and ``(a-b).abs().max()`` both report a *difference* between two
    NaNs, so on a backend that diverges (the global ``mbe`` arm blows GPT-2's
    perplexity up -- E2) they would fail a build that is in fact identical. Comparing
    the raw bytes asks the question this gate actually means: is it the same tensor.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    ba = a.detach().cpu().contiguous().view(torch.uint8)
    bb = b.detach().cpu().contiguous().view(torch.uint8)
    return bool(torch.equal(ba, bb))


def finite_max_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    """``max |a-b|`` over entries finite in both -- a readable diagnostic only.

    The gate is :func:`bitwise_equal`; this number is what a human reads when it
    fails. ``nan`` here with ``bitwise_equal`` true means both sides are non-finite
    in the same places, which is agreement, not disagreement.
    """
    d = (a.detach().float() - b.detach().float()).abs()
    m = torch.isfinite(d)
    return float(d[m].max()) if bool(m.any()) else float("nan")


def compare_params(a: dict, b: dict) -> dict:
    """Max |delta| over every stored tensor, site by site (R5).

    The wrappers exist at identical paths in both arms, so this pairs a per-site
    deepcopy against the shared prototype it was copied from. Both sides are resolved
    with ``get_submodule`` because ``named_modules`` hides the repeats in the
    reference arm (that is the sharing, not a missing site).
    """
    worst, worst_site, checked, missing, differing = 0.0, None, 0, [], []
    for name in a["site_names"]:
        mod_a = a["model"].get_submodule(name)
        try:
            mod_b = b["model"].get_submodule(name)
        except AttributeError:
            missing.append(name)
            continue
        sd_a, sd_b = mod_a.state_dict(), mod_b.state_dict()
        if sd_a.keys() != sd_b.keys():
            missing.append(f"{name}: key mismatch")
            continue
        for k, t_a in sd_a.items():
            t_b = sd_b[k]
            if not torch.is_tensor(t_a) or t_a.shape != t_b.shape:
                missing.append(f"{name}.{k}: shape")
                continue
            checked += 1
            if not bitwise_equal(t_a, t_b):
                differing.append(f"{name}.{k}")
                d = finite_max_delta(t_a, t_b)
                if not (d != d) and d > worst:      # skip nan
                    worst, worst_site = d, f"{name}.{k}"
    return dict(bitwise_equal=(not differing), n_differing=len(differing),
                differing=differing[:10],
                max_abs_delta=worst, worst_tensor=worst_site,
                tensors_checked=checked, unpaired=missing[:10],
                n_unpaired=len(missing))


def compare_spikes(a: dict, b: dict) -> dict:
    ka, kb = a["spikes"]["by_kind"], b["spikes"]["by_kind"]
    kinds = sorted(set(ka) | set(kb))
    rel = {}
    for k in kinds:
        va, vb = ka.get(k, 0.0), kb.get(k, 0.0)
        rel[k] = abs(va - vb) / max(abs(va), 1e-30)
    return dict(
        total_spikes_copy=a["spikes"]["total_spikes"],
        total_spikes_ref=b["spikes"]["total_spikes"],
        total_spikes_equal=(a["spikes"]["total_spikes"]
                            == b["spikes"]["total_spikes"]),
        spikes_per_input_copy=a["spikes"]["spikes_per_input"],
        spikes_per_input_ref=b["spikes"]["spikes_per_input"],
        spikes_per_input_equal=(a["spikes"]["spikes_per_input"]
                                == b["spikes"]["spikes_per_input"]),
        by_kind_max_rel=max(rel.values()) if rel else 0.0,
        by_kind_rel=rel,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random GPT-2 (2 layers, d=32): wiring check only. "
                         "R1 is not evaluated -- the fingerprint is meaningless here")
    ap.add_argument("--backend", default="mbe_pasn")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-steps", type=int, default=_DEF.n_steps)
    ap.add_argument("--n-basis-act", type=int, default=_DEF.n_basis_act,
                    help="6 is the repo default; G.1 says 4 for GELU/Tanh")
    ap.add_argument("--convert-ops", default="all",
                    choices=["all", "both", "activation", "layernorm"],
                    help="conversion scope. 'all' = Stage 2 (the frozen build); "
                         "'both' = Stage 1, GELU + LayerNorm only. Only matters "
                         "when chasing an older record's provenance")
    ap.add_argument("--mbe-readout-order", type=int,
                    default=_DEF.mbe_readout_order,
                    help="global-MBE decoder order. P0.4's 'matched' rows use 2 (to "
                         "match PASN's order-2 decoder); the ConvertConfig default "
                         "is 1. It reaches rsqrt / exp2 / inv only -- identities "
                         "stay order 1 (the FP multiply factorises a pure spike "
                         "sum) and the signed activation path discards it")
    ap.add_argument("--pasn-t-fixed", type=int, default=16)
    # Identity tying is on by default and every GPT-2 record carries it, so the
    # untied network build has never been measured. It is the arm that says what
    # tying is worth on the axis under attack: the identity is the only tied
    # primitive, and it is 45.1% of this model's spikes.
    ap.add_argument("--pasn-id-tied", action=argparse.BooleanOptionalAction,
                    default=_DEF.pasn_id_tied)
    ap.add_argument("--pasn-id-target", default="relative")
    ap.add_argument("--pasn-id-target-rel", type=float, default=1e-2)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--calib-offset", type=int, default=0)
    ap.add_argument("--dataset", default="wikitext-2-raw-v1")
    ap.add_argument("--verbose-fits", action="store_true")
    ap.add_argument("--json", default=None, metavar="PATH")
    ap.add_argument("--tag", default="e21-sharedrefs")
    # Pre-registered thresholds. Exposed so the record says what they were, NOT so
    # they can be relaxed after seeing the result (idea-002 forbids that).
    ap.add_argument("--gate-stored-params", type=int, default=5500)
    ap.add_argument("--gate-by-kind-rel", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[e21] device={device}  model={'smoke' if args.smoke else args.model}")

    arm_copy = build_arm(False, args, device)
    arm_ref = build_arm(True, args, device)

    params = compare_params(arm_copy, arm_ref)
    spikes = compare_spikes(arm_copy, arm_ref)
    logits_same = bitwise_equal(arm_copy["logits"], arm_ref["logits"])
    logit_delta = finite_max_delta(arm_copy["logits"], arm_ref["logits"])
    logits_finite = bool(torch.isfinite(arm_copy["logits"]).all())
    # The copy arm's paths index both arms -- see ``unique_neuron_ids``.
    sharing_copy = unique_neuron_ids(arm_copy, arm_copy["site_names"])
    sharing_ref = unique_neuron_ids(arm_ref, arm_copy["site_names"])

    sc, sr = arm_copy["store"], arm_ref["store"]
    gates = {
        "R1_stored_params": {
            # Pre-registered for the *frozen PASN build* (689 bases x 7.3537). It has
            # no meaning for a smoke model or for another backend, and stretching it
            # to cover them would be the same thing as widening a gate after the
            # fact. Those runs report the number and skip the verdict.
            "value": sr["params"], "threshold": args.gate_stored_params,
            "pass": (None if (args.smoke or args.backend != "mbe_pasn")
                     else sr["params"] <= args.gate_stored_params),
            "note": ("not evaluated: --smoke" if args.smoke else
                     f"not evaluated: threshold is for mbe_pasn, backend is "
                     f"{args.backend}" if args.backend != "mbe_pasn" else ""),
        },
        "R2_spikes_exact": {
            "value": [spikes["spikes_per_input_equal"],
                      spikes["total_spikes_equal"]],
            "pass": bool(spikes["spikes_per_input_equal"]
                         and spikes["total_spikes_equal"]),
        },
        "R3_by_kind": {
            "value": spikes["by_kind_max_rel"], "threshold": args.gate_by_kind_rel,
            "pass": spikes["by_kind_max_rel"] <= args.gate_by_kind_rel,
        },
        "R5_bitwise": {
            # Byte equality, not a tolerance: the two arms read the same tensors or
            # the wiring is wrong. Stated this way it also survives a backend whose
            # logits are non-finite (the global ``mbe`` arm on GPT-2).
            "value": {"params_bitwise_equal": params["bitwise_equal"],
                      "logits_bitwise_equal": logits_same,
                      "params_max_abs_delta": params["max_abs_delta"],
                      "logits_max_abs_delta": logit_delta,
                      "logits_all_finite": logits_finite},
            "pass": bool(params["bitwise_equal"] and logits_same),
        },
    }
    verdict = ("supported" if all(g["pass"] for g in gates.values()
                                  if g["pass"] is not None) else "refuted")

    print("\n=== E21 gate ===")
    print(f"  stored bytes   copy {sc['bytes']:>8,}   ref {sr['bytes']:>8,}   "
          f"({sr['bytes'] / max(sc['bytes'], 1):.3f}x)")
    print(f"  stored params  copy {sc['params']:>8,}   ref {sr['params']:>8,}   "
          f"({sr['params'] / max(sc['params'], 1):.3f}x)")
    print(f"  primitives     copy {sc['primitives']:>8,}   ref "
          f"{sr['primitives']:>8,}   (expected to drop)")
    print(f"  distinct neuron objects: act {sharing_copy['activation_objects']} -> "
          f"{sharing_ref['activation_objects']}, "
          f"ln {sharing_copy['layernorm_objects']} -> "
          f"{sharing_ref['layernorm_objects']}")
    print(f"  spikes/input   copy {spikes['spikes_per_input_copy']:.6f}   ref "
          f"{spikes['spikes_per_input_ref']:.6f}   "
          f"equal={spikes['spikes_per_input_equal']}")
    print(f"  by_kind max rel move  {spikes['by_kind_max_rel']:.3e}")
    print(f"  params bitwise equal  {params['bitwise_equal']} over "
          f"{params['tensors_checked']} tensors "
          f"({params['n_differing']} differing, {params['n_unpaired']} unpaired)")
    print(f"  logits bitwise equal  {logits_same}   "
          f"(max |delta| {logit_delta:.3e}, all finite {logits_finite})")
    if params["unpaired"]:
        print(f"  unpaired: {params['unpaired']}")
    if params["differing"]:
        print(f"  differing: {params['differing']}")
    for name, g in gates.items():
        mark = "--" if g["pass"] is None else ("PASS" if g["pass"] else "FAIL")
        print(f"  [{mark}] {name}")
    print(f"  VERDICT: {verdict}")

    if args.json:
        common = dict(
            model=("smoke: tiny random GPT-2" if args.smoke else args.model),
            smoke=args.smoke, backend=args.backend, scope=args.convert_ops,
            stage=(2 if args.convert_ops == "all" else 1),
            n_basis_act=args.n_basis_act,
            mbe_readout_order=args.mbe_readout_order,
            device=device, epochs=args.epochs, n_steps=args.n_steps,
            block=args.block, calib_offset=args.calib_offset,
            dataset=args.dataset, experiment="E21",
            started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        rows = []
        for arm, store, sharing in ((arm_copy, sc, sharing_copy),
                                    (arm_ref, sr, sharing_ref)):
            rows.append(dict(
                common,
                tag=f"{args.tag}-{arm['label']}",
                arm=arm["label"],
                share_fit_refs=bool(arm["cfg"].share_fit_refs),
                pasn_beta=dict(arm["cfg"].pasn_beta or {}),
                convert_cfg=cfg_snapshot(arm["cfg"]),
                stored_params=store["params"], stored_bytes=store["bytes"],
                stored_bytes_naive=store["bytes_naive"],
                stored_shared_factor=store["shared_factor"],
                stored_by_role=store["by_role"],
                n_primitives=store["primitives"],
                distinct_neuron_objects=sharing,
                spikes_per_token=arm["spikes"]["spikes_per_input"],
                total_spikes=arm["spikes"]["total_spikes"],
                by_kind=arm["spikes"]["by_kind"],
                ops_per_input=arm["spikes"].get("ops_per_input"),
                energy_pj_per_input=arm["spikes"].get("energy_pj_per_input"),
                build_s=arm["build_s"],
                # Accuracy is not measured here: R4 is the GPU eval, and a record
                # that leaves these null is the honest form (no run, no number).
                ppl_ann=None, ppl_snn=None, delta_pct=None,
                status="ok",
                comparison=(dict(gates=gates, verdict=verdict, params=params,
                                 spikes=spikes,
                                 logits_bitwise_equal=logits_same,
                                 logits_max_abs_delta=logit_delta,
                                 logits_all_finite=logits_finite)
                            if arm["label"] == "ref" else None),
            ))
        for row in rows:
            write_record(args.json, row)

    return 0 if verdict == "supported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
