"""E37 -- per-primitive router depth for the tied identity.

E36 tried to *slide* the identity ladder (``e_max`` onto a data quantile) and was
refuted: the top rung then absorbs everything above it and carries ~100% of the
error. This script keeps ``e_max`` where it is -- so nothing overflows -- and only
**lowers ``e_min``**, which is E11b's axis. The reason to revisit it is that E11b
measured the deploy knob's *scope*, and the scope is the whole finding.

**Why depth is nearly free for this primitive.** The identity is positively
homogeneous, so ``tied=True`` fits **one prototype** on the unit residual and
every magnitude bank is a reference to it carrying its own ``2^e`` scale. Only
the near-zero bank is fitted separately -- ``[0, 2^e_min)`` spans every scale
down to zero, so there is no exponent to factor out and it is not homogeneous
(see the ``near-zero banks are not homogeneous`` branch in ``build_mbe_pasn``).
So an extra magnitude bank adds **no bases and no parameters**; it only
reclassifies elements out of the untied ``N=4`` near-zero bank into the tied
``N=1`` ones. Measured on ``id_a`` in isolation: bytes 128 at every depth from
``e_min=-6`` to ``-10``, spikes 13.17 -> 3.69, reconstruction L2 flat at ~0.0099.
That is an **occupancy** move, not a capacity move.

**What E11b actually left open.** Its section 6 says the deploy knob
``pasn_id_e_min`` is global -- it deepens ``id_qk`` and ``id_v`` too, "그쪽은
집중이 없어서 뱅크만 는다" -- and that arm was never run. Only ``id_a`` (the
attention matrix, mean ``1/seq``) and ``idn`` (which carries the row reciprocals
``1/S``) have the concentration. So the scopes are separated here:

  ``control``  every identity at its current depth
  ``ida``      extra depth on ``id_a`` only
  ``conc``     extra depth on ``id_a`` + ``idn`` -- the two concentrated ones
  ``all``      extra depth on all four = what the deploy knob actually does

⚠️ **This does not overturn E11b.** That was real GPT-2 operands at seq 256 and
read iso-accuracy on the whole op, where its own section 3 warns the target grid
saturates (1e-1/3e-2/1e-2 all give the same 66.5-spike build). This is synthetic
seq 32. Two different measurements; the record keeps them apart.

Usage::
    python experiments/e37_id_depth_per_prim.py --smoke
    python experiments/e37_id_depth_per_prim.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch                                                         # noqa: E402

from mbe.mbe_pasn import build_mbe_pasn                              # noqa: E402
from mbe.op_cost import SpikeMeter, op_memory                        # noqa: E402
import e36_id_ladder_anchor as E36                                   # noqa: E402
import op_pareto as OP                                               # noqa: E402

#: Which identities each scope deepens. ``id_qk``/``id_v`` are the Gaussian
#: operands -- no concentration, so deepening them only adds banks.
SCOPES = {
    "control": (),
    "ida":     ("id_a",),
    "conc":    ("id_a", "idn"),
    "all":     ("id_a", "idn", "id_qk", "id_v"),
}

TARGETS = E36.TARGETS
SPAN, EPOCHS, SEED, T_PIN = E36.SPAN, E36.EPOCHS, E36.SEED, E36.T_PIN
DEPLOY_TARGET = 0.01

_CACHE: dict = {}


def build(scope: str, extra: int, target_rel: float, d) -> dict:
    """``make_attention``'s routed build with ``e_min`` lowered by ``extra`` rungs.

    ``e_max`` is left on the domain anchor for every primitive, so no bank is
    ever asked for values above its fitted interval -- the failure mode E36 hit.
    Identities are kept as **distinct objects** per role so ``SpikeMeter`` can
    separate ``idn`` from ``id_a``; sharing them is what double-counts the
    attention row (see E36 section 4).
    """
    qk_max = float(max(d["q"].abs().max(), d["k"].abs().max())) * 1.1
    v_max = float(d["v"].abs().max()) * 1.1
    doms = dict(idn=(0.0, 1.0), id_qk=(-qk_max, qk_max), id_a=(0.0, 1.0),
                id_v=(-v_max, v_max))
    deepened = SCOPES[scope]

    def make(role, name, dom, tied):
        e_min, e_max = OP._binades(dom[0], dom[1], span=SPAN,
                                   beta=OP.BETA.get(name, 0.0))
        if role in deepened:
            e_min -= extra
        key = (role, dom, e_min, e_max, target_rel)
        if key not in _CACHE:
            _CACHE[key] = build_mbe_pasn(
                name, dom, e_min=e_min, e_max=e_max, budget="rule",
                target="relative", target_rel=target_rel,
                beta=OP.BETA.get(name, 0.0), tied=tied, epochs=EPOCHS,
                seed=SEED, t_fixed=T_PIN)
        return _CACHE[key]

    prims = {r: make(r, "identity", doms[r], True) for r in doms}
    prims["exp"] = make("exp", "exp2", (0.0, 1.0), False)
    prims["inv"] = make("inv", "inv", (0.5, 1.0), False)
    return prims


def cell(scope: str, extra: int, target_rel: float, op, d) -> dict:
    t0 = time.time()
    prims = build(scope, extra, target_rel, d)
    fit_s = time.time() - t0
    with torch.no_grad(), SpikeMeter() as meter:
        meter.attach_all(prims)
        out = op["run"](prims, d)
    per_prim = meter.per_element(d["n_out"])
    rec = dict(arm=f"{scope}+{extra}", scope=scope, extra=extra,
               target=target_rel, budget=target_rel, T=T_PIN,
               spikes=per_prim["_total"],
               spikes_by_prim={k: v for k, v in per_prim.items()
                               if k != "_total"},
               eranges={k: [prims[k].router.e_min, prims[k].router.e_max]
                        for k in E36.ID_PRIMS},
               n_banks={k: prims[k].router.n_banks for k in E36.ID_PRIMS},
               stored_bases={k: prims[k].stored_bases() for k in E36.ID_PRIMS},
               fit_seconds=fit_s, **op_memory(prims))
    rec.update(OP.errors(out, d["ref"]))
    return rec


def isolated_depth(samples: dict, target_rel: float = DEPLOY_TARGET) -> dict:
    """The tied identity on its own, depth swept -- the claim depth is ~free.

    Reported per primitive because the two concentrated ones have different
    distributions: ``id_a`` is a softmax row (mean ``1/seq``), ``idn`` is bimodal
    (numerators of order 1 **and** row reciprocals of order ``1/seq``).
    """
    out = {}
    for role in ("id_a", "idn"):
        sample = samples[role]
        rows = []
        for e_min in range(-6, -11, -1):
            n = build_mbe_pasn(
                "identity", (0.0, 1.0), e_min=e_min, e_max=0, budget="rule",
                target="relative", target_rel=target_rel, beta=0.0, tied=True,
                epochs=EPOCHS, seed=SEED, t_fixed=T_PIN)
            with torch.no_grad():
                rows.append(dict(
                    e_min=e_min, n_banks=n.router.n_banks,
                    stored_bases=n.stored_bases(),
                    params=n.num_learnable(), bytes=op_memory({"x": n})["bytes"],
                    near0_occupancy=float((sample.abs() < 2.0 ** e_min)
                                          .float().mean()),
                    mean_spikes=n.mean_spikes(sample),
                    recon_l2=float((n.reconstruct(sample) - sample)
                                   .pow(2).sum().sqrt())))
        out[role] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extra", type=int, default=3)
    ap.add_argument("--json", default="results/e37_id_depth_per_prim.json")
    a = ap.parse_args()

    t0 = time.time()
    op = OP.make_attention()
    cfg = argparse.Namespace(seed=SEED, epochs=EPOCHS, binades=SPAN)
    d = op["data"](cfg)
    samples = E36.operand_samples(d)

    scopes = ("control", "conc") if a.smoke else tuple(SCOPES)
    targets = (DEPLOY_TARGET,) if a.smoke else TARGETS
    depths = () if a.smoke else (1, 2, 4)

    cells = []
    for scope in scopes:
        for tr in targets:
            r = cell(scope, 0 if scope == "control" else a.extra, tr, op, d)
            cells.append(r)
            print(f"  {r['arm']:11s} target={tr:<6} nrmse={r['nrmse']:.4g} "
                  f"spk={r['spikes']:7.2f} bytes={r['bytes']}", flush=True)
    for x in depths:                       # depth curve at the deployment target
        r = cell("conc", x, DEPLOY_TARGET, op, d)
        cells.append(r)
        print(f"  {r['arm']:11s} target={DEPLOY_TARGET:<6} "
              f"nrmse={r['nrmse']:.4g} spk={r['spikes']:7.2f} "
              f"bytes={r['bytes']}", flush=True)

    rec = dict(tag="e37-id-depth-per-prim", experiment="E37", backend="analysis",
               model="op-level", op="attention", convert_cfg=None,
               pasn_beta=None, stored_bytes=None, stored_params=None,
               n_primitives=None, status="ok",
               started=_dt.datetime.now().isoformat(timespec="seconds"),
               span=SPAN, T=T_PIN, epochs=EPOCHS, seed=SEED, extra=a.extra,
               targets=list(targets), smoke=a.smoke, scopes=
               {k: list(v) for k, v in SCOPES.items()},
               baseline_from="results/e33_op_pareto_mbe_tau.json",
               note=("baseline de-double-counted (E36 section 4); e_max left on "
                     "the domain anchor so no bank overflows (E36 section 9)"),
               cells=cells, isolated_depth=isolated_depth(samples),
               sample_stats={k: dict(median=float(v.abs().median()),
                                     mean=float(v.abs().mean()),
                                     max=float(v.abs().max()))
                             for k, v in samples.items()},
               elapsed_s=time.time() - t0, torch=torch.__version__)
    rec["iso_accuracy"] = E36.iso_vs_mbe(cells)

    with open(os.path.join(ROOT, a.json), "w", encoding="utf-8") as f:
        json.dump([rec], f, ensure_ascii=False, indent=1)
    print(f"\nwrote {a.json}  ({rec['elapsed_s']:.1f}s)")
    for arm, v in rec["iso_accuracy"].items():
        s = f"{v['spikes']:.2f}x" if v["spikes"] else "n/a"
        b = f"{v['bytes']:.2f}x" if v["bytes"] else "n/a"
        print(f"  {arm:11s} matched {v['matched']}/{v['n_ref']}  "
              f"spikes {s}  bytes {b}")


if __name__ == "__main__":
    main()
