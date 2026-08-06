"""Why PASN loses ``1/x``, and whether re-placing the router wins it back.

``op_pareto`` measured ``1/x`` as the one operator where PASN is beaten by a
global MBE (0.85x spikes, 1.00x bytes). That is not a tuning miss, it is
structural, and the router says so directly:

    inv domain (0.5, 1.0), e in [-6, 0)  ->  7 banks, 1 reachable,
    20000/20000 inputs land in bank 6

``[0.5, 1)`` **is** one binade -- ``[2^-1, 2^0)`` -- and that is exactly the
interval ``frexp`` normalises a mantissa into. The reciprocal's argument is a
mantissa by construction (``SpikingSoftmax`` splits ``S = m * 2^e`` and asks for
``1/m``), so an *exponent* prefix router has nothing to separate. PASN there is a
global MBE neuron plus per-element router arithmetic it gets nothing for. It did
not lose the comparison; it never entered it.

The fix needs no new router. :class:`PrefixRouter` already routes on
``t = (x - beta) / 2^gamma``, so ``beta=0.5`` puts the binade ladder on the
*offset from the mantissa floor* and the banks become

    [0.5, 0.508) [0.508, 0.516) [0.516, 0.531) ... [0.75, 1.0)

log-dense at ``x=0.5``, which is where ``|f''| = 2/x^3`` is largest (16 at 0.5
against 2 at 1.0, an 8x spread). The density lands on the curvature.

This script measures three things, because the first two do not imply the third:

  1. does ``beta`` actually fix the isolated ``1/x`` Pareto,
  2. what it costs in stored bytes (more reachable banks = more banks stored),
  3. **whether it moves the softmax op** -- ``1/x`` runs once per row against
     ``2^x`` once per element, so its share of the op is ~1/C.

.. warning::
   **Read (3) with its design in mind: it under-reports, and it fooled us.**
   ``score_softmax`` pins ``exp2`` and the identity at ``N=8, T=16`` so that any
   movement is attributable to the reciprocal alone. That is the right control
   for *attribution*, but it holds the operator's cost fixed by construction, so
   the op total can only ever come out ~1.0x. This script reports 79.27 -> 80.33
   and the earlier P0.2 / 실험 5 toy measurement reported 280 -> 275 (1.02x), and
   **both conclusions are artefacts of that pinning**.

   Sweeping the whole op budget instead (``op_pareto.py``, softmax, ``pasn``
   arm) shows what is really going on: at ``b=1, T=8`` the shift leaves the
   spike count alone (4.65 -> 4.63) and improves nrmse ``2.60e-2 -> 6.82e-3``.
   The degenerate reciprocal was not spending spikes, it was capping the
   *accuracy* of the whole operator. At matched accuracy softmax then goes
   **1.88x -> 6.41x**, because PASN can now reach the MBE front from a far
   cheaper build.

   The lesson generalises past this primitive: **a primitive-level change must be
   judged with the rest of the operator free to move**, or the measurement
   answers a question nobody asked.

Usage::  python experiments/inv_router_fix.py --json results/inv_router_fix.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import functions  # noqa: E402
from mbe import spiking_ops as so  # noqa: E402
from mbe.mbe_pasn import build_mbe_pasn  # noqa: E402
from mbe.op_cost import SpikeMeter, op_memory  # noqa: E402

import op_pareto as OP  # noqa: E402

DOMAIN = (0.5, 1.0)          # the reciprocal's domain, fixed by the IEEE split


# --------------------------------------------------------------------------

def build_inv(arm: str, b, t, epochs: int, seed: int, span: int):
    """One ``1/x`` neuron for the named arm."""
    if arm == "mbe":
        return so.calibrate("inv", *DOMAIN, n_basis=int(b), n_steps=t,
                            epochs=epochs, seed=seed)
    # beta=0   -> exponent routing, provably one reachable bank (the bug)
    # beta=0.5 -> the ladder sits on (x - 0.5), so the banks actually partition
    beta = 0.5 if "beta" in arm else 0.0
    e_max = -1 if beta else 0
    kw = dict(e_min=e_max - span, e_max=e_max, beta=beta, epochs=epochs,
              seed=seed)
    if "rule" in arm:
        return build_mbe_pasn("inv", DOMAIN, budget="rule", target="relative",
                              target_rel=b, **kw)
    return build_mbe_pasn("inv", DOMAIN, n_local=int(b), n_near0=int(b),
                          n_steps=t, **kw)


ARMS = ("mbe", "pasn", "pasn_beta", "pasn_rule", "pasn_rule_beta")


def cells(arm, cfg):
    if "rule" in arm:
        return [(r, None) for r in cfg.rule_targets]
    return [(n, t) for n in cfg.Ns for t in cfg.Ts]


# --------------------------------------------------------------------------

@torch.no_grad()
def score_primitive(neuron, seed: int, m: int) -> dict:
    """Held-out MSE / nrmse and metered spikes for the isolated ``1/x``."""
    x, y, _ = functions.sample("inv", m=m, seed=seed + 1, domain=DOMAIN)
    prims = dict(inv=neuron)
    with SpikeMeter() as meter:
        meter.attach_all(prims)
        pred = neuron(x)
    out = dict(op_memory(prims), spikes=meter.per_element(x.numel())["_total"])
    out.update(OP.errors(pred, y))
    return out


@torch.no_grad()
def score_softmax(inv_neuron, exp_n, id_n, logits) -> dict:
    """The same neuron inside the real softmax, priced over the whole op.

    This is the number that decides whether the fix matters: ``1/x`` is invoked
    once per *row* while ``2^x`` is invoked once per *element*, so a primitive
    win is divided by the row width before it reaches the operator.
    """
    prims = dict(exp=exp_n, inv=inv_neuron, idn=id_n)
    sm = so.SpikingSoftmax(exp_n, inv_neuron, id_n, spike_mult=True)
    with SpikeMeter() as meter:
        meter.attach_all(prims)
        out = sm(logits, dim=-1)
    n = out.numel()
    per = meter.per_element(n)
    res = dict(op_memory(prims), spikes=per["_total"],
               inv_share=per["inv"] / max(per["_total"], 1e-12),
               spikes_by_prim={k: v for k, v in per.items() if k != "_total"})
    res.update(OP.errors(out, torch.softmax(logits, dim=-1)))
    return res


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--Ns", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    ap.add_argument("--Ts", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--rule-targets", nargs="+", type=float,
                    default=[3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3])
    ap.add_argument("--span", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--m-eval", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/inv_router_fix.json")
    cfg = ap.parse_args()

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(os.path.dirname(cfg.json) or ".", exist_ok=True)

    g = torch.Generator().manual_seed(cfg.seed + 1)
    logits = torch.randn(128, 64, generator=g) * 3.0
    # the other two softmax primitives are held FIXED across every arm, so any
    # movement in the op number is attributable to the reciprocal alone
    exp_n = so.calibrate("exp2", 0.0, 1.0, n_basis=8, n_steps=16,
                         epochs=cfg.epochs, seed=cfg.seed)
    id_n = so.calibrate_identity(0.0, 1.0, n_basis=8, n_steps=16,
                                 epochs=cfg.epochs, seed=cfg.seed)

    records, t0 = [], time.time()
    for arm in cfg.arms:
        for b, t in cells(arm, cfg):
            try:
                neuron = build_inv(arm, b, t, cfg.epochs, cfg.seed, cfg.span)
            except Exception as exc:                    # noqa: BLE001
                print(f"  {arm:16s} b={b:<6g} FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            prim = score_primitive(neuron, cfg.seed, cfg.m_eval)
            soft = score_softmax(neuron, exp_n, id_n, logits)
            alloc = OP.allocation(dict(inv=neuron))
            rec = dict(arm=arm, budget=b, T=t, primary="nrmse",
                       prim=prim, softmax=soft,
                       n_banks_reachable=_reachable(neuron),
                       **{k: v for k, v in alloc.items()
                          if k != "budget_tables"})
            records.append(rec)
            print(f"  {arm:16s} b={b:<6g} T={OP.tlabel(t):>4s}  "
                  f"1/x: mse={prim['mse']:.2e} nrmse={prim['nrmse']:.2e} "
                  f"sp={prim['spikes']:6.2f} B={prim['bytes']:5d} | "
                  f"softmax: nrmse={soft['nrmse']:.2e} sp={soft['spikes']:6.2f} "
                  f"(1/x is {100 * soft['inv_share']:.1f}%)", flush=True)

    res = dict(meta=dict(vars(cfg), elapsed_s=time.time() - t0), records=records)
    res["summary"] = summarise(records)
    with open(cfg.json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    report(res)
    print(f"\nwrote {cfg.json}")


def _reachable(neuron) -> int:
    r = getattr(neuron, "router", None)
    if r is None:
        return 0
    return sum(1 for bi in range(r.n_banks)
               if r.reachable(bi, *DOMAIN) is not None)


def _flat(records, level, key):
    return [dict(r[level], budget=r["budget"], T=r["T"], arm=r["arm"])
            for r in records if key in r[level]]


def summarise(records) -> dict:
    """Iso-accuracy of every arm against the global MBE front, at both levels."""
    out = {}
    for level in ("prim", "softmax"):
        pts = _flat(records, level, "nrmse")
        base = OP.front([p for p in pts if p["arm"] == "mbe"], "spikes", "nrmse")
        arms = {}
        for arm in sorted({p["arm"] for p in pts}):
            sel = [p for p in pts if p["arm"] == arm]
            arms[arm] = dict(front=OP.front(sel, "spikes", "nrmse"),
                             iso=OP.iso_accuracy(base, sel, "nrmse"))
        out[level] = arms
    return out


def report(res) -> None:
    import statistics
    for level, arms in res["summary"].items():
        print(f"\n=== {level}: iso-accuracy vs global MBE "
              f"(MBE/arm, >1 = arm cheaper) ===")
        for arm, a in arms.items():
            if arm == "mbe":
                continue
            ok = [r for r in a["iso"] if r["matched"]]
            if not ok:
                print(f"  {arm:16s} no MBE point matched "
                      f"(0/{len(a['iso'])})")
                continue
            sp = statistics.median(r["spikes_ratio"] for r in ok)
            by = statistics.median(r["bytes_ratio"] for r in ok)
            print(f"  {arm:16s} spikes {sp:5.2f}x   bytes {by:5.2f}x   "
                  f"({len(ok)}/{len(a['iso'])} matched)")


if __name__ == "__main__":
    main()
