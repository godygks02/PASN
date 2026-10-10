"""E36 -- anchor the identity router's ladder on the data, not on the domain top.

The routed identity's binade ladder is hung from the **domain upper bound**:
``op_pareto._binades`` sets ``e_max = ceil(log2(max|x| over the domain))`` and
``e_min = e_max - span``. For ``id_a`` -- the identity that encodes the attention
matrix for the ``AV`` matmul -- the domain is ``[0, 1]``, so the ladder's top rung
is 1.0. The data is nowhere near there: softmax over ``seq`` positions has mean
``1/seq``, so at ``seq = 32`` the mass sits five rungs below the top and 40.1% of
it falls off the bottom into the near-zero bank -- the one bank the router does
**not** supply an exponent to, and the only one the budget rule gives ``N = 4``.
Measured on the control build: that single bank emits **84.3%** of ``id_a``'s
spikes (27.67 per element against 3.46 for a magnitude bank, 8.0x).

This script slides the whole ladder down instead of extending it:
``e_max = ceil(log2(quantile_p(|sample|)))`` with ``span`` unchanged, so the
**bank count -- and therefore the stored bytes -- does not move.** That is what
separates it from the two neighbouring experiments:

* **E11b** lowered ``e_min`` with ``e_max`` fixed. That *adds* banks and bytes
  (1372 -> 1540 B) and its verdict was **don't**: the frontier crosses and the
  deployment operating point sits where ``-6`` wins 2.4x.
* **E22** (``MEMORY_OVERHEAD.md`` section 4, unstarted) clamps ``e_min``/``e_max``
  to the *observed* range to delete dead banks, and is falsified if a reached
  bank's parameters move at all. Sliding the ladder moves them by construction.

So this is the third move on that axis and it has not been measured. It is also
the one E11b's own section 1 argued could not be measured here -- it read the
mean (0.031 > 0.0156) and concluded "the pathology does not exist at seq 32".
The mean is above the floor; **40.1% of the mass is below it.**

Arms (all at ``T = 16`` pinned, i.e. the ``pasn_rule_T`` arm the freeze build
actually ships -- ``freeze_e1.json`` has ``pasn_t_fixed = 16``):

  ``domain``      control, current behaviour
  ``q99``/``q95``/``q90``   quantile anchor on every routed identity
  ``q95_ida``     quantile anchor on ``id_a`` only, for attribution

Usage::
    python experiments/e36_id_ladder_anchor.py --smoke
    python experiments/e36_id_ladder_anchor.py
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
import op_pareto as OP                                               # noqa: E402

#: The four identities in the attention decomposition. ``exp``/``inv`` stay on the
#: domain anchor: they are not identities, their domains are already tight
#: ([0,1] and [0.5,1)), and ``inv`` carries ``beta``, which owns its placement.
ID_PRIMS = ("idn", "id_qk", "id_a", "id_v")

TARGETS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
SPAN = 6            # op_pareto's --binades default; held fixed so G does not move
EPOCHS = 250        # op_pareto_merged.json meta
SEED = 0
T_PIN = 16          # the freeze build's arm (pasn_t_fixed = 16)


# --------------------------------------------------------------------------
# The anchor policies
# --------------------------------------------------------------------------

def erange_domain(domain, sample=None, span=SPAN):
    """Current behaviour: ladder top from the domain bound (``op_pareto._binades``)."""
    return OP._binades(domain[0], domain[1], span=span, beta=0.0)


def erange_quantile(domain, sample, span=SPAN, q=0.95):
    """Ladder top from a quantile of the operand magnitudes; ``span`` unchanged.

    ``e_max`` is the rung above the ``q``-quantile, so a ``1-q`` tail routes into
    the top bank and is fitted on a widened interval -- that cost is the price of
    the move, which is why the accuracy column has to be read next to the spike
    one. Bank count is ``span`` (+ near-zero) either way, so bytes are held fixed.
    """
    mag = sample.abs().flatten()
    mag = mag[mag > 0]
    if mag.numel() == 0:
        return erange_domain(domain, sample, span)
    cut = float(torch.quantile(mag, q))
    e_max = int(math.ceil(math.log2(max(cut, 1e-30))))
    return e_max - int(span), e_max


POLICIES = {
    "domain": lambda dom, s: erange_domain(dom, s),
    "q99":    lambda dom, s: erange_quantile(dom, s, q=0.99),
    "q95":    lambda dom, s: erange_quantile(dom, s, q=0.95),
    "q90":    lambda dom, s: erange_quantile(dom, s, q=0.90),
}


# --------------------------------------------------------------------------
# Operand samples -- computed exactly, not guessed
# --------------------------------------------------------------------------

def operand_samples(d) -> dict:
    """What each routed identity actually sees, replicating the exact op.

    ``idn`` is the softmax's final FP-multiply identity, so it carries **two**
    populations: the shifted numerators ``exp_x`` (order 1) and the row
    reciprocals ``1/S`` (order ``1/seq``). They are concatenated because one
    ladder has to cover both -- which is itself part of what is measured here.
    """
    scores = (d["q"] @ d["k"].transpose(-1, -2)) * d["scale"]
    x_m = scores - scores.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x_m)
    S = exp_x.sum(dim=-1, keepdim=True)
    attn = exp_x / S
    return dict(
        id_qk=torch.cat([d["q"].flatten(), d["k"].flatten()]),
        id_a=attn.flatten(),
        id_v=d["v"].flatten(),
        idn=torch.cat([exp_x.flatten(), (1.0 / S).flatten()]),
    )


# --------------------------------------------------------------------------
# Build one attention cell under a given anchor policy
# --------------------------------------------------------------------------

_BUILD_CACHE: dict = {}


def build_attention(policy: str, target_rel: float, d, samples,
                    ida_only: bool = False) -> dict:
    """``make_attention``'s routed build with per-primitive ``(e_min, e_max)``.

    Mirrors ``op_pareto.make_attention``'s routed build exactly -- same domains,
    same ``beta``, same ``tied``, same budget rule at ``T = 16`` -- and changes
    only where the ladder is hung.

    ⚠️ **One deliberate difference from ``op_pareto``.** There, ``idn`` and
    ``id_a`` are both ``identity`` on ``[0, 1]`` with identical build arguments,
    so ``_PASN_CACHE`` (and ``spiking_ops._FIT_CACHE`` on the MBE side) hands back
    **the same object** for both. ``SpikeMeter`` keys on the prims dict, so that
    one module's accumulated count is reported under both names and summed --
    the shared identity is charged twice. Verified: splitting the object leaves
    the fit bit-identical (nrmse 0.009308 either way) and drops the metered total
    from 100.79 to 64.91 = 100.79 - 35.88, exactly the shared term. The cache
    here keys on the **build arguments and the role**, so the two stay distinct
    objects and the meter can separate them.
    """
    qk_max = float(max(d["q"].abs().max(), d["k"].abs().max())) * 1.1
    v_max = float(d["v"].abs().max()) * 1.1
    doms = dict(idn=(0.0, 1.0), id_qk=(-qk_max, qk_max), id_a=(0.0, 1.0),
                id_v=(-v_max, v_max))

    def ident(name):
        pol = policy if (not ida_only or name == "id_a") else "domain"
        e_min, e_max = POLICIES[pol](doms[name], samples[name])
        key = (name, doms[name], e_min, e_max, target_rel)
        if key not in _BUILD_CACHE:
            _BUILD_CACHE[key] = build_mbe_pasn(
                "identity", doms[name], e_min=e_min, e_max=e_max, budget="rule",
                target="relative", target_rel=target_rel, beta=0.0, tied=True,
                epochs=EPOCHS, seed=SEED, t_fixed=T_PIN)
        return _BUILD_CACHE[key]

    def other(name, dom):
        beta = OP.BETA.get(name, 0.0)
        e_min, e_max = OP._binades(dom[0], dom[1], span=SPAN, beta=beta)
        key = (name, dom, e_min, e_max, target_rel)
        if key not in _BUILD_CACHE:
            _BUILD_CACHE[key] = build_mbe_pasn(
                name, dom, e_min=e_min, e_max=e_max, budget="rule",
                target="relative", target_rel=target_rel, beta=beta, tied=False,
                epochs=EPOCHS, seed=SEED, t_fixed=T_PIN)
        return _BUILD_CACHE[key]

    return dict(exp=other("exp2", (0.0, 1.0)), inv=other("inv", (0.5, 1.0)),
                idn=ident("idn"), id_qk=ident("id_qk"), id_a=ident("id_a"),
                id_v=ident("id_v"))


# --------------------------------------------------------------------------
# Per-bank measurement -- the table the 연구일지 is built on
# --------------------------------------------------------------------------

@torch.no_grad()
def bank_table(neuron, sample) -> list:
    """Per-bank ``(bank, kind, lo, hi, N, T, fire_rate, spikes_per_element,
    occupancy, spike_share)`` measured on ``sample``.

    Replicates ``MBEPASNNeuron.mean_spikes`` and keeps the per-bank terms instead
    of summing them, so occupancy and spike share are the same quantity the
    aggregate is made of rather than a second estimate of it.
    """
    flat = sample.reshape(-1)
    idx = neuron.router.route(flat)
    n = max(flat.numel(), 1)
    rows, total = [], 0.0
    for bi, bank in enumerate(neuron.bank_mods):
        mask = idx == bi
        cnt = int(mask.sum())
        spec = neuron.router.banks[bi]
        lo, hi = neuron.router.bank_interval(bi)
        nb, nt = bank.cfg.n_basis, bank.cfg.n_steps
        if cnt == 0:
            rows.append([bi, spec["kind"], lo, hi, nb, nt, 0.0, 0.0, 0.0, 0.0])
            continue
        fr = float(bank.firing_rate(neuron._bank_input(flat, mask, spec)))
        per_el = fr * nb * nt
        total += per_el * cnt
        rows.append([bi, spec["kind"], lo, hi, nb, nt, fr, per_el, cnt / n, 0.0])
    for r in rows:
        r[9] = (r[7] * r[8] * n / total) if total > 0 else 0.0
    return rows


# --------------------------------------------------------------------------

def run(smoke: bool) -> dict:
    op = OP.make_attention()
    cfg = argparse.Namespace(seed=SEED, epochs=EPOCHS, binades=SPAN)
    d = op["data"](cfg)
    samples = operand_samples(d)

    arms = [("domain", False), ("q95", False)] if smoke else [
        ("domain", False), ("q99", False), ("q95", False), ("q90", False),
        ("q95", True)]
    targets = (0.01,) if smoke else TARGETS

    cells, tables = [], {}
    for policy, ida_only in arms:
        arm = f"{policy}_ida" if ida_only else policy
        for tr in targets:
            t0 = time.time()
            prims = build_attention(policy, tr, d, samples, ida_only)
            fit_s = time.time() - t0
            with torch.no_grad(), SpikeMeter() as meter:
                meter.attach_all(prims)
                out = op["run"](prims, d)
            per_prim = meter.per_element(d["n_out"])
            rec = dict(arm=arm, policy=policy, ida_only=ida_only, target=tr,
                       T=T_PIN, budget=tr, spikes=per_prim["_total"],
                       spikes_by_prim={k: v for k, v in per_prim.items()
                                       if k != "_total"},
                       eranges={k: [prims[k].router.e_min, prims[k].router.e_max]
                                for k in ID_PRIMS},
                       n_banks={k: prims[k].router.n_banks for k in ID_PRIMS},
                       fit_seconds=fit_s, **op_memory(prims))
            rec.update(OP.errors(out, d["ref"]))
            cells.append(rec)
            print(f"  {arm:9s} target={tr:<6} nrmse={rec['nrmse']:.4g} "
                  f"spk={rec['spikes']:7.2f} bytes={rec['bytes']}", flush=True)
            if tr == 0.01:
                tables[arm] = {p: bank_table(prims[p], samples[p])
                               for p in ("id_a", "idn")}
    stats = {k: dict(n=int(v.numel()), mean=float(v.abs().mean()),
                     median=float(v.abs().median()),
                     p90=float(torch.quantile(v.abs(), 0.90)),
                     p95=float(torch.quantile(v.abs(), 0.95)),
                     p99=float(torch.quantile(v.abs(), 0.99)),
                     max=float(v.abs().max()))
             for k, v in samples.items()}
    return dict(cells=cells, bank_tables=tables, sample_stats=stats)


def iso_vs_mbe(cells: list) -> dict:
    """Restate the attention row of the operator table under each anchor.

    The reference is the **re-fitted** MBE front from E33, not the original one:
    a win over a baseline we fit worse than our own method is not a win.

    The baseline is also **de-double-counted** here. The MBE arm shares its
    identity between ``idn`` and ``id_a`` for the same reason the routed arm does
    (``spiking_ops._FIT_CACHE``), and all 15 attention records in that file carry
    the signature -- ``id_a`` and ``idn`` exactly equal. Subtracting the shared
    term once puts both sides on the same accounting; leaving it in would flatter
    this experiment, since its own builds do not double count.
    """
    path = os.path.join(ROOT, "results", "e33_op_pareto_mbe_tau.json")
    mbe = []
    for r in json.load(open(path, encoding="utf-8"))["records"]:
        if r["op"] != "attention":
            continue
        bp = r["spikes_by_prim"]
        assert abs(bp["id_a"] - bp["idn"]) < 1e-9, "expected the shared identity"
        r = dict(r, spikes=r["spikes"] - bp["id_a"])
        mbe.append(r)
    base = OP.front(mbe, "spikes", "nrmse")
    med = lambda v: sorted(v)[len(v) // 2]                        # noqa: E731
    out = {}
    for arm in dict.fromkeys(c["arm"] for c in cells):
        rows = OP.iso_accuracy(base, [c for c in cells if c["arm"] == arm],
                               "nrmse")
        ok = [r for r in rows if r["matched"]]
        out[arm] = dict(
            matched=len(ok), n_ref=len(rows),
            spikes=med([r["spikes_ratio"] for r in ok]) if ok else None,
            bytes=med([r["bytes_ratio"] for r in ok]) if ok else None,
            rows=rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", default="results/e36_id_ladder_anchor.json")
    a = ap.parse_args()

    t0 = time.time()
    res = run(a.smoke)
    rec = dict(tag="e36-id-ladder-anchor", experiment="E36", backend="analysis",
               model="op-level", op="attention", convert_cfg=None, pasn_beta=None,
               stored_bytes=None, stored_params=None, n_primitives=None,
               status="ok",
               started=_dt.datetime.now().isoformat(timespec="seconds"),
               span=SPAN, T=T_PIN, epochs=EPOCHS, seed=SEED,
               targets=list(TARGETS), smoke=a.smoke,
               baseline_from="results/e33_op_pareto_mbe_tau.json",
               elapsed_s=time.time() - t0, torch=torch.__version__, **res)
    rec["iso_accuracy"] = iso_vs_mbe(res["cells"])

    path = os.path.join(ROOT, a.json)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([rec], f, ensure_ascii=False, indent=1)
    print(f"\nwrote {a.json}  ({rec['elapsed_s']:.1f}s)")
    for arm, v in rec["iso_accuracy"].items():
        s = f"{v['spikes']:.2f}x" if v["spikes"] else "n/a"
        b = f"{v['bytes']:.2f}x" if v["bytes"] else "n/a"
        print(f"  {arm:9s} matched {v['matched']}/{v['n_ref']}  "
              f"spikes {s}  bytes {b}")


if __name__ == "__main__":
    main()
