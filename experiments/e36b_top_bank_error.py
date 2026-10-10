"""E36b -- where the sliding ladder's error actually goes.

E36 slid the ``id_a`` ladder down onto a quantile of the data. The spike half of
the prediction held: the near-zero bank's occupancy fell 40.1% -> 1.9% and its
share of ``id_a``'s spikes fell 84.3% -> 13.8%. The accuracy half did not --
attention nrmse went 0.0093 -> 0.2057. This script says why, on the identity
alone, so the mechanism is measured rather than inferred from the op.

The router clamps the exponent to ``e_max - 1``, so sliding the window down
leaves the **top rung absorbing everything above it**. At ``q95`` that rung is
nominally ``[0.0625, 0.125)`` while ``id_a`` reaches 0.5309 -- 4.2x past where it
was fitted.

Note the two error columns disagree on purpose. The budget rule targets
**relative** error and keeps it small everywhere (mean 0.0017 -> 0.0041); the op
is scored on **nrmse**, which is absolute, and the overflow bank's relative error
lands on the largest values. That mismatch is the whole result.

Usage::  python experiments/e36b_top_bank_error.py
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

import e36_id_ladder_anchor as E36                                   # noqa: E402
import op_pareto as OP                                               # noqa: E402


def probe(policy: str, sample: torch.Tensor, target_rel: float = 0.01) -> dict:
    # NOT under no_grad: the build *fits* the bank prototype and needs the graph.
    e_min, e_max = E36.POLICIES[policy]((0.0, 1.0), sample)
    neuron = E36.build_mbe_pasn(
        "identity", (0.0, 1.0), e_min=e_min, e_max=e_max, budget="rule",
        target="relative", target_rel=target_rel, beta=0.0, tied=True,
        epochs=E36.EPOCHS, seed=E36.SEED, t_fixed=E36.T_PIN)
    with torch.no_grad():
        return _measure(neuron, sample, policy, e_min, e_max, target_rel)


@torch.no_grad()
def _measure(neuron, sample, policy, e_min, e_max, target_rel) -> dict:
    rec = neuron.reconstruct(sample)
    err = (rec - sample).abs()
    tot_sq = float(err.pow(2).sum())
    idx = neuron.router.route(sample)
    banks = []
    for bi in range(neuron.router.n_banks):
        m = idx == bi
        if int(m.sum()) == 0:
            continue
        lo, hi = neuron.router.bank_interval(bi)
        banks.append(dict(
            bank=bi, kind=neuron.router.banks[bi]["kind"], lo=lo, hi=hi,
            occupancy=float(m.float().mean()),
            rel_err=float((err[m] / sample[m].clamp(min=1e-9)).mean()),
            sq_err_share=float(err[m].pow(2).sum()) / tot_sq if tot_sq else 0.0))
    return dict(
        policy=policy, e_min=e_min, e_max=e_max, target_rel=target_rel,
        top_rung=[2.0 ** (e_max - 1), 2.0 ** e_max],
        data_max=float(sample.max()),
        overflow_factor=float(sample.max()) / 2.0 ** e_max,
        rel_err_mean=float((err / sample.clamp(min=1e-9)).mean()),
        l2_err=math.sqrt(tot_sq), banks=banks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/e36b_top_bank_error.json")
    a = ap.parse_args()

    t0 = time.time()
    op = OP.make_attention()
    cfg = argparse.Namespace(seed=E36.SEED, epochs=E36.EPOCHS, binades=E36.SPAN)
    d = op["data"](cfg)
    sample = E36.operand_samples(d)["id_a"]

    probes = [probe(p, sample) for p in ("domain", "q99", "q95")]
    rec = dict(tag="e36b-top-bank-error", experiment="E36b", backend="analysis",
               model="op-level", op="attention", primitive="id_a",
               convert_cfg=None, pasn_beta=None, stored_bytes=None,
               stored_params=None, n_primitives=None, status="ok",
               started=_dt.datetime.now().isoformat(timespec="seconds"),
               span=E36.SPAN, T=E36.T_PIN, epochs=E36.EPOCHS, seed=E36.SEED,
               probes=probes, elapsed_s=time.time() - t0,
               torch=torch.__version__)
    with open(os.path.join(ROOT, a.json), "w", encoding="utf-8") as f:
        json.dump([rec], f, ensure_ascii=False, indent=1)

    print(f"wrote {a.json}  ({rec['elapsed_s']:.1f}s)\n")
    for p in probes:
        top = p["banks"][-1]
        print(f"{p['policy']:7s} e=[{p['e_min']},{p['e_max']}]  "
              f"top rung [{p['top_rung'][0]:.4f},{p['top_rung'][1]:.4f})  "
              f"overflow {p['overflow_factor']:.1f}x  "
              f"L2 {p['l2_err']:.4f}  "
              f"top bank: occ {100*top['occupancy']:.1f}% "
              f"sq-err share {100*top['sq_err_share']:.1f}%")


if __name__ == "__main__":
    main()
