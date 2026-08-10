"""E5 -- learnable-parameter accounting for the paper's MBE against our PASN build.

`PAPER_PLAN.md` section 4.1 quotes three numbers that decide how the memory axis is
presented, and none of them were reproducible from a script. This derives them
from two things only: the **site inventory** of the model we actually convert, and
the paper's **Appendix G.1** hyperparameters.

Closed forms, verified against the sweep data for N in {1,2,4,6,8}:

    general MBE neuron        6N + 1
    polarity-split (GELU/SiLU)  12N + 1

Appendix G.1, verbatim on the point that decides everything:

    "For GELU and Tanh approximation, we employ MBE neurons configured with
     N = 4 and T = 16. **To mitigate approximation errors in identity mapping,
     all learnable parameters in the MBE neuron are fixed as [20].** Using this
     configuration with N = 8, T = 16, we implement function approximations for
     three distinct mathematical operations: the exponential function 2^x over
     [0,1], the reciprocal 1/x within [0.5,1], and the inverse square root
     1/sqrt(x) across [0.5,2]."

So the paper's identity carries **no learnable parameters at all** -- they are
fixed to the values of reference [20]. The main text agrees from the other side:
"The intensity matrix D can be precomputed and shared across the entire network."

That is the premise section 4.1 flagged as load-bearing, and it lands on the
unfavourable side: 218 of the 339 primitives we convert are identities, and on
the paper's own account they are free. **Our tied identity is therefore not a
differentiator on this axis** -- the paper already fixes and shares its own.

Two anchors keep the accounting honest:
  * the inventory must sum to **339** primitives, the count our converter reports;
  * counting the identity as learnable must reproduce **16,611**, the number
    section 4.1 already carried, so the closed forms and inventory agree with
    whatever produced it.

Usage::  python experiments/mbe_param_model.py
"""
from __future__ import annotations

import argparse
import json
import os

# --------------------------------------------------------------------------
# Site inventory -- GPT-2-medium, Stage 2 scope, as our converter builds it.
# Measured: results/bank_usage_audit.json (E11, 339 sites, full enumeration).
# --------------------------------------------------------------------------
INVENTORY = [
    # (op, role, count, paper primitive, is_identity, polarity_split)
    ("activation", "mlp.act", 24, "gelu",     False, True),
    ("layernorm",  "rsqrt",   49, "invsqrt",  False, False),
    ("layernorm",  "id_dev",  49, "identity", True,  False),
    ("layernorm",  "id_istd", 49, "identity", True,  False),
    ("matmul",     "idn",     48, "identity", True,  False),
    ("matmul",     "idn2",    48, "identity", True,  False),
    ("softmax",    "exp",     24, "exp2",     False, False),
    ("softmax",    "inv",     24, "inv",      False, False),
    ("softmax",    "idn",     24, "identity", True,  False),
]

# Appendix G.1: N=4 for GELU/Tanh, N=8 for exp / inv / invsqrt.
PAPER_N = {"gelu": 4, "invsqrt": 8, "exp2": 8, "inv": 8, "identity": 8}


def params(n_basis: int, polarity_split: bool) -> int:
    return (12 * n_basis + 1) if polarity_split else (6 * n_basis + 1)


def main():
    ap = argparse.ArgumentParser()
    # The frozen build (E0, freeze-t16-unif). Bytes are float32, so
    # naive_params = naive_bytes / 4.
    ap.add_argument("--pasn-params", type=int, default=13472)
    ap.add_argument("--pasn-bytes", type=int, default=53888)
    ap.add_argument("--pasn-naive-bytes", type=int, default=120092,
                    help="storage with cross-site sharing removed")
    ap.add_argument("--json", default="results/e5_param_model.json")
    args = ap.parse_args()

    total_sites = sum(r[2] for r in INVENTORY)
    id_sites = sum(r[2] for r in INVENTORY if r[4])
    assert total_sites == 339, f"inventory sums to {total_sites}, expected 339"

    print(f"=== site inventory (anchor: {total_sites} primitives, "
          f"{id_sites} of them identities) ===")
    print(f"{'op':12s} {'role':9s} {'n':>4s} {'primitive':10s} {'paper N':>8s} "
          f"{'form':>9s} {'per-site':>9s} {'subtotal':>9s}")
    rows, learnable_total, nonid_total = [], 0, 0
    for op, role, n, prim, is_id, split in INVENTORY:
        N = PAPER_N[prim]
        per = params(N, split)
        sub = per * n
        learnable_total += sub
        if not is_id:
            nonid_total += sub
        form = f"{'12N+1' if split else '6N+1'}"
        print(f"{op:12s} {role:9s} {n:>4} {prim:10s} {N:>8} {form:>9s} "
              f"{per:>9} {sub:>9}")
        rows.append(dict(op=op, role=role, n=n, primitive=prim, paper_n=N,
                         polarity_split=split, is_identity=is_id,
                         per_site=per, subtotal=sub))

    id_total = learnable_total - nonid_total
    print(f"\n  identities  {id_sites:>3} sites x {params(8, False):>3} = "
          f"{id_total:>6}")
    print(f"  others      {total_sites - id_sites:>3} sites          = "
          f"{nonid_total:>6}")
    print(f"  ANCHOR: identity counted as learnable = {learnable_total} "
          f"(section 4.1 carried 16,611 -> "
          f"{'MATCH' if learnable_total == 16611 else 'MISMATCH'})")

    pasn_naive = args.pasn_naive_bytes // 4
    conventions = {
        "g1_identity_fixed": dict(
            label="paper's G.1 as written -- identity has NO learnable parameters",
            mbe=nonid_total, pasn=args.pasn_params),
        "identity_counted_learnable": dict(
            label="identity counted as a learnable MBE neuron (the old reading)",
            mbe=learnable_total, pasn=args.pasn_params),
        "neither_side_shares": dict(
            label="neutral -- no cross-site sharing on either side",
            mbe=learnable_total, pasn=pasn_naive),
    }
    print(f"\n=== the conventions ===")
    print(f"{'convention':44s} {'MBE':>8s} {'PASN':>8s} {'MBE/PASN':>9s}  verdict")
    for k, c in conventions.items():
        ratio = c["mbe"] / c["pasn"]
        c["ratio"] = ratio
        c["pasn_wins"] = ratio > 1
        print(f"{c['label'][:44]:44s} {c['mbe']:>8,} {c['pasn']:>8,} "
              f"{ratio:>8.2f}x  {'PASN wins' if ratio > 1 else 'PASN loses'}")

    print("\n=== what this means ===")
    print("  * G.1 says the identity's parameters are FIXED, and 218 of our 339")
    print("    primitives are identities. On the paper's own account they are free.")
    print("  * So the honest headline convention is the first row, and we LOSE it.")
    print("  * And our tied identity is NOT a differentiator: the paper already")
    print("    fixes and shares its own ('D ... precomputed and shared across the")
    print("    entire network'). Tying buys us storage against our own untied")
    print("    build (1.99x, P0.4 Block C), not against them.")
    print("  * Parameters are already out of the headline (section 0). This")
    print("    confirms that decision rather than reopening it.")

    out = dict(inventory=rows, total_sites=total_sites, identity_sites=id_sites,
               anchor_16611=learnable_total,
               pasn_params=args.pasn_params, pasn_bytes=args.pasn_bytes,
               pasn_params_no_sharing=pasn_naive,
               paper_n=PAPER_N, conventions=conventions,
               g1_quote=("To mitigate approximation errors in identity mapping, "
                         "all learnable parameters in the MBE neuron are fixed "
                         "as [20]."),
               source="paper appendix G.1 (p.13 image) + results/bank_usage_audit.json")
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
