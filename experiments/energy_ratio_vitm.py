"""ViT-M/16 energy ratio against the MBE paper's G.4 -- the one cost cell it prints.

The paper reports **17.29 %** for ``E_SNN / E_ANN`` on ViT-M/16 (appendix G.4,
Eq. 28). That is the only *cost* number it prints for a whole model, and E8-M
converted the same timm checkpoint, so the cell looks directly fillable. It is
not, and this script exists to say exactly why and to compute what *is*
comparable.

**Three readings had to be pinned before any arithmetic** (all from
``paper/appendix_p15.png``, transcribed in the vault reference note):

1. ``E_MBE = T x eta x N x C x N_h x E_AC``, where the paper defines *"T is the
   timestep, eta is the average firing rate, **N is the number of tokens**, C is
   the number of token channels, and N_h is the number of heads"*. So
   ``N * C * N_h`` is the element count and **the basis count does not appear**.
   Eq. (5)-(8) of the main text give every basis its own membrane potential,
   threshold and Heaviside spike, so a B-basis neuron physically emits B spike
   trains. Their energy model therefore prices a B=8 neuron identically to a
   B=1 one. We report both readings rather than pick one:
   ``as_printed`` (spikes/elem = T*eta) and ``per_basis`` (x B from G.1).

   NOTE this also corrects our own vault note "Table XI 발화율 대조", which built
   its ``T*eta*N`` column with ``N`` = basis count and attributed that product to
   *"논문 자신의 에너지 식에 들어가는 양"*. The paper's formula does not contain
   the basis count. The physics may, but the citation did not.

2. **Eq. 28 is a whole-network accounting, not a nonlinear-operator one.**
   ``E_AC/E_MAC = 0.9/4.6 = 19.57 %``, so 17.29 % forces ``SOPs/FLOPs = 0.884``.
   The converted nonlinear operators alone come to ``SOPs/FLOPs ~ 0.15`` (printed
   below), i.e. **6x short**. 17.29 % is only reachable if the linear/weight
   matmuls are counted as spike operations too -- the standard "the whole network
   is spike-driven" accounting. Our converter leaves weight matmuls exact and
   never counts them, so **our number and their 17.29 % are not the same
   quantity** and must not be printed side by side.

3. What *is* comparable is converted-operator against converted-operator, on the
   same model, same global ``T=16``, same ``E_AC``, same operator set. That is
   what this script computes.

The honest headline is the **paper-currency** row: priced the way the paper
prices itself (spikes x E_AC, nothing else charged), PASN is ~5x cheaper on the
operators either method actually replaces. Our ``strict`` row is *higher* than
their reconstruction, and that is not a loss -- it charges us for threshold
compares, router arithmetic and decoder powers that G.4 charges nobody for. The
compare class scales as ``N*T`` per element, which is precisely where a solved
per-bank budget wins, so **the paper's own currency understates our margin**.

Run::

    python experiments/energy_ratio_vitm.py
    python experiments/energy_ratio_vitm.py --json results/energy_ratio_vitm.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mbe.metrics import OP_ENERGY_PJ, op_energy_pj  # noqa: E402

E_AC = OP_ENERGY_PJ["ac"]    # 0.9 pJ -- identical to the paper's G.4 constant
E_MAC = OP_ENERGY_PJ["mac"]  # 4.6 pJ -- ditto

#: Table XI (appendix G.4), firing rates at T=16, measured by the paper during
#: ViT-M/16 inference. Keys are the paper's own operation names.
PAPER_FIRING = {
    "attention_score": 0.0831,
    "exp2": 0.4694,
    "inv": 0.0374,
    "invsqrt": 0.2527,
    "ln_input_identity": 0.2761,
    "ln_inv_identity": 0.3846,
    "gelu": 0.3822,
}

#: G.1 basis counts. GELU/Tanh use N=4; identity, 2^x, 1/x, 1/sqrt(x) use N=8.
PAPER_BASIS = {
    "attention_score": 8, "exp2": 8, "inv": 8, "invsqrt": 8,
    "ln_input_identity": 8, "ln_inv_identity": 8, "gelu": 4,
}

T = 16


def geometry() -> dict:
    """ViT-M/16 shape. Queried from timm when available, else the known config."""
    cfg = dict(name="vit_medium_patch16_reg4_gap_256", img=256, patch=16,
               depth=12, dim=512, heads=8, mlp=2048, prefix=4, classes=1000)
    try:
        import timm
        m = timm.create_model(cfg["name"], pretrained=False)
        cfg.update(depth=len(m.blocks), dim=m.embed_dim,
                   heads=m.blocks[0].attn.num_heads,
                   mlp=m.blocks[0].mlp.fc1.out_features,
                   prefix=m.num_prefix_tokens, patches=m.patch_embed.num_patches,
                   source="timm")
    except Exception as exc:                                  # pragma: no cover
        cfg.update(patches=(cfg["img"] // cfg["patch"]) ** 2,
                   source=f"constants ({type(exc).__name__})")
    cfg["tokens"] = cfg["patches"] + cfg["prefix"]
    return cfg


def ann_macs(g: dict) -> dict:
    """ANN multiply-accumulates per image. The denominator of Eq. 28."""
    S, d, m = g["tokens"], g["dim"], g["mlp"]
    per_block = dict(
        qkv=S * d * 3 * d,
        qk=S * S * d,          # heads * S * S * d_head == S * S * d
        av=S * S * d,
        proj=S * d * d,
        fc1=S * d * m,
        fc2=S * m * d,
    )
    block = sum(per_block.values())
    total = (g["patches"] * (g["patch"] ** 2 * 3) * d
             + g["depth"] * block
             + d * g["classes"])
    return dict(per_block=per_block, block=block, total=total)


def converted_elements(g: dict) -> dict:
    """Elements each converted operator sees per image, by the paper's op names.

    Stage 2 scope (their Algorithm 1, our converter): activation, LayerNorm,
    Softmax, and activation-x-activation matmul. Weight matmuls stay exact in
    both methods and appear in neither numerator.

    ``attention_score`` is the FP multiply and is measured in *matmul operations*
    (MO in ``E_FP``), not elements -- both QK^T and AV.
    """
    S, d, m, L, H = g["tokens"], g["dim"], g["mlp"], g["depth"], g["heads"]
    return dict(
        gelu=L * S * m,                 # one activation per MLP hidden unit
        exp2=L * H * S * S,             # softmax numerator, per score
        inv=L * H * S,                  # softmax denominator, one per row
        ln_input_identity=L * 2 * S * d,  # 2 LayerNorms per block, per element
        ln_inv_identity=L * 2 * S,      # per token
        invsqrt=L * 2 * S,              # per token
        attention_score=L * 2 * S * S * d,   # MO for E_FP: QK^T and AV
    )


def paper_spikes(elems: dict, per_basis: bool) -> dict:
    """Spikes per image for the paper's neurons, from Table XI + G.1.

    Non-multiply operators use ``E_MBE``: spikes/elem = ``T * eta`` as printed,
    or ``T * eta * B`` under the per-basis reading. The FP multiply uses their
    separate ``E_FP = T^2 * eta_1 * eta_2 * MO``; the per-basis reading is *not*
    applied to it, because coincidence counting across two B-basis operands is
    not specified by the paper and squaring B would be our invention, not theirs.
    """
    out = {}
    for op, n in elems.items():
        if op == "attention_score":
            eta = PAPER_FIRING[op]
            out[op] = (T ** 2) * eta * eta * n
        else:
            mult = PAPER_BASIS[op] if per_basis else 1
            out[op] = T * PAPER_FIRING[op] * mult * n
    return out


def mbe_all_op(elems: dict) -> dict:
    """MBE's *all-op* energy on the nonlinear operators, per image.

    The paper's G.4 charges spikes only. Our own accounting (``neuron_op_cost``)
    charges four more classes, and for a plain MBE neuron all of them are closed
    form -- no run is needed to price MBE the way we price ourselves:

        cmp  = B * T          per element   (Eq. 6: one Heaviside per basis per step)
        ac   = 2 * spikes                   (each spike decrements u and adds into o)
        mac  = B * r          r = 1         (Eq. 8 is a linear readout, sum w(n) o(n))
        poly = B * (r - 1) = 0
        router = 0                          (no routing to pay for)

    ⚠️ **This forces the per-basis reading.** Charging ``B*T`` compares -- one per
    basis per timestep -- is incoherent with the as-printed spike count ``T*eta``,
    which has no basis factor. If every basis has its own threshold it has its own
    spike train. So an all-op comparison cannot be quoted next to the 5.27x
    spike-only figure, which uses the as-printed reading: they are two different
    readings of the same paper.

    🔴 **The FP multiply is excluded and cannot be added without inventing
    structure.** ``E_FP = T^2*eta1*eta2*MO`` is a coincidence model; the paper never
    says how the two operands are reconstructed, so MBE's compare count on that path
    is not derivable. That is exactly the path where our spike-only margin lives
    (93.5%), so **do not compare this number against a PASN figure that includes
    matmul.** Nonlinear operators only, both sides.
    """
    out = {}
    for op, n in elems.items():
        if op == "attention_score":
            continue
        B, eta = PAPER_BASIS[op], PAPER_FIRING[op]
        spikes = T * eta * B * n                  # per-basis reading, see above
        out[op] = dict(cmp=B * T * n, ac=2.0 * spikes, mac=float(B) * n,
                       poly=0.0, router=0.0, bitop=0.0, spikes=spikes)
    return out


def iso_compare_table(path: str = "results/op_pareto_merged.json") -> None:
    """Per-operator compare counts at matched accuracy -- the scope-clean version.

    ``cmp`` is ``bases * T`` per element for both neuron types, so the honest
    comparison is at equal approximation error on the same operator. For each
    operator we take the most accurate MBE point PASN can match or beat, then the
    cheapest PASN point that matches it. ``mbe`` rows carry ``budget`` (= N) and
    ``T``; PASN rows carry the per-element means the budget rule produced.

    Caveat: ``n_mean * t_mean`` is the product of two means, not the mean of the
    product, so it is exact only when the rule hands one (N, T) to every element.
    Treat it as accurate to within the spread of the per-bank budgets.
    """
    try:
        recs = json.load(open(path, encoding="utf-8"))["records"]
    except Exception as exc:                                   # pragma: no cover
        print(f"\n[iso-accuracy compare table unavailable: {exc}]")
        return
    print("\n## Compares at iso-accuracy, per operator (arm = pasn_rule, what we ship)")
    print(f"  {'operator':<22}{'nrmse':>9}{'MBE N*T':>9}{'PASN N*T':>10}{'ratio':>9}")
    ratios = []
    for op in sorted({r["op"] for r in recs}):
        mb = [r for r in recs if r["op"] == op and r["backend"] == "mbe"]
        pa = [r for r in recs if r["op"] == op and r["backend"] == "pasn_rule"]
        pair = None
        for m in sorted(mb, key=lambda r: r["nrmse"]):
            cand = [p for p in pa if p["nrmse"] <= m["nrmse"]]
            if cand:
                pair = (m, min(cand, key=lambda r: r["n_mean"] * r["t_mean"]))
                break
        if pair is None:
            continue
        m, p = pair
        mnt, pnt = m["budget"] * m["T"], p["n_mean"] * p["t_mean"]
        ratios.append(mnt / pnt)
        print(f"  {op:<22}{m['nrmse']:>9.1e}{mnt:>9.0f}{pnt:>10.1f}{mnt/pnt:>8.2f}x")
    if ratios:
        s = sorted(ratios)
        print(f"  -> {len(ratios)}/{len(ratios)} operators, median "
              f"{s[len(s)//2]:.2f}x fewer compares, range "
              f"{s[0]:.2f}x-{s[-1]:.2f}x, at matched or better accuracy.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="results/e8m_vit_medium.json")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    g = geometry()
    macs = ann_macs(g)
    elems = converted_elements(g)

    rec = json.load(open(a.record, encoding="utf-8"))
    rec = (rec if isinstance(rec, list) else rec.get("records", [rec]))[0]

    # ---- our side: measured, normalised per image -------------------------
    # ``spikes_per_input`` / ``ops_per_input`` are per element of the *model
    # input tensor* (3 x H x W for a ViT), so scale by one image's pixels.
    px = 3 * g["img"] * g["img"]
    # The cost report runs one forward, so ``by_kind`` is a batch total. Recover
    # the batch size from the record's own two figures rather than trusting
    # ``batch_size`` (the cost probe need not use the eval batch).
    total_spikes = sum(rec["by_kind"].values())
    n_images = round(total_spikes / rec["spikes_per_input"] / px)
    ours_spikes = total_spikes / n_images
    ours_ops = {k: v * px for k, v in rec["ops_per_input"].items()}
    ours_strict_pj = rec["energy_pj_per_input"] * px
    ours_paper_currency_pj = ours_spikes * E_AC

    e_ann_pj = macs["total"] * E_MAC

    rows = {}
    for name, per_basis in (("as_printed", False), ("per_basis", True)):
        sp = paper_spikes(elems, per_basis)
        rows[name] = dict(spikes=sum(sp.values()), by_op=sp,
                          pj=sum(sp.values()) * E_AC)

    w = 34
    print(f"# ViT-M/16 ({g['source']}): {g['tokens']} tokens, d={g['dim']}, "
          f"{g['depth']} blocks, {g['heads']} heads")
    print(f"# ANN: {macs['total']/1e9:.3f} GMAC/image "
          f"-> E_ANN = {e_ann_pj/1e9:.2f} mJ/image at E_MAC={E_MAC} pJ\n")

    print("## Converted-operator energy per image (same scope, same T=16, same E_AC)")
    print(f"{'':<{w}}{'spikes/img':>14}{'pJ/img':>14}{'% of E_ANN':>12}")
    for name, r in rows.items():
        print(f"{'MBE (paper), ' + name:<{w}}{r['spikes']/1e6:>12.1f}M"
              f"{r['pj']/1e9:>12.3f}mJ{100*r['pj']/e_ann_pj:>11.2f}%")
    print(f"{'PASN, paper currency':<{w}}{ours_spikes/1e6:>12.1f}M"
          f"{ours_paper_currency_pj/1e9:>12.3f}mJ"
          f"{100*ours_paper_currency_pj/e_ann_pj:>11.2f}%")
    print(f"{'PASN, strict (cmp+router+poly)':<{w}}{ours_spikes/1e6:>12.1f}M"
          f"{ours_strict_pj/1e9:>12.3f}mJ{100*ours_strict_pj/e_ann_pj:>11.2f}%")

    print("\n## Ratio, paper's own currency (spikes x E_AC, nothing else charged)")
    for name, r in rows.items():
        print(f"  MBE {name:<12} / PASN = {r['pj']/ours_paper_currency_pj:>6.2f}x")

    print("\n## Why Eq. 28's 17.29% is not this number")
    sop_over_flop = {n: r["spikes"] / macs["total"] for n, r in rows.items()}
    print(f"  E_AC/E_MAC                      = {100*E_AC/E_MAC:.2f}%")
    print(f"  17.29% implies SOPs/FLOPs       = {0.1729*E_MAC/E_AC:.3f}")
    for n, v in sop_over_flop.items():
        print(f"  converted ops only ({n:<10}) = {v:.3f}"
              f"   ({0.1729*E_MAC/E_AC/max(v,1e-9):.1f}x short)")
    print("  -> Eq. 28 must count weight matmuls as spike ops. We do not, so")
    print("     ours is NOT comparable to 17.29%. Compare converted-op rows only.")

    # ---- the split that decides how the headline may be phrased ------------
    # Their side is 93% one term (the FP multiply). Ours is not. Quoting a
    # single aggregate ratio hides that the two halves move in *opposite*
    # directions under the as-printed reading, so both halves get printed.
    ours_fp = rec["by_kind"]["matmul"] / n_images
    ours_rest = ours_spikes - ours_fp
    print("\n## The split that decides the wording -- the halves disagree")
    print(f"{'':<26}{'MBE':>12}{'PASN':>12}{'ratio':>10}")
    for name, r in rows.items():
        fp = r["by_op"]["attention_score"]
        rest = r["spikes"] - fp
        if name == "as_printed":
            print(f"{'FP multiply (act x act)':<26}{fp/1e6:>10.1f}M{ours_fp/1e6:>10.1f}M"
                  f"{fp/ours_fp:>9.2f}x")
        print(f"{'nonlinear ops, ' + name:<26}{rest/1e6:>10.1f}M{ours_rest/1e6:>10.1f}M"
              f"{rest/ours_rest:>9.2f}x")
    print("  -> the FP multiply carries the aggregate win (14.6x). On the")
    print("     nonlinear operators the as-printed reading has us LOSING 1.9x;")
    print("     only the per-basis reading wins them 3.3x. Do not quote the")
    print("     aggregate without saying which half it came from.")

    print("\n## Where the paper's converted-op total sits (as_printed)")
    for op, s in sorted(rows["as_printed"]["by_op"].items(), key=lambda kv: -kv[1]):
        print(f"  {op:<20}{s/1e6:>10.1f}M spikes{100*s/rows['as_printed']['spikes']:>8.1f}%")
    print("  ^ the FP multiply dominates their side; that term rests on their")
    print("    eta=8.31% and the T^2 coincidence model, not on Table XI alone.")

    # ---- compares: the class G.4 charges nobody for --------------------------
    # Eq. (6), s_n[t] = H(u_n[t] - Vth_n[t]), is one Heaviside compare per basis
    # per timestep, so *both* neurons pay ``bases * T`` compares per element
    # whether or not they fire. G.4 charges neither. Our strict row charges us.
    #
    # A network-level compare ratio is NOT computed here, and the reason is a
    # scope trap worth recording: MBE's nonlinear-op compares are countable
    # (B*T*elements) but its FP-multiply path is not -- ``E_FP`` is a coincidence
    # model that never says how the two operands get reconstructed. Our measured
    # ``cmp`` includes that path (matmul is 34% of our spikes). Dividing the two
    # compares a 6-operator total against a 7-operator one and lands near 1.0x,
    # which understates the real gap by roughly the size of the path it drops.
    mbe_cmp_nonlin = sum(PAPER_BASIS[op] * T * n for op, n in elems.items()
                         if op != "attention_score")
    print("\n## Compares (the class G.4 omits for both sides)")
    print(f"  MBE, nonlinear ops only (B*T*elem) = {mbe_cmp_nonlin/1e6:>8.1f}M"
          f" -> {mbe_cmp_nonlin*E_AC/1e9:.3f} mJ")
    print(f"  PASN, all ops (measured)           = {ours_ops['cmp']/1e6:>8.1f}M"
          f" -> {ours_ops['cmp']*E_AC/1e9:.3f} mJ")
    print("  ^ do NOT divide these: different operator sets. The matched")
    print("    statement is per-operator and iso-accuracy, printed below.")
    iso_compare_table()

    # ---- MBE priced the way we price ourselves (nonlinear operators only) ----
    ma = mbe_all_op(elems)
    tot = {k: sum(v[k] for v in ma.values())
           for k in ("cmp", "ac", "mac", "poly", "router", "bitop")}
    mbe_allop_pj = op_energy_pj(tot)
    print("\n## MBE under OUR accounting -- nonlinear operators only, per image")
    print(f"  {'cmp':<8}{tot['cmp']/1e6:>10.1f}M ops{tot['cmp']*E_AC/1e9:>10.3f} mJ")
    print(f"  {'ac':<8}{tot['ac']/1e6:>10.1f}M ops{tot['ac']*E_AC/1e9:>10.3f} mJ")
    print(f"  {'mac':<8}{tot['mac']/1e6:>10.1f}M ops{tot['mac']*E_MAC/1e9:>10.3f} mJ")
    print(f"  {'TOTAL':<8}{'':>10}    {mbe_allop_pj/1e9:>14.3f} mJ")
    print("  ^ per-basis reading (forced -- see mbe_all_op docstring). FP multiply")
    print("    EXCLUDED: not derivable. Do NOT divide this by a PASN number that")
    print("    includes matmul, and do NOT quote it beside the 5.27x spike-only ratio.")

    print("\n## Our strict accounting, by class (the classes G.4 omits)")
    for k, v in sorted(ours_ops.items(), key=lambda kv: -kv[1] * OP_ENERGY_PJ.get(kv[0], E_AC)):
        pj = v * OP_ENERGY_PJ.get(k, E_AC)
        print(f"  {k:<10}{v/1e6:>10.1f}M ops{pj/1e9:>10.3f}mJ"
              f"{100*pj/ours_strict_pj:>8.1f}%")

    if a.json:
        json.dump(dict(geometry=g, ann_macs=macs, elements=elems,
                       e_ann_pj=e_ann_pj, paper=rows,
                       ours=dict(spikes=ours_spikes, ops=ours_ops,
                                 strict_pj=ours_strict_pj,
                                 paper_currency_pj=ours_paper_currency_pj),
                       record=a.record, tag=rec.get("tag")),
                  open(a.json, "w", encoding="utf-8"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
