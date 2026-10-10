"""E42 step 1: the real GPT-2 input distribution of each routed primitive.

Replays the G1 calibration recipe -- gpt2-medium, the first 8 x 512 tokens of the
WikiText-2 train split, ``convert.calibrate`` with Stage 2 markers -- and derives
what each primitive is actually fed, exactly as ``convert.py`` derives it:

  gelu     pooled MLP activation inputs of all 24 blocks (``_shared_fit_slots``),
           domain = [min, max] padded by ``margin``
  exp2     frac(z), z = (logit - rowmax) log2 e        (``_routed_softmax``)
  inv      frexp mantissa of the softmax row sum, [0.5, 1)
  invsqrt  parity-fixed mantissa of the LayerNorm variance, [0.5, 2)
           (``SpikingLayerNorm.forward``)

Softmax rows are pooled over the 24 layers (convert fits per layer; this is the
network-wide mix). Saves ``{name: dict(domain, route, sample)}`` to a .pt file.

  python experiments/e42_gpt2_samples.py --out logs/e42_gpt2_samples.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import make_attention_spikable, make_spikable  # noqa: E402
from gpt2_wikitext import load_wikitext_ids  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--keep", type=int, default=200_000,
                   help="rows kept per slot; G1 used 20k, which leaves ~930 softmax/LN rows")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    t0 = time.time()
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).eval()
    make_spikable(model)
    make_attention_spikable(model)
    cids = load_wikitext_ids(tok, "train", False, max_rows=20000)[: 64 * args.block]
    calib = [cids[i:i + args.block].unsqueeze(0)
             for i in range(0, args.blocks * args.block, args.block)]
    rec = cv.calibrate(model, calib, keep=args.keep)
    shared = cv._shared_fit_slots(model, rec)
    cfg = cv.ConvertConfig()
    print(f"calibrated in {time.time() - t0:.0f}s; kinds: "
          f"{sorted(set(rec.kinds.values()))}", flush=True)

    out = {}
    # GELU: one pooled slot shared by all blocks.
    act = next(shared[n][0] for n, k in rec.kinds.items() if k == "activation")
    s = act.sample
    lo, hi = float(s.min()), float(s.max())
    span = max(hi - lo, 1e-6)
    lo, hi = lo - cfg.margin * span, hi + cfg.margin * span
    e_min, e_max = cv._erange(lo, hi, cfg.pasn_e_min)
    g = torch.Generator().manual_seed(0)
    s = s[torch.randperm(s.numel(), generator=g)[:400_000]]   # domain kept from all
    out["gelu"] = dict(domain=(lo, hi), route=dict(e_min=e_min, e_max=e_max),
                       sample=s.float())

    # Softmax: replay the op's own decomposition, pooled over layers.
    fr, mn = [], []
    for n, k in rec.kinds.items():
        if k != "softmax":
            continue
        slot = rec.ranges[n][0]
        logits, w = slot.sample, slot.width
        if logits is None or w <= 1 or logits.numel() < w:
            continue
        rows = logits[: (logits.numel() // w) * w].reshape(-1, w)
        z = (rows - rows.max(dim=-1, keepdim=True).values) * math.log2(math.e)
        # Causally masked logits are -inf: e^x = 0 there, and they feed no exp2
        # primitive (the spiking op never sees them), so drop them from frac.
        fin = torch.isfinite(z)
        zs = torch.where(fin, z, torch.zeros_like(z))
        zf = torch.floor(zs)
        fr.append((zs - zf)[fin].reshape(-1))
        exp_x = torch.where(fin, torch.ldexp(torch.exp2(zs - zf), zf.to(torch.int64)),
                            torch.zeros_like(z))
        m_, _ = torch.frexp(exp_x.sum(dim=-1))
        mn.append(m_.reshape(-1))
    e_lo, e_hi = cv._erange(0.0, 1.0, cfg.pasn_e_min)
    out["exp2"] = dict(domain=(0.0, 1.0), route=dict(e_min=e_lo, e_max=e_hi),
                       sample=torch.cat(fr).float())
    beta = (cfg.pasn_beta or {}).get("inv", 0.0)
    b_lo, b_hi = cv._erange_beta(0.5, 1.0, beta, cfg.pasn_beta_binades)
    out["inv"] = dict(domain=(0.5, 1.0), route=dict(e_min=b_lo, e_max=b_hi, beta=beta),
                      sample=torch.cat(mn).float())

    # LayerNorm variance mantissa (rsqrt argument).
    ln = next(shared[n][0] for n, k in rec.kinds.items() if k == "layernorm")
    D = ln.width
    x = ln.sample[: (ln.sample.numel() // D) * D].reshape(-1, D)
    dev = x - x.mean(dim=-1, keepdim=True)
    eps = next(model.get_submodule(n).eps for n, k in rec.kinds.items() if k == "layernorm")
    var = (dev * dev).mean(dim=-1) + eps
    m, e = torch.frexp(var)
    m = torch.where(e % 2 != 0, m * 2.0, m)
    out["invsqrt"] = dict(domain=(0.5, 2.0), route=dict(e_min=-2, e_max=1),
                          sample=m.float())

    for k, v in out.items():
        s = v["sample"]
        print(f"{k:8s} n={s.numel():7d} domain={tuple(round(a, 4) for a in v['domain'])} "
              f"route={v['route']} q01/50/99={[round(float(q), 4) for q in torch.quantile(s[:100000], torch.tensor([.01, .5, .99]))]}",
              flush=True)
    torch.save(dict(meta=dict(model=args.model, block=args.block, blocks=args.blocks,
                              keep=args.keep, split="train", calib_offset=0),
                    prims=out), args.out)


if __name__ == "__main__":
    main()
