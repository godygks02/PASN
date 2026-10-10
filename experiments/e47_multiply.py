"""E47: spike-driven multiplication -- weight x variable and variable x variable,
with order-1 and order-2 readouts, on real GPT-2 operands.

Products in a converted Transformer come in two kinds:
  w*x  weight times activation (linear layers). x is spike-encoded; with an
       order-1 readout each spike adds a precomputed w*a[t] (one AC), plus one
       AC for the bias term w*b and one for the routed scale 2^e.
  x*y  activation times activation (QK^T, attention*V). Both operands are
       spike-encoded; the product is the coincidence sum
       sum_{t,t'} s_x[t] s_y[t'] a_x[t] a_y[t'] -- one AC per spike *pair*, plus
       the bias cross terms (s_x + s_y + 1 ACs) and one for the scale.
Order 2 cannot be spike-driven: x^ = w1*o + w2*o^2 + b is not linear in the spike
sum, so the operand must be decoded (3 multiplies) and then multiplied exactly.

Encoding an operand (N*T compares, 2 AC per spike, router) is paid once per
element and amortised over its fan-out (how many products reuse it). Energy is
the E45 strict model (45 nm: AC/cmp 0.9, MAC 4.6, mul 3.7 pJ); memory reads are
not priced, and the w*a[t] table the order-1 w*x path needs is reported, not
priced. Exact baseline: one MAC (4.6 pJ) per product term.

Encoders (all on |operand|, sign handled by XOR):
  mbe        global MBE identity, the paper's N=8, T=16, log-uniform fit, no bias
  pasn_rule  tied routed identity, the converter's default (rule, rel 1e-2, ro1)
  greedy_ro1 build_mbe_pasn_split, hybrid target, order-1 readout
  greedy_ro2 the same, order-2 readout
Fitted on a WikiText-2 *train* block, scored on a *validation* block.

  python experiments/e47_multiply.py --layer 6 --json results/e47_multiply_L6.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe import spiking_ops as so  # noqa: E402
from mbe.mbe_pasn import (MBEPASNNeuron, TiedBank, build_mbe_pasn,  # noqa: E402
                          build_mbe_pasn_split)
from mbe.metrics import OP_ENERGY_PJ, storage_bytes  # noqa: E402
from gpt2_wikitext import load_wikitext_ids  # noqa: E402

E_AC, E_CMP, E_MAC = OP_ENERGY_PJ["ac"], OP_ENERGY_PJ["cmp"], OP_ENERGY_PJ["mac"]
ROUTER_OPS = 6.0


# ----------------------------------------------------------------- capture
@torch.no_grad()
def capture(model, ids, layer):
    """x into mlp.c_fc, W of c_fc, and q, k, v, attention probs of one block."""
    blk = model.transformer.h[layer]
    got = {}
    h1 = blk.mlp.c_fc.register_forward_hook(lambda m, i, o: got.__setitem__("x_fc", i[0].detach()))
    h2 = blk.attn.c_attn.register_forward_hook(lambda m, i, o: got.__setitem__("qkv", o.detach()))
    model(ids.unsqueeze(0))
    h1.remove(); h2.remove()
    H = model.config.n_head
    D = model.config.n_embd // H
    q, k, v = got["qkv"][0].split(model.config.n_embd, dim=-1)
    sh = lambda t: t.reshape(-1, H, D).transpose(0, 1).contiguous()      # (H,S,D)
    q, k, v = sh(q), sh(k), sh(v)
    S = q.shape[1]
    sc = q @ k.transpose(1, 2) / math.sqrt(D)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)
    probs = torch.softmax(sc.masked_fill(mask, float("-inf")), dim=-1)
    W = blk.mlp.c_fc.weight.detach()                                      # (in, out)
    return dict(x_fc=got["x_fc"][0], W=W, q=q, k=k, v=v, probs=probs)


# ----------------------------------------------------------------- encoders
def per_element(n, x):
    """(recon of |x| with sign, spikes per element, encode energy per element)."""
    mag = x.abs().reshape(-1)
    if isinstance(n, MBEPASNNeuron):
        with torch.no_grad():
            rec = n(mag)
        idx = n.router.route(mag)
        spk = torch.zeros_like(mag)
        cmp = torch.zeros_like(mag)
        for bi, bank in enumerate(n.bank_mods):
            sel = idx == bi
            if not bool(sel.any()):
                continue
            inp = n._bank_input(mag, sel, n.router.banks[bi])
            core, xin = (bank.proto, bank._rho(inp)) if isinstance(bank, TiedBank) else (bank, inp)
            with torch.no_grad():
                spk[sel] = core.spike_train(xin).sum(dim=(1, 2))
            cmp[sel] = core.cfg.n_basis * core.cfg.n_steps
        enc = E_CMP * cmp + E_AC * 2 * spk + E_CMP * ROUTER_OPS
        order = max(b.cfg.readout_order for b in n.bank_mods)
    else:
        with torch.no_grad():
            rec = n.reconstruct(mag)
            spk = n.spike_train(mag).sum(dim=(1, 2))
        enc = E_CMP * n.cfg.n_basis * n.cfg.n_steps + E_AC * 2 * spk
        order = n.cfg.readout_order
    rec = (rec.reshape(x.shape) * torch.sign(x))
    dec = torch.zeros_like(enc) if order == 1 else torch.full_like(enc, E_MAC * 3)  # o^2, w1*o, w2*o^2
    return rec, spk.reshape(x.shape), enc.reshape(x.shape), dec.reshape(x.shape), order


def build_encoders(fit, r, k_floor, epochs, max_t, e_min_id=-6):
    mag = fit.abs().reshape(-1)
    hi = float(mag.max()) * 1.05 + 1e-6
    e_min, e_max = cv._erange(0.0, hi, e_min_id)
    rms = float(mag.pow(2).mean().sqrt())
    enc = {}
    t0 = time.perf_counter()
    enc["mbe"] = so.calibrate_identity(0.0, hi, n_basis=8, n_steps=16, log_sample=True,
                                       use_cache=False)
    enc["pasn_rule"] = build_mbe_pasn(
        "identity", (0.0, hi), e_min=e_min, e_max=e_max, near0="signed", tied=True,
        target="relative", target_rel=r, budget="rule", readout_order=1,
        n_steps=max_t, n_local=2, n_near0=2, epochs=epochs)
    for ro in (1, 2):
        enc[f"greedy_ro{ro}"] = build_mbe_pasn_split(
            "identity", (0.0, hi), e_min=e_min, e_max=e_max, m_max=3, n_steps=max_t,
            epochs=epochs, target="hybrid", target_rel=r,
            target_mse=(k_floor * r * rms) ** 2, readout_order=ro, alpha_init="auto")
    return enc, hi, round(time.perf_counter() - t0, 1)


# ----------------------------------------------------------------- products
def nrmse(a, b):
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())


def wx_row(name, enc, X, W):
    """C = X @ W, X (M,K) encoded, W (K,N) exact."""
    rec, spk, e_enc, e_dec, order = per_element(enc, X)
    N = W.shape[1]
    exact = X @ W
    approx = rec @ W
    if order == 1:
        bias_scale = 0.0 if name == "mbe" else 2 * E_AC              # w*b, 2^e
        per_term = float(spk.mean()) * E_AC + bias_scale + float(e_enc.mean()) / N
        per_term_no_table = float(spk.mean()) * E_MAC + bias_scale + float(e_enc.mean()) / N
    else:
        per_term = E_MAC + float((e_enc + e_dec).mean()) / N
        per_term_no_table = per_term
    return dict(op="w*x", operand_spikes=float(spk.mean()), out_nrmse=nrmse(approx, exact),
                operand_nrmse=nrmse(rec, X), pj_per_term=per_term,
                pj_per_term_no_table=per_term_no_table, fanout=N,
                bytes=int(storage_bytes(enc)), readout_order=order)


def xy_row(name, enc_a, enc_b, A, B, opname):
    """C = A @ B with both operands encoded; A (...,M,K), B (...,K,N)."""
    ra, sa, ea, da, oa = per_element(enc_a, A)
    rb, sb, eb, db, ob = per_element(enc_b, B)
    M, K, N = A.shape[-2], A.shape[-1], B.shape[-1]
    exact = A @ B
    approx = ra @ rb
    if oa == 1 and ob == 1:
        coinc = float((sa @ sb).sum()) / (exact.numel() * K)           # E[s_a s_b] per term
        extra = 0.0 if name == "mbe" else (float(sa.mean()) + float(sb.mean()) + 2) * E_AC
        per_term = coinc * E_AC + extra + float(ea.mean()) / N + float(eb.mean()) / M
    else:
        coinc = None
        per_term = E_MAC + float((ea + da).mean()) / N + float((eb + db).mean()) / M
    return dict(op=opname, operand_spikes=[float(sa.mean()), float(sb.mean())],
                coincidences_per_term=coinc, out_nrmse=nrmse(approx, exact),
                pj_per_term=per_term, fanout=[N, M],
                bytes=int(storage_bytes(enc_a)) + int(storage_bytes(enc_b)),
                readout_order=max(oa, ob))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--targets", nargs="+", type=float, default=[1e-2, 3e-3])
    p.add_argument("--floor-k", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-t", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--e-min", type=int, default=-6,
                   help="identity router depth (convert's pasn_id_e_min is -6)")
    p.add_argument("--ops", nargs="+", default=["wx", "qk", "pv"], choices=["wx", "qk", "pv"])
    p.add_argument("--json", required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).eval()
    tr = load_wikitext_ids(tok, "train", False, max_rows=20000)[: args.block]
    va = load_wikitext_ids(tok, "validation", False)[: args.block]
    fit, ev = capture(model, tr, args.layer), capture(model, va, args.layer)

    out = Path(args.json)
    rec = dict(experiment="e47_multiply", status="running", argv=sys.argv[1:],
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), layer=args.layer,
               e_min=args.e_min, ops=args.ops,
               energy=dict(ac=E_AC, cmp=E_CMP, mac=E_MAC, router_ops=ROUTER_OPS),
               exact_pj_per_term=E_MAC, rows=[])
    for r in args.targets:
        encs = {}
        need = {"wx": ["x_fc"], "qk": ["q", "k"], "pv": ["probs", "v"]}
        for key in [k for o in args.ops for k in need[o]]:
            encs[key], hi, bs = build_encoders(fit[key], r, args.floor_k, args.epochs,
                                               args.max_t, args.e_min)
            print(f"[r={r:g}] encoders for {key}: hi={hi:.3g} ({bs}s)", flush=True)
        for name in ("mbe", "pasn_rule", "greedy_ro1", "greedy_ro2"):
            rows = []
            if "wx" in args.ops:
                rows.append(wx_row(name, encs["x_fc"][name], ev["x_fc"], ev["W"]))
            if "qk" in args.ops:
                rows.append(xy_row(name, encs["q"][name], encs["k"][name], ev["q"],
                                   ev["k"].transpose(1, 2), "q*k (QK^T)"))
            if "pv" in args.ops:
                rows.append(xy_row(name, encs["probs"][name], encs["v"][name], ev["probs"],
                                   ev["v"], "p*v (attn*V)"))
            for row in rows:
                row.update(encoder=name, target_rel=r if name != "mbe" else None,
                           e_min=args.e_min if name != "mbe" else None,
                           exact_pj_per_term=E_MAC,
                           vs_exact=E_MAC / row["pj_per_term"])
                rec["rows"].append(row)
                extra = (f" no-table {row['pj_per_term_no_table']:.2f}"
                         if "pj_per_term_no_table" in row else "")
                print(f"r={r:g} {name:10s} {row['op']:12s} out_nrmse={row['out_nrmse']:.3e} "
                      f"spk={row['operand_spikes']} pJ/term={row['pj_per_term']:.2f}{extra} "
                      f"(exact {E_MAC}) bytes={row['bytes']}", flush=True)
            out.write_text(json.dumps(rec, indent=1))
    rec["status"] = "ok"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
