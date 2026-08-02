"""Per-primitive spike firing rates, against the MBE paper's Table XI.

Table XI is the one place the paper reports a quantity we can put our own number
beside without an accuracy caveat: the fraction of (basis, timestep) slots that
actually fire, per operation, at ``T=16``. It is also the quantity their energy
model is built on -- ``E_MBE = T * eta * N * C * N_h * E_AC`` -- so a firing rate
is directly interpretable rather than being a proxy.

**The comparison is cross-model and that is not a detail.** Table XI is measured
on ViT-M/16 during inference; this measures GPT-2. Two of the rows are close to
architecture-independent because their argument is an IEEE field rather than an
activation (``2^x`` sees ``frac(x log2 e)`` in ``[0,1)``, ``1/x`` sees a mantissa
in ``[0.5,1)``), and those are the rows worth reading closely. GELU and the
identities see real activations, so a difference there is the two models'
distributions as much as the two methods.

Firing rate is ``spikes / (N * T)`` per input element, with ``N`` and ``T``
weighted per element by the bank it routes to -- the routed neuron has no single
``(N, T)``, which is the whole point of the budget rule. ``--t-fixed 16`` matches
the paper's global setting and is the default here for that reason.

**A lower firing rate is not automatically better.** Energy is ``T * eta * N``;
a routed neuron can fire a larger *fraction* of a much smaller number of slots
and still win. Read this table next to the spike totals, never instead of them.

    python experiments/firing_rates.py --model gpt2
    python experiments/firing_rates.py --model gpt2-medium --json results/firing.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import (make_spikable, make_attention_spikable,  # noqa: E402
                              convert_gpt2)

#: Paper Table XI (ViT-M/16 inference, T=16), and which of our primitives is the
#: same operation. ``av_matmul`` has no counterpart in their table.
#: ``(firing rate %, our primitive, their N)``. N is from appendix G.1: 4 for
#: GELU/Tanh, 8 for the identity and 2^x / 1/x / 1/sqrt(x). It is needed because
#: a firing rate alone is not comparable across different basis counts -- their
#: own energy model is ``T * eta * N``, so that product is the honest column.
PAPER_TABLE_XI = {
    "2^x":                      (46.94, "softmax exp",        8),
    "LayerNorm_1/x_identity":   (38.46, "layernorm id_istd",  8),
    "GELU":                     (38.22, "activation",         4),
    "LayerNorm_input_identity": (27.61, "layernorm id_dev",   8),
    "1/sqrt(x)":                (25.27, "layernorm rsqrt",    8),
    "Attention_score":          (8.31,  "matmul qk",          8),
    "1/x":                      (3.74,  "softmax inv",        8),
}
PAPER_T = 16


def role(name: str, kind: str) -> str:
    """Group a primitive path onto the operation the paper names."""
    if kind == "activation":
        return "activation"
    tail = name.rsplit(".", 1)[-1]
    if kind == "layernorm":
        return f"layernorm {tail}"
    if kind == "softmax":
        return f"softmax {tail}"
    if kind == "matmul":
        which = "qk" if "qk_matmul" in name else "av"
        return f"matmul {which}"
    return f"{kind} {tail}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--t-fixed", type=int, default=16,
                    help="match the paper's global T (default); 0 = budget rule")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, attn_implementation="eager").eval()

    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception as exc:                              # offline fallback
        print(f"  (wikitext unavailable: {exc}; using a fixed prompt)")
        text = "The quick brown fox jumps over the lazy dog. " * 400
    ids = tok(text, return_tensors="pt").input_ids[0][:a.tokens].unsqueeze(0)

    print(f"marked {make_spikable(model)} GELU, "
          f"{make_attention_spikable(model)} attention blocks")
    cfg = cv.ConvertConfig(epochs=a.epochs, backend="mbe_pasn", spike_mult=True,
                           pasn_t_fixed=a.t_fixed or None)
    convert_gpt2(model, [ids], cfg=cfg, verbose=False)

    rep = cv.spiking_cost_report(model, ids)
    agg = defaultdict(lambda: dict(spikes=0.0, slots=0.0, elems=0.0, n=0))
    for name, e in rep["primitives"].items():
        r = agg[role(name, e["kind"])]
        r["spikes"] += e["spikes"]
        r["slots"] += e["ops"].get("cmp", 0.0)     # cmp == N*T slots, per element
        r["elems"] += e["elements"]
        r["n"] += 1

    ours = {k: 100.0 * v["spikes"] / max(v["slots"], 1e-30) for k, v in agg.items()}
    # spikes per input element = T * eta * N, the paper's own energy quantity
    ours_spk = {k: v["spikes"] / max(v["elems"], 1e-30) for k, v in agg.items()}

    print(f"\n# firing rate (%), {a.model}, "
          f"T={'16 (matched)' if a.t_fixed else 'per-bank rule'}")
    print(f"{'paper op':<26} {'rate: paper':>12} {'ours':>8}   "
          f"{'spikes/elem: paper':>19} {'ours':>9} {'ratio':>8}")
    print("-" * 92)
    for op, (pap, mine, n_paper) in PAPER_TABLE_XI.items():
        got, got_spk = ours.get(mine), ours_spk.get(mine)
        pap_spk = PAPER_T * (pap / 100.0) * n_paper
        if got is None:
            print(f"{op:<26} {pap:11.2f}% {'(none)':>8}")
            continue
        print(f"{op:<26} {pap:11.2f}% {got:7.2f}%   "
              f"{pap_spk:18.2f} {got_spk:9.2f} {pap_spk / max(got_spk, 1e-9):7.2f}x"
              f"  {mine}")
    print("\n  'spikes/elem' is T * eta * N -- the paper's own energy quantity,"
          " with N from G.1\n  (4 for GELU, 8 elsewhere). ratio > 1 means we emit"
          " fewer spikes for that op.\n  The rate columns are NOT comparable on"
          " their own: our N_j is per bank, theirs is fixed.")

    extra = {k: v for k, v in sorted(ours.items()) if k not in
             {m for _, m, _ in PAPER_TABLE_XI.values()}}
    if extra:
        print("\nno counterpart in Table XI:")
        for k, v in extra.items():
            print(f"  {k:<24} rate {v:6.2f}%   spikes/elem {ours_spk[k]:8.2f}   "
                  f"({agg[k]['n']} primitive(s))")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(model=a.model, t_fixed=a.t_fixed, ours_rate=ours,
                           ours_spikes_per_elem=ours_spk,
                           paper_rate={k: v[0] for k, v in PAPER_TABLE_XI.items()},
                           paper_n={k: v[2] for k, v in PAPER_TABLE_XI.items()}),
                      fh, indent=1)
        print(f"\n[json] {a.json}")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
