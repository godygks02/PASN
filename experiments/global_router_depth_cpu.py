"""Small pretrained GPT-2 CPU pilot for one global router-depth offset.

The current PrefixRouter has no literal k-bit parameter. This experiment shifts
both configured e_min values by the same integer, then selects a single shift
on WikiText validation and reports its untouched test-tail result. It is a
Stage-1 pilot, not a full-model Stage-2 or literal adaptive-prefix result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt2_wikitext import load_wikitext_ids, perplexity, storage_report  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mbe import convert as cv  # noqa: E402
from mbe.gpt2_convert import make_spikable, convert_gpt2  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--dataset-arrow-dir", required=True)
    p.add_argument("--json", type=Path, default=Path("results/global_router_depth_cpu.json"))
    p.add_argument("--deltas", type=int, nargs="+", default=[1, 0, -1],
                   help="add this offset to both baseline e_min values; negative opens deeper banks")
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--val-blocks", type=int, default=8)
    p.add_argument("--test-blocks", type=int, default=8)
    p.add_argument("--test-start-block", type=int, default=16,
                   help="first 16 test blocks were examined in an earlier pilot")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--target-mse", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--val-accuracy-tolerance-pp", type=float, default=0.25,
                   help="among candidates this close to best validation conversion loss, minimize spikes")
    a = p.parse_args()
    if min(a.block, a.val_blocks, a.test_blocks, a.epochs, a.n_steps) <= 0:
        p.error("block, window counts, epochs and n-steps must be positive")
    if not a.deltas or len(set(a.deltas)) != len(a.deltas):
        p.error("provide unique depth deltas")
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    tok = GPT2TokenizerFast.from_pretrained(a.model, local_files_only=True)
    arrow = a.dataset_arrow_dir
    val_ids = load_wikitext_ids(tok, "validation", arrow_dir=arrow)
    test_ids = load_wikitext_ids(tok, "test", arrow_dir=arrow)
    val_ids = val_ids[:a.val_blocks * a.block + 1]
    start = a.test_start_block * a.block
    test_ids = test_ids[start:start + a.test_blocks * a.block + 1]
    if val_ids.numel() < a.val_blocks * a.block + 1 or test_ids.numel() < a.test_blocks * a.block + 1:
        p.error("requested validation or test blocks exceed the token stream")
    train_ids = load_wikitext_ids(tok, "train", max_rows=20000, arrow_dir=arrow)
    calib = [train_ids[i:i + a.block].unsqueeze(0) for i in range(0, 8 * a.block, a.block)]
    if any(x.numel() != a.block for x in calib):
        p.error("not enough calibration tokens")

    base = GPT2LMHeadModel.from_pretrained(a.model, local_files_only=True).eval()
    ann_val = perplexity(base, val_ids, a.block, "cpu", progress_every=0)
    ann_test = perplexity(base, test_ids, a.block, "cpu", progress_every=0)
    del base
    print(f"ANN val={ann_val:.5f} test={ann_test:.5f}", flush=True)

    defaults = cv.ConvertConfig()
    rows = []
    for delta in a.deltas:
        t0 = time.perf_counter()
        model = GPT2LMHeadModel.from_pretrained(a.model, local_files_only=True).eval()
        make_spikable(model)
        cfg = cv.ConvertConfig(
            backend="mbe_pasn", epochs=a.epochs, n_steps=a.n_steps,
            spike_mult=True, pasn_budget="rule", pasn_target_mse=a.target_mse,
            pasn_e_min=defaults.pasn_e_min + delta,
            pasn_id_e_min=defaults.pasn_id_e_min + delta,
            verbose_fits=False,
        )
        convert_gpt2(model, calib, cfg=cfg, only={"activation", "layernorm"}, verbose=False)
        build_s = time.perf_counter() - t0
        ppl_val = perplexity(model, val_ids, a.block, "cpu", progress_every=0)
        ppl_test = perplexity(model, test_ids, a.block, "cpu", progress_every=0)
        # A common first window is sufficient for a wiring pilot; not an
        # evaluation-wide energy claim. Full-window metering is a GPU-run gate.
        cost_val = cv.spiking_cost_report(model, val_ids[:a.block].unsqueeze(0))
        cost_test = cv.spiking_cost_report(model, test_ids[:a.block].unsqueeze(0))
        store = storage_report(model)
        routers = [n.router for _, _, n in cv._spiking_primitives(model)
                   if hasattr(n, "router")]
        row = dict(delta=delta, pasn_e_min=cfg.pasn_e_min,
                   pasn_id_e_min=cfg.pasn_id_e_min,
                   ppl_val=ppl_val, ppl_test=ppl_test,
                   val_conversion_loss_pp=100 * (ppl_val / ann_val - 1),
                   test_conversion_loss_pp=100 * (ppl_test / ann_test - 1),
                   val_spikes_first_block=cost_val["spikes_per_input"],
                   test_spikes_first_block=cost_test["spikes_per_input"],
                   stored_bytes=store["bytes"], n_routers=len(routers),
                   total_banks=sum(r.n_banks for r in routers), build_s=build_s)
        rows.append(row)
        print(f"delta={delta:+d} banks={row['total_banks']} "
              f"val_loss={row['val_conversion_loss_pp']:+.3f}pp "
              f"test_loss={row['test_conversion_loss_pp']:+.3f}pp "
              f"val_spikes={row['val_spikes_first_block']:.0f} "
              f"bytes={row['stored_bytes']} build={build_s:.1f}s", flush=True)
        del model

    best_loss = min(r["val_conversion_loss_pp"] for r in rows)
    eligible = [r for r in rows if r["val_conversion_loss_pp"] <=
                best_loss + a.val_accuracy_tolerance_pp]
    chosen = min(eligible, key=lambda r: (r["val_spikes_first_block"],
                                          r["stored_bytes"],
                                          r["val_conversion_loss_pp"]))
    result = dict(experiment="global_router_depth_cpu_stage1",
                  interpretation="common e_min shift, not literal prefix k",
                  model=a.model, dataset="wikitext-2-raw-v1", scope="activation+layernorm",
                  args=vars(a) | {"json": str(a.json)},
                  ann_val=ann_val, ann_test=ann_test, rows=rows,
                  chosen_delta=chosen["delta"],
                  selection_rule="lowest validation first-block spikes within tolerance of best validation conversion loss")
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"selected delta={chosen['delta']:+d}; saved {a.json}", flush=True)


if __name__ == "__main__":
    main()
