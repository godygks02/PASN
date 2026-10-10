# Vast.ai experiment proposal: fitted per-bank N/T on pretrained GPT-2

Status: design only. No Vast.ai instance has been rented. This proposal does not
change the frozen paper build or `PAPER_PLAN.md` priorities.

## 1. Question and existing evidence

Question: after offline fitting selects each bank's number of MBE bases `N_j` and
timesteps `T_j`, does the *whole converted network* use fewer spikes or less
modeled energy at the same conversion accuracy than the existing analytic rule?
Inference remains input-routed; architecture selection itself happens offline.
The prefix depth, bank boundaries, beta, and identity tying stay fixed so that
the experiment isolates `(N_j, T_j)` selection.
Here `k` means the routing prefix length in `PASN_method.md`. The current
`PrefixRouter` uses a fixed sign/exponent binade partition with a configured
`e_min`; it has no per-bank prefix-length parameter. This experiment tests
adaptive `N` and `T` only. Adaptive `k` would require a separate router
partition search (global depth or variable-depth split/merge tree), including
router/extra-bank costs and a new matched-network ablation.

The local pretrained `gpt2` Stage-2 pilot at `T <= 4, N <= 2`, 10 fit epochs,
16 x 64 test tokens found 2.91% fewer spikes and 15.40% fewer stored bytes for
`fit`, but **worse** PPL (99.4714 vs rule 99.1079, same ANN 98.2637). Its first
four blocks had the opposite accuracy ordering. See
`results/fit_budget_gpt2_stage2_report.md`. This motivates a larger independent
evaluation; it is not evidence that `fit` preserves network accuracy.

## 2. Experimental arms and estimand

Use the same pretrained checkpoint, calibration samples, router, operation set,
maximum T, fit epochs, fitting seed, evaluation tokens, hardware and cost meter.

| Arm | Per-bank budget | Purpose |
|---|---|---|
| R | `pasn_budget=rule` | Existing analytic allocation; direct control |
| F | `pasn_budget=fit` | Fitted offline `(N_j,T_j)` selector |
| A, optional | R with `pasn_t_fixed=16` | Historic fixed-T anchor; not the primary control |
| P, secondary | Permute F's selected `(N,T)` pairs among compatible banks of the same primitive, preserving its budget histogram | Test whether *placement* learned by fitting matters beyond total budget |

For P, hold the fitted basis parameters fixed, permute only the architecture
assignment where legal, and use several preregistered permutation seeds. If
architecture and fitted parameters cannot be separated consistently, rebuild
each permuted architecture with the same fitting budget and call it a separate
control. Do not treat P as part of the primary R/F comparison.

Primary result: the paired difference in relative conversion loss,
`100 * (PPL_SNN/PPL_ANN - 1)`, together with total measured model spikes per
scored token on the *same* held-out windows. Report stored bytes and modeled
operation energy (router/comparison/MAC/state traffic included under a stated
cost model) as separate axes. Report GPU inference latency separately: different
`T_j` values do not imply lower end-to-end latency if the implementation waits
for the largest T. Offline build time and peak VRAM are separate costs.

Do not call F a win merely because its standalone bank MSE is lower. The primary
claim requires a matched network PPL and fewer total spikes; report all points
when no accuracy match exists. Do not mix spike-only and full-operation energy.

## 3. Required code gates before renting

1. Add `--eval-split validation|test` and `--eval-start-token` to
   `experiments/gpt2_wikitext.py`. Selection/tuning uses WikiText-2 validation.
   The first 16 x 64 test blocks were inspected in the CPU pilot; omit the
   corresponding first 1024 test tokens from the untouched primary test tail.
   A full-test number can be supplementary with that exposure disclosed.
2. Record, for every fitted primitive and reachable bank: occupancy, all tested
   `(N,T)`, selection MSE/spikes, selected candidate, target feasibility, and
   whether a tied magnitude prototype or an ordinary bank was used. Record
   aggregate counts of infeasible banks. Save complete `ConvertConfig`, commit
   SHA, package versions, model/tokenizer revision and dataset fingerprint.
3. Count spikes and operation classes over the *evaluation windows*, not only
   the first sample batch currently used by `spiking_cost_report`. Keep the
   old one-batch cost as a diagnostic, labelled as such.
4. Benchmark the selector's candidate-fit time. At `T=16, N<=4`, the current
   grid is 4 x 7 = 28 fits per bank; do not launch that full grid blindly.
   Predefine a smaller grid or a staged search, including the rule-selected
   `(N,T)` as a candidate, and validate that pruning does not change the selected
   bank budgets on a representative op sample. Freeze this selector before
   opening the final test split.
5. Verify on a tiny model that R and F use identical calibration token IDs and
   produce different bank-budget tables. Make target misses explicit; a failed
   local target must not be silently counted as a feasible architecture.
6. Implement P only if the architecture permutation has a clear interpretation
   under the tied magnitude banks. Predeclare compatible groups and check that
   each shuffled assignment is valid before using its network result.

## 4. GPU stages and stop conditions

**G0, environment and baseline.** Rent one reliable on-demand Vast.ai GPU and
keep all paired arms on that same host. Start with one GPU with at least 24 GB
VRAM for `gpt2-medium` (345M), ample CPU/RAM for calibration, and at least
80 GB disk for the environment/cache/logs. Check `nvidia-smi`, dependencies,
model loading and ANN PPL on the canonical recipe. The historical full-test
`gpt2-medium` ANN reference is about 21.7058 at context/stride 1024; a short
validation prefix is a timing check, not an absolute-PPL reproduction gate.

**G1, cost and mechanism pilot.** On `gpt2-medium`, Stage 2 (`--convert-ops all`),
run R and F at T cap 8, `N<=2`, a fixed limited fit budget and 32 preselected
validation windows. Measure build seconds/candidate, peak VRAM, PPL, full-model
spikes, modeled energy and stored bytes. Use the same calibration offset 0.
If a bank target is infeasible, report it. If F loses >0.1 percentage point in
relative conversion loss *and* saves <5% full-model spikes, stop the expensive
T=16 test and diagnose the selector. G1 is a spending gate, not a result to
publish as the final claim.

**G2, primary comparison.** Only after G1 passes, run R and F on
`gpt2-medium` at T cap 16 with `--epochs 300`, Stage 2, the frozen fitting
grid and target MSE `1e-5`. Choose any alternate target from a declared set
on validation only; run both arms under the same selection protocol. Lock the
configs, then evaluate the untouched WikiText-2 test tail with context 1024
and stride 1024. The primary record is PPL conversion loss versus eval-wide
spikes/energy/bytes, not an isolated best point selected after seeing test PPL.
Use the same GPU box for both runs. For a meaningful efficiency claim, predeclare
network noninferiority of at most +0.1 percentage point conversion loss and
at least 10% fewer evaluation-wide spikes, with modeled total energy no worse.
If the threshold fails, report the Pareto trade-off without a win claim.

**G3, robustness and scale.** If G2 passes, repeat R/F with two additional
calibration offsets (e.g. 8 and 16), holding the selector hyperparameters fixed
but allowing each build to choose its own bank budgets. Report mean and range
of paired differences across builds, plus how often each bank's budget changes.
Then test the chosen settings on `gpt2-large` (774M), on a fresh validation/test
protocol; prefer a >=48 GB VRAM offer after measuring peak memory on medium.
`gpt2-xl` (1.5B) is conditional on large fitting and evaluation within the
agreed GPU-hour budget. Report per-model results rather than pooling them.

The pilot must provide actual seconds per fit and per evaluation window before
approving G2 or G3. Prior repository measurements put a frozen `gpt2-medium`
full-test T=16 point around 2.4-3.3 GPU hours on earlier hosts; the exhaustive
F build has no defensible duration estimate until G1 times it.

## 5. Run configuration and provenance

Existing paired CLI knobs (the validation/diagnostic flags in §3 must be added
before final runs):

```bash
COMMON="--model gpt2-medium --backend mbe_pasn --convert-ops all \
  --n-steps 16 --epochs 300 --pasn-target-mse 1e-5 \
  --pasn-fit-max-n 4 --stride 1024 --eval-mode sliding"
python experiments/gpt2_wikitext.py $COMMON --pasn-budget rule \
  --json results/adaptive_rule.json --tag rule
python experiments/gpt2_wikitext.py $COMMON --pasn-budget fit \
  --json results/adaptive_fit.json --tag fit
```

Before each run save Git SHA, exact command, offer ID/GPU model, CUDA/PyTorch
versions, model revision, calibration token hash, evaluation token hash and
`ConvertConfig`. Write logs and JSON after each arm and copy them out before
destroying the instance. Never infer network energy from the parameter table or
from spike counts alone.

## 6. Vast.ai budget and lifecycle

Select an on-demand, verified/high-reliability offer after checking its live
GPU, storage and bandwidth rates. Use one host for paired cost/latency numbers.
Estimate the next stage from G1 wall time:

`expected cost = GPU-hours * live $/GPU-hour + disk GB-hours * live storage rate
                 + transfer GB * live bandwidth rate`.

Agree on a hard dollar or GPU-hour cap before renting. Store results outside the
instance after each arm. Vast.ai bills storage while an instance exists even
when stopped; delete it after files are verified locally. Interruptible rental
is unsuitable for the uncheckpointed candidate search unless resumable fitting
and external result sync have first been tested.
