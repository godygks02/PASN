# Global router-depth CPU pilot

## Setup

- Cached pretrained `gpt2` (12 layers); WikiText-2 raw, Stage 1 GELU + LayerNorm conversion.
- One global depth offset applied to both `pasn_e_min` and `pasn_id_e_min`.
  `delta=+1` makes the near-zero cutoff coarser; `delta=-1` opens deeper banks.
  This is a proxy for global prefix length, not a literal learned `k`-bit router.
- Same 8 train calibration blocks (64 tokens each) for each candidate; rule budget,
  `T<=4`, 6 fitting epochs, absolute target MSE `1e-3`.
- Select on 8 validation blocks. Evaluate on 8 test blocks starting at block 16,
  after the previously inspected first 16 test blocks.
- Selection: among candidates within 0.25 percentage points of the best
  validation conversion loss, minimize spikes on the first validation block.
  Spikes are a one-block diagnostic, not evaluation-wide cost.

## Results

| Global offset | `(e_min, id_e_min)` | Banks across 87 routed primitives | Val conversion loss | Test conversion loss | Val spikes / first block | Stored bytes |
|---:|---:|---:|---:|---:|---:|---:|
| +1 | (-2, -5) | 1,015 | **-0.568 pp** | -4.376 pp | 147,387 | **13,904** |
| 0 | (-3, -6) | 1,089 | +1.324 pp | **-4.900 pp** | 146,995 | 14,672 |
| -1 | (-4, -7) | 1,163 | +0.501 pp | -3.223 pp | **145,892** | 15,440 |

ANN PPL: 57.8116 validation, 83.0067 test. The validation rule selected
`delta=+1`. The selected candidate used 768 fewer stored bytes than the
default offset but 0.27% more spikes on the measured validation block.
On the held-out test slice, the default offset had the best PPL. The selected
candidate's test conversion loss was 0.524 percentage points worse than the
default offset's, despite winning on the short validation slice.

## Interpretation

The experiment confirms that an offline procedure can choose one shared router
depth from model-level validation results, and that this changes bank count,
storage, spikes and PPL. It does **not** establish an accuracy or spike win.
The ranking changes across the short validation and test slices, and Stage 1
leaves attention exact. A literal adaptive `k` claim requires a router whose
prefix depth is an explicit candidate, then a longer validation and Stage 2
network evaluation with evaluation-wide cost metering.

Reproduction script: `experiments/global_router_depth_cpu.py`.
Machine-readable output: `results/global_router_depth_cpu.json`.
