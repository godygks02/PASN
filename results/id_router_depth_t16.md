# E11b -- identity router depth (gpt2-medium, layer 21, seq 256)

## 1. Does the mass actually split? (necessary condition)

| id e_min | banks | reached | top-1 | n_eff |
|---:|---:|---:|---:|---:|
| -6 | 7 | 7 | 0.9714 | 1.19 |
| -8 | 9 | 9 | 0.9088 | 1.56 |
| -10 | 11 | 11 | 0.7655 | 2.55 |
| -12 | 13 | 13 | 0.5476 | 4.69 |


## 2. Iso-accuracy cost of the whole operator

Cheapest build reaching each nrmse, by router depth. Spikes per output element; bytes stored.

| nrmse <= | -6: spikes / bytes | -8: spikes / bytes | -10: spikes / bytes | -12: spikes / bytes |
|---|---|---|---|---|
| 1e-01 | 66.50 / 1372 | 159.11 / 1428 | 209.21 / 1484 | 226.03 / 1540 |
| 3e-02 | 256.70 / 2164 | 159.11 / 1428 | 209.21 / 1484 | 226.03 / 1540 |
| 1e-02 | 284.71 / 2404 | 317.50 / 2508 | 209.21 / 1484 | 226.03 / 1540 |


## 3. Raw frontier (every cell)

The iso-accuracy table above depends on where the thresholds fall; this does not.

| target | -6: nrmse / spikes | -8: nrmse / spikes | -10: nrmse / spikes | -12: nrmse / spikes |
|---|---|---|---|---|
| 1e-01 | 6.59e-02 / 66.5 | 1.83e-02 / 159.1 | 8.32e-03 / 209.2 | 8.23e-03 / 226.0 |
| 3e-02 | 6.59e-02 / 66.5 | 1.83e-02 / 159.1 | 8.32e-03 / 209.2 | 8.23e-03 / 226.0 |
| 1e-02 | 6.59e-02 / 66.5 | 1.83e-02 / 159.1 | 8.32e-03 / 209.2 | 8.23e-03 / 226.0 |
| 3e-03 | 2.97e-02 / 256.7 | 1.15e-02 / 276.0 | 6.01e-03 / 287.2 | 4.87e-03 / 300.8 |
| 1e-03 | 5.78e-03 / 284.7 | 3.56e-03 / 317.5 | 3.42e-03 / 339.3 | 3.41e-03 / 349.2 |


## 4. Verdict vs the shipped default (e_min = -6)

| nrmse <= | best depth | spikes ratio | bytes ratio |
|---|---:|---:|---:|
| 1e-01 | -8 | 0.42x | 0.96x |
| 3e-02 | -8 | 1.61x | 1.52x |
| 1e-02 | -10 | 1.36x | 1.62x |

⚠️ A ratio below 1.00x means the deeper router **costs** more. Report it either way -- a negative result closes the item and pins the freeze at the shipped default.

