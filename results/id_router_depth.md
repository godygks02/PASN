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
| 1e-01 | 37.98 / 1372 | 119.43 / 1428 | 143.29 / 1484 | 122.70 / 1540 |
| 3e-02 | 250.72 / 2164 | 141.69 / 1428 | 143.29 / 1484 | 122.70 / 1540 |
| 1e-02 | 282.87 / 2404 | 315.62 / 2508 | 281.55 / 2372 | 295.19 / 2476 |


## 3. Raw frontier (every cell)

The iso-accuracy table above depends on where the thresholds fall; this does not.

| target | -6: nrmse / spikes | -8: nrmse / spikes | -10: nrmse / spikes | -12: nrmse / spikes |
|---|---|---|---|---|
| 1e-01 | 1.56e-01 / 10.8 | 1.11e-01 / 14.1 | 1.10e-01 / 18.1 | 1.13e-01 / 22.0 |
| 3e-02 | 7.01e-02 / 38.0 | 3.24e-02 / 119.4 | 2.93e-02 / 143.3 | 2.93e-02 / 122.7 |
| 1e-02 | 6.60e-02 / 49.1 | 1.93e-02 / 141.7 | 1.02e-02 / 191.7 | 1.02e-02 / 208.7 |
| 3e-03 | 2.99e-02 / 250.7 | 1.18e-02 / 270.2 | 6.25e-03 / 281.6 | 5.07e-03 / 295.2 |
| 1e-03 | 5.98e-03 / 282.9 | 3.68e-03 / 315.6 | 3.50e-03 / 337.4 | 3.48e-03 / 347.4 |


## 4. Verdict vs the shipped default (e_min = -6)

| nrmse <= | best depth | spikes ratio | bytes ratio |
|---|---:|---:|---:|
| 1e-01 | -8 | 0.32x | 0.96x |
| 3e-02 | -12 | 2.04x | 1.41x |
| 1e-02 | -10 | 1.00x | 1.01x |

⚠️ A ratio below 1.00x means the deeper router **costs** more. Report it either way -- a negative result closes the item and pins the freeze at the shipped default.

