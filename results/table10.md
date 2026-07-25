# Table X reproduction (MBE MSE vs N)

MSE on M samples, T=16. `8*` = N=8 without decay (ablation).
Each cell: **ours** (paper). GELU uses the signed (polarity-split) neuron with a balanced split; the others use the plain MBE neuron. Readout (w, bias) is solved in closed form during fitting.

| Func | N=1 | N=2 | N=4 | N=6 | N=8 | N=8* |
|---|---|---|---|---|---|---|
| GELU | 8.1e-04 (7.1e-03) | 8.1e-04 (4.1e-03) | 2.1e-04 (2.3e-04) | 1.3e-04 (1.7e-04) | 1.7e-04 (1.0e-04) | 1.4e-02 (2.8e-01) |
| invsqrt | 1.3e-03 (1.4e-03) | 1.9e-04 (1.2e-03) | 2.6e-05 (3.1e-04) | 8.5e-06 (1.0e-04) | 5.0e-06 (4.9e-05) | 2.2e-04 (2.8e-03) |
| inv | 2.2e-03 (8.6e-03) | 2.2e-04 (2.1e-04) | 2.2e-05 (1.1e-03) | 4.2e-06 (2.2e-03) | 4.6e-06 (4.4e-05) | 8.2e-04 (4.5e-03) |
| 2^x | 7.2e-04 (8.9e-04) | 8.1e-04 (4.5e-04) | 4.1e-04 (4.0e-04) | 7.2e-05 (2.4e-04) | 6.5e-05 (5.3e-05) | 1.2e-03 (9.5e-03) |
