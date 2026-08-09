# E11 -- bank utilisation audit (gpt2-medium)

## 1. Degeneracy

**104 of 339 sites flagged.**

| site | op | banks | reached | top-1 | n_eff | key range | binades | beta | flags |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `transformer.h.21.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9920 | 1.06 | [8.4e-06, 1] | 16.86 | 0 | top_heavy |
| `transformer.h.19.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9911 | 1.07 | [5.98e-06, 1] | 17.35 | 0 | top_heavy |
| `transformer.h.18.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9909 | 1.07 | [5.89e-06, 1] | 17.38 | 0 | top_heavy |
| `transformer.h.6.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9905 | 1.07 | [8.77e-06, 1] | 16.80 | 0 | top_heavy |
| `transformer.h.16.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9903 | 1.07 | [6.01e-06, 1] | 17.35 | 0 | top_heavy |
| `transformer.h.20.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9900 | 1.07 | [7.21e-06, 1] | 17.08 | 0 | top_heavy |
| `transformer.h.17.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9894 | 1.08 | [5.46e-06, 1] | 17.48 | 0 | imbalanced |
| `transformer.h.9.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9891 | 1.08 | [5.46e-06, 1] | 17.48 | 0 | imbalanced |
| `transformer.h.7.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9887 | 1.08 | [6.87e-06, 1] | 17.15 | 0 | imbalanced |
| `transformer.h.14.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9884 | 1.08 | [5.36e-06, 1] | 17.51 | 0 | imbalanced |
| `transformer.h.22.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9882 | 1.08 | [7.08e-06, 1] | 17.11 | 0 | imbalanced |
| `transformer.h.5.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9876 | 1.09 | [5.82e-06, 1] | 17.39 | 0 | imbalanced |
| `transformer.h.12.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9874 | 1.09 | [6.1e-06, 1] | 17.32 | 0 | imbalanced |
| `transformer.h.15.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9872 | 1.09 | [4.96e-06, 1] | 17.62 | 0 | imbalanced |
| `transformer.h.13.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9865 | 1.10 | [4.79e-06, 1] | 17.67 | 0 | imbalanced |
| `transformer.h.8.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9860 | 1.10 | [6.54e-06, 1] | 17.22 | 0 | imbalanced |
| `transformer.h.11.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9859 | 1.10 | [4.45e-06, 1] | 17.78 | 0 | imbalanced |
| `transformer.h.10.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9858 | 1.10 | [5.24e-06, 1] | 17.54 | 0 | imbalanced |
| `transformer.h.4.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9851 | 1.11 | [6.3e-06, 1] | 17.28 | 0 | imbalanced |
| `transformer.h.23.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9827 | 1.12 | [2.25e-06, 1] | 18.76 | 0 | imbalanced |
| `transformer.h.3.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9808 | 1.13 | [3.36e-06, 1] | 18.18 | 0 | imbalanced |
| `transformer.h.2.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9735 | 1.17 | [3.33e-06, 1] | 18.20 | 0 | imbalanced |
| `transformer.h.1.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9735 | 1.16 | [1.67e-06, 1] | 19.19 | 0 | imbalanced |
| `transformer.h.15.ln_2.rsqrt` | layernorm | 4 | 2 | 0.9727 | 1.13 | [0.501, 2] | 1.99 | 0 | imbalanced |
| `transformer.h.0.attn.av_matmul.idn` | matmul | 8 | 8 | 0.9726 | 1.16 | [9.64e-07, 1] | 19.99 | 0 | imbalanced |
| `transformer.h.21.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9684 | 1.21 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.16.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9606 | 1.26 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.19.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9597 | 1.26 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.6.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9597 | 1.26 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.18.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9586 | 1.27 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.17.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9536 | 1.30 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.20.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9528 | 1.31 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.15.ln_1.rsqrt` | layernorm | 4 | 2 | 0.9492 | 1.22 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.7.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9481 | 1.33 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.9.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9477 | 1.33 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.16.ln_1.rsqrt` | layernorm | 4 | 2 | 0.9453 | 1.24 | [0.521, 1.51] | 1.54 | 0 | imbalanced |
| `transformer.h.19.ln_1.rsqrt` | layernorm | 4 | 2 | 0.9453 | 1.24 | [0.505, 1.92] | 1.93 | 0 | imbalanced |
| `transformer.h.5.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9444 | 1.36 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.22.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9366 | 1.40 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.14.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9364 | 1.40 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.12.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9361 | 1.40 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.4.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9353 | 1.42 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.15.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9324 | 1.43 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.13.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9278 | 1.46 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.1.ln_1.rsqrt` | layernorm | 4 | 2 | 0.9258 | 1.30 | [0.529, 1.55] | 1.55 | 0 | imbalanced |
| `transformer.h.14.ln_2.rsqrt` | layernorm | 4 | 2 | 0.9219 | 1.32 | [0.503, 1.99] | 1.99 | 0 | imbalanced |
| `transformer.h.8.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9203 | 1.49 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.19.ln_2.rsqrt` | layernorm | 4 | 2 | 0.9141 | 1.34 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.10.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9129 | 1.54 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.11.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.9112 | 1.55 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.5.ln_2.rsqrt` | layernorm | 4 | 2 | 0.9082 | 1.36 | [0.503, 1.99] | 1.98 | 0 | imbalanced |
| `transformer.h.6.ln_1.rsqrt` | layernorm | 4 | 2 | 0.9043 | 1.37 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.3.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.8995 | 1.65 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.6.ln_2.rsqrt` | layernorm | 4 | 2 | 0.8711 | 1.47 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.23.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.8621 | 1.86 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.2.attn.attn_softmax.idn` | softmax | 7 | 7 | 0.8562 | 1.95 | [1.4e-45, 1] | 149.00 | 0 | imbalanced |
| `transformer.h.16.ln_2.rsqrt` | layernorm | 4 | 2 | 0.8535 | 1.52 | [0.567, 1.56] | 1.46 | 0 | imbalanced |
| `transformer.h.18.ln_2.rsqrt` | layernorm | 4 | 2 | 0.8516 | 1.52 | [0.722, 1.97] | 1.45 | 0 | imbalanced |
| `transformer.h.1.ln_2.rsqrt` | layernorm | 4 | 2 | 0.8477 | 1.53 | [0.62, 1.82] | 1.56 | 0 | imbalanced |
| `transformer.h.20.ln_1.rsqrt` | layernorm | 4 | 2 | 0.8398 | 1.55 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.7.ln_1.rsqrt` | layernorm | 4 | 2 | 0.8379 | 1.56 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.14.ln_1.rsqrt` | layernorm | 4 | 2 | 0.8184 | 1.61 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.21.ln_2.rsqrt` | layernorm | 4 | 2 | 0.8145 | 1.62 | [0.504, 1.99] | 1.98 | 0 | imbalanced |
| `transformer.h.5.ln_1.rsqrt` | layernorm | 4 | 2 | 0.8086 | 1.63 | [0.502, 1.98] | 1.98 | 0 | imbalanced |
| `transformer.h.21.ln_1.rsqrt` | layernorm | 4 | 2 | 0.8027 | 1.64 | [0.504, 2] | 1.99 | 0 | imbalanced |
| `transformer.h.22.ln_2.rsqrt` | layernorm | 4 | 2 | 0.7891 | 1.67 | [0.501, 1.98] | 1.98 | 0 | imbalanced |
| `transformer.h.4.ln_2.rsqrt` | layernorm | 4 | 2 | 0.7754 | 1.70 | [0.502, 2] | 1.99 | 0 | imbalanced |
| `transformer.h.13.ln_2.rsqrt` | layernorm | 4 | 2 | 0.7480 | 1.76 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.7.ln_2.rsqrt` | layernorm | 4 | 2 | 0.7461 | 1.76 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.17.ln_1.rsqrt` | layernorm | 4 | 2 | 0.7402 | 1.77 | [0.618, 1.65] | 1.42 | 0 | imbalanced |
| `transformer.h.18.ln_1.rsqrt` | layernorm | 4 | 2 | 0.7266 | 1.80 | [0.692, 1.89] | 1.45 | 0 | imbalanced |
| `transformer.h.8.ln_1.rsqrt` | layernorm | 4 | 2 | 0.7148 | 1.82 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.23.ln_2.rsqrt` | layernorm | 4 | 2 | 0.6992 | 1.84 | [0.501, 1.99] | 1.99 | 0 | imbalanced |
| `transformer.h.13.ln_1.rsqrt` | layernorm | 4 | 2 | 0.6836 | 1.87 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.4.ln_1.rsqrt` | layernorm | 4 | 2 | 0.6523 | 1.91 | [0.508, 1.94] | 1.93 | 0 | imbalanced |
| `transformer.h.2.ln_1.rsqrt` | layernorm | 4 | 2 | 0.6504 | 1.91 | [0.5, 1.9] | 1.92 | 0 | imbalanced |
| `transformer.h.3.ln_2.rsqrt` | layernorm | 4 | 2 | 0.6348 | 1.93 | [0.51, 1.96] | 1.94 | 0 | imbalanced |
| `transformer.h.12.ln_2.rsqrt` | layernorm | 4 | 2 | 0.6289 | 1.93 | [0.501, 2] | 1.99 | 0 | imbalanced |
| `transformer.h.0.ln_2.rsqrt` | layernorm | 4 | 2 | 0.6270 | 1.94 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.8.ln_2.rsqrt` | layernorm | 4 | 2 | 0.6211 | 1.94 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.20.ln_2.rsqrt` | layernorm | 4 | 2 | 0.6191 | 1.94 | [0.502, 2] | 1.99 | 0 | imbalanced |
| `transformer.h.2.ln_2.rsqrt` | layernorm | 4 | 2 | 0.6133 | 1.95 | [0.511, 1.96] | 1.94 | 0 | imbalanced |
| `transformer.ln_f.rsqrt` | layernorm | 4 | 2 | 0.6113 | 1.95 | [0.502, 2] | 1.99 | 0 | imbalanced |
| `transformer.h.9.ln_1.rsqrt` | layernorm | 4 | 2 | 0.5938 | 1.96 | [0.501, 2] | 1.99 | 0 | imbalanced |
| `transformer.h.12.ln_1.rsqrt` | layernorm | 4 | 2 | 0.5918 | 1.97 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.0.ln_1.rsqrt` | layernorm | 4 | 2 | 0.5820 | 1.97 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.11.ln_2.rsqrt` | layernorm | 4 | 2 | 0.5703 | 1.98 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.10.ln_1.rsqrt` | layernorm | 4 | 2 | 0.5625 | 1.98 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.9.ln_2.rsqrt` | layernorm | 4 | 2 | 0.5547 | 1.99 | [0.5, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.23.ln_1.rsqrt` | layernorm | 4 | 2 | 0.5410 | 1.99 | [0.5, 1.99] | 2.00 | 0 | imbalanced |
| `transformer.h.17.ln_2.rsqrt` | layernorm | 4 | 2 | 0.5312 | 2.00 | [0.647, 1.66] | 1.36 | 0 | imbalanced |
| `transformer.h.3.ln_1.rsqrt` | layernorm | 4 | 2 | 0.5273 | 2.00 | [0.529, 1.92] | 1.86 | 0 | imbalanced |
| `transformer.h.22.ln_1.rsqrt` | layernorm | 4 | 2 | 0.5254 | 2.00 | [0.502, 1.98] | 1.98 | 0 | imbalanced |
| `transformer.h.11.ln_1.rsqrt` | layernorm | 4 | 2 | 0.5215 | 2.00 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.10.ln_2.rsqrt` | layernorm | 4 | 2 | 0.5117 | 2.00 | [0.501, 2] | 2.00 | 0 | imbalanced |
| `transformer.h.7.ln_2.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00817, 0.239] | 4.87 | 0 | imbalanced |
| `transformer.h.8.ln_1.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00806, 0.24] | 4.90 | 0 | imbalanced |
| `transformer.h.8.ln_2.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00806, 0.233] | 4.86 | 0 | imbalanced |
| `transformer.h.9.ln_1.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00802, 0.233] | 4.86 | 0 | imbalanced |
| `transformer.h.9.ln_2.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00802, 0.228] | 4.83 | 0 | imbalanced |
| `transformer.h.10.ln_1.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00798, 0.228] | 4.84 | 0 | imbalanced |
| `transformer.h.10.ln_2.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00798, 0.223] | 4.81 | 0 | imbalanced |
| `transformer.h.11.ln_1.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00798, 0.225] | 4.82 | 0 | imbalanced |
| `transformer.h.11.ln_2.id_istd` | layernorm | 11 | 2 | 0.5020 | 2.00 | [0.00798, 0.219] | 4.78 | 0 | imbalanced |

**Reading a flagged row.** `binades` is the width of the observed routing key in powers of two -- the number of ranges the ladder *can* split into. Below ~1 the router cannot partition at all and the bank is a global neuron; the fix is to re-anchor `beta` on the domain's hard end, as `beta=0.5` did for `1/x`.


## 2. Load distribution

| op | sites | banks (mean) | reached (mean) | n_eff (elem-weighted) |
|---|---:|---:|---:|---:|
| activation | 24 | 22.0 | 15.5 | 6.59 |
| layernorm | 147 | 11.7 | 7.2 | 4.03 |
| matmul | 96 | 10.2 | 10.0 | 2.70 |
| softmax | 72 | 6.0 | 6.0 | 2.33 |

## 3. MoE contract -- network level

Stored = bases held (shared/tied prototypes counted once). Active = `sum_b p_b * N_b`, the bases one input fires, element-weighted across sites.

| op | sites | stored bases | active bases / input | paper N (all active) | ratio |
|---|---:|---:|---:|---:|---:|
| activation | 24 | 792 | 1.01 | 4 | 3.96x |
| layernorm | 147 | 392 | 1.50 | 8 | 5.32x |
| matmul | 96 | 288 | 1.78 | 8 | 4.49x |
| softmax | 72 | 360 | 1.58 | 8 | 5.08x |
| **all** | 339 | **1832** | **1.57** | - | - |

⚠️ `active bases` is not an energy figure on its own -- energy is `T*eta*N`, and `T_j` varies per bank. The `active_slots` field in the JSON carries the `N_j*T_j` version. Quote ratios, not absolutes (함정 13).

