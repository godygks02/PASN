# PASN: Prefix-Adaptive Spiking Neuron

## 1. Method Overview

PASN is a range-adaptive extension of the Multi-Basis Encoding (MBE) neuron.
An MBE neuron uses one global set of temporal spiking bases for the entire input
domain. PASN retains the same MBE dynamics but organizes the bases into multiple
range-specialized banks. A deterministic prefix router selects one bank for
each input, and only the bases in that bank are executed.

The structural difference is:

```text
MBE:
input x → one global MBE basis bank → decoded output

PASN:
input x → floating-point prefix router → one selected MBE basis bank
        → decoded output
```

PASN does not replace the internal MBE basis equation. It changes how MBE bases
are allocated and activated across the input domain.

## 2. Baseline MBE Neuron

Let an MBE neuron contain \(N\) temporal bases. For basis \(n\), the decoding
kernel, reset kernel, and firing threshold are time-dependent exponential
functions:

\[
P_n[t]
=
\alpha_{P,n}
\exp\left(-\frac{t\Delta t}{\tau_{P,n}}\right),
\qquad
P\in\{d,r,V_{\mathrm{th}}\}.
\]

Here, \(d_n[t]\) is the decoding kernel, \(r_n[t]\) is the reset kernel, and
\(V_{\mathrm{th},n}[t]\) is the firing threshold. Given the initial membrane
state \(u_n[0]=x\), each basis evolves as

\[
s_n[t]
=
H\left(u_n[t]-V_{\mathrm{th},n}[t]\right),
\]

\[
u_n[t+1]
=
u_n[t]-s_n[t]r_n[t],
\]

where \(H(\cdot)\) is the Heaviside step function. The decoded output of the
global MBE neuron is

\[
\hat f_{\mathrm{MBE}}(x)
=
\sum_{n=1}^{N}
w_n
\sum_{t=0}^{T-1}
s_n[t]d_n[t].
\]

All inputs use the same parameters:

\[
\Theta_{\mathrm{MBE}}
=
\{\theta_n\}_{n=1}^{N},
\]

where \(\theta_n\) denotes all parameters associated with basis \(n\).
Therefore, every input is represented using the same global basis bank,
regardless of its numerical range.

The exact MBE parameter constraints, initialization, surrogate gradient, and
signed-input treatment are properties of the baseline MBE implementation.
PASN uses the same choices inside every bank so that the comparison changes
only the bank organization and routing mechanism.

## 3. Prefix-Based Range Routing

PASN divides the input domain into \(R\) ordered numerical ranges. The range is
selected using the leading varying bits of an order-preserving representation
of the floating-point input.

For an FP32 input \(x\), let \(b(x)\) be its unsigned 32-bit representation. A
monotonic key can be constructed as

\[
q(x)=
\begin{cases}
\sim b(x), & \text{if } x<0,\\
b(x)\oplus \texttt{0x80000000}, & \text{if } x\ge 0,
\end{cases}
\]

where \(\sim\) is bitwise inversion. This transformation makes the unsigned key
order consistent with the numerical order of finite floating-point values.

For a configured input interval, the router ignores the common leading bits and
uses the next \(k\) varying bits as a prefix:

\[
z(x)=\operatorname{PrefixRoute}_k(q(x)),
\qquad
z(x)\in\{0,\ldots,R-1\},
\qquad
R\le 2^k.
\]

The router is fixed and contains no learned classifier or gating network. The
original floating-point value is passed to the selected MBE bank without
quantizing or truncating its numerical value. The prefix is used only for bank
selection.

## 4. Range-Specialized MBE Banks

PASN assigns an independent MBE parameter bank to every routed range:

\[
\mathcal B_j
=
\{\theta_{j,n}\}_{n=1}^{N_j},
\qquad
j\in\{0,\ldots,R-1\}.
\]

A bank is the complete set of MBE bases available for one input range. It is
not a single basis. For example, if PASN has eight ranges and two bases per
range, it stores eight banks, each containing two MBE bases.

For input \(x\), only bank \(\mathcal B_{z(x)}\) is active. Its spiking dynamics
are identical to those of the baseline MBE neuron:

\[
s_{j,n}[t]
=
H\left(
u_{j,n}[t]-V_{\mathrm{th},j,n}[t]
\right),
\]

\[
u_{j,n}[t+1]
=
u_{j,n}[t]-s_{j,n}[t]r_{j,n}[t].
\]

The PASN output is

\[
\hat f_{\mathrm{PASN}}(x)
=
\sum_{n=1}^{N_{z(x)}}
w_{z(x),n}
\sum_{t=0}^{T-1}
s_{z(x),n}[t]d_{z(x),n}[t].
\]

Parameters belonging to unselected banks do not participate in the forward
pass for that input.

## 5. Training

Given training pairs \(\{(x_i,y_i)\}_{i=1}^{M}\), each input is first assigned
to a bank by the fixed prefix router:

\[
z_i=z(x_i).
\]

The PASN parameters are optimized with

\[
\mathcal L
=
\frac{1}{M}
\sum_{i=1}^{M}
\ell\left(
\hat f_{\mathrm{PASN}}(x_i;\Theta_{z_i}),
y_i
\right).
\]

Because routing is deterministic, each sample updates only its selected bank.
All banks may be trained jointly in one model or calibrated independently using
the samples routed to each range. The two procedures use the same method as
long as the banks do not share parameters.

If the input distribution is highly imbalanced, the training sampler or loss
weighting may be balanced across ranges. This changes the calibration protocol,
not the PASN architecture. The routing boundaries remain fixed during training.

## 6. Signed Inputs

Prefix routing and signed-input spiking dynamics are separate concerns.
The order-preserving key allows negative and positive inputs to be routed in
their correct numerical order. It does not, by itself, make an MBE basis capable
of representing negative values.

Therefore, PASN follows this consistency rule:

> Any signed-input mechanism used by the newly implemented MBE baseline must be
> applied identically to every PASN bank.

If the MBE implementation uses polarity-specific bases, signed-magnitude
decomposition, or another signed extension, PASN preserves that mechanism after
range selection. The prefix router must not be credited for behavior provided
by the underlying signed MBE implementation.

## 7. Relationship to MBE

PASN can be interpreted as a conditional MBE neuron with a fixed, bit-derived
hard router.

| Property | MBE | PASN |
|---|---|---|
| MBE basis dynamics | Temporal exponential bases | Same |
| Number of banks | One global bank | Multiple range-specialized banks |
| Bank selection | None | Deterministic FP-prefix routing |
| Active banks per input | One | One |
| Parameter sharing across ranges | Full | None by default |
| Learned routing parameters | None | None |
| Active bases per input | \(N\) | \(N_{z(x)}\) |
| Stored basis sets | \(N\) | \(\sum_r N_r\) |

The global MBE neuron is the special case of PASN with one range:

\[
R=1
\quad\Longrightarrow\quad
\hat f_{\mathrm{PASN}}(x)=\hat f_{\mathrm{MBE}}(x).
\]

This relation provides a direct implementation check: setting the PASN router
to a single bank must reproduce the output of the newly implemented MBE neuron
when both models use identical parameters.

## 8. Method Characteristics

PASN is designed to exchange global parameter sharing for local specialization.
With \(R\) banks and \(N_{\mathrm{local}}\) bases per bank, its total basis
storage is

\[
N_{\mathrm{stored}}=R\,N_{\mathrm{local}},
\]

while the number of active bases for one input is

\[
N_{\mathrm{active}}=N_{\mathrm{local}}.
\]

Thus, PASN may use fewer active bases than a global MBE configuration with
\(N_{\mathrm{global}}>N_{\mathrm{local}}\), but it generally stores more
parameters across all banks. It also introduces prefix extraction, bank
selection, and bank-addressing overhead.

These are structural properties of the method. Whether local specialization
improves approximation accuracy or produces a favorable hardware trade-off
must be evaluated separately and is not assumed by the method definition.

---

# PASN v2 — Concretization: Adaptive-Budget Prefix Routing

Sections 1–8 leave the router (§3) and per-bank budget (§4) abstract; taken
literally they describe input binning, which is incremental over MBE. This
concretization pins down a specific, hardware-native instantiation that turns
PASN into an **input-adaptive-compute** neuron. It supplies the paper's novelty
beyond MBE. The abstract properties above (R=1 reduction, deterministic routing,
one active bank) are all preserved.

## 9. The three concrete ideas

1. **The FP exponent prefix is a free, exact coarse code.** Reading the IEEE-754
   sign + leading exponent bits routes ``x`` into a *binade* (a power-of-two
   magnitude interval). This is the abstract order-preserving prefix of §3, made
   concrete. Crucially it gives **logarithmically dense ranges near zero** — the
   exact region where MBE fails on GELU/SiLU (concentrated curvature). A uniform
   global domain cannot buy this concentration; the FP representation gives it for
   free.
2. **Banks approximate only the intra-binade residual.** Any ``x`` factors as
   ``x = σ · 2^e · (1+ρ)`` with mantissa residual ``ρ ∈ [0,1)``. Routing fixes
   ``(σ, e)`` exactly at **zero spike cost**; bank ``(σ,e)`` need only approximate
   ``g_{σ,e}(ρ) = f(σ·2^e·(1+ρ))`` on the unit interval ``ρ∈[0,1)``. A smooth 1-D
   map on a unit interval needs far fewer bases/steps than the global ``f`` — the
   origin of the spike savings.
3. **Heterogeneous per-bank budget ``(N_j, T_j)``.** Flat tails (``f≈const`` over a
   whole binade) take ``N_j=1``; high-curvature near-zero binades take larger
   ``N_j``. Compute is *allocated by range*, not spent uniformly.

## 10. Router (concrete)

Given a calibrated domain ``[lo, hi]`` and bounds ``e_min ≤ e_max``:

- For ``x`` with ``|x| ≥ 2^{e_min}``: ``σ = sign(x)``, ``e =
  clip(⌊log₂|x|⌋, e_min, e_max−1)``; bank key ``(σ, e)``. Magnitude bank ``(σ,e)``
  covers ``|x| ∈ [2^e, 2^{e+1})`` and normalises ``u[0] = (|x| − 2^e)/2^e = ρ``.
- For ``|x| < 2^{e_min}``: a single **near-zero bank** covering the signed interval
  ``(−2^{e_min}, 2^{e_min})``, fed the raw ``x`` (this is the only bank that sees
  mixed signs — and it is exactly the smooth GELU/SiLU bend, so one small MBE
  neuron handles it).

The router is deterministic and parameter-free (``clip(⌊log₂|·|⌋)`` = an exponent
read). ``R = (#signs)·(e_max−e_min) + 1`` banks.

## 11. Sign subsumption

Because ``σ`` is the top routed bit, **every magnitude bank sees single-sign
inputs**; only the one near-zero bank straddles zero. The polarity 4-term split
of the baseline signed mechanism ([[Signed MBE Neuron]]) is therefore needed only
inside that single small bank, not globally. Routing structurally absorbs the sign
problem for the tails (§6's consistency rule, made cheap).

## 12. Evaluation protocol (the headline claim)

PASN is **not** evaluated by a single MSE. For each nonlinearity, sweep the budget
and plot the frontier of **approximation MSE vs. mean spikes per input**
(``= firing_rate · N_{j(x)} · T_{j(x)}``), alongside timestep ``T`` and stored
parameters ``Σ_j N_j``. The contribution is: **PASN's MSE–spike Pareto frontier
dominates the global MBE neuron** on GELU / SiLU / 1/x / 2^x / 1/√x, and — plugged
into the Phase-4 conversion ([[Phase 4 - Conversion Framework]]) — matches ANN
accuracy at fewer spikes (lower energy). Honest cost: PASN stores more parameters
and adds prefix-extraction/bank-addressing overhead (report both).

## 13. Reduction check (unchanged)

``e_min = e_max`` over the whole domain, one sign, identity residual normalisation
⇒ one bank = the baseline MBE neuron ⇒ bit-identical output for identical
parameters ([[MBE Neuron Core]] ``test_r1_reduces_to_single_bank``, to be
upgraded to the real router).
