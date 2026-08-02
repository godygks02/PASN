# PASN: Prefix-Adaptive Spiking Neuron

## 0. Positioning (read first)

PASN is proposed as a **distinct training-free ANN→SNN conversion neuron**, not as
an incremental improvement over the MBE neuron. It combines two ingredients:
(1) **multi-basis exponential-decay temporal encoding** — the substrate introduced
by MBE (Wang et al., AAAI-26) — and (2) a **deterministic floating-point-prefix
router** that partitions the input domain into power-of-two (binade) ranges, each
served by its own local basis bank.

We do **not** claim to beat MBE, and we do not rely on our own MBE reimplementation
as the baseline: our reimplementation reproduces MBE's *function-approximation*
numbers (Table X) but does **not** reproduce its near-lossless downstream
conversion, because a single global multi-basis neuron has a structural near-zero
accuracy floor on GELU (it cannot resolve the near-zero curvature and the mid-range
tail simultaneously — see the reproduction notes). Instead, PASN is **evaluated
against the MBE paper's published benchmark numbers** (Tab. 1–4), on the same
models/datasets, using our own reproduced ANN baselines.

The ``R=1`` reduction (a one-bank PASN equals a single multi-basis neuron) is kept
as a *structural relationship* to the multi-basis family, not as an improvement
claim. Our internal MBE-vs-PASN function-approximation experiments are **ablations**
that isolate the contribution of the prefix-routed local banks over a single global
bank of the same substrate.

## 1. Method Overview

PASN organizes multi-basis temporal spiking bases into multiple range-specialized
local banks. A deterministic prefix router selects one bank per input, and only the
bases in that bank are executed. (A single global multi-basis bank — one bank for
the whole domain — is the ``R=1`` special case.)

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

   **Measured on GPT-2 (Stage 2), the allocation is carried by ``N_j``, not by
   ``T_j``.** Forcing every bank to the rule's basis cap costs **3.26x spikes and
   2.55x storage**; forcing every bank to a global ``T=16`` costs **1.06x spikes
   and nothing at all in storage**. The rule already picks ``T=16`` wherever the
   budget actually is — the identities inside matmul and LayerNorm span decades —
   and only trims ``T`` on the flat activation tails, which are 3% of the network's
   spikes. So the honest form of this claim is **"per-range *basis* allocation"**;
   the timestep half is real but secondary. The two axes are complementary rather
   than redundant: the rule gives matmul/LayerNorm the maximum ``T`` while cutting
   their ``N``, and does the opposite for activations.

   **The rule presumes a bank the router has already narrowed.** Where the router
   degenerates — an argument that is *already* a single binade, such as the IEEE
   mantissa in ``[0.5,1)`` that ``1/x`` receives inside the softmax — the rule is
   extrapolating outside the regime its constants were fitted on, and it
   under-provisions ``N``. Measured: it selects ``(N=2, T=16)`` for 17.32 spikes
   per element where ``(N=3, T=8)`` reaches *better* error for **8.76** — 1.98x,
   identical across three seeds. Fewer bases force the bank to fire on nearly every
   timestep, so spikes are **not monotone** in ``(N, T)`` there. Such primitives
   should be searched rather than predicted. (Elsewhere the rule holds up: on
   ``2^x`` and ``1/sqrt(x)`` it is spike-optimal, on the identity within 1.08x.)

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

## 12. Evaluation protocol and measured results

The baseline is the **MBE paper's published tables**, never our own MBE
reimplementation (§0). Our reimplementation is retained only as an *internal
ablation* — the arm with routing removed — and its numbers are not quoted as a
comparison to the literature.

Three levels, all matched, all measured (2026-08-02):

| level | published | PASN | matching |
|---|---|---|---|
| **network** (Tab. 3, GPT-2-medium × WikiText-2) | **+1.57%** relative conversion loss | **−0.14%** | operator set verified identical (their Algorithm 1); global ``T=16`` both sides; PASN uses **fewer** bases |
| **operator** (Tab. XI, firing rates) | 7 primitives | **2.2–10.3x fewer spikes/element on 6 of 7** | ``T=16``; compared through ``T·η·N``, their own energy quantity |
| **function** (Tab. X, MSE vs N) | per-function MSE | reproduced and beaten | no scope question at this level |

**Report ``T·η·N``, never the firing rate alone.** A routed bank holds far fewer
``(basis, timestep)`` slots, so it fires a *larger fraction* of a much smaller
number: on ``1/sqrt(x)`` PASN fires 65.65% against the paper's 25.27% and still
emits 3.08x fewer spikes. Firing rate is not an efficiency metric on its own.

**Units.** Tab. 3 prints ``22.69 (+0.35)`` and the prose calls it "0.35%
conversion loss", but 0.35 is the *absolute* perplexity difference; relative it is
**+1.57%**. Our figure is relative and belongs beside that.

**Honest costs, all measured rather than asserted:**

- **Router arithmetic is real**: ~6 operations plus one bit-select per element,
  **15.9%** of PASN's modelled energy on GPT-2 — bought by removing **7.55x** of
  the threshold comparisons, which are paid every timestep whether or not a
  neuron fires.
- **Memory is a wash against a global MBE** (35168 vs 38148 bytes, 1.08x). The
  1.99x memory result belongs to *tying the identity*, a different comparison;
  do not merge them.
- **One operator is a regression**: ``1/x`` costs 3.8x the published figure
  (§9.3). It is the cheapest primitive by element count, so the effect on totals
  is ≈0.3%, but it must be disclosed.
- Energy is a **model** (45 nm; ``E_AC`` 0.9 pJ, ``E_MAC`` 4.6 pJ — the paper's own
  constants), and **memory traffic is uncounted on both sides**. Bank switching
  breaks locality, so that omission may favour us.

**Open assumption**: the paper does not state its perplexity recipe (stride,
context) anywhere in the appendix, so both figures are read relative to their own
ANN baseline (ours 21.706, theirs 22.34).

## 13. Reduction check (unchanged)

``e_min = e_max`` over the whole domain, one sign, identity residual normalisation
⇒ one bank = the baseline MBE neuron ⇒ bit-identical output for identical
parameters ([[MBE Neuron Core]] ``test_r1_reduces_to_single_bank``, to be
upgraded to the real router).

## 14. Open extensions (measured openings, not speculation)

The core above is settled. These are the places the measurements say the method is
*not* finished; each carries the number that says so.

1. **Mantissa-prefix routing.** The router reads sign and exponent. An argument
   that is already a mantissa — ``1/x`` on ``[0.5,1)`` inside the softmax — has one
   reachable binade, so routing does nothing for it, and that is the single
   operator where we lose (§12). Reading mantissa bits is the same idea one prefix
   deeper, not a bolt-on, and it is the only structural fix for that class.
2. **Budget search where the router degenerates.** §9.3: predicted ``(2,16)``,
   optimal ``(3,8)``, 1.98x. Detecting a degenerate router is trivial (one
   reachable range) and searching a 28-point grid is cheap at build time.
3. **The τ block is untouched and is where memory actually lives.** Four state
   tensors per basis (``tau_r``, ``tau_vth``, ``tau_d``, ``dt``) are **54–61%** of
   stored bytes — a property of the *neuron*, orthogonal to routing, and the axis
   on which we are merely level with a global MBE. Banks fit scaled versions of
   related shapes, so sharing τ across banks is the obvious untried move.
4. **Attention-specific structure.** Attention is **87.7%** of a converted GPT-2's
   spikes (matmul 45.1% + softmax 42.6%); activations are 3.0%. Two concrete
   openings: the softmax output is decoded to FP and immediately re-encoded for
   ``attn·V`` (a fusion would skip a round trip already priced at 6.3 spikes per
   activation), and ``inv_S`` is broadcast to ``S×S`` *before* reconstruction when
   ``S`` values would do — a **bit-identical** fix worth ≈14% of total spikes.
5. **No network-level operating point.** The identity budget ``r`` moved spikes
   4.2x at Stage 1 but only **1.90x** at Stage 2, because it bites hardest on
   LayerNorm, which full conversion reduces to ~9%. A conversion method should be
   able to trade accuracy for energy at the network level; presently it cannot.
6. **The router's own parameters are hand-set.** ``e_min`` and ``pasn_id_e_min``
   (−3 / −6) came from the toy and were never solved for, while the method's whole
   argument is that budgets should be solved rather than tuned.

Priority is not the order above: (4) sits on 87.7% of the budget, (1) and (2) fix
a 0.3% operator but repair the method's *story*, and (3) is the only route to a
memory claim.
