# Temporal Stacking & Space–Time Coupling — Numeric Walkthrough

> Plain, by-hand-numeric explanation of: the existing method (per-token nonlinear MLA),
> temporal predictability (phase4), survive-MLA (phase5), space+time coupling (gate1), and the
> design of **Experiment (b): DeltaKV on the nonlinear-MLA latent** (`phase8_deltakv_stack.py`).

---

## Part 0 — The existing method, numerically (per-token nonlinear MLA)

The method compresses **each token independently**. For one token it does:

```
1. take the hidden state x  (the input to k_proj / v_proj)
2. make ONE shared latent  z  (dim d_c)  — this is all we cache
3. rebuild BOTH K and V from z
```

Tiny example. Say a token's true Key and Value (head_dim = 2 each):

```
K = [3, 1]     V = [2, 4]      stack them:   s = [K ; V] = [3, 1, 2, 4]   (4 numbers)
```

Compress `s` to `d_c = 2` numbers and rebuild:

```
encode:   z = s · Vk            (Vk = top-2 PCA directions of [K;V])   -> z = [z1, z2]   (CACHE THIS)
decode:   ŝ = z · Vkᵀ  +  corr(z)                                       -> ŝ = [K̂ ; V̂]
          └ linear = MLA ┘  └ nonlinear bend (Linear→GELU→Linear) ┘
```

Two facts that matter for everything below:

- **`z` is computed straight from `x`** by a folded matrix `z = x·W_z − bz` (where `W_z` is the
  model's own `k_proj`/`v_proj` weights merged with the PCA basis). So the latent is a *linear*
  function of the token's hidden state.
- The result earlier: nonlinear `corr` beats linear MLA, and the margin **explodes at aggressive
  compression** (7B, τ=0.75: ppl **33.3** nonlinear vs **155.7** linear).

**The limitation we attack now:** every `z_t` is cached on its own. But the tokens are a
*sequence* — `z_t` and `z_{t-1}` are correlated. Per-token MLA throws that away.

---

## Part 1 — Temporal predictability (phase4): "cache the surprise, not the value"

Idea: if `z_t` (or the raw key `k_t`) is **predictable from recent tokens**, don't store the
whole thing — store only the **innovation** (the part the predictor got wrong). This is an axis
**orthogonal** to MLA, so it could *stack* on top.

### The intuition in one example

Suppose successive keys (1-D for simplicity) drift slowly:

```
k =  5.0 , 5.3 , 5.5 , 5.9 , 6.2          (each token ≈ previous token)
```

- **Per-token caching** stores 5.0, 5.3, 5.5, 5.9, 6.2 — values around 5–6, a wide range.
- **Predict** each from the previous (`ẑ_t = k_{t-1} + typical_drift`) and store only the
  **innovation** `e_t = k_t − ẑ_t`:

```
predictor learns drift ≈ +0.3 per step, so:
   e =  (5.3 − 5.3) , (5.5 − 5.6) , (5.9 − 5.8) , (6.2 − 6.2)  =  0.0 , −0.1 , +0.1 , 0.0
```

The innovations are **tiny** (±0.1) compared to the keys (~6). To the same precision, a tiny
number needs **far fewer bits** than a number near 6. That bit saving is the whole prize.

### How phase4 measures it — FVE (Fraction of Variance Explained)

```
FVE = 1 −  ‖ k_t − k̂_t ‖²  /  ‖ k_t − mean ‖²
            └ predictor's error ┘   └ "just guess the average" error ┘
```

- FVE = 1 → previous keys predict `k_t` perfectly (innovation ≈ 0).
- FVE = 0 → previous keys are useless (innovation = the whole key; nothing to save).

On the drift example a *fitted* linear predictor gets **FVE ≈ 0.96** (96% of the key explained by
history). 

### The control (so we don't fool ourselves)

Re-run the **same fit** but **shuffle** the pairing — predict `k_t` from a *random* earlier key.
Now history is meaningless, so a good fit should score **FVE ≈ 0**. That shuffled score is the
**overfitting floor**. The real signal is:

```
temporal headroom = FVE_real − FVE_null
```

**phase4's actual result (Llama-2-7B):** mean `FVE_lin` real ≈ **0.35**, null ≈ **−0.07** →
headroom ≈ **+0.42**. And the nonlinear gain (`FVE_nl − FVE_lin`) ≈ **+0.002** → essentially
zero. **Verdict: keys are ~35–50% temporally predictable, and the predictor only needs to be
LINEAR.** (A linear AR predictor is cheap — good news for cost.)

---

## Part 2 — Does it survive MLA? (phase5)

Worry: maybe MLA *already used up* the temporal redundancy. If so, the latent `z` would be
temporally **white** (FVE ≈ 0) and stacking buys nothing.

phase5 measures the temporal FVE **of the MLA latent `z@d_c`** as `d_c` shrinks:

```
                temporal FVE (real − null)
raw key:                 +0.427
z @ d_c=2048:            +0.312
z @ d_c=1024:            +0.314
z @ d_c=512:             +0.327
z @ d_c=256:             +0.347      ← still ~0.35, basically the raw-key headroom
```

**The latent stays ~35% predictable even after aggressive MLA.** So the temporal structure
**survives** — MLA and temporal prediction are largely *orthogonal* axes. **They stack.** This is
the green light for Experiment (b).

---

## Part 3 — Space + time coupling (gate1)

A different, more ambitious question: instead of "compress space (MLA) then predict in time
separately," can we compress the **2-D (time × dimension) block jointly** and win because energy
lives in *combinations* a separable scheme can't reach?

### The 2-D transform

Take a block of `Tb` tokens → a `Tb × C` matrix `X` (rows = time, cols = key-dimension). Apply
**two** transforms:

```
A = Utᵀ · X · Vk          Ut = temporal KLT (Tb×Tb) ,  Vk = spatial PCA (C×C)
```

`A[i, j]` = how much energy is in **temporal-mode i × spatial-mode j**. Square-and-average over
blocks → an energy **grid** `E[i, j]`. Compressing = keep the `N` strongest cells of that grid,
zero the rest, invert.

### The whole point: which cells to keep

```
(A) per-token MLA : keep ALL time rows × top-r spatial cols   (a full-width rectangle)
(B) best rectangle: top-Kt time × top-dc spatial  (drop weak time modes too)
(S) separable     : the cells you'd pick if energy were  time-marginal ⊗ space-marginal  (independent)
(C) joint optimal : the top-N cells of the ACTUAL grid, wherever they sit (non-rectangular)
```

**Coupling = does (C) beat (S)?** — i.e., is the true energy grid *more* than the outer product
of its margins?

### Numeric example (2×2 grid, keep N=2 cells)

```
Separable grid (rank-1):  E = [ 8  4 ]   = outer([4,1],[2,1])
                              [ 2  1 ]
   margins already explain it -> joint and separable pick the SAME 2 cells -> coupling = 0.

Coupled grid (diagonal):  E = [ 8  1 ]    margins: time=[9,9], space=[9,9]
                              [ 1  8 ]
   separable surrogate = outer(margins) = uniform -> picks 2 cells worth  8+1 = 9
   joint picks the true top-2 (the diagonal)      -> 8+8 = 16
   joint keeps 16/18 = 89% energy vs separable 9/18 = 50%  ->  COUPLING IS REAL.
```

The coupled grid's energy sits on the **diagonal** (time-mode *i* pairs with spatial-mode *i*) —
structure a rectangle/separable support physically cannot capture.

### gate1's actual verdict

```
   r    comp |  (A)MLA  (C)joint |  CPL = (S)−(C)
   64    64x |  19.143   21.070  |   +1.176     <- coupling real (joint beats separable)
  128    32x |   9.183    9.193  |   +0.207        AND the gain widens at aggressive r
  256    16x |   6.528    6.462  |   +0.037
  512     8x |   5.976    5.936  |   +0.003
```

- **Coupling is statistically REAL** and grows as you compress harder → **GATE #1 = PASS**.
- **BUT** look at (A) vs (C): at 64× the joint codec (21.07) is **worse** than plain per-token MLA
  (19.14). The coupling is real but the **block transform** (which also adds a 16-token latency)
  **does not beat just doing per-token MLA.** Honest negative for the *block* form.

**Why this points to Experiment (b):** gate1 exploited time via a **block transform** (heavy,
high-latency). The *streaming* way to exploit the same temporal structure is **prediction** —
cache only the innovation of each token's latent. That's lighter, causal, and is exactly what
phase4/phase5 said is alive. The field's name for it is **DeltaKV**.

---

## Part 4 — Experiment (b): DeltaKV on the nonlinear-MLA latent

**Claim to test:** at a matched cache budget (bits/token), caching the **temporal innovation** of
the nonlinear-MLA latent beats caching the latent per-token — i.e. temporal stacking pays off
*on top of* the nonlinear method.

### The codec (per layer)

```
per-token (baseline):     cache  Q_B(z_t)                        bits = d_c · B
DeltaKV (ours):           predict  ẑ_t = z_{t-1}·P + c           (linear AR, fit on calibration)
                          cache    Q_B'(e_t),  e_t = z_t − ẑ_t   bits = d_c · B'
                          decode   z_t = ẑ_t + Q_B'(e_t)         (closed loop: uses decoded history)
   then reconstruct K,V = z·Vkᵀ + corr(z)   (the SAME nonlinear decoder, untouched)
```

Because `Var(e) ≈ (1 − FVE)·Var(z) ≈ 0.65·Var(z)`, the innovation needs fewer bits for the same
fidelity. For a Gaussian source the saving is exactly:

```
ΔR = ½ · log2( Var(z) / Var(e) ) = ½ · log2( 1 / (1−FVE) )  bits/dim
   with FVE ≈ 0.35  →  ½·log2(1.54) ≈ 0.31 bits/dim  saved, for free, at equal quality.
```

### What we measure

- **Diagnostic (cheap):** latent temporal FVE (real vs shuffled-null) and the predicted bits/dim
  saved — confirms phase5's headroom *on the exact latent the method uses*.
- **Downstream (the real test):** WikiText-2 perplexity vs **bits/token**, two curves
  (per-token vs DeltaKV). **WIN = DeltaKV curve sits below** (lower ppl at equal bits, or equal
  ppl at fewer bits).

### Controls (keep it honest)

- **Null:** shuffle the latent's temporal order before fitting `P` → predictor useless →
  innovation = the whole latent → DeltaKV must collapse onto the per-token curve. If DeltaKV
  "wins" under the null, the win is a bug.
- **Wiring check:** at large `B` (≈16 bits) both codecs ≈ the unquantized nonlinear-MLA ppl.
- **Closed vs open loop:** the faithful decoder predicts from the *decoded* history (no drift
  cheating). v1 runs the cheap **open-loop** gate first (predict from true history); only if that
  wins do we pay for the sequential closed-loop version.

### Win condition (the gate)

```
DeltaKV beats per-token at matched bits/token on perplexity, AND the gap survives the null.
  -> temporal stacking is a real, free lever on top of nonlinear MLA  -> build closed-loop + RD curve.
ELSE -> per-token nonlinear MLA already captures it (like gate1's block result) -> stop.
```

Implemented in `phase8_deltakv_stack.py` (v1: diagnostic + open-loop perplexity gate).

---

## Part 5 — Results (Llama-2-7B, WikiText-2)

Run: `KV_TAU=0.9  KV_CALIB=60000  KV_PPL_CHUNKS=40  KV_BITS=2,3,4,6,16  KV_CLIP=4.0`
(full output: `results/phase8_deltakv_stack.txt`; an under-calibrated 8k-token smoke run is in
`results/phase8_deltakv_stack_smoke.txt` and should be ignored — its lossless B16 ppl is 43.6,
i.e. the reconstruction was broken by too little calibration, not by quantization).

### Wiring check — PASS

```
baseline (full K+V) ppl = 5.942      nonlinear-MLA @ lossless(B16) ppl = 6.863   (mean d_c = 1143)
```

At B=16 the codec is ≈ the unquantized nonlinear MLA (6.86 vs 5.94), so the bit-budget rows below
are trustworthy.

### Diagnostic — PASS (the latent IS temporally predictable)

```
mean temporal headroom (FVE_real − FVE_null) = +0.332      mean bits/dim saved = 0.178
FVE_null ≈ −0.02 per layer  (the shuffled control is properly destroyed)
FVE_real peaks at layer 6 (0.408); strong through the early/mid stack, tapering late.
```

This confirms phase5 on the *exact* latent the method caches: there is ~0.18 bits/dim of
temporal headroom, in principle free.

### Downstream gate — FAIL (the headroom does not convert)

```
bits | per-token |  DELTA  | DELTA-null |  read
-----+-----------+---------+------------+------------------------------------------------
  2  | 18072.02  | 4295.80 |  14833.02  |  both unusable (ppl in thousands)
  3  |   397.99  |  390.34 |    367.06  |  DELTA "wins" by 2%, but NULL beats DELTA -> spurious
  4  |    94.11  |  140.07 |     78.55  |  DELTA WORSE than per-token
  6  |    58.26  |   87.65 |     59.10  |  DELTA WORSE than per-token
 16  |     6.863 |   6.863 |     6.863  |  lossless tie
```

The script's auto-line ("DELTA beats per-token at 2/4 budgets") is misleading: applying the
**stated win gate** — DELTA < per-token AND the gain absent under the shuffled null — *no usable
budget passes*. At B=3 the shuffled null (367) actually beats the real predictor (390); at B=4/6
DELTA is flatly worse than per-token.

### Why it fails, despite the headroom

1. **Innovation tail-clipping.** The delta quantizer clips at ±CLIP·sd_e. When the linear AR
   predictor misses, `e_t` has heavy tails that exceed the clip → large errors that outweigh the
   variance reduction. This dominates at B=4/6, where DELTA goes *worse*.
2. **The real lever is centering, not prediction.** DELTA-null ≤ per-token at B=3/4 (367<398,
   78<94). Under the null `P≈0`, so `ẑ_t ≈ c` (the per-dim mean) — i.e. DELTA-null is just
   *mean-centered* per-token quantization, and that alone helps because the mid-rise quantizer is
   symmetric about 0. The temporal predictor on top adds nothing net.

### Verdict — lands in the "ELSE" branch

**Per-token nonlinear MLA already captures the temporal redundancy** (same story as gate1's block
result). Open-loop DeltaKV is a **negative result** at usable budgets. Since closed-loop predicts
from the *reconstructed* history (strictly noisier than this open-loop upper bound), it can only do
worse — so **the closed-loop codec is not warranted.**

**Cheap real win salvaged from the null:** add per-dim **mean-centering** to the per-token
quantizer (what DELTA-null accidentally demonstrated) — ≈17% ppl drop at B=4 (94→79), no predictor
needed.
