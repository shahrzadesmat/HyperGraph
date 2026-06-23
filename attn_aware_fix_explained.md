# Attention-Aware KV Compression — the Fix to phase3, with Numerics & Proof

> **What this is.** `phase3_query_aware.py` tested the right idea ("compress keys to preserve
> the attention *score* `q·k`, not the key *vector*") but implemented the wrong math and blew up
> (ppl **1631 / nan**). This doc shows (1) exactly what was wrong, (2) the correct objective and
> its closed-form solution — a **whitened generalized eigenproblem** — with two fully worked
> numeric examples, and (3) a short optimality proof. Implemented in
> [phase9_attn_aware.py](phase9_attn_aware.py).
>
> Companion to [attention_aware_explained.md](attention_aware_explained.md) (the K/V whitening
> recipe) and [baseline_comparison.md](baseline_comparison.md) (the baseline ladder).

---

## 0. TL;DR — one line

```
phase3 (BUG):   k̂ = U_q U_qᵀ k          U_q = top-r eigenvectors of Σ_q = E[q qᵀ]   (query covariance ALONE)
phase9 (FIX):   k̂ = Σ_q^{-1/2} W Wᵀ Σ_q^{1/2} k     W = top-r eigenvectors of  Σ_q^{1/2} C_k Σ_q^{1/2}
```

phase3 projected the key onto the **query subspace**, ignoring where the keys actually live
(`C_k`). The fix **whitens by `Σ_q^{1/2}`, does PCA there, and un-whitens** — the provably optimal
rank-`r` linear codec for the **attention-logit error**. Both encode and decode fold into
matmuls → **zero inference overhead vs. MLA**.

---

## 1. The bug in phase3

The attention logit per head is `s = qᵀk`. We cache `r` numbers per key and rebuild `k̂`. phase3
chose the key subspace as the **top-r eigenvectors of the query 2nd-moment `Σ_q = E[qqᵀ]`** and
did an orthogonal projection ([phase3_query_aware.py:113-117](phase3_query_aware.py#L113-L117)):

```python
U = Uq[l][:, :, :R]            # top-r eigvecs of Cq  (QUERY covariance only)
z    = einsum('nhd,hdr->nhr', k, U)   # encode
khat = einsum('nhr,hdr->nhd', z, U)   # decode  ->  k̂ = U_q U_qᵀ k
```

Two things are wrong with `k̂ = U_q U_qᵀ k`:

1. **It throws away all key energy outside the query's top-r subspace.** If the keys have large
   variance in a direction the queries read only *moderately*, that direction is deleted — even
   though it contributes a lot to `qᵀk`. `C_k` never enters the choice of subspace.
2. **It optimizes neither error.** It is the optimal codec for *neither* `‖Δk‖²` (that's key-PCA)
   *nor* the logit error `Δkᵀ Σ_q Δk` (that's the whitened problem below). It is a heuristic that
   happens to be right only when `Σ_q` and `C_k` share eigenvectors — which they do not.

Result: phase3's query-aware arm gave **ppl 1631 / nan** and its score-error was *worse* than
plain key-PCA at every rank (the "reduction" column was negative). The instinct was correct; the
optimization was not.

`phase6_output_aware_v.py` has the milder version of the same mistake for V: it keeps the value
eigenbasis and only **re-ranks** directions by the diagonal gain `diag(EᵀGE)`
([phase6_output_aware_v.py:61-62](phase6_output_aware_v.py#L61-L62)) — it never **rotates** the
basis, so it captures none of the off-diagonal coupling and reports ≈ +0.0% (a false negative).

---

## 2. The objective: what error attention actually feels

The model never reads `k`; it reads the logit `qᵀk`. With `Δk = k − k̂`, the squared logit error,
averaged over the query distribution, is

```
E_q[(qᵀΔk)²] = Δkᵀ E[q qᵀ] Δk = Δkᵀ Σ_q Δk = ‖ Σ_q^{1/2} Δk ‖²
```

So the right loss is a **Σ_q-weighted (Mahalanobis) norm**, not the Euclidean `‖Δk‖²`. Plain
MLA / key-PCA minimize `‖Δk‖²` — the `Σ_q = I` special case, i.e. the **wrong inner product**.
This is also *why the nonlinear residual looked "too small"*: it was trained under `‖Δk‖²`, which
is dominated by high-variance channels that linear PCA already nails — there is nothing left for a
bend to do **in that metric**. Change the metric and the budget moves to logit-relevant
directions.

---

## 3. Worked example A — *which direction to keep* (diagonal, by hand)

Three orthogonal directions `e1,e2,e3` (head_dim = 3), compress to **r = 1** (keep one). Each
direction has a key variance `λ_k` and a query energy `λ_q`. The **logit energy** of a direction
is `λ_k · λ_q` (how much it contributes to `E[(qᵀk)²]`).

| direction | `λ_k` (key var) | `λ_q` (query energy) | **logit = `λ_k·λ_q`** |
|---|---|---|---|
| e1 | **100** | 3 | 300 |
| e2 | 60 | 8 | **480** |
| e3 | 4 | **40** | 160 |

Total logit energy = 300 + 480 + 160 = **940**. Each method keeps one direction, drops the rest:

| method | keeps | logic | **dropped logit / 940** |
|---|---|---|---|
| key-PCA (MLA) | e1 | max `λ_k` (most key variance) | 640/940 = **68.1 %** |
| queryWrong (phase3) | e3 | max `λ_q` (most query energy) | 780/940 = **83.0 %** |
| **attnAware (ours)** | e2 | **max `λ_k·λ_q`** (most logit energy) | 460/940 = **48.9 %** |

The story matches the real data exactly: **key-PCA is decent** (high-variance key directions *are*
read), **queryWrong is catastrophic** (it keeps `e3`, a near-null *key* direction the model can't
reconstruct anything from — phase3's 1631 ppl), and **attnAware wins** by keeping the direction
that actually carries the score. attnAware reduces to "rank directions by `λ_k·λ_q`" only because
this example is diagonal; the general case needs a rotation — Example B.

---

## 4. Worked example B — *the whitening rotation* (2×2, exact)

Now `Σ_q` is **not** diagonal in the key basis, so the optimal directions are a **rotation** of
both the key axes and the query axes — the part phase3/phase6 could never reach. head_dim = 2,
compress to **r = 1**.

```
C_k = [100   0 ]      (keys vary along x1)        Σ_q = [ 8   8 ]      (queries read a tilted dir)
      [  0   9 ]                                         [ 8  18 ]
```

**Step 1 — the three candidate directions.**
```
key-PCA      keeps eigvec of C_k    = [1.000, 0.000]      (the x1 axis)
queryWrong   keeps eigvec of Σ_q    = [0.485, 0.875]      (the query's top axis)
```

**Step 2 — whiten.** `Σ_q^{1/2}` and its inverse (note `Σ_q^{1/2}·Σ_q^{1/2} = Σ_q` exactly):
```
Σ_q^{1/2}  = [2.558  1.208]        Σ_q^{-1/2} = [ 0.455  -0.135]
             [1.208  4.067]                     [-0.135   0.286]
```

**Step 3 — PCA in whitened space.** Form `M = Σ_q^{1/2} C_k Σ_q^{1/2}` and take its top eigvec:
```
M = [667.30  353.06]      eig(M) = [880.20 , 81.80]      W_top = [-0.856, -0.516]
    [353.06  294.70]
```

**Step 4 — fold into encode/decode.** `E = Σ_q^{1/2} W`, `D = Σ_q^{-1/2} W`:
```
encode  z   = Eᵀ k ,  E_col = [-2.814, -3.134]      (the r numbers actually cached)
decode  k̂ = D z   ,  D_col = [-0.320, -0.032]
```

**Step 5 — score it.** relative logit error `= tr((I−P)ᵀ Σ_q (I−P) C_k) / tr(Σ_q C_k)`,
`tr(Σ_q C_k) = 962`:

| method | keeps direction | **rel. logit error** |
|---|---|---|
| key-PCA (MLA) | `[1.00, 0.00]` | 162.0/962 = **16.8 %** |
| queryWrong (phase3) | `[0.49, 0.87]` | 280.3/962 = **29.1 %** |
| **attnAware (ours)** | whitened | **81.8/962 = 8.5 %** |

attnAware **halves** key-PCA's logit error (16.8 → 8.5 %), and queryWrong is the **worst** (29.1 %) —
reproducing phase3 in miniature. The kept direction is neither the key axis nor the query axis: it
is the rotated whitened axis no axis-aligned method can name.

**The punchline (this *is* the proof, see §5).** Notice `tr(M) = 880.2 + 81.8 = 962 = tr(Σ_q C_k)`,
and attnAware's residual `81.8` is **exactly the eigenvalue of `M` it dropped**. The eigenvalues of
`M` *are* the per-direction logit energies; keeping the top one and dropping the rest is optimal
water-filling. No other rank-1 linear codec can beat `81.8/962`.

---

## 5. The proof (optimality of whitened PCA for the logit metric)

**Claim.** Among all rank-`r` linear codecs `k̂ = D Eᵀ k` (`E, D ∈ ℝ^{d×r}`), the one minimizing the
expected logit error `J = E_k[‖Σ_q^{1/2}(k − k̂)‖²]` is

```
W = top-r eigenvectors of  M = Σ_q^{1/2} C_k Σ_q^{1/2},
E = Σ_q^{1/2} W ,   D = Σ_q^{-1/2} W ,
```

and its optimal value is `J* = Σ_{j>r} μ_j`, the sum of the **dropped** eigenvalues of `M`.

**Proof.** Substitute the whitened key `g = Σ_q^{1/2} k` and `ĝ = Σ_q^{1/2} k̂`. Then
`J = E‖g − ĝ‖²`, and `ĝ = Σ_q^{1/2} D Eᵀ Σ_q^{-1/2} g` is an arbitrary rank-`r` linear map of `g`.
Minimizing the reconstruction MSE of `g` by a rank-`r` linear map is **ordinary PCA on `g`**
(Eckart–Young): the optimum is `ĝ = W Wᵀ g` with `W` = top-`r` eigenvectors of
`E[g gᵀ] = Σ_q^{1/2} C_k Σ_q^{1/2} = M`, and residual `E‖g − ĝ‖² = Σ_{j>r} μ_j`. Mapping back,
`k̂ = Σ_q^{-1/2} ĝ = Σ_q^{-1/2} W Wᵀ Σ_q^{1/2} k`, i.e. `z = Wᵀ Σ_q^{1/2} k` (encode) and
`k̂ = Σ_q^{-1/2} W z` (decode). ∎

**Corollaries.**
- **attnAware ≤ key-PCA, always.** key-PCA (`E = D = `eigvecs of `C_k`) is a *feasible* rank-`r`
  linear codec, hence its `J` ≥ `J*`. So on the score-error diagnostic attnAware **cannot lose** —
  a built-in correctness check (`phase9` flags it if it ever does). This is the opposite of
  phase3, and is how you know the bug is fixed.
- **Right budget ledger.** `tr(M) = tr(Σ_q C_k)` = total logit energy, and dropping direction `j`
  costs exactly `μ_j`. So ranks should be allocated across heads/layers by the **whitened spectrum
  `μ`** (logit energy), not by the key-variance spectrum MLA uses. That is a second, free lever.
- **`P = D Eᵀ = Σ_q^{-1/2} W Wᵀ Σ_q^{1/2}` is an oblique projector** (`P² = P`, rank `r`), and at
  `r = d` it is `I` (lossless) — the `phase9` wiring check.

---

## 5b. Why it's zero-overhead at inference (the folding)

The whitening matrices `Σ_q^{±1/2}` look like extra work, but they **never appear at runtime** — they
fold offline into the projection matrices the model already applies. The key `k` is itself
`k = W_K x` (the `k_proj`), so:

```
ENCODE   z = Wᵀ Σ_q^{1/2} k = (Wᵀ Σ_q^{1/2} W_K) x = W_enc · x        W_enc precomputed [r × d_model]
DECODE   k̂ = Σ_q^{-1/2} W z = (Σ_q^{-1/2} W) z      = W_dec · z        W_dec precomputed [d × r]
```

`W_enc` and `W_dec` are **collapsed once during calibration** into single dense matrices; `Σ_q^{1/2}`
and `Σ_q^{-1/2}` are never materialized during generation. So at inference:

| step | plain MLA | attnAware (ours) | cost difference |
|---|---|---|---|
| cache per token | latent `z` (`r` floats) | latent `z` (`r` floats) | **identical** |
| make `z` from `x` | one matmul `W_z·x` | one matmul `W_enc·x` | **same shape, same FLOPs** |
| rebuild `k̂` from `z` | one matmul (up-proj) | one matmul `W_dec·z` | **same shape, same FLOPs** |

attnAware is **the same two matmuls as MLA with different (precomputed) numbers in them** — same cache
size, same FLOPs, same memory traffic. The win is paid entirely **offline** (one extra eigendecomp per
head at calibration). Contrast the **nonlinear** residual `corr(z) = Linear→GELU→Linear`, which is a
genuine extra per-token MLP at *every* decode step — the **3–5 % decode-FLOP** surcharge phase2b
charged. attnAware buys a lower-error codec for **free**, where the nonlinear path buys it for a fee.

---

## 6. What changed in code: phase3 → phase9

| | phase3 (bug) | phase9 (fix) |
|---|---|---|
| subspace | eigvecs of `Σ_q` only | eigvecs of `Σ_q^{1/2} C_k Σ_q^{1/2}` |
| encode | `z = U_qᵀ k` | `z = Wᵀ Σ_q^{1/2} k` |
| decode | `k̂ = U_q z` | `k̂ = Σ_q^{-1/2} W z` |
| uses `C_k`? | **no** | yes |
| optimal for `Δkᵀ Σ_q Δk`? | no | **yes (proof §5)** |

The only new numerical care is `Σ_q^{-1/2}`: floor the query eigenvalues at `rcond·λ_max`
(`KV_RCOND`, default `1e-3`) before the inverse-sqrt so near-null query directions don't blow up
the decode. `Σ_q^{1/2}` and `Σ_q^{-1/2}` use the **same** floored eigenvalues, so `P² = P` and the
`r = d` wiring check stays exact.

**The V analog (same structure, drop-in).** A value error reaches the output through `W_O`, so the
metric is `‖W_{O,h} Δv‖² = Δvᵀ G_h Δv` with `G_h = W_{O,h}ᵀ W_{O,h}`. Replace `Σ_q → G_h`: whiten by
`G_h^{1/2}`, PCA on `G_h^{1/2} C_v G_h^{1/2}`, un-whiten. This **rotates** the basis (what phase6's
diagonal re-rank skipped). Both `Σ_q` (one running `qqᵀ` sum) and `W_O` (a weight you already have)
are available at calibration.

**RoPE note (K only).** `K` is reconstructed pre-RoPE but met post-rotation, so the exact weighting
is the position-averaged `Σ_q^{eff} = E_Δ[R_Δᵀ Σ_q R_Δ]`, which is block-diagonal within RoPE's 2-D
frequency blocks. Pre-RoPE `Σ_q` is a first-cut proxy; the perplexity surgery applies real RoPE +
softmax downstream, so **Metric 2 is exact regardless**.

---

## 7. How to run

```bash
# attention-aware K compression vs key-PCA(=MLA) vs phase3's buggy query-aware
QA_CALIB=60000 QA_PPL_CHUNKS=60 QA_RANKS=8,16,32,48,64,96 QA_PPL_RANKS=32,64 \
  python phase9_attn_aware.py        # or: sbatch run_phase9.sh
```

**Read the output.**
- *Score-error table:* attnAware **must** be ≤ key-PCA at every rank (proof §5). If not, the
  inverse-sqrt is mis-wired — raise `KV_RCOND`. queryWrong > key-PCA reproduces phase3.
- *Wiring checks:* key-PCA@`r=d` and attnAware@`r=d` must both equal baseline ppl (`P = I`).
- *Perplexity table:* **WIN = attnAware ppl < key-PCA ppl** at matched `r`. If yes, the lever to
  beat linear KV compression is the **objective** (preserve the logit via the `Σ_q`-whitened
  generalized eigenproblem) — not encoder nonlinearity. That is the Tier-1 result; Tier-2 then
  re-fits the nonlinear residual `corr(z)` under this same `Σ_q` metric to test whether curvature
  reappears once it is measured in the inner product the model actually uses.
