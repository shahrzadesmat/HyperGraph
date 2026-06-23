# Phase-12: Joint K+V at a Matched TOTAL Budget — Result + the Shared-Latent Insight

> **Goal.** phase11 showed the 2nd-order output/Fisher metric `M̄` beats KQ-SVD for the KEY basis, but
> K-only ("16× on K") is only ~1.9× on the *total* cache. phase12 compresses **both K and V** at a
> matched **total** budget and asks: does the Fisher advantage survive on the full cache, and is there a
> *usable* high-compression operating point? **Answer: the ordering survives (fisher > kqsvd > mla at
> every usable budget), but separate K/V low-rank is a lossy codec — the absolute numbers want a
> *shared* [K;V] latent.** This doc gives the result and explains, with a worked example, *why* shared
> beats separate and how the Fisher metric plugs into it.
> Code: [phase12_joint_kv.py](phase12_joint_kv.py). Builds on [phase11_explained.md](phase11_explained.md).

---

## 1. What phase12 runs

Compress K **and** V, each per head to rank `r` (total cacheX = `head_dim/r`, counting both). Three
full stacks:

| stack | K metric | V metric | = |
|---|---|---|---|
| `mla` | keyPCA (Euclidean) | valuePCA (Euclidean) | MLA / Palu-style |
| `kqsvd` | `Σ_q` (1st-order logit) | `G = W_OᵀW_O` (output) | KQ-SVD's K & V |
| `fisher` | `M̄` (2nd-order output) | `G` | **this work** |

V uses the same output metric `G` in both `kqsvd` and `fisher`, so the **K metric (`Σ_q` vs `M̄`) is the
only differentiator.** WikiText-2 perplexity, real RoPE+softmax. Baseline (full K+V) = 5.739.

---

## 2. Results (Llama-2-7B, run 19544962)

```
 rank  cacheX(total) |    mla     kqsvd    fisher | Δ(kqsvd−fisher)  Δ(mla−fisher)
    8       16.0x     |  541.61   609.32   579.45 |     +29.87          −37.84       (all garbage)
   16        8.0x     |  154.91   108.07   105.85 |      +2.21          +49.05       FISHER BEST
   32        4.0x     |   29.67    21.88    21.09 |      +0.80          +8.58        FISHER BEST
   64        2.0x     |    8.49     7.50     7.49 |      +0.01          +1.00        FISHER BEST
```

**Two findings:**
1. **The ordering holds on the full cache:** `fisher > kqsvd > mla` at all *usable* budgets (3/4 — it
   loses only at 16× total, where every method is >500 ppl garbage). At 4×, fisher beats mla by **+8.6**
   and kqsvd by **+0.8**.
2. **But separate low-rank K+V is lossy.** Even 2× total is **7.49** (vs baseline 5.74, a +1.75
   penalty) — *not* near-lossless. Compare phase2b's **shared-latent** MLA: **6.15 @4.8×**. So
   compressing K and V *separately* is markedly worse than a shared latent. The relative *objective*
   claim is confirmed; the absolute compression is modest.

---

## 3. The insight: separate K/V wastes budget — a shared [K;V] latent is the fix

### The two ways to spend the budget

```
SEPARATE (phase12):  z_K = E_Kᵀ k  (r_K)  ;  z_V = E_Vᵀ v  (r_V)   →  cache = r_K + r_V
SHARED  (phase2b/MLA): z = Eᵀ [k;v]  (d_c, ONE latent)            →  cache = d_c, rebuilds BOTH
```

### Why shared is more efficient — Reason 1: K and V are correlated

Both come from the **same** hidden state `x`: `k = x W_Kᵀ`, `v = x W_Vᵀ`. So `[k;v]` lives in a subspace
of dimension ≤ rank of `x`'s contribution — it does **not** fill its `2·dh` dims. Separate compression
can't look across the K/V boundary, so it **stores the shared `x`-structure twice** (once in `z_K`,
once in `z_V`). The shared latent stores it **once**.

**Worked toy (per head, 2-D), at matched total budget = 2.** Let `k = [x1, x2]`, `v = [x1, x2]`
(maximally correlated), `Var(x1)=4, Var(x2)=1`:

```
SEPARATE  r_K=r_V=1 : each keeps its top dir x1, drops x2.
   k̂=[x1,0],  v̂=[x1,0]   → x2 LOST in both, x1 stored twice.   error² = 2·Var(x2) = 2

SHARED    d_c=2 : stack [x1,x2,x1,x2].  Its 4×4 covariance has eigenvalues  [8, 2, 0, 0]
   (genuinely rank-2).  Top-2 dirs [1,0,1,0],[0,1,0,1] span the data exactly.
   → reconstruct BOTH k and v PERFECTLY.   error = 0
```

Same budget (2): separate loses 20% of the signal, shared is **lossless** — purely by exploiting that
`k` and `v` share `x`. (Real K/V are *partially* correlated, so the gain is smaller — but this is
exactly why phase2b's shared MLA beats phase12's separate K/V.)

### Reason 2: flexible allocation

Separate forces a rigid split (`r_K` vs `r_V`; phase12 fixed them equal). The shared latent's `d_c`
directions are chosen by the **joint** spectrum, so capacity flows automatically to wherever the
combined K+V energy is — layer by layer, no manual split.

### Provable
Joint PCA of `[K;V]` picks the top-`d_c` eigvecs of the full `2dh×2dh` covariance, **including
directions that mix K and V**. Separate is the *block-diagonal special case* (directions confined to
one block). So **joint ≤ separate error at matched total budget**, always — separate is a constrained,
strictly-worse version.

---

## 4. How the Fisher metric plugs into the shared latent (the next combination)

Keep the shared-latent efficiency **and** add the 2nd-order objective via a **block-diagonal** metric:

```
P = blkdiag( M̄^{1/2} ,  G^{1/2} )            M̄ = output-Fisher metric for K (phase11)
                                              G  = W_OᵀW_O (output metric for V)

whiten  s' = P·[k;v] ;  W = top-d_c eigvecs of  P·Cov([K;V])·P ;  z = Wᵀ P[k;v] ;  [k̂;v̂] = P⁻¹W z
```

This gets **both levers at once**:

| | exploits K-V redundancy | spends budget on output-relevance |
|---|---|---|
| phase2b (shared, Euclidean) | ✅ | ❌ |
| phase11/12 (Fisher, **separate**) | ❌ | ✅ |
| **shared + Fisher (proposed phase13)** | ✅ | ✅ |

Both levers bite hardest at **aggressive** compression (small `d_c`): you can't afford to duplicate the
shared `x`-structure, *and* every dimension must be output-relevant (recall phase11's gain **widened**
under compression). So the combination should push the **near-lossless frontier to a higher total
cacheX** than either Euclidean-MLA (phase2b) or separate-Fisher (phase12).

**Build path:** extend phase2b (it already has the shared `[K;V]` latent with the correctness-checked
`z = x·W_z` folding) — replace `eigh(Cov)` with `eigh(P·Cov·P)`, capture `M̄` (needs real attention
weights → eager, like phase11) and `G` (from `W_O`). Drop the nonlinear `corr` (phase10 killed it).
**The one subtlety:** phase2b *standardizes* `[K;V]` first, so the effective standardized-space metric
is `diag(sd)·P²·diag(sd)`, whose square root is **not** `diag(sd)·P` (they don't commute) — the
whitening must be derived carefully in standardized space (numpy wiring-check before the GPU run).

---

## 5. Honest framing

The **big** efficiency win here is the shared latent itself — which is *MLA*, already established. The
**novel-and-ours** part stays the 2nd-order Fisher metric (a ~6% basis nudge, 94% subspace overlap with
KQ-SVD — see [phase11_explained.md](phase11_explained.md)). So the right claim is: *"we put the
closed-form output-Fisher metric on the standard shared-latent MLA architecture, and it extends the
near-lossless frontier vs. Euclidean MLA"* — not "we invented a new high-compression codec." phase12's
own numbers (fisher > kqsvd > mla at every usable budget) are the evidence that the metric helps on the
full cache; the shared latent is the carrier that makes the absolute numbers good.

## Reproduce
`sbatch run_phase12.sh` → `results/phase12_joint_kv.txt` (job 19544962; ~8 min; eager attention).
Env: `P12_RANKS`, `P12_CALIB_SEQS`, `P12_SEQLEN`, `KV_RCOND`.
