# Relation to Prior Work: phase9 (attention-aware linear KV) ≈ KQ-SVD

> **Bottom line for the thesis.** The Tier-1 *linear* attention-aware codec (phase9) is **not novel**
> — it is mathematically equivalent to **KQ-SVD** (Lesens, Rakhshan & Rabusseau,
> [arXiv:2512.05916](https://arxiv.org/abs/2512.05916), Dec 2025). Do **not** claim the linear method
> as a contribution. Cite KQ-SVD, present phase9 as an independent (covariance-form) re-derivation and a
> validated building block, and move the novelty to the cell KQ-SVD leaves empty (nonlinear × attention
> metric, and RoPE). This doc gives the equivalence proof, a point-by-point diff, and the citation
> language so a reviewer can't blindside you.

---

## 1. The equivalence (proof)

**KQ-SVD's objective** (their Eq., keys; per layer, per head): with key matrix `K ∈ ℝ^{T_k×d}`,
query matrix `Q ∈ ℝ^{T_q×d}`, and rank-`R` codec `K̂ = K A Bᵀ` (`A,B ∈ ℝ^{d×R}`),

```
min_{A,B}  ‖ K A Bᵀ Qᵀ − K Qᵀ ‖_F²        (error in the attention-score matrix K Qᵀ)
```

**phase9's objective** (logit error, per head): with `Σ_q = E[q qᵀ]`, `Δk = k − k̂`,

```
min  E_k[ Δkᵀ Σ_q Δk ]                      (expected squared attention-logit error)
```

**They are the same objective.** Expand KQ-SVD with `Δk_i = k_i − k̂_i`:

```
‖(K − K̂) Qᵀ‖_F² = Σ_{i,j} (Δk_iᵀ q_j)²
                  = Σ_i Δk_iᵀ ( Σ_j q_j q_jᵀ ) Δk_i
                  = Σ_i Δk_iᵀ (Qᵀ Q) Δk_i
                  = T_q · Σ_i Δk_iᵀ Σ_q Δk_i        since Σ_q = (1/T_q) Qᵀ Q.
```

So KQ-SVD minimizes exactly `Σ_i Δk_iᵀ Σ_q Δk_i` — phase9's objective, summed over calibration keys.
Both are solved in closed form (Eckart–Young) and yield the **same optimal rank-`R` reconstruction**:
- **KQ-SVD:** SVD of the data matrix `K Qᵀ`, top-`R` left singular vectors.
- **phase9:** eigenvectors of the covariance form `Σ_q^{1/2} C_k Σ_q^{1/2}`, with whitened encode/decode.

These are the sample-space (`KQᵀ`) and population-space (`Σ_q, C_k`) views of one optimum. Identical method.

---

## 2. Point-by-point diff

| axis | phase9 (ours) | KQ-SVD | same? |
|---|---|---|---|
| objective | min `Δkᵀ Σ_q Δk` (logit error) | min `‖K̂Qᵀ − KQᵀ‖²` | **identical (§1)** |
| solution form | eigh of `Σ_q^{1/2} C_k Σ_q^{1/2}` (whitening) | SVD of `K Qᵀ` (data matrix) | same optimum, different algebra |
| granularity | per (layer, head) | per (layer, head) | **same** |
| calibration | offline, WikiText-2 | offline, C4 (128 seqs) | same idea |
| values (V) | whiten by `(W_OᵀW_O)^{1/2}` | min `‖V A Bᵀ W_O − V W_O‖²` | **same W_O insight** |
| optimality claim | proof §5 (dropped eigenvalues) | Thm 2/3 (Eckart–Young) | same |
| "keys-alone is suboptimal" | keyPCA baseline | K-SVD baseline (Thm 1) | same message |
| RoPE | position-averaged `Σ_q^eff` (proposed, not yet run) | **not handled** | **gap KQ-SVD leaves open** |
| nonlinear decoder | proposed (Tier-2) | **none (purely linear)** | **gap KQ-SVD leaves open** |
| models shown | Llama-2-7B (+13B planned) | Llama2-7B/13B, Llama3-8B, Mistral-7B | they're broader |

**Minor genuine differences (not enough to claim novelty):**
- KQ-SVD applies the basis to *queries* at runtime (`V̂_K`) and absorbs the V basis into `W_O`; phase9
  reconstructs `k̂` and runs vanilla attention. Both preserve the same score; phase9's framing makes the
  "zero-overhead = same two matmuls as MLA" point cleaner, but it's a presentation difference.
- KQ-SVD's other baseline, **EigenAttention** (vertical `[K;Q]` concatenation + joint SVD), differs from
  phase9's **queryWrong** (project K onto the query subspace = phase3's bug). Their Thm 4 shows
  EigenAttention degrades to K-SVD under K/Q norm imbalance — a different failure mode than phase3's.

---

## 3. What KQ-SVD does **not** do (the open space)

1. **RoPE.** KQ-SVD "assumes standard attention without position embeddings." Real K is rotated before
   the score (`logit = qᵀ R_Δ k`). A RoPE-correct attention-aware basis — using the position-averaged
   `Σ_q^eff = E_Δ[R_Δᵀ Σ_q R_Δ]`, block-diagonal in RoPE's 2-D frequency blocks — is **not in KQ-SVD**.
   ⚠️ but partly contested: [EliteKV](https://arxiv.org/abs/2503.01586) already does RoPE-aware low-rank
   KV via per-head frequency selection + light uptraining. The *calibration-only `Σ_q`-averaging* variant
   is a narrower, still-open slice.
2. **Nonlinearity.** KQ-SVD is purely linear. **No one fits a nonlinear KV decoder under the attention
   metric.** [KV-CAR](https://arxiv.org/abs/2512.06727) is nonlinear but trained on *Euclidean*
   reconstruction. This is the empty cell — **Tier-2**.

---

## 4. The 2×2 that positions the thesis

| | Euclidean metric `‖Δk‖²` | Attention metric `Δkᵀ Σ_q Δk` |
|---|---|---|
| **Linear** | PCA · MLA · Palu · ASVD · K-SVD | **KQ-SVD** = our phase9 |
| **Nonlinear** | KV-CAR · our phase1/2b "ours" | **← OPEN — Tier-2 (phase10)** |

The contribution is **(a) filling the empty cell** and **(b) the diagnostic that explains why the prior
nonlinear cell underdelivered**: nonlinear KV curvature looks "too small" only because it was measured in
the Euclidean inner product; the model's inner product is `Σ_q`. We re-measure curvature there. Either
outcome is a result — a positive fills the cell; a negative ("the objective is the entire lever;
nonlinearity is a red herring even in the right metric") closes the 2×2 cleanly.

---

## 5. Other neighbors (one-liners for related work)

- [Expected Attention](https://arxiv.org/abs/2510.00636) (Oct 2025) — uses **pre-RoPE query mean+cov**
  for KV compression, but for **token eviction** (importance scoring), not a low-rank projection basis.
- [CARE](https://arxiv.org/abs/2603.17946) (Mar 2026) — activation-aware low-rank for GQA→MLA
  conversion + **adjusted-rank allocation** (overlaps our budget-allocation lever); input-activation
  reconstruction, not the query metric.
- [EliteKV](https://arxiv.org/abs/2503.01586) (Mar 2025) — RoPE-frequency-aware joint low-rank KV
  (light uptraining).
- [KV-CAR](https://arxiv.org/abs/2512.06727) (Dec 2025) — nonlinear per-head autoencoder, Euclidean loss.

---

## 6. Citation language (drop-in)

> *"Our linear attention-aware key codec minimizes the query-weighted reconstruction error
> `E[Δkᵀ Σ_q Δk]`, equivalent to the attention-score error minimized by KQ-SVD
> [Lesens et al., 2025], which we recover here in closed form as a whitened generalized eigenproblem.
> We treat this linear codec as a validated baseline and study the previously unexplored regime of a
> **nonlinear** decoder trained under the same attention metric (and its RoPE-correct extension), which
> KQ-SVD and prior low-rank methods do not address."*
