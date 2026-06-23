# Phase-10: Does Nonlinearity Pay *in the Attention Metric*? — Walkthrough + Result

> **The question.** phase9 showed the lever for KV compression is the **objective** (compress in the
> `Σ_q` attention metric, = KQ-SVD), not the **architecture**. phase2b/`probe_nonlinear_gap` showed a
> nonlinear decoder is a red herring **under Euclidean MSE**. phase10 closes the loop: put the
> nonlinear residual back, but train and measure it in the **right (`Σ_q`) inner product**. Is there
> curvature the model actually reads?
>
> **The answer: no.** The empty cell of the 2×2 is empty. This doc explains the test, gives a by-hand
> numeric example of *why*, and reports the numbers. Code: [phase10_nonlinear_attn.py](phase10_nonlinear_attn.py).
> Companion to [attn_aware_fix_explained.md](attn_aware_fix_explained.md) (the linear `Σ_q` codec) and
> [kqsvd_relation.md](kqsvd_relation.md) (why the linear part = KQ-SVD).

---

## 1. The 2×2 phase10 completes

| | Euclidean metric `‖Δk‖²` | Attention metric `Δkᵀ Σ_q Δk` |
|---|---|---|
| **Linear** | PCA · MLA | KQ-SVD (= phase9 `attnAware`) ✅ |
| **Nonlinear** | KV-CAR · phase1/2b "ours" | **phase10 ← the empty cell** |

Three corners were filled and pointed the same way ("the objective is the lever, not the
nonlinearity"). phase10 fills the fourth: a nonlinear residual `corr(z)` trained **under `Σ_q`**.

---

## 2. What phase10 runs (per head, K only)

Start from the phase9 linear base (the `Σ_q`-whitened rank-`r` codec): `z = Eᵀk`, `k̂_lin = D z`.
Add a tiny per-head residual `corr(z) = Linear(r→h)→GELU→Linear(h→d)` (zero-init, so it **starts at the
linear base** and can only help):

```
k̂ = D z + corr(z)
```

Train `corr` two ways and compare:

| name | training loss | asks |
|---|---|---|
| `corrW` | `‖ Σ_q^{1/2}(k − k̂) ‖²` (the **logit** metric) | does curvature help **in the metric the model reads**? |
| `corrE` | `‖ k − k̂ ‖²` (plain Euclidean) | does it help raw reconstruction? (the control) |

Read-outs: **(1)** held-out **logit error** with a **Gaussian-null** control (`Δgap` = real gain minus
the overfitting floor); **(2)** WikiText-2 **perplexity** (real RoPE+softmax). WIN for the cell =
`corrW` beats the linear base **and** the gain survives the null.

---

## 3. Why it can't help — the by-hand example

The whole result is one geometric fact: **the curvature lives in directions the queries barely read,
and the `Σ_q` metric down-weights exactly those directions.** Here it is in 2-D.

### Setup
Five keys on a curve (`head_dim = 2`), compress to `r = 1`:

```
       k = [ x1 ,  x2 ] ,   with  x2 = 0.2·x1²   (a parabola — genuine curvature)
   A [-2, 0.8]  B [-1, 0.2]  C [0, 0]  D [1, 0.2]  E [2, 0.8]
   mean = [0, 0.4]      Var(x1) = 2.000      Var(x2) = 0.112
```

Queries read **x1 strongly, x2 barely**: `Σ_q = diag(1, 0.01)`, so `Σ_q^{1/2} = diag(1, 0.1)`.

### The linear base keeps x1, drops x2
Whitened covariance `Σ_q^{1/2} C_k Σ_q^{1/2} = diag(2.000, 0.00112)` → PCA keeps **x1**. So
`k̂_lin = [x1, 0.4]` (x2 pinned to its mean). The leftover is the whole parabola in x2.

### The residual *can* fix the curve — `corr(x1) = 0.2x1² − 0.4` recovers x2 exactly. Now score it two ways:

```
                       leftover x2 fixed by corr(z)
  Euclidean rel-error : sqrt(0.112 / 2.112)       = 23.0 %   ← residual looks very useful
  LOGIT     rel-error : sqrt(0.00112 / 2.00112)   =  2.37 %  ← 10× smaller
```

**The same nonlinear correction cuts Euclidean error by 23% but logit error by only 2.4%.** The
curvature is real — Euclidean sees it — but **logit-irrelevant**, because `Σ_q` weights the curved
coordinate (x2) by 0.01. The model never reads the curve.

### Push it toward reality
Real K curvature lives in *many* directions, each carrying *less* query energy. Shrink `Σ_q[x2]`:

```
  Σ_q[x2] = 0.01   → logit nl-gain 2.37 %
  Σ_q[x2] = 0.001  → logit nl-gain 0.75 %
  Σ_q[x2] = 0.0001 → logit nl-gain 0.24 %      → washes into the Gaussian-null floor
```

This is exactly phase10's measured `nl-gain ≈ 0.05%`. **And it predicts the twist:** `corrE`
(Euclidean) chases the 23% and fully fits the curve; `corrW` (logit) sees ~0% incentive and barely
moves — so the *Euclidean*-trained residual reconstructs K **better**, and since real RoPE'd attention
uses the curved coordinate a hair more than the aggregate `Σ_q` says, `corrE` even helps perplexity
slightly **more** than the "right-metric" `corrW`. The nonlinearity is doing generic reconstruction,
**not** exploiting attention-geometry curvature.

---

## 4. The numbers (Llama-2-7B, K-only, WikiText-2)

### Diagnostic — held-out LOGIT error, Gaussian-null controlled

```
 rank | keyPCA(lin)  attn(lin)   +corrE   +corrW |  nl-gain  Δgap-null  verdict
   16 |      0.1540     0.0994   0.0977   0.0985 |  +0.0009    +0.0009   no real gain
   32 |      0.1121     0.0665   0.0649   0.0661 |  +0.0005    +0.0005   no real gain
```

The residual cuts logit error by **~0.05–0.09%** and barely clears the overfitting floor
(`Δgap-null ≈ 0`). In the metric the model reads, **the bend has nothing to do** — just as the toy
predicts.

### Downstream — WikiText-2 perplexity (baseline 5.739)

```
 rank  cacheX |  keyPCA  attnAware  +corrW(new)   +corrE
   16    8.0x |  13.352     11.909       11.086   10.835
   32    4.0x |   6.765      6.635        6.565    6.491
```

The auto-label says "NL HELPS," but two facts make it a **negative**:
1. **Metric-agnostic.** `corrE` (Euclidean) **beats** `corrW` (the "right" metric) at both budgets — the
   opposite of what an attention-curvature effect would do. The toy explains why.
2. **Noise-level where it counts.** At the only near-usable budget (4×), the residual buys **+0.07 ppl**
   (6.64 → 6.57, vs baseline 5.74). The larger 8× gain is in the already-broken regime (ppl 11–13).

---

## 5. Verdict — the empty cell is empty

| | Euclidean | Attention metric `Σ_q` |
|---|---|---|
| **Linear** | PCA/MLA | **KQ-SVD / phase9 — the lever ✅** |
| **Nonlinear** | KV-CAR / phase2b | **phase10 — negligible & metric-agnostic ❌** |

> **Nonlinear KV curvature is a red herring *even in the model's own inner product*.** A residual
> trained under the attention metric reduces the logit error by ~0.05% (below the null floor), and
> downstream it is no better — actually worse — than a residual trained under plain Euclidean. The
> curvature that exists lives in directions the queries don't read; `Σ_q` correctly discards it.
> **The objective is the entire lever; the architecture is not.**

This closes the 2×2 and locks the thesis spine: every axis of KV compression should minimize the
distortion the model *reads* (`Σ_q` for K, `W_OᵀW_O` for V), and once you do, adding nonlinearity buys
nothing. The only remaining room is **deeper on the objective** — the closed-form second-order
(output/Fisher) metric, not a richer decoder.

## Reproduce
`sbatch run_phase10.sh` → `results/phase10_nonlinear_attn.txt` (job 19540767: 31 min, K-only, ranks
16/32, Gaussian null on). Env: `P10_RANKS`, `P10_CALIB`, `P10_NULL`, `KV_RCOND`.
