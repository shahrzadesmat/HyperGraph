# Phase-11: The Second-Order (Output / Fisher) Metric — Walkthrough + Result

> **The idea.** phase9 (= KQ-SVD) compresses keys to preserve the **logit** `qᵀk` — a *first-order*
> proxy. But the model reads the attention **output** (after softmax + value mixing). phase11 derives
> the **closed-form second-order (output/Fisher) metric** and uses it to pick the low-rank key basis.
> It **beats KQ-SVD at every budget** (job 19542312). This doc explains first- vs second-order, the
> math, a by-hand example, and the numbers. Code: [phase11_fisher_metric.py](phase11_fisher_metric.py).
> Sits on top of [attn_aware_fix_explained.md](attn_aware_fix_explained.md) (the linear `Σ_q` codec)
> and [kqsvd_relation.md](kqsvd_relation.md) (why phase9 = KQ-SVD).

---

## 1. First-order vs second-order — what "order" means

A compressed key affects the answer through a **chain**:

```
key  →  logit qᵀk  →  softmax  →  attention weights a  →  Σ a·v (×W_O)  →  output  →  next word
```

"Order" = how far down the chain you account for the error of compression.

- **First-order (KQ-SVD):** look only at the *first link* — the logit. Preserve `qᵀk`, weight every key
  equally. Linear/straight-line approximation. Ignores the curved softmax and the value mixing.
- **Second-order (phase11):** follow the error to the **output**, through the *curved* softmax and the
  value sum. Two facts appear that first-order misses:
  1. **how much attention a key actually gets** (`a²`) — an ignored key's error is harmless;
  2. **how distinctive its value is** (`‖W_O(v−v̄)‖²`) — if a key's value equals the local average,
     mis-weighting it barely moves the output.

"Second-order" because the output distortion is, to leading order, a **quadratic** (Hessian/Fisher)
form in the key errors — and that quadratic is the metric below.

---

## 2. The math (closed form)

Compress keys → small error `Δk_n`. Propagate to the output `o_m = W_O Σ_n a_mn v_n` (keys treated
independently — the OBS/GPTQ approximation), with `ō_m = Σ_n a_mn v_n`:

```
Δo_m = Σ_n a_mn (q_mᵀΔk_n / √d) · W_O(v_n − ō_m)
```

Take `E_m‖Δo_m‖²`. Grouping by key, it becomes a **reweighted query second moment** (per head):

```
M̄ = Σ_m c_m q_m q_mᵀ ,   c_m = Σ_n a_mn² · ‖W_O(v_n − ō_m)‖²
```

`c_m` = **output-sensitivity of query m** = (attention concentration `a²`) × (value spread it sees,
through `W_O`). Then the SAME whitened generalized eigenproblem as phase9, with `M̄` for `Σ_q`:

```
W = top-r eigvecs of  M̄^{1/2} C_k M̄^{1/2} ,   encode z = Wᵀ M̄^{1/2} k ,   decode k̂ = M̄^{-1/2} W z
```

**KQ-SVD is the special case `c_m ≡ 1`** (every query equally important). phase11 is its second-order
generalization — same machinery, a smarter weight. Zero inference overhead (folds into matmuls).

---

## 3. By-hand example — why the weight matters

Two query types reading two key directions `x1, x2`; compress each key to **r = 1** (keep one
direction). `W_O = I` for clarity.

```
Query A  reads x1 ;  the values it attends to are DISTINCT  (+1 and −1)
Query B  reads x2 ;  FREQUENT (10×), but the values it reads are IDENTICAL (3 and 3)
Keys vary 4× more in x2 :   C_k = diag(1, 4)
```

**Step 1 — the weights `c_m`** (`c = Σ_n a_n² ‖v_n − ō‖²`, attention `a=[.5,.5]`):

```
c_A = .5²(+1−0)² + .5²(−1−0)² = 0.5      (distinct values → query A's reading matters)
c_B = .5²(3−3)²  + .5²(3−3)²  = 0.0      (identical values → query B can't change the output!)
```

**Step 2 — the two metrics:**

```
Σ_q  (KQ-SVD, frequency-weighted)  = 1·qAqAᵀ + 10·qBqBᵀ = [[1, 0],[0, 10]]   → weights x2 10×
M̄   (fisher, output-weighted)     = 0.5·qAqAᵀ + 0·qBqBᵀ = [[0.5,0],[0, 0]]  → weights x1 only
```

**Step 3 — which direction each method keeps** (top eigvec of `metric^{1/2} C_k metric^{1/2}`):

| method | metric | keeps | why |
|---|---|---|---|
| keyPCA | Euclidean | **x2** | keys vary most in x2 |
| KQ-SVD | `Σ_q` | **x2** | query B reads x2 *a lot* (it's frequent) |
| **fisher** | `M̄` | **x1** | only query A's reading changes the output |

**Step 4 — the output consequence.** Only query A's reading affects the output (B reads identical
values, so its output is invariant to key errors). keyPCA and KQ-SVD spend their one rank on **x2** —
the direction a *frequent but output-irrelevant* query reads — and **destroy query A's logit → broken
output**. fisher spends it on **x1** → query A preserved → **output exact.**

> The lesson: first-order (`Σ_q`) is fooled by a frequent query that *can't actually change the
> output*; second-order (`M̄`) zeroes it (`c_B=0`) and spends the budget where the output truly moves.
> Real K is softer (`c` varies continuously), but the mechanism is exactly this.

---

## 4. Results — Llama-2-7B, WikiText-2 (run 19542312)

K-only, V left full; pre-RoPE `q` proxy with the **real** attention weights; perplexity exact
(real RoPE + softmax). Baseline (full K) = **5.739**. Calibration 40,960 tokens; ranks 8/16/32/64.

```
 rank  cacheX |   keyPCA  attnAware(=KQ-SVD)  fisher(new) | Δ(KQ-SVD − fisher)  verdict
    8   16.0x |   30.067         27.906          26.107   |       +1.799        FISHER WINS
   16    8.0x |   12.637         11.847          11.186   |       +0.661        FISHER WINS
   32    4.0x |    6.766          6.626           6.583   |       +0.043        FISHER WINS
   64    2.0x |    5.951          5.889           5.882   |       +0.007        FISHER WINS
```

**fisher beats KQ-SVD at 4/4 ranks, and the gain widens under compression** (+0.007 @2× → **+1.80
@16×**). Full ladder at 16×: MLA 30.07 → KQ-SVD 27.91 → **fisher 26.11**. Wiring check passed
(fisher@r=128 = 5.739 = baseline).

**Honest caveats.**
- Gain is small at usable budgets (+0.04 @4×); the large gains (8×/16×) are where all methods are
  already degraded (K-only). Same shape as phase9: real, monotone, but the high-compression *operating
  point* needs the V arm + joint K+V to be usable.
- Pre-RoPE `q` proxy; the independent-keys (OBS) approximation in deriving `M̄`.
- The script's "subspace overlap = 0.000" diagnostic is **unreliable** here (it assumes orthonormal
  columns, but the decode bases `Σ_q^{-1/2}W` are oblique) — ignore it; the perplexity is the truth.

---

## 5. Why it's novel (the positioning)

| method | objective | order | solution | vs phase11 |
|---|---|---|---|---|
| KQ-SVD ([2512.05916](https://arxiv.org/abs/2512.05916)) | attention **score** | 1st | closed-form | phase11 = its 2nd-order generalization (`c_m≡1`) |
| StiefAttention ([2601.21686](https://arxiv.org/abs/2601.21686)) | layer **output** | — | **learned MLP** | phase11 is the **closed-form** counterpart |
| Attention Matching ([2602.16284](https://arxiv.org/abs/2602.16284)) | output + mass | 1st | token **subset** + LSQ | different axis (token compaction, not low-rank) |
| Palu / ReCalKV | reconstruction | — | Fisher for **rank allocation** | phase11 uses Fisher for the **basis**, not allocation |

**The gap phase11 fills:** a *closed-form second-order (output/Fisher) metric* for the low-rank **key
basis**. No neighbor does this — KQ-SVD is first-order, StiefAttention is learned, Attention Matching
is token-selection, Palu/ReCalKV use Fisher only to *allocate rank*. It is **incremental** (the
"second-order KQ-SVD") and lives in a fast-moving space, but as of this check it is **not done**.

## Reproduce
`sbatch run_phase11.sh` → `results/phase11_fisher_metric.txt` (job 19542312; 15 min; needs eager
attention for the real weights). Env: `P11_RANKS`, `P11_CALIB_SEQS`, `P11_SEQLEN`, `KV_RCOND`.
