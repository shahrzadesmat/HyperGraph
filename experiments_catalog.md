# KV-Compression Experiment Catalog (phase1–12 + feeder probes)

> One lab-notebook index for the **nonlinear / attention-aware KV-cache compression** thread on
> Llama-2-7B (some 13B / TinyLlama / Qwen). Every experiment → hypothesis → key numbers → verdict.
> Deeper write-ups: [kv_surgery_explanation.md](kv_surgery_explanation.md) (phase1),
> [baseline_comparison.md](baseline_comparison.md) (phase2b + baselines),
> [temporal_stack_explained.md](temporal_stack_explained.md) (phase4/5/gate1/phase8),
> [attention_aware_explained.md](attention_aware_explained.md) (the Σ_q method),
> [attn_aware_fix_explained.md](attn_aware_fix_explained.md) (phase3 bug → phase9 fix).

## The shared harness
Every KV experiment uses the same loop: **capture** K/V (and sometimes Q) activations on WikiText-2
*train* (calibration, frozen model) → **fit** a per-layer codec on calibration only → **surgery**
(forward hooks on `k_proj`/`v_proj` replace the output, pre-RoPE, with the reconstruction) →
**WikiText-2 *test* perplexity**. Budget = `d_c` stored floats/token; `cacheX = 2C/d_c`. Wins are
judged at **matched cache**, never matched params.

---

## Summary table

| # | experiment | one-line hypothesis | verdict |
|---|---|---|---|
| — | `probe_redundancy_llm` | K/V/MLP activations are low effective-rank (redundant) | ✅ redundancy real, **mostly linear** |
| — | `probe_nonlinear3` | curved decoder beats linear at fixed bottleneck, vs Gaussian null | ✅ gap survives null **in recon space** |
| — | `probe_nonlinear_gap_llm` | …and it's exploitable **at matched param budget** | ❌ **0/24** — it's "more params", not curvature |
| 1 | `phase1_kv_surgery` | nonlinear recon translates to ppl (uniform budget) | ✅ near-lossless @2×, ✗ both break @4× |
| 2a | `phase2_kv_curve` | per-layer rank allocation fixes the 4× cliff | ✅ curve, superseded by 2b |
| 2-GQA | `phase2_kv_gqa` | nonlinear redundancy survives on top of GQA (Qwen2.5) | ⏳ run-candidate |
| 2b | `phase2b_mla_joint` | nonlinear-MLA > linear-MLA at matched `d_c` | ⚠️ wins 5/5 but **only big where ppl≫6** |
| — | `baseline_linear_mla` | ours vs svd/asvd/pca/mla (full run) | ⚠️ +0.05 ppl @4×, +10 @16× (unusable) |
| — | `baseline_kvcar` | ours vs KV-CAR (nonlinear-vs-nonlinear) | ✅ 4/4 (KV-CAR calib-only, broken) |
| 4 | `phase4_temporal` | keys are temporally predictable | ✅ +0.42 headroom, **linear** predictor |
| 5 | `phase5_survive_mla` | temporal headroom survives MLA → stacks | ✅ stays ~0.31–0.35 as `d_c`↓ |
| gate1 | `gate1_joint_st` | space-time **coupling** is real & worth a block codec | ⚠️ coupling real, **block loses to per-token** |
| 8 | `phase8_deltakv_stack` | cache the temporal **innovation** of the latent | ❌ open-loop DeltaKV fails at usable bits |
| 7 | `phase7_prefix_kv` | shared prefixes compress far harder (cross-request) | 🔎 256× @ KL≈0.4 — exploratory |
| 3 | `phase3_query_aware` | compress K to preserve the **score**, not the vector | ❌ **bug**: ppl 1631 (used `eigvecs(Σ_q)` alone) |
| 6 | `phase6_output_aware_v` | compress V in **output** space (W_O), not value-var | ❌ ~0 gain (diagonal re-rank, **no rotation**) |
| 9 | `phase9_attn_aware` | phase3 done right: **Σ_q-whitened generalized eig** | ✅ beats MLA 4/4, gain widens ↑compression; **= KQ-SVD** (not novel) |
| 10 | `phase10_nonlinear_attn` | does nonlinearity pay **in the Σ_q metric**? (empty cell) | ❌ negligible & metric-agnostic — **closes the 2×2** |
| 11 | `phase11_fisher_metric` | **2nd-order** output/Fisher metric `M̄` beats KQ-SVD | ✅✨ **the novel positive** — wins 4/4, robust, generalizes 13B |
| 12 | `phase12_joint_kv` | Fisher on the **full K+V** cache, matched total budget | ⚠️ ordering holds; separate K/V lossy → wants **shared latent** |

Legend: ✅ pass · ❌ negative · ⚠️ real-but-not-useful · ✨ novel · 🔎 open/promising · ⏳ not-yet-run.

---

## A. Foundation — is there *nonlinear* structure to exploit?

### `probe_redundancy_llm` — how redundant, and of what kind
**Hypothesis:** activations are low effective-rank.
**Result (TinyLlama):** effective-rank / C ≈ **0.01–0.09** everywhere (e.g. `mlp_hidden` eff/C≈0.02–0.09,
`attn_out` ≈0.01–0.06). Massive redundancy — but dominated by **low-rank linear** structure
(pairwise-dup small, the bulk is the low eff-rank itself).
**Verdict:** ✅ redundancy is real and large, ⇒ low-rank (MLA-style) compression is well-motivated;
says nothing yet about *nonlinear* gains.

### `probe_nonlinear3` — curved vs flat decoder, against a Gaussian null
**Hypothesis:** a nonlinear decoder reconstructs better than the best linear (PCA) one at a fixed
bottleneck `k`, and the gap is **not** overfitting (covariance-matched Gaussian-null control,
frozen PCA encoder, doc-level split, block-bootstrap CI).
**Result:** the curved-vs-flat reconstruction gap **survives the null** on K/V (and Q) in both a ViT
and 7B — i.e. nonlinear geometry exists *in reconstruction-error space*.
**Verdict:** ✅ but it's a **reconstruction** finding — "so what?" deferred to phase1 (downstream) and
the param-matched probe (below).

### `probe_nonlinear_gap_llm` — is the gap exploitable at matched *cost*?
**Hypothesis:** `AE@r` (nonlinear bottleneck `r`) beats `PCA@r_eq` where `r_eq` gives linear the
**same parameter budget**.
**Result:** **0/24** wins, mean gap **−0.281**. At equal params, linear PCA (rank ~2055) crushes the
AE (e.g. `attn_out` L0: AE@32 = 0.572 vs PCA@r_eq = 0.161).
**Verdict:** ❌ the nonlinear "win" at fixed `r` is just **more parameters**, not exploitable curvature
— *for the weight/param metric*. (Caveat for the KV-cache metric: there params are amortized/free and
cache is the cost, so this probe's ledger is the conservative one — phase2b uses the cache ledger.)

---

## B. The method — does nonlinear KV compression translate downstream?

### `phase1_kv_surgery` — THE exploitability gate (uniform budget)
**Result:** @2× linear **9.11 → nonlinear 6.81** (baseline 5.74; nonlinear near-lossless, cuts the
penalty ~68%); @4× **both break** (~1900–2500, uniform allocation too aggressive).
**Verdict:** ✅ at 2×, nonlinear translates downstream — "it's a method." The 4× cliff ⇒ need per-layer
rank allocation (phase2).

### `phase2_kv_curve` (2a) — per-layer rank allocation
**Change:** rank `k_l` per (layer,site) by variance-retention `τ` (sensitive layers auto-get more),
sweep `τ` → (compression, ppl) curve; also charges the nonlinear decoder's decode-FLOPs/token.
**Verdict:** ✅ removes the uniform-4× cliff; **superseded by phase2b** (MLA-faithful joint latent).

### `phase2_kv_gqa` — does it survive GQA?
**Hypothesis:** on Qwen2.5-7B (GQA, KV already small: 512≪3584) there's still nonlinear redundancy to
compress further. **Verdict:** ⏳ run-candidate (modern-LLM relevance check).

### `phase2b_mla_joint` (2b) — nonlinear-MLA vs **linear-MLA** at matched `d_c`  ⭐ the core result
One shared latent `z` (dim `d_c`) reconstructs **both** K and V; `z` folds from the model's own
`k_proj`/`v_proj` (correctness-checked). Linear = `z·Vkᵀ` (= faithful MLA); nonlinear = `+ corr(z)`.
**Result (7B, baseline 5.739):**

| τ | cacheX | ppl linear-MLA | ppl nonlinear | Δ |
|---|---|---|---|---|
| 0.95 | 4.8× | 6.153 | 6.046 | **+0.11** |
| 0.90 | 7.2× | 7.161 | 6.636 | +0.53 |
| 0.85 | 9.9× | 10.857 | 8.055 | +2.80 |
| 0.80 | 13.5× | 30.074 | 11.788 | +18.3 |
| 0.75 | 18.1× | 155.730 | 33.342 | +122 |

(13B mirrors: +0.12 @5.2×, +2.4 @8.9×, +496 @14.8×.)
**Verdict:** ⚠️ nonlinear wins 5/5 — **but the gain is ~0 in the usable regime (ppl≈6) and only
explodes where both are already broken (ppl≫11).** This is the crux: *the nonlinear curvature is too
small where it would matter.* Decode-FLOP overhead ≈ 3–5%.

### `baseline_linear_mla` — ours vs the linear ladder (full run, **done**, job 19510775)
**Result (7B, baseline 5.942):**

| d_c | cacheX | svd | asvd | pca | mla | **ours** |
|---|---|---|---|---|---|---|
| 2048 | 4× | 1528 | 8.81 | 31.7 | 6.18 | **6.13** |
| 1024 | 8× | 5582 | 16.83 | 220 | 7.53 | **6.98** |
| 512 | 16× | 5911 | 63.99 | 1507 | 20.69 | **10.41** |
| 256 | 32× | nan | 251 | 6570 | 2199 | **73.7** |
**Verdict:** ⚠️ ours = best, but **+0.05 @4×, +0.54 @8×** (negligible where usable); the big margins
(16×/32×) are in garbage-ppl territory. Confirms phase2b on real named baselines. ASVD is the only
respectable diagonal-linear; plain SVD/global-PCA are non-starters.

### `baseline_kvcar` — nonlinear vs nonlinear (full run, **done**, job 19510776)
**Result:** ours beats KV-CAR **4/4** (KV-CAR 593–1944 ppl across budgets).
**Verdict:** ✅ but with a fairness asterisk — this KV-CAR is **calibration-only** (frozen model), while
the original co-trains the AE with cross-entropy; its numbers here are essentially untrained. Use as
"per-head AE doesn't survive all-layer calibration-only surgery," not as a knockout.

---

## C. The temporal axis — an orthogonal lever (cache the *change*)

### `phase4_temporal` — are keys predictable from recent keys?
**Result:** FVE_real ≈ **0.35** vs shuffled-null ≈ −0.07 → **headroom +0.42**; nonlinear-predictor gain
**+0.002** (≈0). **Verdict:** ✅ keys are ~35% temporally predictable and a **linear** AR predictor
suffices (cheap). Cache the innovation, not the value.

### `phase5_survive_mla` — does MLA already eat that redundancy?
**Result:** temporal headroom of the **MLA latent** stays **+0.31 → +0.35** as `d_c` shrinks
2048→256 (vs +0.43 raw). **Verdict:** ✅ temporal structure **survives** MLA → the two axes are
orthogonal → they *should* stack. Green-light for phase8.

### `gate1_joint_st` — is space–time **coupling** real, and worth a block codec?
2-D (time×dim) transform; compare per-token MLA (A) vs best rectangle (B) vs separable-surrogate (S)
vs joint-optimal (C). **Result:** pure coupling **CPL = S−C = +1.18 @64×**, shrinking to +0.003 @8×
(grows toward aggressive r) → **GATE PASS**. BUT (A)MLA **19.14** < (C)joint **21.07** @64×.
**Verdict:** ⚠️ coupling is statistically real, **but the block codec loses to plain per-token MLA**
(and adds 16-token latency). Points to *streaming* prediction (DeltaKV) instead of a block transform.

### `phase8_deltakv_stack` — DeltaKV on the nonlinear-MLA latent
Cache `Q(e_t)`, `e_t = z_t − ẑ_t`, linear AR predictor. **Diagnostic PASS:** latent headroom +0.33,
**0.178 bits/dim** free. **Downstream FAIL:** at usable budgets DELTA is **worse** than per-token
(B=4: 140 vs 94; B=6: 88 vs 58); the apparent B=3 "win" (390<398) is **beaten by the shuffled null**
(367) → spurious. **Verdict:** ❌ open-loop DeltaKV doesn't convert (innovation tail-clipping; the real
lever was mean-**centering**, not prediction). Closed-loop ⊂ open-loop upper bound ⇒ not warranted.
**Salvage:** per-dim mean-centering alone = ~17% ppl drop @B=4.

---

## D. The cross-request axis (exploratory)

### `phase7_prefix_kv` — do shared prefixes compress harder?
**Result:** KL(full-prefix ‖ compressed-prefix) at probe positions: **256× → mean KL 0.405**, 128× →
0.31, 64× → 0.67 (single-sequence KV is only ~2–3× near-lossless). Low worst-vs-mean spread (transfers
across diverse probes). **Verdict:** 🔎 a shared **prefix** tolerates *much* more compression than
per-request KV — a real cross-request lever, but KL≈0.4 ≠ lossless; needs a downstream-quality test to
promote.

---

## E. The objective pivot — preserve the *logit*, not the *vector*

### `phase3_query_aware` — the right instinct, the wrong math  ❌
Compress K onto the **query subspace** to keep `q·k`: `k̂ = U_q U_qᵀ k`, `U_q = eigvecs(Σ_q)` **alone**.
**Result:** score-error **worse** than key-PCA at every rank; ppl **1631 / nan**.
**Verdict:** ❌ **bug** — projecting onto the query subspace deletes key energy and ignores `C_k`. Its own
footer states the thesis correctly: *"the lever is the OBJECTIVE, not encoder nonlinearity."* (Fixed in
phase9.)

### `phase6_output_aware_v` — V in output space  ❌ (milder bug)
Keep the **value eigenbasis**, only **re-rank** directions by diagonal output-gain `diag(EᵀGE)`,
`G=W_OᵀW_O`. **Result:** ~**+0.0%** output-error reduction; +0.30 ppl @4× (ppl 25.5, unusable regime).
**Verdict:** ❌ negative — but it never **rotates** the basis (the off-diagonal of `G` is dropped), so the
result is a diagonal-only artifact, not a test of the full output-whitened codec.

### `phase9_attn_aware` — phase3 done right  ✅ (but = KQ-SVD, not novel)
The corrected codec: `W` = top-r eigvecs of `M = Σ_q^{1/2} C_k Σ_q^{1/2}`; encode `z = Wᵀ Σ_q^{1/2} k`,
decode `k̂ = Σ_q^{-1/2} W z` — the **provably optimal** rank-r linear codec for the logit error
`Δkᵀ Σ_q Δk`. **Result (jobs 19528000/19528589, 7B, baseline 5.739):** attnAware beats keyPCA(=MLA)
**4/4**, gain widens under compression — +0.06@2× → **+1.62@16×** (16×: MLA 31.8→attn 30.2); score-error
−57..−90%; queryWrong(phase3)=nan reproduced. **Zero inference overhead** (folds into matmuls).
**Verdict:** ✅ the lever is the objective — BUT this exact method is **KQ-SVD** ([2512.05916](https://arxiv.org/abs/2512.05916),
proof of equivalence in [kqsvd_relation.md](kqsvd_relation.md)). Use as a **validated baseline**, not a
contribution. Explainer: [attn_aware_fix_explained.md](attn_aware_fix_explained.md).

### `phase10_nonlinear_attn` — does nonlinearity pay in the Σ_q metric?  ❌ (closes the 2×2)
Nonlinear residual `corr(z)` on the attnAware base, trained under the whitened (Σ_q) loss vs Euclidean,
Gaussian-null controlled. **Result (job 19540767):** diagnostic nl-gain ≈ **0.05%** (below the null
floor); downstream gain is **metric-AGNOSTIC** (Euclidean-trained `corrE` ≥ Σ_q-trained `corrW`) and
noise-level at usable budgets (+0.07 @4×). **Verdict:** ❌ curvature is a red herring **even in the right
inner product** — it lives in low-query-energy directions Σ_q discards. The 2×2 (linear/nonlinear ×
Euclid/Σ_q) is closed. Explainer: [phase10_explained.md](phase10_explained.md).

### `phase11_fisher_metric` — the 2nd-order output/Fisher metric  ✅✨ THE NOVEL POSITIVE
KQ-SVD preserves the **logit** (1st order). The model reads the **output** (after softmax + value mix).
Closed-form 2nd-order metric `M̄ = Σ_m c_m q qᵀ`, `c_m = Σ_n a_mn²‖W_O(v_n−ō_m)‖²` (attention-mass² ×
value-distinctiveness); same whitened eigenproblem with `M̄` for `Σ_q`. **Results:** beats KQ-SVD **4/4**
on 7B (job 19542312; +0.007@2× → +1.80@16×), **robust at 60k** calib (job 19544961; +2.34@16×, overlap
0.94 = a ~6% basis nudge), and **replicates on Llama-2-13B** (job 19556686; 4/4, +2.64@16×, where KQ-SVD
even falls *below* keyPCA). c_m math validated 4e-14 vs naive. **Cross-checked NOT done** — distinct from
KQ-SVD (1st-order), StiefAttention (learned MLP), Attention-Matching (token compaction), Palu/ReCalKV
(Fisher for *allocation*). **Verdict:** ✅✨ genuine novel positive, but **incremental** ("2nd-order
KQ-SVD", 6% nudge) in a hot space. KQ-SVD = special case `c_m≡1`. Explainer:
[phase11_explained.md](phase11_explained.md).

### `phase12_joint_kv` — Fisher on the full K+V cache  ⚠️ (ordering holds, separate is lossy)
Compress K and V each to rank r (total cacheX=`dh/r`): mla vs kqsvd(Σq-K+G-V) vs fisher(M̄-K+G-V).
**Result (job 19544962):** `fisher > kqsvd > mla` at all **usable** budgets (3/4; loses only at 16× where
all >500 ppl). BUT separate K/V is **lossy** — 2× total = 7.49 (vs phase2b shared MLA 6.15 @4.8×).
**Verdict:** ⚠️ the objective ordering survives on the full cache; absolute compression wants a **shared
[K;V] latent** (the next combination, "phase13"). Explainer: [phase12_explained.md](phase12_explained.md).

---

## Where the thread stands (the storyline)

1. Redundancy is real but **mostly linear** (`redundancy`, `nonlinear3`, `nonlinear_gap`).
2. A nonlinear decoder helps downstream only **near-losslessly at 2×**; at usable budgets the gain → ~0
   (`phase1`, `phase2b`, `baseline_linear_mla`) — and it **stays a red herring even in the Σ_q metric**
   (`phase10`). **Nonlinearity is not the lever.**
3. Temporal & coupling structure is **real but already captured** by per-token MLA (`gate1`, `phase8`).
4. **The lever is the OBJECTIVE** (the model reads `q·k`, not `k`). phase3 tried it (bug), **phase9** fixed
   it — but that's **KQ-SVD** (already published). The novel step is going **2nd-order**: **phase11**'s
   closed-form output/Fisher metric `M̄` beats KQ-SVD, robustly and across model sizes — the **one novel
   positive**. It's incremental (a 6% basis nudge); the deployable absolute number needs the **shared-latent
   Fisher** (phase13). Thesis spine = the unified metric-map + honest negatives (phase10) + phase11.
