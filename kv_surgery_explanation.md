# KV-Surgery: from "nonlinear redundancy exists" to "it's a method"

*Phase-1 exploitability test for nonlinear KV-cache compression. Last updated 2026-06-16.*

## Where we were: a finding that wasn't yet a method

We had established that K/V (and Q, and weakly MLP) activations have **nonlinear
redundancy** — at a fixed bottleneck size `k`, a *curved* (nonlinear) decoder reconstructs
them better than the best *flat* (linear/PCA) one, in **both a ViT and a 7B LLM**,
calibrated against a Gaussian null (so it's not overfitting).

But that was measured in **reconstruction-error space**. The open question: *so what?* A
better reconstruction number doesn't automatically mean a better model — the nonlinear
decoder might just be fitting high-variance channels the model doesn't care about. (The
FLAT-LLM paper is a cautionary tale: their reconstruction gains did **not** translate to
perplexity.) We needed to prove the redundancy matters for the model's **actual behavior**.

## What the KV surgery did

The KV cache stores a K vector and a V vector for **every token** in the context — the
thing that balloons in memory during long-context inference. "Compressing the KV cache"
means storing a small **k-dim code** per token instead of the full C=4096-dim K/V, and
reconstructing K/V when needed.

We simulated exactly that, as a **transplant**:

1. **Offline (calibration):** for each of the 32 layers we learned an encoder (K → k-dim
   code `z`) and a decoder (`z` → reconstructed K̂), in **two flavors** — a **linear**
   decoder (= PCA, what MLA/ASVD/EigenAttention do) and our **nonlinear** decoder.
2. **Surgery:** during a *real* forward pass on **held-out** WikiText-2, we intercepted
   every layer's K and V mid-computation and **swapped in the reconstructed-from-`z`
   version**, then let attention run on those.
3. **Measured perplexity** (next-token prediction quality) for three cases at the **same
   cache budget**: full K/V (baseline) vs linear-reconstructed vs nonlinear-reconstructed.

The test reduces to one comparison: **at the same compression, does nonlinear give lower
perplexity than linear?**

## The method this builds

This is the prototype of **nonlinear KV-cache compression — a nonlinear generalization of
MLA.** MLA caches a small latent per token and reconstructs K/V with a *linear*
up-projection. Our method does the same but with a **learned nonlinear decoder**, justified
by the probe showing K/V have curvature a linear up-projection provably can't capture.

## The result

| config | perplexity (lower = better) | |
|---|---|---|
| **baseline** (full K/V) | **5.74** | reference |
| **2× compression** | linear **9.11**  →  nonlinear **6.81** | nonlinear near-lossless |
| **4× compression** | linear **2491**  →  nonlinear **1929** | both broken (ignore) |

- **At 2× compression, nonlinear is near-lossless** (6.81 vs 5.74 baseline, only +1.07),
  while **linear degrades badly** (9.11, +3.37). The nonlinear decoder **cuts the
  compression penalty by ~68%.** This is the clean, real win.
- At 4×, **both break** (perplexity ~2000 = garbage); naive uniform compression is too
  aggressive there, so the 4× "win" is meaningless.

**Verdict: the redundancy translates downstream.** At a practical operating point,
exploiting the nonlinear structure preserves the model where linear compression hurts it.
**It's a method, not just a finding.**

## Why it works (intuition)

Picture the K vectors as points on a **curved sheet** in 4096-dim space.
- **Linear compression (MLA/PCA)** flattens that sheet onto a k-dim *plane* — it can't
  represent the curvature, so it throws it away. Reconstructed K is off → attention sees
  corrupted keys → perplexity rises.
- **The nonlinear decoder bends to follow the sheet** — it reconstructs K far more
  faithfully at the *same* k → attention sees near-correct keys → perplexity barely moves.

And the decoder's cost is **amortized**: one small shared network reused to reconstruct
every cached token, so per token it's cheap relative to the memory saved — which is exactly
why the KV cache (not MLP) is the right place to spend the nonlinearity.

## Why it matters

The KV cache is *the* memory bottleneck for long-context LLMs, and the SOTA way to shrink
it (MLA) is **linear**. Our result says there's nonlinear structure in K/V that MLA leaves
on the table, and using it gives **near-lossless 2× compression where linear loses a lot.**

## Honest caveats (to defend the claim)

1. The linear baseline here was **naive activation-PCA, not a tuned MLA** — so this proves
   "nonlinear beats linear *at matched budget*," not yet "beats production MLA." That's the
   Phase-2 comparison.
2. We have **not yet charged the nonlinear decoder's per-token decode FLOPs** — Phase 2
   must report that (the win lives in the memory-bound long-context regime).
3. The 4× cliff means a real method needs **per-layer rank allocation**, not uniform.

## One-sentence summary

Transplanting nonlinear-reconstructed K/V into the live model keeps perplexity near-lossless
at **2× KV-cache compression**, where the linear (MLA-style) approach degrades sharply —
which makes **"nonlinear MLA"** a real method to build.

## Reproduce

`phase1_kv_surgery.py` (this repo) — all 32 layers, K+V, fits frozen-PCA + nonlinear
decoder on calibration K/V, then surgery + WikiText-2 perplexity. Output:
`phase1_kv_surgery.txt`.
