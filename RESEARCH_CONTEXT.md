# HyperGraph — Research Context & Handoff

_Session export · 2026-06-14 · target venue: AAAI_

This document captures the full state of the novelty discussion so work can resume
in a fresh session (the cloud container is ephemeral). It is a handoff, not a paper.

---

## 1. What the project is

Structured compression of transformers. Two phases exist in the repo:

- **Phase 1 (ViT):** "Typed Pruning Hypergraph" — generalizes **Isomorphic Pruning**
  (Fang et al., ECCV 2024) with three knobs in `hypergraph.py`:
  - `S_min` — block/depth removal (drop low-sensitivity blocks)
  - `theta` — group blocks by importance, one prune ratio per group
  - `alpha` — "functional coupling" edges that let important neighbours boost a block
  - Setting `S_min=0, theta=1, alpha=0` reproduces Isomorphic Pruning exactly.
  - Pipeline: `hypergraph.py`, `prune_vit.py`, on DeiT-Small / ImageNet.
- **Phase 2 (LLM):** diagnostic probes on Llama-2-7B (hook-based, training-free):
  error propagation, QK/OV circuits, cross-block / cross-type redundancy, allocation
  showdown. These are PROBES, not a full pruning pipeline.

## 2. The novelty problem (why we were stuck)

Every signal invented so far collapses onto a simpler existing baseline:
- `probe_alloc_Llama-2-7b-hf.txt`: amplification-aware **JOINT beats FISHER by only
  +0.48%** — a near-tie with a plain loss-gradient allocator. This is the killer.
- `probe_circuit_alloc`: depth-flip risks "reproducing ASVD" if `depth<flat ≈ 0`.
- α-coupling / amplification ≈ gradient sensitivity → no separable contribution.

Conclusion: neither "global allocation" nor "a better importance signal" can be the
headline — both are taken (ARA, FLAT-LLM, the ICLR'26 global method, Fisher).

## 3. Strategic decisions made this session

1. **Target = LLM** (not ViT). User's call: LLMs are the higher-impact / newer target.
   (Note: training-free SVD-style compression makes this feasible without 7B fine-tuning,
   reusing the hook-based low-rank + WikiText-ppl harness in `probe_circuit_alloc.py`.)
2. **Direction = hypergraph "set-level / higher-order" redundancy**, specifically the
   CROSS-MODULE form (see §5), as the one defensible white space.

## 4. The core idea (plain)

- **Pairwise redundancy** (what graph methods see): two channels are near-duplicates →
  delete one. An edge connects exactly 2 nodes.
- **Set-level / higher-order redundancy:** a SET of channels is jointly redundant even
  though no pair is alike. Example: channels A=[1,0,1,0], B=[0,1,0,1], C=[1,1,1,1]=A+B.
  No pair correlates, yet C is removable. Only visible when you look at the SET.
- A **hyperedge** wraps a set (3+ nodes); weight `w(e) = |e| − effrank(e)` = #removable
  dims. A **hypergraph** is the natural structure; pairwise graphs structurally cannot
  represent it.

## 5. CRITICAL correction (made late in session) — where the real difference is

Within-layer set redundancy (the `HO_gap` on one MLP layer) is **already captured by
SVD methods** (ASVD/SVD-LLM/FLAT-LLM) — intra-matrix low-rank is what SVD does by
definition. So the within-layer story beats *pairwise channel pruning* (GOHSP/DepGraph,
a ViT-era baseline) but is **weak against the LLM SVD baselines.**

**The real, defensible differentiator is CROSS-MODULE redundancy** — hyperedges that
span DIFFERENT matrices / layers / sub-block types, which per-matrix SVD cannot reach
("SVD can only be applied independently to each linear module" — even ARA states this).

Evidence already in the repo for the cross-module form:
- `probe_crosstype_result.txt`: MLP output reconstructible from **attention**
  (held-out err ≈ 0.72, control ≈ 1.0). SVD decomposes MLP & attn separately → blind to it.
- `probe_crossblock_result.txt`: a late block reconstructible from **earlier** blocks'
  neurons (held-out err < 1.0 at small K). Per-matrix SVD can't share rank across blocks.

→ The go/no-go **premise probe for the LLM target is cross-module redundancy on Llama**
  (extend crosstype/crossblock with a train/test split), NOT the within-layer HO_gap.

## 6. Existing evidence (real numbers from probe files)

- `probe_redundancy_result.txt` (DeiT-Small / ViT):
  - MLP hidden: 1536 channels, eff_rank ≈ 150 (eff/C ≈ 0.10), pairwise duplicates ≈ 2,
    **HO_gap ≈ 1380**. Attn out: eff/C ≈ 0.21, HO_gap ≈ 300.
  - NOTE: this is ViT. For the LLM target it must be re-measured on Llama, and as the
    CROSS-MODULE form (§5), not within-layer.
- `probe_alloc_Llama-2-7b-hf.txt` (Llama-2-7B, keep 50%, WikiText ppl, training-free):
  - No-comp 6.87 · uniform 17.39 · local 19.21 · FLAT-LLM 14.56 · fisher 14.02 · joint 13.96
  - JOINT vs FISHER = +0.48% (the tie that motivated the pivot).
- `probe_errorprop_Llama-2-7b-hf.txt`: layer compression error amplifies 0.7×–30× to the
  output, depth-dependent (early layers amplify, late layers damp).

## 7. Differentiation from the baselines (the comparison table)

All three SVD baselines = low-rank factorization, ONE matrix at a time:
- **ASVD** — scale weight columns by activation magnitude, then SVD; sensitivity rank search.
- **SVD-LLM** — "whiten" activations so truncation minimizes OUTPUT loss; per-matrix.
- **FLAT-LLM** — head-wise PCA on attention value/output activations; per-head.
Shared limit: can't cross the nonlinear boundary between modules.

- **LatentLLM** (arXiv 2505.18413, MERL, CVPR-W 2025; AAAI ext. "Activation-Aware Transform
  to Multi-Head Latent Attention"): compresses **pairs JOINTLY** — joint QK (minimize
  attention-map error), joint VO (attention-output), joint up-down (MLP output) — producing
  a Multi-Head Latent Attention (MLA) model. "Joint > independent" because what's used is
  the PRODUCT (W_Q W_Kᵀ), not each matrix. ⚠️ This PARTIALLY OCCUPIES the "joint/cross-module"
  cell AND overlaps the QK/OV circuit angle. BUT its joint is limited to FIXED,
  architecturally-adjacent pairs WITHIN one layer — it does NOT do cross-layer or
  cross-type (MLP↔attn, block→block) redundancy. That gap is our opening.
  (Method reconstructed from abstract + secondary sources; full PDF was not accessible —
  verify exact result tables from the paper, e.g. ~/Downloads/2505.18413v1.pdf.)

Other near-misses to cite & wall off:
- **Gator** (arXiv 2205.15404, 2022): uses the word "hypergraph" for pruning, but for
  STRUCTURAL coupling (must-cut-together by topology) = our E_s, NOT redundancy.
- **O-information / higher-order MI** (arXiv 2211.00416, 2022; 2024 ext.): set-level
  synergy/redundancy among neuron GROUPS, but importance-only, tiny nets, no method,
  not hypergraph-framed, never beats SVD SOTA.
- **GOHSP** (AAAI'23, 2301.05345): graph (pairwise) head ranking — not hypergraph.
- ⚠️ **Caveat to pre-empt:** "Pruning Low-Rank Heads → catastrophic collapse" (2602.02195):
  low internal rank ≠ unimportant. Our signal is RECONSTRUCTIBILITY-FROM-OTHERS, not a
  head being internally low-rank. Keep these distinct.

## 8. The reviewer-proof claim (scoped)

> A hypergraph **redundancy** formulation where hyperedges = jointly rank-deficient channel
> sets that SPAN layers and sub-block types (MLP↔attention, block→block), used as a
> rank-allocation / compression primitive for LLMs. We generalize joint compression beyond
> LatentLLM's fixed within-layer pairs to arbitrary, data-discovered cross-boundary
> hyperedges — redundancy that per-matrix SVD (ASVD/SVD-LLM/FLAT-LLM) and fixed-pair joint
> SVD (LatentLLM) structurally cannot reach — and show it beats them at matched compression.

LatentLLM is actually HELPFUL: it's published proof that "joint > independent" is real and
publishable. We must just frame our hyperedge as **cross-boundary and data-discovered**, or
we look like a re-run of LatentLLM.

## 9. Curated reading list (2024–2026)

Direct competitors (rank allocation): ARA (2510.19389), Global Rank & Sparsity Opt
(2505.03801, ICLR'26), From Local to Global (2510.18030), PGSVD (2510.05544),
FLRC (2510.09332), FLAT-LLM (2505.23966), Týr-the-Pruner (2503.09657),
IO-SVD (2605.15626), LatentLLM (2505.18413).
Layer importance / depth / error-prop: AlphaPruning (2410.10912) [spectral, orthogonal to
Fisher — promising], Rethinking Layer Redundancy (2604.24938), Maximum Redundancy Pruning
(2503.18377), LoRP (2605.27786), Prune&Comp (2507.18212).
Circuit / mechanistic: Capability-Guided Compression (2603.16440), Talking Heads (2406.09519),
Transformer Circuits (transformer-circuits.pub).
Higher-order / hypergraph: O-information (2211.00416), Structural Reducibility of Hypergraphs
(2601.02603), Gator (2205.15404).

## 10. Deliverables produced this session

- `make_redundancy_slides.py` — python-pptx generator (Isomorphic-Pruning visual style).
- `render_redundancy.py` — matplotlib slide→PNG previewer.
- `hypergraph_redundancy.pptx` — 10-slide AAAI pitch deck (topic → background → gap →
  related-work landscape → method → worked example → pipeline → expected results →
  positioning). NOTE: slide 9's ppl bars are TARGETS/placeholders; only the right-hand
  evidence panel (HO_gap etc.) is real. Slide 9 currently uses LLM ppl — consistent with
  the LLM target. The within-layer framing on slide 4 should be upgraded to the
  CROSS-MODULE framing (§5) before submission.

## 11. Next steps / open experiments (all training-free, run on cluster w/ GPU + Llama weights)

1. **Premise gate (do first):** cross-module redundancy probe on Llama-2-7B — extend
   `probe_crosstype.py` / `probe_crossblock.py` with a train/test split; show MLP-from-attn
   and block-from-block reconstructibility is large and survives held-out test. Go/no-go.
2. **Hyperedge finder:** correlation-clustering → per-cluster effective-rank test →
   weighted cross-boundary hyperedges `w(e)=|e|−effrank(e)`.
3. **Allocator + compress:** spend a global rank budget ∝ hyperedge redundancy; project via
   the existing hook machinery (`probe_circuit_alloc.py` style).
4. **Showdown table:** WikiText/C4 ppl + zero-shot, matched compression, vs uniform, ASVD,
   SVD-LLM, FLAT-LLM, ARA, **and LatentLLM**. Models: Llama-2-7B → 13B / Llama-3-8B.

### Environment notes
- This repo's probes reference CUDA + `/work/hdd/bdjd/hypergraph_pruning/` paths → meant to
  run on the user's cluster (the cloud container has no GPU / no gated Llama weights).
- WebFetch is heavily blocked here (arXiv/most hosts 403); WebSearch works. To read a PDF,
  put it inside the container (commit+push, then pull) — the container cannot see the user's
  laptop filesystem.
- Dev branch: `claude/quirky-albattani-ql0cnd`.
