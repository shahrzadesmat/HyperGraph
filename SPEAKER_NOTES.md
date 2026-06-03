# Speaker Notes — Typed Pruning Hypergraph (idea deck)

Companion notes for `hypergraph_slides.pptx`. Idea-only talk (no results yet).
One section per slide — what to say out loud.

---

## Slide 1 — Title / setup

Set expectations up front: today I'm explaining the **idea**, not results — those are still running.

The setup in one breath:
- We take a **pretrained Vision Transformer** (DeiT-Small).
- We want to **cut its compute ~45%** (MACs) with minimal accuracy loss.
- Every pruning method answers two questions: **which** parts to remove, and **how much**.

Our method answers them with a **typed dependency graph** that *generalizes* Isomorphic Pruning (Fang et al., ECCV'24).

---

## Slide 2 — Background: how the field ranks what to prune (recreates VainF Fig. 2)

Before our method, here's how pruning decides what to cut:

- **(a) Network** — just the stacked layers.
- **(b) Local pruning** ranks importance *within each layer separately*. Problem: a globally useless layer still keeps half its channels.
- **(c) Global pruning** ranks *all* parameters together. Problem: it's unfair — attention and MLP weights live on different scales, so comparing them directly biases the result.
- **(d) Isomorphic pruning** is the fix: **group parameters by structural type** (all attention together, all MLP together) and rank *within* each group, where importance scores are actually comparable.

**(d) is the baseline we build on.**

---

## Slide 3 — Isomorphic pruning in one picture

Concretely, isomorphic pruning has **two structure types — attention and MLP — and assigns one ratio to each**.

- Inside attention it shrinks the **head dimension**; inside MLP it shrinks the **hidden dimension**.
- It **never touches the residual stream** (the embedding dim that flows through the whole network) — that's shared across all layers, and cutting it destroys the pretrained features. *(We learned this the hard way — pruning the residual collapsed zero-shot accuracy to random.)*
- **Every one of the 12 blocks gets the same ratio.**

Simple and strong. The weakness: it treats a **critical block identically to a near-redundant one**.

---

## Slide 4 — What uniform pruning misses (the gap)

Uniform treatment misses three things — and these motivate our three parameters:

1. **Not all blocks are needed.** Some barely change the output and could be removed *entirely* (depth), not just shrunk.
2. **Blocks differ in importance.** One ratio for all → over-prunes the important ones and wastes budget on the redundant ones.
3. **Blocks are functionally coupled.** Their importance rises and falls together, so their pruning decisions should influence each other.

Isomorphic pruning treats every block **in isolation** — it can't express any of these.

---

## Slide 5 — THE key slide: VainF's graph vs our graph

Spend time here. Both methods are dependency graphs; the contrast is the whole contribution.

**How to read the graph (define each edge cleanly):**
- **Residual stream** (solid gray line): data flowing from one block to the next. *Both methods.*
- **E_s — Structural edge** (the coupled-squares icon under each node): weights **inside one block** that must be pruned together — e.g. qkv ↔ proj ↔ mlp share a dimension. *Both methods have these (inherited from the dependency graph).*
- **E_f — Functional edge** (dashed orange arc, labeled w_ij): links **two different blocks** whose importance moves together, letting importance flow between them. **New — our addition.**

**LEFT = VainF's graph:** block nodes + structural edges only. No edges between blocks, every block kept, one ratio per type.

**RIGHT = our graph H = (V′, E_s, E_f):** same nodes, same structural edges, plus three additions:
- **S_min** removes a redundant block entirely → node set V′ is smaller (Block 2, the ✕).
- **θ** groups blocks by importance, each group its own ratio → the green vs red coloring.
- **α** adds the **functional edges** (dashed arcs) → importance flows between coupled blocks.

**Takeaway:** we keep every edge VainF has, and add depth removal, importance groups, and inter-block edges.

---

## Slide 6 — The three parameters, defined

- **S_min — depth-pruning threshold.** A threshold on **block sensitivity**: how much the output changes when a block is skipped.
  - `S(i) = mean ‖f(x) − f_bypass(x)‖ / ‖f(x)‖`
  - remove block i if `S(i) < S_min`
  - *Graph effect:* shrinks node set V′ — redundant blocks deleted entirely.

- **θ (theta) — grouping threshold.** Blocks with similar importance merge into one group; each group gets its own ratio.
  - group i, j if `|Î_i − Î_j| < θ`
  - `θ = 1` → one group (uniform, = VainF); `θ = 0` → every block separate
  - *Graph effect:* partitions blocks into importance groups — important groups pruned less.

- **α (alpha) — coupling strength.** Adds functional edges between blocks of similar importance and lets them boost each other.
  - `w_ij = min(I_i, I_j) / max(I_i, I_j)`
  - `I↑(j) = I(j) · (1 + α · Σ w_ij·I(i)/Z)`
  - *Graph effect:* creates E_f — a block's importance rises with its coupled neighbours'.

**Punchline (land this):** set `S_min = 0, θ = 1, α = 0` and our method collapses **exactly** to isomorphic pruning. So VainF is a **special case** of our framework — one corner of the parameter space. Our claim: better points exist, and these three knobs let us find them. Results will tell us by how much.

---

## Slide 7 — Worked example: building the graph one component at a time

Walk down the four rows — each row adds one component to the row above. Note these are **illustrative numbers** to show the mechanism, not measured results.

Setup (5 blocks):
- Sensitivity `S`: B0=0.90, B1=0.50, **B2=0.35**, B3=0.60, B4=0.85
- Importance `I`: B0=0.45, B1=0.22, B3=0.25, B4=0.40

Rows:
1. **Start = VainF.** All 5 blocks kept, one uniform ratio, no edges between blocks (all blue).
2. **+ S_min = 0.40.** B2's sensitivity 0.35 < 0.40 → B2 is removed entirely (✕). The residual chain now skips it. Node set V′ shrank from 5 to 4.
3. **+ θ (grouping).** Group the survivors by importance: {B0, B4} are high (≈0.45, 0.40) → **green**, pruned *less*; {B1, B3} are low (≈0.22, 0.25) → **red**, pruned *more*. Same MAC budget, allocated by importance.
4. **+ α (coupling).** Add functional edges between similar blocks — B0–B4 and B1–B3 (dashed orange). Coupled blocks boost each other's importance, sharpening the group assignment.

**Key line:** the graph is **built up, not rebuilt** — each knob adds structure on top of the previous, and turning all three off returns you to VainF (row 1).
