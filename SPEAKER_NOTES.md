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

## Slide 3 — Our method in the same view (parallels the background slide)

Same 4-column layout as the previous slide, but now each step is **our** contribution — so collaborators see our method in the visual language they just learned.

- **(a) Pretrained** — the stacked layers, untouched.
- **(b) + S_min (depth)** — one redundant block is removed entirely (✕), not just shrunk.
- **(c) + θ (groups)** — survivors are split into importance groups that get *different* ratios: green group pruned less (small slice), red group pruned more (big slice).
- **(d) + α (coupling)** — add functional edges (dashed, w_ij) between same-group blocks so they boost each other.

Key parallel: the field's view ends at "(d) Isomorphic" on the previous slide; **our view starts there and keeps going**. Turn all three off → back to isomorphic.

---

## Slide 4 — Isomorphic pruning in one picture

Concretely, isomorphic pruning has **two structure types — attention and MLP — and assigns one ratio to each**.

- Inside attention it shrinks the **head dimension**; inside MLP it shrinks the **hidden dimension**.
- It **never touches the residual stream** (the embedding dim that flows through the whole network) — that's shared across all layers, and cutting it destroys the pretrained features. *(We learned this the hard way — pruning the residual collapsed zero-shot accuracy to random.)*
- **Every one of the 12 blocks gets the same ratio.**

Simple and strong. The weakness: it treats a **critical block identically to a near-redundant one**.

---

## Slide 5 — Our pruning in one picture (parallels the isomorphic one-picture slide)

Direct contrast to the previous slide. There, all 12 blocks were identical; here they're treated by importance.

- **Greyed rows** (blocks 3, 4, 5) — removed entirely by **S_min** (depth).
- **Green-tagged blocks** (left bar) — important group → **thin** pruned slice (pruned less).
- **Red-tagged blocks** — redundant group → **thick** pruned slice (pruned more). That's **θ** giving each group its own ratio.
- **Dashed orange arc (w_ij)** between two green blocks — a functional edge from **α**.
- Attention stays blue, MLP stays orange — same two structure types, same untouched residual stream — only the *amount* per block changes.

Line to say: "same picture as isomorphic, but the budget is now spent where it costs least."

---

## Slide 6 — What uniform pruning misses (the gap)

One ratio for every block cannot express three concrete differences. Each card maps a gap → the parameter that fixes it (right-hand chip).

1. **Some blocks are nearly dead weight.** Block *sensitivity* (output change when the block is bypassed) varies widely — a few blocks are close to identity, yet uniform pruning still keeps and shrinks them, spending MAC budget on blocks that contribute almost nothing.  *isomorphic: depth ignored* → **fix: S_min** (remove them entirely).
2. **Blocks are not equally important.** *Taylor importance* (|grad × weight|) differs block-to-block. One global ratio over-prunes the critical blocks (accuracy drops) and under-prunes the redundant ones (budget wasted) — the cut lands in the wrong places.  *isomorphic: width is flat* → **fix: θ** (per-group ratios).
3. **Block importances are not independent.** Some blocks' importance scores are correlated (rise/fall together). Pruning each in isolation ignores that a surviving block may depend on a neighbour you just pruned.  *isomorphic: coupling ignored* → **fix: α** (functional edges).

Land it: isomorphic treats every block in isolation; our three parameters each remove one of these blind spots.

---

## Slide 7 — THE key slide: VainF's graph vs our graph

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

## Slide 8 — The three parameters, defined

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

## Slide 9 — Worked example: building the graph one component at a time

Walk down the four rows — each row adds one component to the row above. Note these are **illustrative numbers** to show the mechanism, not measured results.

Setup (5 blocks) — values are shown on the slide, under each node:
- Sensitivity `S`: B0=0.90, B1=0.85, **B2=0.35**, B3=0.55, B4=0.60
- Importance `I`: B0=0.45, B1=0.42, B3=0.24, B4=0.22  (groups come out contiguous: {B0,B1} high, {B3,B4} low)

Rows:
1. **Start = VainF.** All 5 blocks kept, one uniform ratio, no edges (all blue).
2. **+ S_min = 0.40.** B2's sensitivity `S=0.35 < 0.40` → removed entirely (✕); the residual chain skips it. V′ shrinks 5→4. (Point at the S values under each node.)
3. **+ θ (grouping).** Survivors group by importance — the slide draws labeled brackets: **Group A {B0,B1}** (I≈0.45,0.42) → r↓ pruned less (green); **Group B {B3,B4}** (I≈0.24,0.22) → r↑ pruned more (red). Same MAC budget, allocated by importance.
4. **+ α (coupling).** Functional edges couple **within** each group — B0–B1 and B3–B4 (dashed orange w_ij) — so coupled blocks boost each other, sharpening the grouping.

**Key line:** the graph is **built up, not rebuilt** — each knob adds structure on top of the previous, and turning all three off returns you to VainF (row 1).
