"""
Hypergraph construction for structured pruning.
Three parameters that modify the dependency graph:
  S_min  -- sensitivity threshold: blocks below this are removed entirely (depth pruning)
  theta  -- merge threshold: controls which blocks share a pruning ratio
  alpha  -- functional coupling: lets important neighbours boost a block's importance

When S_min=0, theta=1.0, alpha=0.0 the output reproduces standard isomorphic pruning
(one ratio for all attention blocks, one ratio for all MLP blocks).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

MAX_PRUNE_RATIO = 0.85   # hard cap: never prune more than 85% of any layer via width


# ---------------------------------------------------------------------------
# Step 1 — block sensitivity  (S_min parameter)
# ---------------------------------------------------------------------------

def compute_block_sensitivities(model: nn.Module,
                                 calib_loader,
                                 device: torch.device) -> Dict[int, float]:
    """
    For each block i, bypass its learned transform and measure how much the
    model output changes.  S(i) near 1 = critical.  S(i) near 0 = redundant.

    Formula:
        S(i) = mean_over_batch( ||f(x) - f_bypass(x)||_2 / ||f(x)||_2 )
    """
    model.eval()
    inputs = next(iter(calib_loader))[0].to(device)

    with torch.no_grad():
        ref_out = model(inputs)          # (B, num_classes)

    sensitivities = {}
    for i, block in enumerate(model.blocks):
        # hook: return block input unchanged (pure residual bypass)
        handle = block.register_forward_hook(lambda m, inp, out: inp[0])
        with torch.no_grad():
            bypass_out = model(inputs)
        handle.remove()

        diff = (ref_out - bypass_out).norm(dim=-1)   # (B,)
        norm = ref_out.norm(dim=-1)                  # (B,)
        s    = (diff / (norm + 1e-8)).mean().item()
        sensitivities[i] = min(s, 1.0)

    return sensitivities


# ---------------------------------------------------------------------------
# Step 2 — Taylor importance scores per block
# ---------------------------------------------------------------------------

def compute_taylor_scores(model: nn.Module,
                           calib_loader,
                           criterion: nn.Module,
                           device: torch.device) -> Dict[int, Dict[str, float]]:
    """
    One forward+backward pass on a single calibration batch.
    Returns {block_idx: {"attn": score, "mlp": score}}.

    Score = sum of |grad * weight| for all weights in that sub-module.
    This is the standard Taylor importance criterion.
    """
    model.eval()
    model.zero_grad()

    inputs, labels = next(iter(calib_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    loss = criterion(model(inputs), labels)
    loss.backward()

    scores = {}
    for i, block in enumerate(model.blocks):
        attn_score = sum(
            (p.grad * p).abs().sum().item()
            for p in [block.attn.qkv.weight, block.attn.proj.weight]
            if p.grad is not None
        )
        mlp_score = sum(
            (p.grad * p).abs().sum().item()
            for p in [block.mlp.fc1.weight, block.mlp.fc2.weight]
            if p.grad is not None
        )
        scores[i] = {"attn": attn_score, "mlp": mlp_score}

    model.zero_grad()
    return scores


# ---------------------------------------------------------------------------
# Step 3 — functional edges  (alpha parameter)
# ---------------------------------------------------------------------------

def compute_functional_edges(scores: Dict[int, Dict],
                              threshold: float = 0.3) -> Dict[Tuple, float]:
    """
    Directed edge (i -> j) exists when:
      - block i comes before block j
      - their combined importance scores are similar (ratio > threshold)

    Edge weight = min/max ratio of combined scores.
    High weight = the two blocks track each other's importance closely.
    """
    edges = {}
    block_ids = sorted(scores.keys())

    for i in block_ids:
        si = scores[i]["attn"] + scores[i]["mlp"]
        for j in block_ids:
            if i >= j:
                continue
            sj = scores[j]["attn"] + scores[j]["mlp"]
            sim = min(si, sj) / (max(si, sj) + 1e-8)
            if sim > threshold:
                edges[(i, j)] = sim

    return edges


def apply_alpha_boost(scores: Dict[int, Dict],
                       edges: Dict[Tuple, float],
                       alpha: float) -> Dict[int, Dict]:
    """
    Pass 1 of cascaded importance:
        I_up(j) = I(j) * (1 + alpha * sum_{i->j in E_f} w_ij * I(i)/total_I)

    Predecessor contributions are normalized by the total score so that
    factor stays within [1, 1+alpha], preventing score explosion.
    When alpha=0 this returns the original scores unchanged.
    """
    if alpha == 0.0:
        return {k: dict(v) for k, v in scores.items()}   # copy, no change

    total_I = sum(s["attn"] + s["mlp"] for s in scores.values()) + 1e-8

    boosted = {}
    for j, s in scores.items():
        incoming = sum(
            w * (scores[i]["attn"] + scores[i]["mlp"]) / total_I
            for (i, k), w in edges.items() if k == j
        )
        factor = 1.0 + alpha * incoming   # factor in [1, 1+alpha]
        boosted[j] = {
            "attn": s["attn"] * factor,
            "mlp":  s["mlp"]  * factor,
        }
    return boosted


# ---------------------------------------------------------------------------
# Step 4 — theta grouping  (theta parameter)
# ---------------------------------------------------------------------------

def form_groups_by_theta(scores: Dict[int, Dict],
                          theta: float) -> Dict[str, List[List[int]]]:
    """
    Group blocks by importance similarity.
    Two blocks i and j merge into the same group if:
        |norm_score_i - norm_score_j| < theta

    theta=1.0  =>  all blocks in one group  (isomorphic pruning baseline)
    theta=0.0  =>  every block in its own group
    """
    block_ids = sorted(scores.keys())

    # normalise per sub-module type
    max_a = max(s["attn"] for s in scores.values()) + 1e-8
    max_m = max(s["mlp"]  for s in scores.values()) + 1e-8
    norm = {i: {"attn": scores[i]["attn"] / max_a,
                "mlp":  scores[i]["mlp"]  / max_m}
            for i in block_ids}

    def greedy_group(key: str) -> List[List[int]]:
        groups: List[List[int]] = []
        used = set()
        for i in block_ids:
            if i in used:
                continue
            group = [i]
            used.add(i)
            for j in block_ids:
                if j in used:
                    continue
                if abs(norm[i][key] - norm[j][key]) < theta:
                    group.append(j)
                    used.add(j)
            groups.append(sorted(group))
        return groups

    return {
        "attn_groups": greedy_group("attn"),
        "mlp_groups":  greedy_group("mlp"),
    }


# ---------------------------------------------------------------------------
# Step 5 — budget allocation
# ---------------------------------------------------------------------------

def allocate_ratios(groups: Dict[str, List[List[int]]],
                     boosted_scores: Dict[int, Dict],
                     sensitivities: Dict[int, float],
                     surviving_blocks: Dict[int, float],
                     baseline_macs_g: float,
                     target_macs_g: float) -> Dict[int, Dict[str, float]]:
    """
    Given theta-groups and boosted importance scores, compute one pruning
    ratio per group and assign it to each block in that group.

    Less important groups get higher pruning ratios.
    Returns {block_idx: {"attn": ratio, "mlp": ratio}}.

    Budget accounting:
        total_reduction = 1 - target/baseline
        depth savings   = n_removed / n_total  (blocks already deleted)
        width r_base    = (total_reduction × n_total - n_removed) / n_survive
        → only the remaining MAC budget is distributed via width pruning
    """
    total_reduction = max(0.0, 1.0 - (target_macs_g / baseline_macs_g))
    n_total   = len(sensitivities)
    n_survive = len(surviving_blocks)
    n_removed = n_total - n_survive

    if n_survive == 0:
        return {}

    # Width-only reduction needed after depth pruning already saved n_removed/n_total
    r_base = max(0.0, (total_reduction * n_total - n_removed) / n_survive)
    r_base = min(r_base, MAX_PRUNE_RATIO)

    ratios: Dict[int, Dict] = {}

    def _group_ratio(group_blocks: List[int], key: str) -> float:
        live = [i for i in group_blocks if i in surviving_blocks]
        if not live:
            return 0.0
        total_imp = sum(boosted_scores[i][key] for i in surviving_blocks) + 1e-8
        avg_imp   = sum(boosted_scores[i][key] for i in live) / len(live)
        # norm_imp = 1.0 when this group has exactly average importance
        norm_imp  = avg_imp / total_imp * len(surviving_blocks)
        # r = r_base × (2 − norm_imp):
        #   norm_imp = 1.0 → r = r_base  (average group → exactly the width target)
        #   norm_imp → 0   → r = 2×r_base (unimportant group pruned twice as hard)
        #   norm_imp → 2   → r ≈ 0        (very important group barely pruned)
        # Proof budget holds: mean over groups of (2 − norm_imp) = 2 − 1 = 1 → mean r = r_base ✓
        raw = r_base * (2.0 - norm_imp)
        return max(0.05, min(raw, 0.90))

    for group in groups["attn_groups"]:
        r = _group_ratio(group, "attn")
        for i in group:
            if i not in ratios:
                ratios[i] = {}
            ratios[i]["attn"] = r

    for group in groups["mlp_groups"]:
        r = _group_ratio(group, "mlp")
        for i in group:
            if i not in ratios:
                ratios[i] = {}
            ratios[i]["mlp"] = r

    return ratios


# ---------------------------------------------------------------------------
# Main entry: build_hypergraph
# ---------------------------------------------------------------------------

def build_hypergraph(model: nn.Module,
                      calib_loader,
                      criterion: nn.Module,
                      device: torch.device,
                      baseline_macs_g: float,
                      target_macs_g: float,
                      S_min: float = 0.0,
                      theta: float = 1.0,
                      alpha: float = 0.0) -> Dict:
    """
    Full pipeline.  Returns a dict with everything the pruner needs.

    Correctness check:
        S_min=0, theta=1.0, alpha=0.0  =>  one group for all attn blocks,
        one group for all MLP blocks, with ratio = f(target_macs).
        This should reproduce isomorphic pruning.
    """
    # --- S_min: remove redundant blocks ---
    sensitivities    = compute_block_sensitivities(model, calib_loader, device)
    surviving_blocks = {i: s for i, s in sensitivities.items() if s >= S_min}
    removed_blocks   = {i: s for i, s in sensitivities.items() if s <  S_min}

    print(f"\n[Hypergraph] Block sensitivities:")
    for i, s in sorted(sensitivities.items()):
        tag = " <- REMOVE" if i in removed_blocks else ""
        print(f"  Block {i:2d}: {s:.3f}{tag}")

    # --- Taylor scores on surviving blocks only ---
    scores = compute_taylor_scores(model, calib_loader, criterion, device)
    scores = {i: s for i, s in scores.items() if i in surviving_blocks}

    # --- alpha: functional edges + importance boost ---
    edges         = compute_functional_edges(scores, threshold=0.3)
    boosted_scores = apply_alpha_boost(scores, edges, alpha)

    print(f"\n[Hypergraph] Functional edges (alpha={alpha}):")
    if edges:
        for (i, j), w in sorted(edges.items()):
            print(f"  b{i}_attn -> b{j}_attn  weight={w:.3f}")
    else:
        print("  (none above threshold)")

    # --- theta: group surviving blocks ---
    groups = form_groups_by_theta(boosted_scores, theta)

    print(f"\n[Hypergraph] Groups (theta={theta}):")
    print(f"  Attn groups: {groups['attn_groups']}")
    print(f"  MLP  groups: {groups['mlp_groups']}")

    # --- allocate per-group ratios ---
    ratios = allocate_ratios(
        groups, boosted_scores, sensitivities,
        surviving_blocks, baseline_macs_g, target_macs_g
    )

    print(f"\n[Hypergraph] Per-block pruning ratios:")
    for i in sorted(surviving_blocks):
        print(f"  Block {i:2d}: attn={ratios[i]['attn']:.3f}  mlp={ratios[i]['mlp']:.3f}")
    if removed_blocks:
        print(f"  Removed entirely: {sorted(removed_blocks.keys())}")

    return {
        "sensitivities":    sensitivities,
        "surviving_blocks": surviving_blocks,
        "removed_blocks":   removed_blocks,
        "scores":           scores,
        "boosted_scores":   boosted_scores,
        "edges":            edges,
        "groups":           groups,
        "ratios":           ratios,
    }
