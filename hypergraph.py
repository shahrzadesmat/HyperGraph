"""
Hypergraph construction for structured pruning.
Three parameters that modify the dependency graph:
  S_min  -- sensitivity threshold: blocks below this are removed entirely (depth pruning)
  theta  -- merge threshold: controls which blocks share a pruning ratio
  alpha  -- functional coupling: lets important neighbours boost a block's importance

Foundation (S_min=0, theta=1.0, alpha=0.0):
  Reproduces the VainF isomorphic pruning structure — one group for all attn blocks
  at r_attn_base, one group for all MLP blocks at r_mlp_base, where
  r_attn_base = r_mlp_base * head_scale (default 0.2, matching VainF DeiT-Small).
  Our three parameters extend this foundation.
"""

import math

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


def compute_taylor_channels(model: nn.Module,
                            calib_loader,
                            criterion: nn.Module,
                            device: torch.device) -> Dict[int, Dict[str, "torch.Tensor"]]:
    """
    Per-CHANNEL Taylor importance (data-driven), matching VainF's criterion.
    Accumulated as |grad * weight| per batch over the calibration set.

    Returns {block_idx: {"qkv": (3*embed,), "fc1": (hidden,), "fc2": (hidden,)}}
    where each entry is the per-output-channel (or per-input-channel for fc2)
    importance used to decide WHICH channels to keep during pruning.
    """
    model.eval()
    accum = None

    for inputs, labels in calib_loader:
        model.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        loss = criterion(model(inputs), labels)
        loss.backward()

        if accum is None:
            accum = {i: {"qkv": 0.0, "fc1": 0.0, "fc2": 0.0}
                     for i in range(len(model.blocks))}

        for i, block in enumerate(model.blocks):
            qkv = block.attn.qkv
            fc1 = block.mlp.fc1
            fc2 = block.mlp.fc2
            # per qkv output row (3*embed,), per fc1 output row (hidden,),
            # per fc2 input col (hidden,)
            accum[i]["qkv"] = accum[i]["qkv"] + (qkv.weight.grad * qkv.weight).abs().sum(dim=1).detach()
            accum[i]["fc1"] = accum[i]["fc1"] + (fc1.weight.grad * fc1.weight).abs().sum(dim=1).detach()
            accum[i]["fc2"] = accum[i]["fc2"] + (fc2.weight.grad * fc2.weight).abs().sum(dim=0).detach()

    model.zero_grad()
    for i in accum:
        for k in accum[i]:
            accum[i][k] = accum[i][k].cpu()
    return accum


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

    def connected_components(key: str) -> List[List[int]]:
        # Build adjacency: edge between i and j if |norm_i - norm_j| < theta
        # Uses BFS connected components so grouping is transitive and
        # order-independent (unlike the old greedy seed approach).
        adj: Dict[int, List[int]] = {i: [] for i in block_ids}
        for idx_i, i in enumerate(block_ids):
            for j in block_ids[idx_i + 1:]:
                if abs(norm[i][key] - norm[j][key]) < theta:
                    adj[i].append(j)
                    adj[j].append(i)

        groups: List[List[int]] = []
        visited: set = set()
        for start in block_ids:
            if start in visited:
                continue
            component: List[int] = []
            queue = [start]
            visited.add(start)
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbour in adj[node]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
            groups.append(sorted(component))
        return groups

    return {
        "attn_groups": connected_components("attn"),
        "mlp_groups":  connected_components("mlp"),
    }


# ---------------------------------------------------------------------------
# Step 5 — budget allocation
# ---------------------------------------------------------------------------

def allocate_ratios(groups: Dict[str, List[List[int]]],
                     boosted_scores: Dict[int, Dict],
                     sensitivities: Dict[int, float],
                     surviving_blocks: Dict[int, float],
                     baseline_macs_g: float,
                     target_macs_g: float,
                     head_scale: float = 0.2) -> Dict[int, Dict[str, float]]:
    """
    Given theta-groups and boosted importance scores, compute one pruning
    ratio per group and assign it to each block in that group.

    Foundation (matches VainF isomorphic pruning):
        r_mlp_base  — base ratio for all MLP groups
        r_attn_base = r_mlp_base * head_scale  — base ratio for attn groups
                      (head_scale=0.2 matches VainF DeiT-Small: 0.1/0.5)

    Our three parameters modulate around this foundation:
        S_min  → depth_factor changes (fewer blocks → higher r_mlp_base)
        theta  → multiple groups each weighted by norm_imp around their base
        alpha  → boosted_scores change norm_imp, shifting ratios toward
                 important blocks

    Less important groups get higher pruning ratios (r = base × (2 − norm_imp)).
    Budget proof: mean(2 − norm_imp) = 1 → mean r = r_base ✓

    Returns {block_idx: {"attn": ratio, "mlp": ratio}}.
    """
    n_total   = len(sensitivities)
    n_survive = len(surviving_blocks)

    if n_survive == 0:
        return {}

    target_ratio = target_macs_g / baseline_macs_g
    depth_factor = n_survive / n_total

    # MACs scale as (1-r)^2 because pruning embed_dim changes both input and
    # output of every downstream linear simultaneously.
    # Solve: depth_factor × (1 - r_mlp_base)^2 = target_ratio
    r_mlp_base  = max(0.0, 1.0 - math.sqrt(max(0.0, target_ratio / depth_factor)))
    r_mlp_base  = min(r_mlp_base, MAX_PRUNE_RATIO)
    r_attn_base = r_mlp_base * head_scale   # attn pruned proportionally less

    ratios: Dict[int, Dict] = {}

    def _group_ratio(group_blocks: List[int], key: str, base_r: float) -> float:
        live = [i for i in group_blocks if i in surviving_blocks]
        if not live or base_r == 0.0:
            return 0.0
        total_imp = sum(boosted_scores[i][key] for i in surviving_blocks) + 1e-8
        avg_imp   = sum(boosted_scores[i][key] for i in live) / len(live)
        norm_imp  = avg_imp / total_imp * len(surviving_blocks)
        raw = base_r * (2.0 - norm_imp)
        return max(0.0, min(raw, MAX_PRUNE_RATIO))

    for group in groups["attn_groups"]:
        r = _group_ratio(group, "attn", r_attn_base)
        for i in group:
            if i not in ratios:
                ratios[i] = {}
            ratios[i]["attn"] = r

    for group in groups["mlp_groups"]:
        r = _group_ratio(group, "mlp", r_mlp_base)
        for i in group:
            if i not in ratios:
                ratios[i] = {}
            ratios[i]["mlp"] = r

    print(f"\n[Hypergraph] Base ratios: r_mlp_base={r_mlp_base:.3f}  "
          f"r_attn_base={r_attn_base:.3f}  (head_scale={head_scale})")

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
                      alpha: float = 0.0,
                      edge_threshold: float = 0.3,
                      head_scale: float = 0.2) -> Dict:
    """
    Full pipeline.  Returns a dict with everything the pruner needs.

    With S_min=0, theta=1.0, alpha=0.0, head_scale=0.2:
        Reproduces VainF isomorphic pruning — one attn group at r_attn_base,
        one MLP group at r_mlp_base, r_attn_base = r_mlp_base * 0.2.
    Each parameter extends this foundation independently.
    """
    # --- S_min: remove redundant blocks ---
    sensitivities    = compute_block_sensitivities(model, calib_loader, device)
    surviving_blocks = {i: s for i, s in sensitivities.items() if s >= S_min}
    removed_blocks   = {i: s for i, s in sensitivities.items() if s <  S_min}

    print(f"\n[Hypergraph] Block sensitivities:")
    for i, s in sorted(sensitivities.items()):
        tag = " <- REMOVE" if i in removed_blocks else ""
        print(f"  Block {i:2d}: {s:.3f}{tag}")

    # --- Taylor scores on surviving blocks only (scalar, for grouping/ratios) ---
    scores = compute_taylor_scores(model, calib_loader, criterion, device)
    scores = {i: s for i, s in scores.items() if i in surviving_blocks}

    # --- per-channel Taylor (data-driven, for WHICH channels to keep) ---
    taylor_channels = compute_taylor_channels(model, calib_loader, criterion, device)

    # --- alpha: functional edges + importance boost ---
    edges         = compute_functional_edges(scores, threshold=edge_threshold)
    boosted_scores = apply_alpha_boost(scores, edges, alpha)

    print(f"\n[Hypergraph] Functional edges (alpha={alpha}, edge_threshold={edge_threshold}):")
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
        surviving_blocks, baseline_macs_g, target_macs_g,
        head_scale=head_scale,
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
        "taylor_channels":  taylor_channels,
    }
