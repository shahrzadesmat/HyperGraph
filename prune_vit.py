"""
ViT pruning execution using hypergraph-guided per-block ratios.
Reuses the coupling logic from the existing PruningAgent but applies
different ratios per block (MLP) and sensitivity-weighted selection (attention).

Attention constraint: all blocks share the same embedding dimension because
of residual connections, so attention pruning is a single global decision.
The novel part is that the WHICH channels to keep is weighted by block
sensitivity -- important blocks have more influence on the global selection.

MLP is truly per-block (local hidden dimension), so each theta-group gets
its own ratio independently.
"""

import torch
import torch.nn as nn
from typing import Dict


MIN_HEAD_DIM   = 8     # minimum dimension per head after pruning
MIN_MLP_DIM    = 64    # minimum MLP hidden dimension after pruning
MAX_PRUNE      = 0.85  # hard cap on pruning ratio


# ---------------------------------------------------------------------------
# Depth pruning: remove entire blocks
# ---------------------------------------------------------------------------

def remove_blocks(model: nn.Module, removed_blocks: Dict[int, float]) -> nn.Module:
    """
    Remove all blocks whose index is in removed_blocks.
    Rebuilds model.blocks as a new ModuleList without the removed indices.
    """
    if not removed_blocks:
        return model

    keep_indices = [i for i in range(len(model.blocks))
                    if i not in removed_blocks]
    model.blocks = nn.ModuleList([model.blocks[i] for i in keep_indices])
    print(f"[Depth prune] Removed blocks {sorted(removed_blocks.keys())}. "
          f"Remaining: {len(model.blocks)}")
    return model


# ---------------------------------------------------------------------------
# Attention pruning — one global decision, sensitivity-weighted
# ---------------------------------------------------------------------------

def prune_attention_global(model: nn.Module,
                            hg: Dict,
                            ratios: Dict[int, Dict],
                            device: torch.device,
                            index_map: Dict[int, int]) -> torch.Tensor:
    """
    Prune the shared attention embedding dimension across all surviving blocks.

    Selection is sensitivity-weighted: channels that are important in
    high-sensitivity blocks are preferred in the global keep set.

    Returns keep_indices for downstream embedding update.
    """
    surviving = hg["surviving_blocks"]
    sensitivities = hg["sensitivities"]

    if not surviving:
        return None

    # Use first surviving block to get dimensions (use remapped index)
    first_orig  = sorted(surviving.keys())[0]
    first_block = model.blocks[index_map[first_orig]]
    embed_dim   = first_block.attn.proj.out_features
    num_heads   = first_block.attn.num_heads
    head_dim    = embed_dim // num_heads

    # Target number of heads: sensitivity-weighted average ratio
    avg_ratio = sum(ratios[i]["attn"] for i in surviving) / len(surviving)
    avg_ratio = min(avg_ratio, MAX_PRUNE)
    heads_to_keep = max(1, int(num_heads * (1.0 - avg_ratio)))
    # ensure head_dim stays valid
    while heads_to_keep * head_dim < MIN_HEAD_DIM and heads_to_keep < num_heads:
        heads_to_keep += 1

    print(f"[Attn prune] {num_heads} heads -> {heads_to_keep} heads "
          f"(avg ratio={avg_ratio:.3f})")

    # Sensitivity-weighted per-channel importance across all surviving blocks
    # Important blocks contribute more to the global channel selection.
    channel_importance = torch.zeros(embed_dim, device=device)

    for i in surviving:
        block  = model.blocks[index_map[i]]   # remap original → new index
        weight = sensitivities[i]             # high sensitivity = high influence
        w_qkv  = block.attn.qkv.weight.data  # (3*embed_dim, embed_dim)

        # importance per embedding dimension = Q+K+V row norms
        for h in range(num_heads):
            q_norm = w_qkv[h * head_dim                  : (h+1) * head_dim,                  :].norm(dim=1).mean()
            k_norm = w_qkv[h * head_dim + embed_dim      : (h+1) * head_dim + embed_dim,      :].norm(dim=1).mean()
            v_norm = w_qkv[h * head_dim + 2 * embed_dim  : (h+1) * head_dim + 2 * embed_dim,  :].norm(dim=1).mean()
            channel_importance[h * head_dim : (h+1) * head_dim] += (q_norm + k_norm + v_norm) * weight

    # Select top heads based on aggregated importance
    head_importance = torch.stack([
        channel_importance[h * head_dim : (h+1) * head_dim].sum()
        for h in range(num_heads)
    ])
    keep_heads = head_importance.topk(heads_to_keep).indices.sort().values

    keep_idx = torch.cat([
        torch.arange(h * head_dim, (h+1) * head_dim, device=device)
        for h in keep_heads
    ])
    qkv_keep = torch.cat([keep_idx,
                           keep_idx + embed_dim,
                           keep_idx + 2 * embed_dim])

    new_embed_dim = len(keep_idx)

    # Apply to ALL surviving blocks uniformly (residual stream constraint)
    for i in surviving:
        block = model.blocks[index_map[i]]   # remap original → new index
        qkv   = block.attn.qkv
        proj  = block.attn.proj

        # QKV: prune output rows (kept head channels) AND input cols (new embed_dim)
        qkv.weight = nn.Parameter(qkv.weight.data[qkv_keep, :][:, keep_idx])
        if qkv.bias is not None:
            qkv.bias = nn.Parameter(qkv.bias.data[qkv_keep])
        qkv.out_features = len(qkv_keep)
        qkv.in_features  = new_embed_dim

        # proj: prune input cols (kept attention output) AND output rows (new embed_dim)
        proj.weight = nn.Parameter(proj.weight.data[keep_idx, :][:, keep_idx])
        if proj.bias is not None:
            proj.bias = nn.Parameter(proj.bias.data[keep_idx])
        proj.in_features  = new_embed_dim
        proj.out_features = new_embed_dim

        # update head count and scale
        block.attn.num_heads = heads_to_keep
        block.attn.head_dim  = head_dim
        block.attn.scale     = head_dim ** -0.5

    return keep_idx


# ---------------------------------------------------------------------------
# MLP pruning — per-block ratios
# ---------------------------------------------------------------------------

def prune_mlp_per_block(model: nn.Module,
                         hg: Dict,
                         ratios: Dict[int, Dict],
                         device: torch.device,
                         index_map: Dict[int, int]):
    """
    Prune MLP hidden dimension independently per surviving block.
    Each block gets the ratio assigned to its theta-group.
    fc1.out_features == fc2.in_features constraint maintained.
    """
    surviving = hg["surviving_blocks"]

    for i in surviving:
        block = model.blocks[index_map[i]]   # remap original → new index
        ratio = min(ratios[i]["mlp"], MAX_PRUNE)
        if ratio <= 0.0:
            continue

        fc1 = block.mlp.fc1
        fc2 = block.mlp.fc2
        hidden_dim  = fc1.out_features
        keep_hidden = max(MIN_MLP_DIM, int(hidden_dim * (1.0 - ratio)))

        # joint importance: fc1 output norm * fc2 input norm
        fc1_imp = fc1.weight.data.norm(dim=1)           # (hidden_dim,)
        fc2_imp = fc2.weight.data.norm(dim=0)           # (hidden_dim,)
        joint   = fc1_imp * fc2_imp

        keep_idx = joint.topk(keep_hidden).indices.sort().values

        # prune fc1 rows (output channels)
        fc1.weight = nn.Parameter(fc1.weight.data[keep_idx, :])
        if fc1.bias is not None:
            fc1.bias = nn.Parameter(fc1.bias.data[keep_idx])
        fc1.out_features = keep_hidden

        # prune fc2 columns (input channels)
        fc2.weight = nn.Parameter(fc2.weight.data[:, keep_idx])
        fc2.in_features = keep_hidden

    print(f"[MLP prune]  Per-block MLP ratios applied to {len(surviving)} blocks")


# ---------------------------------------------------------------------------
# Embedding update after attention pruning
# ---------------------------------------------------------------------------

def update_global_embeddings(model: nn.Module,
                               keep_idx: torch.Tensor,
                               device: torch.device):
    """
    After attention pruning changes the embedding dimension, update:
      - patch_embed projection
      - positional embedding
      - cls_token
      - final LayerNorm
    """
    if keep_idx is None:
        return

    new_dim = len(keep_idx)

    # patch embedding projection
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
        proj = model.patch_embed.proj
        proj.weight = nn.Parameter(proj.weight.data[keep_idx, :, :, :])
        if proj.bias is not None:
            proj.bias = nn.Parameter(proj.bias.data[keep_idx])
        proj.out_channels = new_dim

    # positional embedding  (B, num_patches+1, embed_dim)
    if hasattr(model, 'pos_embed'):
        model.pos_embed = nn.Parameter(model.pos_embed.data[:, :, keep_idx])

    # cls token  (1, 1, embed_dim)
    if hasattr(model, 'cls_token'):
        model.cls_token = nn.Parameter(model.cls_token.data[:, :, keep_idx])

    # final norm
    if hasattr(model, 'norm'):
        _prune_layernorm(model.norm, keep_idx)

    # block-level LayerNorms inside surviving blocks
    for block in model.blocks:
        for attr in ['norm1', 'norm2']:
            if hasattr(block, attr):
                _prune_layernorm(getattr(block, attr), keep_idx)

    # classifier head: input features reduced to new_dim
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head.weight = nn.Parameter(model.head.weight.data[:, keep_idx])
        model.head.in_features = new_dim

    # MLP fc1 input and fc2 output also connect to the residual stream,
    # so they must be sliced to new_dim after embed_dim changes.
    for block in model.blocks:
        if hasattr(block, 'mlp'):
            fc1 = block.mlp.fc1
            fc2 = block.mlp.fc2
            # fc1: weight (hidden, embed_dim) → (hidden, new_dim)
            fc1.weight = nn.Parameter(fc1.weight.data[:, keep_idx])
            fc1.in_features = new_dim
            # fc2: weight (embed_dim, hidden) → (new_dim, hidden)
            fc2.weight = nn.Parameter(fc2.weight.data[keep_idx, :])
            if fc2.bias is not None:
                fc2.bias = nn.Parameter(fc2.bias.data[keep_idx])
            fc2.out_features = new_dim

    print(f"[Embeddings] Updated to new embed_dim={new_dim}")


def _prune_layernorm(ln: nn.LayerNorm, keep_idx: torch.Tensor):
    if hasattr(ln, 'weight') and ln.weight is not None:
        ln.weight = nn.Parameter(ln.weight.data[keep_idx])
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias = nn.Parameter(ln.bias.data[keep_idx])
    ln.normalized_shape = (len(keep_idx),)


# ---------------------------------------------------------------------------
# Main entry: prune_vit
# ---------------------------------------------------------------------------

def prune_vit(model: nn.Module,
               hg: Dict,
               ratios: Dict[int, Dict],
               device: torch.device) -> nn.Module:
    """
    Full pruning pipeline:
      1. Remove depth-pruned blocks (S_min)
      2. Prune attention globally (sensitivity-weighted)
      3. Prune MLP per block (theta-grouped ratios)
      4. Update global embeddings

    After this, validate with a forward pass.
    """
    model = model.to(device)

    # Step 1 — depth pruning
    model = remove_blocks(model, hg["removed_blocks"])

    # Build mapping: original block index → new position in re-indexed model.blocks
    # (remove_blocks compacts the list; subsequent steps must use new positions)
    original_surviving = sorted(hg["surviving_blocks"].keys())
    index_map: Dict[int, int] = {orig: new for new, orig in enumerate(original_surviving)}

    # Step 2 — attention width pruning (global, sensitivity-weighted)
    keep_idx = prune_attention_global(model, hg, ratios, device, index_map)

    # Step 3 — MLP width pruning (per-block)
    prune_mlp_per_block(model, hg, ratios, device, index_map)

    # Step 4 — fix global embeddings
    update_global_embeddings(model, keep_idx, device)

    # Validate
    try:
        test = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            model(test)
        print("[Validation] Forward pass OK")
    except Exception as e:
        raise RuntimeError(f"Model broken after pruning: {e}")

    return model
