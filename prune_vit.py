"""
ViT pruning execution using hypergraph-guided per-block ratios.

IMPORTANT — the residual stream is NEVER pruned.
The embedding dimension (residual stream, 384 for DeiT-Small) is shared by
every block through the residual connections and was pretrained as a single
coherent feature space. Pruning it scrambles the representation across all
layers at once and destroys the model (zero-shot collapses to random).

So we prune ONLY inside each block, leaving embed_dim fixed:
  - Attention: shrink head_dim within each head (keep all heads). qkv output
    rows and proj input columns shrink; qkv input and proj output stay at
    embed_dim. This matches VainF isomorphic head-dim pruning.
  - MLP: shrink the hidden dimension only. fc1 input and fc2 output stay at
    embed_dim.

The novelty (S_min depth pruning, theta grouping, alpha coupling) lives in the
per-block *ratios*, not in touching the residual stream.
"""

import types

import torch
import torch.nn as nn
from typing import Dict


MIN_HEAD_DIM   = 8     # minimum dimension per head after pruning
MIN_MLP_DIM    = 64    # minimum MLP hidden dimension after pruning
MAX_PRUNE      = 0.85  # hard cap on pruning ratio


# ---------------------------------------------------------------------------
# Patched attention forward (timm reshapes to input embed dim, which breaks
# after head_dim pruning; reshape to num_heads*head_dim instead).
# ---------------------------------------------------------------------------

def _patched_attn_forward(self, x):
    B, N, C = x.shape
    inner = self.num_heads * self.head_dim
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if hasattr(self, 'q_norm') and self.q_norm is not None:
        q, k = self.q_norm(q), self.k_norm(k)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, inner)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ---------------------------------------------------------------------------
# Depth pruning: remove entire blocks
# ---------------------------------------------------------------------------

def remove_blocks(model: nn.Module, removed_blocks: Dict[int, float]) -> nn.Module:
    """
    Remove all blocks whose index is in removed_blocks.
    Rebuilds model.blocks as a new Sequential without the removed indices.
    Must be Sequential (not ModuleList) because timm calls self.blocks(x) directly.
    """
    if not removed_blocks:
        return model

    keep_indices = [i for i in range(len(model.blocks))
                    if i not in removed_blocks]
    model.blocks = nn.Sequential(*[model.blocks[i] for i in keep_indices])
    print(f"[Depth prune] Removed blocks {sorted(removed_blocks.keys())}. "
          f"Remaining: {len(model.blocks)}")
    return model


# ---------------------------------------------------------------------------
# Attention pruning — head_dim within each head, embed_dim untouched
# ---------------------------------------------------------------------------

def prune_attention_head_dim(model: nn.Module,
                             hg: Dict,
                             ratios: Dict[int, Dict],
                             device: torch.device,
                             index_map: Dict[int, int]):
    """
    Per-block head-dim pruning. embed_dim (residual stream) stays fixed.

    For each surviving block, keep all num_heads heads but reduce head_dim by
    the block's attention ratio. Within each head we keep the top-k dimensions
    by combined Q+K+V row norm (the same kept indices are applied to Q, K, V so
    dot products stay aligned). proj input columns are sliced to match the new
    per-head V output; proj output stays at embed_dim.
    """
    surviving = hg["surviving_blocks"]

    for i in surviving:
        block = model.blocks[index_map[i]]
        attn  = block.attn
        qkv   = attn.qkv
        proj  = attn.proj

        embed_dim = qkv.in_features                 # stays fixed (e.g. 384)
        num_heads = attn.num_heads
        head_dim  = qkv.out_features // (3 * num_heads)

        r = min(ratios[i]["attn"], MAX_PRUNE)
        new_head_dim = max(MIN_HEAD_DIM, int(round(head_dim * (1.0 - r))))
        if new_head_dim >= head_dim or r <= 0.0:
            # still patch forward so reshape uses num_heads*head_dim uniformly
            attn.forward = types.MethodType(_patched_attn_forward, attn)
            continue

        w = qkv.weight.data                         # (3*embed_dim, embed_dim)

        keep_rows      = []                         # qkv output rows to keep
        proj_keep_cols = []                         # proj input cols to keep
        for h in range(num_heads):
            # importance of each of this head's head_dim channels = Q+K+V norms
            imp = torch.zeros(head_dim, device=w.device)
            for part in range(3):                   # 0=Q, 1=K, 2=V
                base = part * embed_dim + h * head_dim
                imp += w[base:base + head_dim, :].norm(dim=1)
            keep = imp.topk(new_head_dim).indices.sort().values

            for part in range(3):
                base = part * embed_dim + h * head_dim
                keep_rows.append(base + keep)
            # proj input column block for head h (concatenated V outputs)
            proj_keep_cols.append(h * head_dim + keep)

        keep_rows      = torch.cat(keep_rows)
        proj_keep_cols = torch.cat(proj_keep_cols)

        # qkv: prune output rows, keep input cols (= embed_dim)
        qkv.weight = nn.Parameter(w[keep_rows, :])
        if qkv.bias is not None:
            qkv.bias = nn.Parameter(qkv.bias.data[keep_rows])
        qkv.out_features = len(keep_rows)

        # proj: prune input cols, keep output rows (= embed_dim)
        proj.weight = nn.Parameter(proj.weight.data[:, proj_keep_cols])
        proj.in_features = len(proj_keep_cols)

        # update head structure + reshape patch
        attn.head_dim = new_head_dim
        attn.scale    = new_head_dim ** -0.5
        attn.forward  = types.MethodType(_patched_attn_forward, attn)

    print(f"[Attn prune] Per-block head_dim pruning applied to {len(surviving)} "
          f"blocks (embed_dim unchanged)")


# ---------------------------------------------------------------------------
# MLP pruning — per-block hidden dim, embed_dim untouched
# ---------------------------------------------------------------------------

def prune_mlp_per_block(model: nn.Module,
                        hg: Dict,
                        ratios: Dict[int, Dict],
                        device: torch.device,
                        index_map: Dict[int, int]):
    """
    Prune MLP hidden dimension independently per surviving block.
    fc1 input (embed_dim) and fc2 output (embed_dim) stay fixed — only the
    hidden dimension shrinks, so the residual stream is untouched.
    """
    surviving = hg["surviving_blocks"]

    for i in surviving:
        block = model.blocks[index_map[i]]
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
# Main entry: prune_vit
# ---------------------------------------------------------------------------

def prune_vit(model: nn.Module,
              hg: Dict,
              ratios: Dict[int, Dict],
              device: torch.device) -> nn.Module:
    """
    Full pruning pipeline (residual stream never touched):
      1. Remove depth-pruned blocks (S_min)
      2. Prune attention head_dim per block (embed_dim fixed)
      3. Prune MLP hidden dim per block (embed_dim fixed)

    After this, validate with a forward pass.
    """
    model = model.to(device)

    # Step 1 — depth pruning
    model = remove_blocks(model, hg["removed_blocks"])

    # Map original block index → new position after compaction
    original_surviving = sorted(hg["surviving_blocks"].keys())
    index_map: Dict[int, int] = {orig: new for new, orig in enumerate(original_surviving)}

    # Step 2 — attention head_dim pruning (embed_dim unchanged)
    prune_attention_head_dim(model, hg, ratios, device, index_map)

    # Step 3 — MLP hidden pruning (embed_dim unchanged)
    prune_mlp_per_block(model, hg, ratios, device, index_map)

    # Validate
    try:
        test = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            model(test)
        print("[Validation] Forward pass OK")
    except Exception as e:
        raise RuntimeError(f"Model broken after pruning: {e}")

    return model
