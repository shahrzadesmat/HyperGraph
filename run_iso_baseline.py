"""
Isomorphic pruning baseline — faithful replication of arxiv 2407.04616.

Key property: layers of the same structural type (isomorphic structures) get
the same pruning ratio.  Different structural types get different ratios.

  Group 1 — MLP hidden dim : r_mlp  (binary-searched to hit target MACs)
  Group 2 — Attention QKV  : r_attn = r_mlp * HEAD_SCALE  (always smaller)

HEAD_SCALE=0.2 matches the paper's DeiT-Small run (0.1/0.5).

Within each group every block gets exactly r — not a global budget distributed
by importance — so all 12 MLP blocks get the same ratio (truly isomorphic).
global_pruning=False enforces this; global_pruning=True would let importance
redistribute the budget across blocks unevenly.

No S_min, no theta, no alpha.
"""

import argparse
import copy
import json
import math
import os
import time

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


HEAD_SCALE = 0.2   # r_attn = r_mlp * HEAD_SCALE  (paper: 0.1/0.5 for DeiT-Small)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="deit_small_patch16_224")
    p.add_argument("--pretrained",      action="store_true", default=True)
    p.add_argument("--ckpt",            default=None)
    p.add_argument("--data_path",       required=True)
    p.add_argument("--val_resize",      type=int, default=256)
    p.add_argument("--val_crop",        type=int, default=224)
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--calib_size",      type=int, default=1024)
    p.add_argument("--calib_batch",     type=int, default=64)
    p.add_argument("--target_macs_g",   type=float, required=True)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=0.05)
    p.add_argument("--warmup_epochs",   type=int,   default=3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--output_dir",      default="./results/iso_paper_baseline")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--skip_finetune",   action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_loaders(args):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.val_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.val_resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.val_crop),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dst = ImageFolder(os.path.join(args.data_path, "train"), transform=train_tf)
    val_dst   = ImageFolder(os.path.join(args.data_path, "val"),   transform=val_tf)

    g = torch.Generator()
    g.manual_seed(args.seed)
    calib_indices = torch.randperm(len(train_dst), generator=g)[:args.calib_size].tolist()
    calib_dst = Subset(train_dst, calib_indices)

    calib_loader = DataLoader(calib_dst, batch_size=args.calib_batch,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dst, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    return calib_loader, train_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return correct / total, loss_sum / total


# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------

def finetune(model, train_loader, val_loader, args, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc  = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(args.epochs):
        model.train()
        ep_loss = ep_correct = ep_total = 0
        t0 = time.time()

        for images, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss    += loss.item() * images.size(0)
            ep_correct += (logits.detach().argmax(1) == labels).sum().item()
            ep_total   += images.size(0)

        train_acc = ep_correct / ep_total
        val_acc, val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1:3d} | train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} | val_loss={val_loss:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.0f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, best_path)

    print(f"\n[Fine-tune] Best val_acc={best_acc:.4f}  saved to {best_path}")
    return best_acc


# ---------------------------------------------------------------------------
# MAC counter
# ---------------------------------------------------------------------------

def count_macs(model, device, crop_size=224):
    example = torch.randn(1, 3, crop_size, crop_size, device=device)
    macs, params = tp.utils.count_ops_and_params(model, example)
    return macs, params


# ---------------------------------------------------------------------------
# Isomorphic pruning
# ---------------------------------------------------------------------------

def _build_ignored(model):
    ign = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    if hasattr(model, 'patch_embed'):
        ign += list(model.patch_embed.modules())
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        ign.append(model.head)
    return ign


def _get_qkv_modules(model):
    return [block.attn.qkv for block in model.blocks
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv')]


def isomorphic_prune(model, calib_loader, target_macs_g, device, args):
    """
    True isomorphic pruning:
      - MLP hidden dim group  → ratio r_mlp  (same for all 12 blocks)
      - Attention QKV group   → ratio r_attn = r_mlp * HEAD_SCALE (same for all 12)
      - global_pruning=False  → each block gets exactly its ratio (not redistributed)
      - Taylor importance within each group ranks which channels to drop
    Binary search finds r_mlp that hits target MACs.
    """
    target_macs = target_macs_g * 1e9
    criterion   = nn.CrossEntropyLoss()
    example_input = torch.randn(1, 3, args.val_crop, args.val_crop, device=device)

    # ── Binary search using fast MagnitudeImportance (no backward needed) ────
    def try_ratio(r_mlp):
        r_attn = r_mlp * HEAD_SCALE
        m = copy.deepcopy(model).to(device)
        prd = {qkv: r_attn for qkv in _get_qkv_modules(m)}
        pruner = tp.pruner.MetaPruner(
            m,
            torch.randn(1, 3, args.val_crop, args.val_crop, device=device),
            importance=tp.importance.MagnitudeImportance(p=1),
            global_pruning=False,
            pruning_ratio=r_mlp,
            pruning_ratio_dict=prd,
            ignored_layers=_build_ignored(m),
            iterative_steps=1,
        )
        pruner.step()
        macs, _ = count_macs(m, device, args.val_crop)
        del m, pruner
        return macs

    lo, hi = 0.0, 0.85
    best_r_mlp = 0.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if try_ratio(mid) <= target_macs:
            best_r_mlp = mid
            hi = mid
        else:
            lo = mid

    best_r_attn = best_r_mlp * HEAD_SCALE
    print(f"[Iso] Binary search → r_mlp={best_r_mlp:.4f}  r_attn={best_r_attn:.4f}  "
          f"(target={target_macs_g:.2f}G)")

    # ── Accumulate real Taylor gradients for final pruning ───────────────────
    # Keep grads alive — GroupTaylorImportance reads .grad directly
    model.train()
    model.zero_grad()
    for images, labels in tqdm(calib_loader, desc="Taylor grads", leave=False):
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()
    # Do NOT zero_grad here

    # ── Apply final pruning with Taylor importance ────────────────────────────
    prd = {qkv: best_r_attn for qkv in _get_qkv_modules(model)}

    final_pruner = tp.pruner.MetaPruner(
        model,
        example_input,
        importance=tp.importance.GroupTaylorImportance(),
        global_pruning=False,
        pruning_ratio=best_r_mlp,
        pruning_ratio_dict=prd,
        ignored_layers=_build_ignored(model),
        iterative_steps=1,
    )
    final_pruner.step()
    model.zero_grad()

    return model, best_r_mlp, best_r_attn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[Setup] Loading {args.model} (pretrained={args.pretrained})")
    model = timm.create_model(args.model, pretrained=args.pretrained)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
    model = model.to(device)

    base_macs, base_params = count_macs(model, device, args.val_crop)
    print(f"\n[Baseline] MACs={base_macs/1e9:.3f}G  Params={base_params/1e6:.1f}M")

    print("\n[Data] Building loaders...")
    calib_loader, train_loader, val_loader = build_loaders(args)

    print("\n[Baseline] Evaluating before pruning...")
    base_acc, base_loss = evaluate(model, val_loader, device)
    print(f"  Accuracy={base_acc:.4f}  Loss={base_loss:.4f}")

    print(f"\n[Iso Prune] target_macs_g={args.target_macs_g}")
    model, best_r_mlp, best_r_attn = isomorphic_prune(
        model, calib_loader, args.target_macs_g, device, args)

    pruned_macs, pruned_params = count_macs(model, device, args.val_crop)
    actual_reduction = 1.0 - pruned_macs / base_macs
    print(f"\n[Pruned]  MACs={pruned_macs/1e9:.3f}G  Params={pruned_params/1e6:.1f}M  "
          f"Reduction={actual_reduction*100:.1f}%")

    print("\n[Zero-shot] Evaluating pruned model...")
    zs_acc, zs_loss = evaluate(model, val_loader, device)
    print(f"  Accuracy={zs_acc:.4f}  Loss={zs_loss:.4f}")

    ft_acc = None
    if not args.skip_finetune:
        print(f"\n[Fine-tune] Training for {args.epochs} epochs")
        ft_acc = finetune(model, train_loader, val_loader, args, device)

    results = {
        "model":             args.model,
        "method":            "isomorphic",
        "r_mlp":             round(best_r_mlp, 4),
        "r_attn":            round(best_r_attn, 4),
        "head_scale":        HEAD_SCALE,
        "target_macs_g":     args.target_macs_g,
        "baseline_macs_g":   round(base_macs / 1e9, 4),
        "pruned_macs_g":     round(pruned_macs / 1e9, 4),
        "mac_reduction":     round(actual_reduction, 4),
        "baseline_params_m": round(base_params / 1e6, 2),
        "pruned_params_m":   round(pruned_params / 1e6, 2),
        "baseline_acc":      round(base_acc, 4),
        "zeroshot_acc":      round(zs_acc, 4),
        "finetuned_acc":     round(ft_acc, 4) if ft_acc is not None else None,
    }

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Model          : {args.model}")
    print(f"  Method         : isomorphic (MLP={best_r_mlp:.3f}, Attn={best_r_attn:.3f})")
    print(f"  MACs           : {base_macs/1e9:.3f}G -> {pruned_macs/1e9:.3f}G  ({actual_reduction*100:.1f}% off)")
    print(f"  Params         : {base_params/1e6:.1f}M -> {pruned_params/1e6:.1f}M")
    print(f"  Baseline acc   : {base_acc:.4f}")
    print(f"  Zero-shot acc  : {zs_acc:.4f}  (drop: {(base_acc-zs_acc)*100:.2f}pp)")
    if ft_acc is not None:
        print(f"  Fine-tuned acc : {ft_acc:.4f}  (drop: {(base_acc-ft_acc)*100:.2f}pp)")
    print(f"  Results saved  : {out_json}")
    print("=" * 60)

    torch.save(model, os.path.join(args.output_dir, "pruned_model.pth"))


if __name__ == "__main__":
    main()
