"""
Standalone ablation runner for Typed Pruning Hypergraph on ViT models.

Ablation ladder (add one component at a time):
  Baseline: --S_min 0.0 --theta 1.0 --alpha 0.0   (reproduces isomorphic pruning)
  +S_min:   --S_min 0.15 --theta 1.0 --alpha 0.0
  +theta:   --S_min 0.15 --theta 0.3 --alpha 0.0
  +alpha:   --S_min 0.15 --theta 0.3 --alpha 0.3   (full method)

Usage:
  python run.py --model deit_base_patch16_224 --data_path /path/to/imagenet \
                --target_macs_g 9.0 --S_min 0.0 --theta 1.0 --alpha 0.0 \
                --epochs 30 --output_dir ./results/baseline
"""

import argparse
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

from hypergraph import build_hypergraph
from prune_vit import prune_vit


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument("--model",          default="deit_base_patch16_224")
    p.add_argument("--pretrained",     action="store_true", default=True)
    p.add_argument("--ckpt",           default=None, help="Optional .pth checkpoint")

    # Data
    p.add_argument("--data_path",      required=True)
    p.add_argument("--val_resize",     type=int, default=256)
    p.add_argument("--val_crop",       type=int, default=224)
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--num_workers",    type=int, default=4)

    # Calibration (subset of train used for sensitivity + Taylor)
    p.add_argument("--calib_size",     type=int, default=1024,
                   help="Number of training images for calibration")
    p.add_argument("--calib_batch",    type=int, default=64)

    # Pruning
    p.add_argument("--target_macs_g",  type=float, required=True,
                   help="Target MACs in GigaOps, e.g. 9.0")
    p.add_argument("--S_min",           type=float, default=0.0)
    p.add_argument("--theta",           type=float, default=1.0)
    p.add_argument("--alpha",           type=float, default=0.0)
    p.add_argument("--edge_threshold",  type=float, default=0.3,
                   help="Min importance similarity for a functional edge to form. "
                        "Higher = sparser graph (fewer edges, stronger signal per edge).")

    # Fine-tuning
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.05)
    p.add_argument("--warmup_epochs",  type=int,   default=3)
    p.add_argument("--label_smoothing",type=float, default=0.1)

    # Misc
    p.add_argument("--output_dir",     default="./results/run")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--skip_finetune",  action="store_true",
                   help="Only prune and zero-shot evaluate, skip fine-tuning")

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

    # calibration subset (fixed seed for reproducibility)
    g = torch.Generator()
    g.manual_seed(args.seed)
    calib_indices = torch.randperm(len(train_dst), generator=g)[:args.calib_size].tolist()
    calib_dst = Subset(train_dst, calib_indices)

    calib_loader = DataLoader(calib_dst, batch_size=args.calib_batch,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)
    train_loader  = DataLoader(train_dst, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader    = DataLoader(val_dst, batch_size=args.batch_size,
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
# Fine-tune (simple AdamW + cosine, no distributed/wandb/pbench)
# ---------------------------------------------------------------------------

def finetune(model, train_loader, val_loader, args, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(args.epochs):
        model.train()
        ep_loss = ep_correct = ep_total = 0
        t0 = time.time()

        for images, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                ep_loss    += loss.item() * images.size(0)
                ep_correct += (model(images).argmax(1) == labels).sum().item()
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
# MAC counting helper
# ---------------------------------------------------------------------------

def count_macs(model, device):
    example = torch.randn(1, 3, 224, 224, device=device)
    macs, params = tp.utils.count_ops_and_params(model, example)
    return macs, params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- load model ---
    print(f"\n[Setup] Loading {args.model} (pretrained={args.pretrained})")
    model = timm.create_model(args.model, pretrained=args.pretrained)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        print(f"  Loaded checkpoint from {args.ckpt}")
    model = model.to(device)

    # --- baseline MACs ---
    base_macs, base_params = count_macs(model, device)
    print(f"\n[Baseline] MACs={base_macs/1e9:.3f}G  Params={base_params/1e6:.1f}M")

    # --- data ---
    print("\n[Data] Building loaders...")
    calib_loader, train_loader, val_loader = build_loaders(args)

    # --- baseline accuracy ---
    print("\n[Baseline] Evaluating before pruning...")
    base_acc, base_loss = evaluate(model, val_loader, device)
    print(f"  Accuracy={base_acc:.4f}  Loss={base_loss:.4f}")

    # --- build hypergraph ---
    print(f"\n[Hypergraph] S_min={args.S_min}  theta={args.theta}  "
          f"alpha={args.alpha}  edge_threshold={args.edge_threshold}")
    criterion_for_taylor = nn.CrossEntropyLoss()
    hg = build_hypergraph(
        model, calib_loader, criterion_for_taylor, device,
        baseline_macs_g = base_macs / 1e9,
        target_macs_g   = args.target_macs_g,
        S_min           = args.S_min,
        theta           = args.theta,
        alpha           = args.alpha,
        edge_threshold  = args.edge_threshold,
    )

    # --- prune ---
    print("\n[Prune] Applying pruning...")
    model = prune_vit(model, hg, hg["ratios"], device)

    # --- post-prune MACs ---
    pruned_macs, pruned_params = count_macs(model, device)
    actual_ratio = 1.0 - pruned_macs / base_macs
    print(f"\n[Pruned]  MACs={pruned_macs/1e9:.3f}G  Params={pruned_params/1e6:.1f}M  "
          f"Reduction={actual_ratio*100:.1f}%")

    # --- zero-shot accuracy ---
    print("\n[Zero-shot] Evaluating pruned model (no fine-tuning)...")
    zs_acc, zs_loss = evaluate(model, val_loader, device)
    print(f"  Accuracy={zs_acc:.4f}  Loss={zs_loss:.4f}")

    # --- fine-tune ---
    ft_acc = None
    if not args.skip_finetune:
        print(f"\n[Fine-tune] Training for {args.epochs} epochs "
              f"(lr={args.lr}, wd={args.weight_decay})")
        ft_acc = finetune(model, train_loader, val_loader, args, device)

    # --- results summary ---
    results = {
        "model":          args.model,
        "S_min":           args.S_min,
        "theta":           args.theta,
        "alpha":           args.alpha,
        "edge_threshold":  args.edge_threshold,
        "target_macs_g":  args.target_macs_g,
        "baseline_macs_g":  round(base_macs / 1e9, 4),
        "pruned_macs_g":    round(pruned_macs / 1e9, 4),
        "mac_reduction":    round(actual_ratio, 4),
        "baseline_params_m": round(base_params / 1e6, 2),
        "pruned_params_m":   round(pruned_params / 1e6, 2),
        "baseline_acc":   round(base_acc, 4),
        "zeroshot_acc":   round(zs_acc, 4),
        "finetuned_acc":  round(ft_acc, 4) if ft_acc is not None else None,
        "surviving_blocks": sorted(hg["surviving_blocks"].keys()),
        "removed_blocks":   sorted(hg["removed_blocks"].keys()),
        "num_groups_attn":  len(hg["groups"]["attn_groups"]),
        "num_groups_mlp":   len(hg["groups"]["mlp_groups"]),
    }

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Model          : {args.model}")
    print(f"  S_min / theta / alpha : {args.S_min} / {args.theta} / {args.alpha}")
    print(f"  MACs           : {base_macs/1e9:.3f}G -> {pruned_macs/1e9:.3f}G  ({actual_ratio*100:.1f}% off)")
    print(f"  Params         : {base_params/1e6:.1f}M -> {pruned_params/1e6:.1f}M")
    print(f"  Baseline acc   : {base_acc:.4f}")
    print(f"  Zero-shot acc  : {zs_acc:.4f}  (drop: {(base_acc-zs_acc)*100:.2f}pp)")
    if ft_acc is not None:
        print(f"  Fine-tuned acc : {ft_acc:.4f}  (drop: {(base_acc-ft_acc)*100:.2f}pp)")
    print(f"  Results saved  : {out_json}")
    print("=" * 60)

    # --- save pruned model ---
    pruned_path = os.path.join(args.output_dir, "pruned_model.pth")
    torch.save(model, pruned_path)
    print(f"  Pruned model   : {pruned_path}")


if __name__ == "__main__":
    main()
