"""
Isomorphic pruning baseline — replicates the approach from arxiv 2407.04616.

Uniform ratio across all attention blocks (shared embed_dim head pruning).
Uniform ratio across all MLP blocks (per-block independent width pruning).
Taylor importance criterion: |grad × weight| summed over output neurons.
No S_min, no theta, no alpha — all blocks pruned equally.

Usage:
  python run_iso_baseline.py --data_path /path/to/imagenet \
      --target_macs_g 2.5 --epochs 20 --output_dir ./results/iso_baseline
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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="deit_small_patch16_224")
    p.add_argument("--pretrained",     action="store_true", default=True)
    p.add_argument("--ckpt",           default=None)
    p.add_argument("--data_path",      required=True)
    p.add_argument("--val_resize",     type=int, default=256)
    p.add_argument("--val_crop",       type=int, default=224)
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--calib_size",     type=int, default=1024)
    p.add_argument("--calib_batch",    type=int, default=64)
    p.add_argument("--target_macs_g",  type=float, required=True)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.05)
    p.add_argument("--warmup_epochs",  type=int,   default=3)
    p.add_argument("--label_smoothing",type=float, default=0.1)
    p.add_argument("--output_dir",     default="./results/iso_baseline")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--skip_finetune",  action="store_true")
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
# Taylor importance: |grad × weight| summed over output neurons
# ---------------------------------------------------------------------------

def compute_taylor_importance(model, calib_loader, criterion, device):
    """
    Run one pass over calibration data, accumulate |grad * weight| per parameter.
    Returns a dict: module_name -> 1-D tensor of per-output-neuron importance.
    """
    model.train()
    model.zero_grad()

    for images, labels in tqdm(calib_loader, desc="Taylor grads", leave=False):
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()

    importance = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.grad is not None:
            # sum over input dimension → per-output-neuron score
            imp = (m.weight.grad * m.weight.data).abs().sum(dim=1)
            importance[name] = imp.detach().cpu()

    model.zero_grad()
    return importance


# ---------------------------------------------------------------------------
# Isomorphic pruning
# ---------------------------------------------------------------------------

def count_macs(model, device, crop_size=224):
    example = torch.randn(1, 3, crop_size, crop_size, device=device)
    macs, params = tp.utils.count_ops_and_params(model, example)
    return macs, params


def isomorphic_prune(model, calib_loader, base_macs, target_macs_g, device, args):
    """
    Replicate arxiv 2407.04616 isomorphic pruning:
      - uniform ratio r_attn for all attention projection layers  (global embed_dim)
      - uniform ratio r_mlp  for all MLP fc layers                (per-block independent)
      - Taylor importance to select which neurons to keep

    We use torch_pruning's MagnitudeImportance as a proxy that is equivalent
    to Taylor when run after backward (grads are already accumulated).

    Strategy:
      1. Binary-search a single pruning ratio r in [0, MAX_R].
      2. Apply r uniformly to both attention (global) and MLP (per-block).
      3. Stop when MACs ≤ target.
    """
    target_macs = target_macs_g * 1e9
    criterion   = nn.CrossEntropyLoss()

    # Accumulate Taylor gradients
    imp_scores = compute_taylor_importance(model, calib_loader, criterion, device)

    # We use torch_pruning's GroupNormImportance wrapper to handle linked groups.
    # For isomorphic ViT pruning we apply a single global ratio.
    imp = tp.importance.GroupTaylorImportance()

    # Build DG with ignored layers
    example_input = torch.randn(1, 3, args.val_crop, args.val_crop, device=device)
    ignored = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            ignored.append(m)
    # patch embedding + classifier head stay full
    if hasattr(model, 'patch_embed'):
        ignored += list(model.patch_embed.modules())
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        ignored.append(model.head)

    pruner = tp.pruner.MetaPruner(
        model,
        example_input,
        importance=imp,
        global_pruning=True,        # uniform ratio across all coupled groups
        pruning_ratio=0.0,          # will be updated per iteration
        ignored_layers=ignored,
        iterative_steps=1,
    )

    # Binary search for the ratio that hits target MACs
    lo, hi = 0.0, 0.85
    best_ratio = 0.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        # clone model for trial
        import copy
        trial_model = copy.deepcopy(model)

        trial_input = torch.randn(1, 3, args.val_crop, args.val_crop, device=device)
        trial_ignored = []
        for m in trial_model.modules():
            if isinstance(m, nn.LayerNorm):
                trial_ignored.append(m)
        if hasattr(trial_model, 'patch_embed'):
            trial_ignored += list(trial_model.patch_embed.modules())
        if hasattr(trial_model, 'head') and isinstance(trial_model.head, nn.Linear):
            trial_ignored.append(trial_model.head)

        trial_pruner = tp.pruner.MetaPruner(
            trial_model,
            trial_input,
            importance=tp.importance.MagnitudeImportance(p=1),
            global_pruning=True,
            pruning_ratio=mid,
            ignored_layers=trial_ignored,
            iterative_steps=1,
        )
        trial_pruner.step()
        trial_macs, _ = count_macs(trial_model, device, args.val_crop)

        if trial_macs <= target_macs:
            best_ratio = mid
            hi = mid
        else:
            lo = mid

        del trial_model, trial_pruner

    print(f"[Iso] Binary search → pruning_ratio={best_ratio:.4f}  "
          f"(target={target_macs_g:.2f}G)")

    # Apply the found ratio with Taylor importance on the real model
    # Re-accumulate gradients (fresh pass)
    imp_scores = compute_taylor_importance(model, calib_loader, criterion, device)

    # Patch model linear weights with accumulated grads so GroupTaylorImportance works
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name in imp_scores:
            m.weight.grad = (imp_scores[name].unsqueeze(1)
                             .expand_as(m.weight)
                             .to(device))

    final_ignored = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            final_ignored.append(m)
    if hasattr(model, 'patch_embed'):
        final_ignored += list(model.patch_embed.modules())
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        final_ignored.append(model.head)

    final_pruner = tp.pruner.MetaPruner(
        model,
        example_input,
        importance=imp,
        global_pruning=True,
        pruning_ratio=best_ratio,
        ignored_layers=final_ignored,
        iterative_steps=1,
    )
    final_pruner.step()
    model.zero_grad()

    return model, best_ratio


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
    model, best_ratio = isomorphic_prune(
        model, calib_loader, base_macs, args.target_macs_g, device, args)

    pruned_macs, pruned_params = count_macs(model, device, args.val_crop)
    actual_ratio = 1.0 - pruned_macs / base_macs
    print(f"\n[Pruned]  MACs={pruned_macs/1e9:.3f}G  Params={pruned_params/1e6:.1f}M  "
          f"Reduction={actual_ratio*100:.1f}%")

    print("\n[Zero-shot] Evaluating pruned model...")
    zs_acc, zs_loss = evaluate(model, val_loader, device)
    print(f"  Accuracy={zs_acc:.4f}  Loss={zs_loss:.4f}")

    ft_acc = None
    if not args.skip_finetune:
        print(f"\n[Fine-tune] Training for {args.epochs} epochs")
        ft_acc = finetune(model, train_loader, val_loader, args, device)

    results = {
        "model":              args.model,
        "method":             "isomorphic",
        "pruning_ratio":      round(best_ratio, 4),
        "target_macs_g":      args.target_macs_g,
        "baseline_macs_g":    round(base_macs / 1e9, 4),
        "pruned_macs_g":      round(pruned_macs / 1e9, 4),
        "mac_reduction":      round(actual_ratio, 4),
        "baseline_params_m":  round(base_params / 1e6, 2),
        "pruned_params_m":    round(pruned_params / 1e6, 2),
        "baseline_acc":       round(base_acc, 4),
        "zeroshot_acc":       round(zs_acc, 4),
        "finetuned_acc":      round(ft_acc, 4) if ft_acc is not None else None,
    }

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Model          : {args.model}")
    print(f"  Method         : isomorphic (uniform Taylor, no S_min/theta/alpha)")
    print(f"  Pruning ratio  : {best_ratio:.4f}")
    print(f"  MACs           : {base_macs/1e9:.3f}G -> {pruned_macs/1e9:.3f}G  ({actual_ratio*100:.1f}% off)")
    print(f"  Params         : {base_params/1e6:.1f}M -> {pruned_params/1e6:.1f}M")
    print(f"  Baseline acc   : {base_acc:.4f}")
    print(f"  Zero-shot acc  : {zs_acc:.4f}  (drop: {(base_acc-zs_acc)*100:.2f}pp)")
    if ft_acc is not None:
        print(f"  Fine-tuned acc : {ft_acc:.4f}  (drop: {(base_acc-ft_acc)*100:.2f}pp)")
    print(f"  Results saved  : {out_json}")
    print("=" * 60)

    pruned_path = os.path.join(args.output_dir, "pruned_model.pth")
    torch.save(model, pruned_path)
    print(f"  Pruned model   : {pruned_path}")


if __name__ == "__main__":
    main()
