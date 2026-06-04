"""
Run hyperedge-based MLP pruning (v1) and compare against VainF / theta+alpha.

Reuses data loading, fine-tuning and evaluation from run.py.
Prunes only MLP hidden via higher-order reconstruction hyperedges, with a single
variance-tolerance tau binary-searched to hit the MAC budget.
"""
import argparse, json, os, time
import timm, torch
import torch.nn as nn
import torch_pruning as tp

from run import build_loaders, evaluate, finetune, count_macs
from hyperedge_prune import (collect_mlp_stats, pivoted_cholesky,
                             prune_mlp_hyperedge, calibrate_tau)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deit_small_patch16_224")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--val_resize", type=int, default=256)
    p.add_argument("--val_crop", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--calib_size", type=int, default=1024)
    p.add_argument("--calib_batch", type=int, default=64)
    p.add_argument("--target_macs_g", type=float, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--output_dir", default="./results/hyperedge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_finetune", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[Setup] Loading {args.model}")
    model = timm.create_model(args.model, pretrained=args.pretrained).to(device)
    base_macs, base_params = count_macs(model, device, args.val_crop)
    print(f"[Baseline] MACs={base_macs/1e9:.3f}G  Params={base_params/1e6:.1f}M")

    print("\n[Data] Building loaders...")
    calib_loader, train_loader, val_loader = build_loaders(args)

    print("\n[Baseline] Evaluating...")
    base_acc, _ = evaluate(model, val_loader, device)
    print(f"  Accuracy={base_acc:.4f}")

    print("\n[Hyperedge] Collecting MLP activation statistics...")
    stats = collect_mlp_stats(model, calib_loader, device)

    print("[Hyperedge] Pivoted-Cholesky column selection per block...")
    perms, var_curves = {}, {}
    for b in stats:
        cov, _ = stats[b]
        perm, var = pivoted_cholesky(cov)
        perms[b], var_curves[b] = perm, var
        # effective rank ~ #pivots to reach 99% variance
        k99 = int(torch.searchsorted(var, torch.tensor(0.99)).item())
        print(f"  Block {b:2d}: {len(perm):4d} pivots (rank), k@99%var={k99}")

    print(f"\n[Hyperedge] Calibrating tau to {args.target_macs_g}G...")
    tau = calibrate_tau(model, stats, perms, var_curves, args.target_macs_g,
                        device, count_macs, crop=args.val_crop)
    print(f"  tau (variance retained) = {tau:.4f}")

    print("\n[Hyperedge] Applying fold+prune...")
    kmap = prune_mlp_hyperedge(model, stats, perms, var_curves, tau)
    print(f"  kept hidden per block: {[kmap[b] for b in sorted(kmap)]}")

    pruned_macs, pruned_params = count_macs(model, device, args.val_crop)
    red = 1.0 - pruned_macs / base_macs
    print(f"\n[Pruned] MACs={pruned_macs/1e9:.3f}G  Params={pruned_params/1e6:.1f}M  "
          f"Reduction={red*100:.1f}%")

    # validate forward
    with torch.no_grad():
        model(torch.randn(1, 3, args.val_crop, args.val_crop, device=device))
    print("[Validation] Forward OK")

    print("\n[Zero-shot] Evaluating pruned model...")
    zs_acc, _ = evaluate(model, val_loader, device)
    print(f"  Zero-shot acc={zs_acc:.4f}  (drop {100*(base_acc-zs_acc):.2f}pp)")

    ft_acc = None
    if not args.skip_finetune:
        print(f"\n[Fine-tune] {args.epochs} epochs")
        ft_acc = finetune(model, train_loader, val_loader, args, device)

    results = {
        "model": args.model, "method": "hyperedge_mlp_v1",
        "tau": round(tau, 4),
        "target_macs_g": args.target_macs_g,
        "baseline_macs_g": round(base_macs/1e9, 4),
        "pruned_macs_g": round(pruned_macs/1e9, 4),
        "mac_reduction": round(red, 4),
        "baseline_params_m": round(base_params/1e6, 2),
        "pruned_params_m": round(pruned_params/1e6, 2),
        "baseline_acc": round(base_acc, 4),
        "zeroshot_acc": round(zs_acc, 4),
        "finetuned_acc": round(ft_acc, 4) if ft_acc is not None else None,
        "kept_hidden": [kmap[b] for b in sorted(kmap)],
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"  Method        : hyperedge_mlp_v1")
    print(f"  MACs          : {base_macs/1e9:.3f}G -> {pruned_macs/1e9:.3f}G ({red*100:.1f}% off)")
    print(f"  Zero-shot acc : {zs_acc:.4f}")
    if ft_acc is not None:
        print(f"  Fine-tuned    : {ft_acc:.4f}")
    print(f"  vs VainF=0.6880  iso_baseline=0.6914  theta+alpha=0.6997")
    print("="*60)
    torch.save(model, os.path.join(args.output_dir, "pruned_model.pth"))


if __name__ == "__main__":
    main()
