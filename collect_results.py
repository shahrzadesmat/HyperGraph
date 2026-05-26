"""
Collect results.json files from multiple run directories and print a sorted table.

Usage:
  python collect_results.py results/sweep_smin_*
  python collect_results.py results/sweep_theta_*
  python collect_results.py results/sweep_alpha_*
  python collect_results.py results/*          # all runs
"""

import json
import sys
import glob
import os

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def main():
    patterns = sys.argv[1:] or ["results/*/results.json"]
    paths = []
    for pat in patterns:
        if os.path.isdir(pat):
            paths.append(os.path.join(pat, "results.json"))
        elif pat.endswith(".json"):
            paths.append(pat)
        else:
            paths.extend(glob.glob(os.path.join(pat, "results.json")))

    rows = []
    for p in sorted(paths):
        r = load(p)
        if r is None:
            continue
        rows.append(r)

    if not rows:
        print("No results found.")
        return

    # Determine which columns vary (to show in table)
    params = ["S_min", "theta", "alpha", "edge_threshold"]
    varying = [k for k in params if len({r.get(k) for r in rows}) > 1]
    if not varying:
        varying = params

    # Sort by finetuned_acc desc (fall back to zeroshot_acc)
    rows.sort(key=lambda r: r.get("finetuned_acc") or r.get("zeroshot_acc") or 0, reverse=True)

    # Header
    col_w = 9
    header_params = [f"{k:>{col_w}}" for k in varying]
    header_fixed  = ["mac_reduc", "zs_acc", "ft_acc", "base_acc"]
    print("  ".join(header_params + [f"{h:>9}" for h in header_fixed]))
    print("-" * (col_w * len(varying) + 11 * len(header_fixed) + 20))

    for r in rows:
        param_cols = [f"{r.get(k, '?'):>{col_w}}" for k in varying]
        mac_r  = r.get("mac_reduction", float("nan"))
        zs     = r.get("zeroshot_acc",  float("nan"))
        ft     = r.get("finetuned_acc") or float("nan")
        base   = r.get("baseline_acc",  float("nan"))
        val_cols = [
            f"{mac_r:>9.3f}",
            f"{zs:>9.4f}",
            f"{ft:>9.4f}",
            f"{base:>9.4f}",
        ]
        print("  ".join(param_cols + val_cols))

    print()
    best = rows[0]
    print(f"Best config (by {'finetuned' if best.get('finetuned_acc') else 'zero-shot'} acc):")
    for k in varying:
        print(f"  {k} = {best.get(k)}")
    print(f"  finetuned_acc = {best.get('finetuned_acc')}")
    print(f"  mac_reduction = {best.get('mac_reduction'):.3f}")


if __name__ == "__main__":
    main()
