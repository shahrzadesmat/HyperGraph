"""
Probe: does DeiT-Small have HIGHER-ORDER redundancy that pairwise methods miss?

For each block we look at the activations of its prunable channels and ask two
questions:

  (1) PAIRWISE redundancy — how many channels are near-duplicates of another
      channel? (max |correlation| > 0.9).  This is all a graph method
      (GOHSP / DepGraph) can find.

  (2) TOTAL redundancy — how many dimensions are actually wasted?  We measure
      the *effective rank* of the channel activations (participation ratio of
      the covariance eigenvalues).  C channels but effective rank r << C means
      C - r dimensions are redundant.

The HIGHER-ORDER gap = (total redundant dims) - (pairwise-duplicate channels).
If this gap is large, there is lots of redundancy that ONLY a set-level
(hypergraph) method can exploit — pairwise edges can't see it.

Writes a table to probe_redundancy_result.txt.
"""

import torch, timm, os, json
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

DATA = "/work/hdd/bdjd/imagenet_10pct"
N_IMAGES = 256          # calibration images
MAX_TOKENS = 8000       # subsample this many token-vectors for covariance
CORR_DUP = 0.9          # |corr| above this = pairwise near-duplicate
OUT = "/work/hdd/bdjd/hypergraph_pruning/probe_redundancy_result.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("deit_small_patch16_224", pretrained=True).eval().to(device)

# ---- data ----
tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ds = ImageFolder(os.path.join(DATA, "val"), transform=tf)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:N_IMAGES].tolist()
loader = DataLoader(Subset(ds, idx), batch_size=64, num_workers=4)

# ---- hooks: capture MLP hidden activations (post-GELU) and attention output ----
acts = {}   # name -> list of (tokens, C) tensors
def mk_hook(name):
    def hook(m, i, o):
        acts.setdefault(name, []).append(o.detach().reshape(-1, o.shape[-1]).cpu())
    return hook
def mk_prehook(name):
    def hook(m, i):
        acts.setdefault(name, []).append(i[0].detach().reshape(-1, i[0].shape[-1]).cpu())
    return hook

handles = []
for b, blk in enumerate(model.blocks):
    handles.append(blk.mlp.act.register_forward_hook(mk_hook(f"b{b:02d}_mlp_hidden")))
    handles.append(blk.attn.proj.register_forward_pre_hook(mk_prehook(f"b{b:02d}_attn_out")))

with torch.no_grad():
    for x, _ in loader:
        model(x.to(device))

for h in handles: h.remove()

# ---- analyze each captured tensor ----
def effective_rank(eigs):
    eigs = eigs.clamp(min=0)
    s = eigs.sum()
    if s <= 0: return 0.0
    return float((s*s) / (eigs*eigs).sum())   # participation ratio

def analyze(X):
    # X: (tokens, C)
    if X.shape[0] > MAX_TOKENS:
        sel = torch.randperm(X.shape[0])[:MAX_TOKENS]
        X = X[sel]
    X = X - X.mean(0, keepdim=True)
    C = X.shape[1]
    cov = (X.T @ X) / (X.shape[0]-1)            # C x C
    eigs = torch.linalg.eigvalsh(cov)
    eff = effective_rank(eigs)
    # correlation matrix
    d = torch.sqrt(torch.diag(cov)).clamp(min=1e-8)
    corr = cov / (d[:,None]*d[None,:])
    corr.fill_diagonal_(0)
    max_per_ch = corr.abs().max(dim=1).values
    n_pairwise_dup = int((max_per_ch > CORR_DUP).sum())   # channels with a near-twin
    mean_max_corr = float(max_per_ch.mean())
    total_redundant = C - eff                              # wasted dimensions
    higher_order_gap = total_redundant - n_pairwise_dup
    return dict(C=C, eff_rank=round(eff,1), eff_frac=round(eff/C,3),
                mean_max_corr=round(mean_max_corr,3), n_pairwise_dup=n_pairwise_dup,
                total_redundant=round(total_redundant,1),
                higher_order_gap=round(higher_order_gap,1))

lines = []
def log(s): lines.append(s); print(s)

log(f"{'layer':<18} {'C':>5} {'eff_rank':>9} {'eff/C':>6} {'meanMaxCorr':>12} "
    f"{'pairDup':>8} {'totRedund':>10} {'HO_gap':>8}")
log("-"*82)
for name in sorted(acts.keys()):
    X = torch.cat(acts[name], 0)
    r = analyze(X)
    log(f"{name:<18} {r['C']:>5} {r['eff_rank']:>9} {r['eff_frac']:>6} "
        f"{r['mean_max_corr']:>12} {r['n_pairwise_dup']:>8} {r['total_redundant']:>10} "
        f"{r['higher_order_gap']:>8}")

# summary
mlp = [analyze(torch.cat(acts[n],0)) for n in acts if 'mlp' in n]
attn= [analyze(torch.cat(acts[n],0)) for n in acts if 'attn' in n]
log("")
log("INTERPRETATION:")
log(f"  MLP  hidden: mean eff/C = {sum(r['eff_frac'] for r in mlp)/len(mlp):.3f}  "
    f"(lower = more redundant)")
log(f"  MLP  mean pairwise-duplicate channels = {sum(r['n_pairwise_dup'] for r in mlp)/len(mlp):.0f} / "
    f"{mlp[0]['C']}")
log(f"  MLP  mean HIGHER-ORDER gap = {sum(r['higher_order_gap'] for r in mlp)/len(mlp):.0f} dims "
    f"(redundancy invisible to pairwise methods)")
log(f"  Attn out: mean eff/C = {sum(r['eff_frac'] for r in attn)/len(attn):.3f}")
log(f"  Attn mean HIGHER-ORDER gap = {sum(r['higher_order_gap'] for r in attn)/len(attn):.0f} dims")
log("")
log("If HO_gap >> pairDup, pairwise graph methods (GOHSP/DepGraph) leave most")
log("redundancy on the table -> a set-level (hypergraph) method has real headroom.")

open(OUT, "w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
