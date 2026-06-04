"""
Redundancy vs model scale/depth.  Usage: python probe_scale.py <timm_model>

Tests whether a BIGGER/DEEPER model has more harvestable redundancy than
DeiT-Small (which turned out near-independent).

Part A  WITHIN-LAYER (width): pivoted-Cholesky column subset on each block's MLP
        hidden covariance -> fraction of channels needed for 90% / 99% variance.
        Lower fraction = more within-layer redundancy (helps theta+alpha grouping).

Part B  CROSS-BLOCK (depth): held-out, centered reconstruction of each block's
        residual contribution from its 6 preceding blocks. Low err + control ~1.0
        = real cross-block redundancy (helps the cross-block idea).
"""
import os, sys, torch, timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from hyperedge_prune import pivoted_cholesky

MODEL = sys.argv[1] if len(sys.argv) > 1 else "deit_small_patch16_224"
DATA = "/work/hdd/bdjd/imagenet_10pct"
N_IMAGES = 160
MAX_TOKENS = 16000
TRAIN_FRAC = 0.7
RIDGE = 1e-2
PRECEDING = 6
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_scale_{MODEL.split('_')[0]}_{MODEL.split('_')[1] if len(MODEL.split('_'))>1 else ''}.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(MODEL, pretrained=True).eval().to(device)
N = len(model.blocks)
Hdim = model.blocks[0].mlp.fc1.out_features
Edim = model.blocks[0].mlp.fc2.out_features

tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ds = ImageFolder(os.path.join(DATA, "val"), transform=tf)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:N_IMAGES].tolist()
loader = DataLoader(Subset(ds, idx), batch_size=24, num_workers=4)

# Part A: accumulate hidden covariance incrementally (no token storage).
# Part B: store per-block delta tokens (embed-dim, smaller).
G = {b: torch.zeros(Hdim, Hdim, dtype=torch.float64, device=device) for b in range(N)}
S = {b: torch.zeros(Hdim, dtype=torch.float64, device=device) for b in range(N)}
ncov = [0]
delta = {b: [] for b in range(N)}
handles = []
def mk_cov(b):
    def hook(m, i, o):
        X = o.detach().reshape(-1, o.shape[-1]).double()
        G[b] += X.t() @ X; S[b] += X.sum(0)
        if b == 0: ncov[0] += X.shape[0]
    return hook
def mk_delta(b):
    def hook(m, i, o):
        delta[b].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
    return hook
for b, blk in enumerate(model.blocks):
    handles.append(blk.mlp.act.register_forward_hook(mk_cov(b)))
    handles.append(blk.mlp.register_forward_hook(mk_delta(b)))
with torch.no_grad():
    for x, _ in loader:
        model(x.to(device))
for h in handles: h.remove()

lines = []
def log(s): lines.append(s); print(s)
log(f"MODEL={MODEL}   blocks={N}  embed={Edim}  mlp_hidden={Hdim}")

# ---- Part A: within-layer column-subset ----
log("\nPART A  within-layer: fraction of MLP channels needed (pivoted Cholesky)")
log(f"{'blk':>4} {'k@90%':>8} {'k@99%':>8} {'frac@99':>8}")
fr99 = []
for b in range(N):
    mu = S[b] / ncov[0]
    cov = G[b]/ncov[0] - torch.outer(mu, mu)
    cov = 0.5*(cov+cov.t())
    perm, var = pivoted_cholesky(cov)
    k90 = int(torch.searchsorted(var, torch.tensor(0.90)).item())
    k99 = int(torch.searchsorted(var, torch.tensor(0.99)).item())
    fr99.append(k99/Hdim)
    log(f"{b:>4} {k90:>8} {k99:>8} {k99/Hdim:>8.3f}")
log(f"mean frac@99 = {sum(fr99)/len(fr99):.3f}  (LOWER = more within-layer redundancy)")

# ---- Part B: cross-block held-out ----
sel = torch.randperm(torch.cat(delta[0],0).shape[0])[:MAX_TOKENS]
D = {b: torch.cat(delta[b],0)[sel].to(device) for b in range(N)}
n = D[0].shape[0]; ntr = int(TRAIN_FRAC*n)
pm = torch.randperm(n, device=device); trI, teI = pm[:ntr], pm[ntr:]
def sc(X):
    mu = X[trI].mean(0, keepdim=True); return X[trI]-mu, X[teI]-mu
Dtr, Dte = {}, {}
for b in range(N): Dtr[b], Dte[b] = sc(D[b])
def fe(Ftr, Ytr, Fte, Yte, shuffle=False):
    if shuffle: Ftr = Ftr[torch.randperm(Ftr.shape[0], device=Ftr.device)]
    p = Ftr.shape[1]; Gm = Ftr.t()@Ftr; lam = RIDGE*torch.trace(Gm)/p
    M = torch.linalg.solve(Gm+lam*torch.eye(p, device=Ftr.device), Ftr.t()@Ytr)
    return float((Yte-Fte@M).norm()/Yte.norm())
log(f"\nPART B  cross-block: reconstruct block from {PRECEDING} preceding blocks (held-out)")
log(f"{'blk':>4} {'err':>8} {'control':>8}")
ee = []
for i in range(1, N):
    prev = [j for j in range(max(0,i-PRECEDING), i)]
    Ftr = torch.cat([Dtr[j] for j in prev],1); Fte = torch.cat([Dte[j] for j in prev],1)
    e = fe(Ftr, Dtr[i], Fte, Dte[i]); c = fe(Ftr, Dtr[i], Fte, Dte[i], shuffle=True)
    ee.append(e)
    log(f"{i:>4} {e:>8.3f} {c:>8.3f}")
log(f"mean cross-block err = {sum(ee)/len(ee):.3f}  (LOWER = more cross-block redundancy; control ~1.0)")
log("\nCompare to DeiT-Small: within-layer frac@99~0.82, cross-block err~0.6-0.9 (both weak).")
open(OUT,"w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
