"""
Cross-block redundancy probe — TRAIN/TEST split + sparsity curve.

Fixes the overfit-optimism of the previous Part B: we now FIT the reconstruction
on TRAIN tokens and MEASURE on HELD-OUT tokens. Generalization to unseen tokens
is the honest test of real redundancy.

For each LATE block i, reconstruct its residual contribution Δ_i from EARLIER
blocks' hidden NEURONS:
  * err_test_full  : held-out error using ALL earlier neurons
  * err_test_ctrl  : held-out error with TRAIN dictionary row-shuffled
                     (proper control -> should be ~1.0; shuffle can't generalize)
  * sparsity curve : held-out error using only the top-K earlier neurons
                     (selected by covariance with the target) for small K.

If a SMALL K of earlier neurons reconstructs a late block on HELD-OUT tokens with
low error, the cross-block redundancy is real AND cheaply exploitable.
"""
import os, torch, timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

DATA = "/work/hdd/bdjd/imagenet_10pct"
N_IMAGES = 200
MAX_TOKENS = 24000
TRAIN_FRAC = 0.7
RIDGE = 1e-2
LATE = [8, 9, 10, 11]
KS = [64, 128, 256, 512, 1024, 2048]
OUT = "/work/hdd/bdjd/hypergraph_pruning/probe_crossblock_result.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("deit_small_patch16_224", pretrained=True).eval().to(device)

tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ds = ImageFolder(os.path.join(DATA, "val"), transform=tf)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:N_IMAGES].tolist()
loader = DataLoader(Subset(ds, idx), batch_size=32, num_workers=4)

N = len(model.blocks)
delta = {b: [] for b in range(N)}
hidden = {b: [] for b in range(N)}
handles = []
def mk(store, b):
    def hook(m, i, o):
        store[b].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
    return hook
for b, blk in enumerate(model.blocks):
    handles.append(blk.mlp.register_forward_hook(mk(delta, b)))
    handles.append(blk.mlp.act.register_forward_hook(mk(hidden, b)))
with torch.no_grad():
    for x, _ in loader:
        model(x.to(device))
for h in handles: h.remove()

ntok = torch.cat(delta[0], 0).shape[0]
sel = torch.randperm(ntok)[:min(MAX_TOKENS, ntok)]
Draw = {b: torch.cat(delta[b], 0)[sel].to(device) for b in range(N)}
Hraw = {b: torch.cat(hidden[b], 0)[sel].to(device) for b in range(N)}
n = Draw[0].shape[0]
ntr = int(TRAIN_FRAC * n)
perm = torch.randperm(n, device=device)
tr, te = perm[:ntr], perm[ntr:]

def center(X):  # center by TRAIN mean, apply to both splits
    mu = X[tr].mean(0, keepdim=True)
    return X[tr] - mu, X[te] - mu

Dtr, Dte = {}, {}
Htr, Hte = {}, {}
for b in range(N):
    Dtr[b], Dte[b] = center(Draw[b])
    Htr[b], Hte[b] = center(Hraw[b])

lines = []
def log(s): lines.append(s); print(s)

def fit_eval(Ftr, Ytr, Fte, Yte, shuffle=False):
    if shuffle:
        Ftr = Ftr[torch.randperm(Ftr.shape[0], device=Ftr.device)]
    p = Ftr.shape[1]
    G = Ftr.t() @ Ftr
    lam = RIDGE * torch.trace(G) / p
    M = torch.linalg.solve(G + lam*torch.eye(p, device=Ftr.device), Ftr.t() @ Ytr)
    return float((Yte - Fte @ M).norm() / Yte.norm())

log(f"CROSS-BLOCK, train/test split (n_train={ntr}, n_test={n-ntr})")
log(f"late block reconstructed from EARLIER hidden neurons (held-out error)")
log("")
header = f"{'blk':>4} {'#neur':>7} {'full':>6} {'ctrl':>6} | " + " ".join(f"K={k}".rjust(8) for k in KS)
log(header); log("-"*len(header))
for i in LATE:
    Ftr = torch.cat([Htr[j] for j in range(i)], 1)
    Fte = torch.cat([Hte[j] for j in range(i)], 1)
    Ytr, Yte = Dtr[i], Dte[i]
    e_full = fit_eval(Ftr, Ytr, Fte, Yte)
    e_ctrl = fit_eval(Ftr, Ytr, Fte, Yte, shuffle=True)
    # selection scores: covariance of each earlier neuron with the target
    scores = (Ftr.t() @ Ytr).norm(dim=1)            # (#neurons,)
    order = torch.argsort(scores, descending=True)
    row = []
    for k in KS:
        sub = order[:k]
        row.append(fit_eval(Ftr[:, sub], Ytr, Fte[:, sub], Yte))
    cells = " ".join(f"{v:8.3f}" for v in row)
    log(f"{i:>4} {Ftr.shape[1]:>7} {e_full:6.3f} {e_ctrl:6.3f} | {cells}")

log("")
log("READ: 'ctrl' (shuffled, held-out) should be ~1.0 -> any low full/K error is")
log("REAL generalizable cross-block redundancy. The K columns show how FEW earlier")
log("neurons reconstruct a late block on unseen tokens. Low error at small K =")
log("cheap, exploitable cross-block redundancy = the novel result.")

open(OUT, "w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
