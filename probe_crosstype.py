"""
Cross-TYPE redundancy probe (attention <-> MLP), honest held-out.

Isomorphic pruning ASSUMES attention and MLP are different structure types with
incomparable importance, so it never compares across them. This probe tests that
assumption directly: can a block's MLP contribution to the residual be
reconstructed from the ATTENTION contribution (same / neighbouring blocks)?

If yes (low held-out error, control ~1.0), attn and MLP do OVERLAPPING work ->
cross-type redundancy that isomorphic pruning is structurally blind to.

Both contributions live in the same 384-d residual space, so they are directly
comparable. Train/test split + centering from the start (no overfit optimism).
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
OUT = "/work/hdd/bdjd/hypergraph_pruning/probe_crosstype_result.txt"

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
attn = {b: [] for b in range(N)}   # attention residual contribution (384)
mlp  = {b: [] for b in range(N)}   # mlp residual contribution (384)
handles = []
def mk(store, b):
    def hook(m, i, o):
        store[b].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
    return hook
for b, blk in enumerate(model.blocks):
    handles.append(blk.attn.register_forward_hook(mk(attn, b)))
    handles.append(blk.mlp.register_forward_hook(mk(mlp, b)))
with torch.no_grad():
    for x, _ in loader:
        model(x.to(device))
for h in handles: h.remove()

ntok = torch.cat(attn[0], 0).shape[0]
sel = torch.randperm(ntok)[:min(MAX_TOKENS, ntok)]
A = {b: torch.cat(attn[b], 0)[sel].to(device) for b in range(N)}
Mp = {b: torch.cat(mlp[b], 0)[sel].to(device) for b in range(N)}
n = A[0].shape[0]; ntr = int(TRAIN_FRAC*n)
perm = torch.randperm(n, device=device); trI, teI = perm[:ntr], perm[ntr:]

def split_center(X):
    mu = X[trI].mean(0, keepdim=True)
    return X[trI]-mu, X[teI]-mu
Atr, Ate, Mtr, Mte = {}, {}, {}, {}
for b in range(N):
    Atr[b], Ate[b] = split_center(A[b])
    Mtr[b], Mte[b] = split_center(Mp[b])

def fit_eval(Ftr, Ytr, Fte, Yte, shuffle=False):
    if shuffle: Ftr = Ftr[torch.randperm(Ftr.shape[0], device=Ftr.device)]
    p = Ftr.shape[1]; G = Ftr.t()@Ftr
    lam = RIDGE*torch.trace(G)/p
    M = torch.linalg.solve(G+lam*torch.eye(p, device=Ftr.device), Ftr.t()@Ytr)
    return float((Yte-Fte@M).norm()/Yte.norm())

lines=[];
def log(s): lines.append(s); print(s)
log(f"CROSS-TYPE redundancy: reconstruct MLP contribution from ATTENTION (held-out)")
log(f"n_train={ntr} n_test={n-ntr}")
log(f"{'blk':>4} {'mlp<-attn_self':>15} {'mlp<-attn_local':>16} {'control':>9}")
log("-"*48)
es, el = [], []
for i in range(N):
    e_self = fit_eval(Atr[i], Mtr[i], Ate[i], Mte[i])
    loc = [j for j in (i-1, i, i+1) if 0 <= j < N]
    Ftr = torch.cat([Atr[j] for j in loc], 1); Fte = torch.cat([Ate[j] for j in loc], 1)
    e_loc = fit_eval(Ftr, Mtr[i], Fte, Mte[i])
    e_ctrl = fit_eval(Atr[i], Mtr[i], Ate[i], Mte[i], shuffle=True)
    es.append(e_self); el.append(e_loc)
    log(f"{i:>4} {e_self:>15.3f} {e_loc:>16.3f} {e_ctrl:>9.3f}")
log(f"mean: self={sum(es)/len(es):.3f}  local={sum(el)/len(el):.3f}  (control ~1.0)")
log("")
log("READ: low held-out err + control ~1.0 -> MLP work is partly reproducible from")
log("ATTENTION -> cross-type redundancy exists -> isomorphic's within-type")
log("assumption leaves redundancy unexploited. If err ~ 1.0, attn & MLP are")
log("independent and isomorphic's assumption holds (idea dead).")
open(OUT,"w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
