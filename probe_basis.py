"""
Basis-vs-unit probe.  Usage: python probe_basis.py [timm_model]

Tests the root-cause hypothesis: the redundancy in transformer MLP layers is
DENSE (rotated directions), not unit-aligned (channels). If so, BASIS pruning
(low-rank projection) recovers far more redundancy than UNIT pruning (channel
subset) at the same kept-dimension budget -> existing structured pruning leaves
that gap on the table.

For each MLP layer, at kept-dim budgets k, compare held-out reconstruction error
of the post-GELU hidden activations under:
  (a) CHANNEL subset  : keep best k channels (pivoted Cholesky), LS-reconstruct
                        the rest  -> what ThiNet/GOHSP/isomorphic/theta+alpha do
  (b) LOW-RANK         : keep top-k PCA directions, project                -> basis pruning

err_channel >> err_lowrank at the same k  =>  dense redundancy is real and only
reachable by basis pruning.  The gap = headroom every unit-removal method misses.
Train/test split, centered (honest, no overfit).
"""
import os, sys, torch, timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from hyperedge_prune import pivoted_cholesky

MODEL = sys.argv[1] if len(sys.argv) > 1 else "deit_small_patch16_224"
DATA = "/work/hdd/bdjd/imagenet_10pct"
N_IMAGES = 160
MAX_TOKENS = 14000
TRAIN_FRAC = 0.7
RIDGE = 1e-3
FRACS = [0.05, 0.10, 0.20, 0.30, 0.50]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_basis_{MODEL.split('_')[0]}_{MODEL.split('_')[1]}.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(MODEL, pretrained=True).eval().to(device)
N = len(model.blocks)
H = model.blocks[0].mlp.fc1.out_features

tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ds = ImageFolder(os.path.join(DATA, "val"), transform=tf)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:N_IMAGES].tolist()
loader = DataLoader(Subset(ds, idx), batch_size=32, num_workers=4)

hid = {b: [] for b in range(N)}
handles = []
def mk(b):
    def hook(m, i, o):
        hid[b].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
    return hook
for b, blk in enumerate(model.blocks):
    handles.append(blk.mlp.act.register_forward_hook(mk(b)))
with torch.no_grad():
    for x, _ in loader:
        model(x.to(device))
for h in handles: h.remove()

ntok = torch.cat(hid[0],0).shape[0]
sel = torch.randperm(ntok)[:min(MAX_TOKENS, ntok)]
n = sel.shape[0]; ntr = int(TRAIN_FRAC*n)
pm = torch.randperm(n); trI, teI = pm[:ntr], pm[ntr:]

lines=[]
def log(s): lines.append(s); print(s)
log(f"BASIS vs UNIT pruning — held-out reconstruction error of MLP hidden")
log(f"MODEL={MODEL}  blocks={N}  hidden={H}  n_train={ntr} n_test={n-ntr}")
ks = [max(1, int(f*H)) for f in FRACS]
log(f"{'blk':>4} | " + " ".join(f"k={k}".center(15) for k in ks))
log(f"{'':>4} | " + " ".join("chan / lowrank".center(15) for _ in ks))
log("-"*(7+16*len(ks)))

gap_acc = {k:[] for k in ks}
for b in range(N):
    Hall = torch.cat(hid[b],0)[sel].to(device)
    mu = Hall[trI].mean(0, keepdim=True)
    Htr = Hall[trI]-mu; Hte = Hall[teI]-mu
    cov = (Htr.t()@Htr).double()
    perm, _ = pivoted_cholesky(cov, maxk=max(ks))         # channel order (train), capped
    evals, evecs = torch.linalg.eigh(cov)                 # ascending
    nrm = Hte.norm()
    cells=[]
    for k in ks:
        # (a) channel subset: reconstruct all hidden from k kept channels
        K = torch.tensor(perm[:k], device=device)
        Xk = Htr[:, K]
        W = torch.linalg.solve(Xk.t()@Xk + RIDGE*torch.eye(k,device=device)*(Xk.t()@Xk).diagonal().mean(),
                               Xk.t()@Htr)                # (k, H) reconstruction map
        err_ch = float((Hte - Hte[:,K]@W).norm()/nrm)
        # (b) low-rank: project onto top-k PCA directions
        Vk = evecs[:, -k:].float()                        # (H, k) top-k
        err_lr = float((Hte - (Hte@Vk)@Vk.t()).norm()/nrm)
        gap_acc[k].append(err_ch - err_lr)
        cells.append(f"{err_ch:.3f}/{err_lr:.3f}".center(15))
    log(f"{b:>4} | " + " ".join(cells))

log("")
log("mean error gap (channel - lowrank), per k:")
for k in ks:
    g = sum(gap_acc[k])/len(gap_acc[k])
    log(f"  k={k:5d} ({k/H*100:.0f}% of hidden):  gap = {g:+.3f}")
log("")
log("READ: large positive gap = low-rank (basis) reconstructs far better than")
log("channel subset at the same budget -> redundancy is DENSE, unreachable by")
log("any unit-removal method -> basis-coupling pruning has real headroom.")
open(OUT,"w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
