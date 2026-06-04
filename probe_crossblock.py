"""
Cross-block redundancy probe (tightened).

Improvements over v1:
  * CENTER every signal (remove per-channel mean) so the intercept can't trivially
    inflate reconstruction -> the shuffled control now lands near 1.0 and the real
    cross-block signal is clean.
  * Add NEURON-granularity test: reconstruct a LATE block's residual contribution
    from EARLIER blocks' actual hidden NEURONS (the units we'd fold), not just from
    their 384-d block outputs.

Part A (block-level): can Δ_i be reconstructed from other blocks' Δ_j ?
Part B (neuron-level): can a late block's Δ_i be reconstructed from earlier blocks'
                       post-GELU hidden neurons ?  (fine-grained, causal)

err near 0 AND control near 1.0  ->  real, non-trivial cross-block redundancy.
"""
import os, torch, timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

DATA = "/work/hdd/bdjd/imagenet_10pct"
N_IMAGES = 160
MAX_TOKENS = 20000
RIDGE = 1e-3
LATE = [8, 9, 10, 11]
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
delta = {b: [] for b in range(N)}   # MLP residual contribution (384)
hidden = {b: [] for b in range(N)}  # post-GELU hidden neurons (1536)
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
for h in handles:
    h.remove()

# stack + subsample + CENTER, move to GPU
def finalize(store, sel):
    out = {}
    for b in range(N):
        X = torch.cat(store[b], 0)[sel].to(device)
        out[b] = X - X.mean(0, keepdim=True)        # center
    return out

ntok = torch.cat(delta[0], 0).shape[0]
sel = torch.randperm(ntok)[:min(MAX_TOKENS, ntok)]
D = finalize(delta, sel)
H = finalize(hidden, sel)
n = D[0].shape[0]

lines = []
def log(s): lines.append(s); print(s)


def ls_err(F, Y, shuffle=False):
    """Relative error reconstructing centered Y (n,d) from centered features F (n,p)."""
    if shuffle:
        F = F[torch.randperm(F.shape[0], device=F.device)]
    p = F.shape[1]
    G = F.t() @ F
    lam = RIDGE * torch.trace(G) / p
    M = torch.linalg.solve(G + lam*torch.eye(p, device=F.device), F.t() @ Y)
    return float((Y - F @ M).norm() / Y.norm())


# ---- Part A: block-level (centered) ----
log(f"PART A  block-level cross-block reconstruction (centered, n={n})")
log(f"{'block':>5} {'err_allOthers':>14} {'err_earlierOnly':>16} {'err_control':>12}")
log("-"*52)
ea, ec = [], []
for i in range(N):
    others  = torch.cat([D[j] for j in range(N) if j != i], 1)
    e_all   = ls_err(others, D[i])
    e_ctrl  = ls_err(others, D[i], shuffle=True)
    if i > 0:
        earlier = torch.cat([D[j] for j in range(i)], 1)
        e_caus  = ls_err(earlier, D[i]); ec.append(e_caus); cs = f"{e_caus:.3f}"
    else:
        cs = "  n/a"
    ea.append(e_all)
    log(f"{i:>5} {e_all:>14.3f} {cs:>16} {e_ctrl:>12.3f}")
log(f"mean: all-others={sum(ea)/len(ea):.3f}   earlier-only={sum(ec)/len(ec):.3f}   (control should be ~1.0)")

# ---- Part B: neuron-level for late blocks (from earlier hidden neurons) ----
log("")
log(f"PART B  late blocks reconstructed from EARLIER hidden NEURONS (causal, centered)")
log(f"{'block':>5} {'#earlierNeurons':>16} {'err_neuron':>11} {'err_control':>12}")
log("-"*48)
for i in LATE:
    Fe = torch.cat([H[j] for j in range(i)], 1)     # earlier hidden neurons
    e  = ls_err(Fe, D[i])
    ctrl = ls_err(Fe, D[i], shuffle=True)
    log(f"{i:>5} {Fe.shape[1]:>16} {e:>11.3f} {ctrl:>12.3f}")

log("")
log("READ: centered err near 0 with control near 1.0 -> genuine cross-block")
log("redundancy. Part B shows whether late blocks are reproducible from earlier")
log("NEURONS (the actual fold units) -> exploitable for cross-block pruning.")

open(OUT, "w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
