"""
Cross-block redundancy probe.

Premise to test: transformer blocks do OVERLAPPING work — a block's contribution
to the shared residual stream can be reconstructed from OTHER blocks' contributions.
If true, there is cross-block redundancy that every per-layer method (ThiNet/CP/
DepGraph/GOHSP) is structurally blind to.

For each block i we take its MLP contribution to the residual  Δ_i  (tokens × 384)
and ask: how well can a SINGLE fixed linear map reproduce Δ_i from
   (a) ALL other blocks' contributions {Δ_j : j≠i}
   (b) only EARLIER blocks {Δ_j : j<i}   (causal — usable for a forward fold)
   (c) CONTROL: the same dictionary with rows shuffled (breaks token alignment)

err near 0  -> Δ_i is reproducible from other blocks  -> cross-block redundancy real.
control err should stay near 1.0 (else low err is a trivial artifact).

Writes a table to probe_crossblock_result.txt.
"""
import os, torch, timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

DATA = "/work/hdd/bdjd/imagenet_10pct"
N_IMAGES = 160
MAX_TOKENS = 16000
RIDGE = 1e-3
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

# capture each block's MLP contribution to the residual (Δ_mlp, in residual space R^384)
deltas = {b: [] for b in range(len(model.blocks))}
handles = []
def mk(b):
    def hook(m, i, o):
        deltas[b].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
    return hook
for b, blk in enumerate(model.blocks):
    handles.append(blk.mlp.register_forward_hook(mk(b)))

with torch.no_grad():
    for x, _ in loader:
        model(x.to(device))
for h in handles:
    h.remove()

N = len(model.blocks)
D = {b: torch.cat(deltas[b], 0) for b in range(N)}     # each (tokens, 384)
ntok = D[0].shape[0]
if ntok > MAX_TOKENS:
    sel = torch.randperm(ntok)[:MAX_TOKENS]
    D = {b: D[b][sel].to(device) for b in range(N)}
else:
    D = {b: D[b].to(device) for b in range(N)}
n = D[0].shape[0]
embed = D[0].shape[1]


def ls_err(dict_blocks, target, shuffle=False):
    """Relative error of reconstructing target (n,384) from a fixed linear map
    over the concatenated dictionary blocks (+ intercept). Ridge-regularized."""
    if not dict_blocks:
        return float("nan")
    Dm = torch.cat([D[j] for j in dict_blocks], dim=1)          # (n, 384*|dict|)
    if shuffle:
        Dm = Dm[torch.randperm(n, device=Dm.device)]
    ones = torch.ones(n, 1, device=Dm.device)
    Dm = torch.cat([Dm, ones], dim=1)
    G = Dm.t() @ Dm
    lam = RIDGE * torch.trace(G) / G.shape[0]
    M = torch.linalg.solve(G + lam*torch.eye(G.shape[0], device=Dm.device), Dm.t() @ target)
    resid = target - Dm @ M
    return float(resid.norm() / target.norm())


lines = []
def log(s): lines.append(s); print(s)

log(f"cross-block reconstruction of each block's MLP contribution  (n={n} tokens)")
log(f"{'block':>5} {'err_allOthers':>14} {'err_earlierOnly':>16} {'err_control':>12}")
log("-"*52)
ea_all, ea_caus = [], []
for i in range(N):
    others  = [j for j in range(N) if j != i]
    earlier = [j for j in range(i)]
    e_all  = ls_err(others, D[i])
    e_caus = ls_err(earlier, D[i]) if earlier else float("nan")
    e_ctrl = ls_err(others, D[i], shuffle=True)
    ea_all.append(e_all)
    if earlier: ea_caus.append(e_caus)
    cs = f"{e_caus:.3f}" if earlier else "  n/a"
    log(f"{i:>5} {e_all:>14.3f} {cs:>16} {e_ctrl:>12.3f}")

log("")
log(f"mean err (all-others)     = {sum(ea_all)/len(ea_all):.3f}")
log(f"mean err (earlier-only)   = {sum(ea_caus)/len(ea_caus):.3f}  (causal, foldable)")
log("")
log("READ: low err (<~0.3) AND control err near 1.0  ->  a block's work is")
log("largely reproducible from OTHER blocks -> cross-block redundancy is REAL")
log("and (for earlier-only) causally foldable. If err ~ control ~ 1.0, blocks")
log("are independent and the cross-block idea is dead.")

open(OUT, "w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
