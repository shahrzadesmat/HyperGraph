"""
Multiplicative/gated set-redundancy probe.   python probe_nonlinear_mult.py <model>

The plain-MLP probe found ~0 nonlinear gap. But a generic MLP underfits PRODUCTS,
and the activations come from a MULTIPLICATIVE architecture (SwiGLU: hidden =
SiLU(gate) * up). So this tests an architecture-matched operation: can a GATED
(multiplicative) decoder reconstruct channels where linear low-rank AND a generic
MLP cannot?  Run head-to-head, same data, both PCA-initialized (gap >= 0).

Per MLP layer, matched bottleneck k:
  PCA          : linear low-rank (= FLAT-LLM)
  +MLP corr    : generic nonlinear decoder  (Linear->GELU->Linear)   -> gap_mlp
  +GATED corr  : multiplicative decoder  (SiLU(Wg z) (*) (Wu z))->Wd  -> gap_gate

  gap_gate >> gap_mlp >> 0 -> MULTIPLICATIVE redundancy a plain MLP misses (real find).
  gap_gate ~ gap_mlp ~ 0   -> no nonlinear redundancy of either kind -> all linear.
"""
import os, sys, math, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

MODEL=sys.argv[1] if len(sys.argv)>1 else "meta-llama/Llama-2-7b-hf"
IS_VIT=("deit" in MODEL) or ("vit" in MODEL)
MAX_TOK=8000; FRACS=[0.05,0.10]; EPOCHS=500; BS=512
tag=MODEL.split("/")[-1]; OUT=f"/work/hdd/bdjd/hypergraph_pruning/probe_nonlinear_mult_{tag}.txt"
device=torch.device("cuda")

# ---------- capture MLP hidden ----------
acts={}
if IS_VIT:
    import timm
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    model=timm.create_model(MODEL,pretrained=True).eval().to(device)
    Nb=len(model.blocks); LAYERS=[1,Nb//2,Nb-3]
    tf=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds=ImageFolder("/work/hdd/bdjd/imagenet_10pct/val",transform=tf)
    idx=torch.randperm(len(ds),generator=torch.Generator().manual_seed(0))[:80].tolist()
    loader=DataLoader(Subset(ds,idx),batch_size=32,num_workers=4); hs=[]
    for l in LAYERS:
        acts[l]=[]
        def mk(l):
            def h(m,i,o): acts[l].append(o.detach().reshape(-1,o.shape[-1]).float().cpu())
            return h
        hs.append(model.blocks[l].mlp.act.register_forward_hook(mk(l)))
    with torch.no_grad():
        for x,_ in loader: model(x.to(device))
    for h in hs: h.remove()
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16).eval().to(device)
    layers=model.model.layers; Nl=len(layers); LAYERS=[5,Nl//2,Nl-6]; hs=[]
    for l in LAYERS:
        acts[l]=[]
        def mk(l):
            def h(m,i): acts[l].append(i[0].detach().reshape(-1,i[0].shape[-1]).float().cpu())
            return h
        hs.append(layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)))
    raw=open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt").read(400000)
    ids=tok(raw,return_tensors="pt").input_ids[0][:48*256].view(48,256).to(device)
    with torch.no_grad():
        for s in range(0,48,8): model(ids[s:s+8])
    for h in hs: h.remove()
del model; torch.cuda.empty_cache()

class MLPCorr(nn.Module):
    def __init__(s,k,h,C):
        super().__init__(); s.net=nn.Sequential(nn.Linear(k,h),nn.GELU(),nn.Linear(h,C))
        nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)
    def forward(s,z): return s.net(z)

class GatedCorr(nn.Module):                       # SwiGLU-style multiplicative
    def __init__(s,k,h,C):
        super().__init__(); s.Wg=nn.Linear(k,h); s.Wu=nn.Linear(k,h); s.Wd=nn.Linear(h,C)
        nn.init.zeros_(s.Wd.weight); nn.init.zeros_(s.Wd.bias)
    def forward(s,z): return s.Wd(F.silu(s.Wg(z))*s.Wu(z))

def recon_err(X,rec): return float((X-rec).norm()/X.norm())
def train_corr(Xtr,Xte,Vk,corr):
    We=nn.Parameter(Vk.clone()); Wd=nn.Parameter(Vk.t().contiguous().clone()); corr=corr.to(device)
    opt=torch.optim.Adam([We,Wd]+list(corr.parameters()),lr=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS); ntr=Xtr.shape[0]
    for ep in range(EPOCHS):
        perm=torch.randperm(ntr,device=device)
        for b in range(0,ntr,BS):
            xb=Xtr[perm[b:b+BS]]; z=xb@We; rec=z@Wd+corr(z)
            loss=((rec-xb)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    with torch.no_grad(): z=Xte@We; e=recon_err(Xte,z@Wd+corr(z))
    del We,Wd,corr; torch.cuda.empty_cache(); return e

lines=[]; lg=lambda s:(lines.append(s),print(s))
lg(f"MULTIPLICATIVE vs MLP set-redundancy   MODEL={MODEL}  layers={LAYERS}")
lg(f"{'layer':>5} {'C':>6} {'k':>6} {'PCA':>8} {'+MLP':>8} {'+GATED':>8} | {'gap_mlp':>8} {'gap_gate':>9}")
lg("-"*72)
gm=[]; gg=[]
for l in LAYERS:
    X=torch.cat(acts[l],0); acts[l]=None
    X=X[torch.randperm(X.shape[0])[:min(MAX_TOK,X.shape[0])]]
    mu=X.mean(0,keepdim=True); sd=X.std(0,keepdim=True)+1e-6; Xn=((X-mu)/sd).to(device)
    n=Xn.shape[0]; ntr=int(0.7*n); pm=torch.randperm(n); Xtr=Xn[pm[:ntr]]; Xte=Xn[pm[ntr:]]; C=Xn.shape[1]
    for f in FRACS:
        k=max(2,int(f*C)); hh=min(max(2*k,512),4096)
        _,_,Vh=torch.linalg.svd(Xtr,full_matrices=False); Vk=Vh[:k].t().contiguous()
        pca=recon_err(Xte,Xte@Vk@Vk.t())
        em=train_corr(Xtr,Xte,Vk,MLPCorr(k,hh,C)); eg=train_corr(Xtr,Xte,Vk,GatedCorr(k,hh,C))
        a=(pca-em)/pca*100; b=(pca-eg)/pca*100; gm.append(a); gg.append(b)
        lg(f"{l:>5} {C:>6} {k:>6} {pca:>8.4f} {em:>8.4f} {eg:>8.4f} | {a:>+7.1f}% {b:>+8.1f}%")
    del Xn,Xtr,Xte; torch.cuda.empty_cache()
lg("")
lg(f"MEAN gap_mlp (generic) = {np.mean(gm):+.1f}%    MEAN gap_gate (multiplicative) = {np.mean(gg):+.1f}%")
lg("gap_gate >> gap_mlp > 0 -> MULTIPLICATIVE redundancy a plain MLP/low-rank miss.")
lg("gap_gate ~ gap_mlp ~ 0  -> no nonlinear redundancy of either operation -> all linear.")
open(OUT,"w").write("\n".join(lines)+"\n"); print("Saved:",OUT)
