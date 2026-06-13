"""
Q/K/V nonlinear redundancy probe (robust).   python probe_nonlinear_qkv.py <model>

Captures the Q, K, V projection activations (what ASVD/EigenAttention/QSVD/MLA
compress linearly) and asks: does a nonlinear predictor reconstruct them better
than linear low-rank at matched bottleneck?  PCA-init residual AE -> gap >= 0.

  gap% ~ 0  -> Q/K/V redundancy is LINEAR -> low-rank methods are optimal.
  gap% >> 0 -> nonlinear redundancy in Q/K/V beyond linear rank.
"""
import os, sys, math, torch, numpy as np
import torch.nn as nn

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
IS_VIT = ("deit" in MODEL) or ("vit" in MODEL)
MAX_TOK=8000; FRACS=[0.05,0.10]; EPOCHS=500; BS=512
tag=MODEL.split("/")[-1]; OUT=f"/work/hdd/bdjd/hypergraph_pruning/probe_nonlinear_qkv_{tag}.txt"
device=torch.device("cuda")

Q={}; K={}; V={}
if IS_VIT:
    import timm
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    model=timm.create_model(MODEL,pretrained=True).eval().to(device)
    Nb=len(model.blocks); LAYERS=[1,Nb//2,Nb-3]; dim=model.blocks[0].attn.qkv.out_features//3
    tf=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds=ImageFolder("/work/hdd/bdjd/imagenet_10pct/val",transform=tf)
    idx=torch.randperm(len(ds),generator=torch.Generator().manual_seed(0))[:80].tolist()
    loader=DataLoader(Subset(ds,idx),batch_size=32,num_workers=4); hs=[]
    for l in LAYERS:
        Q[l]=[];K[l]=[];V[l]=[]
        def mk(l):
            def h(m,i,o):
                o=o.reshape(-1,o.shape[-1]).detach().float().cpu()
                Q[l].append(o[:,:dim]);K[l].append(o[:,dim:2*dim]);V[l].append(o[:,2*dim:])
            return h
        hs.append(model.blocks[l].attn.qkv.register_forward_hook(mk(l)))
    with torch.no_grad():
        for x,_ in loader: model(x.to(device))
    for h in hs: h.remove()
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16).eval().to(device)
    layers=model.model.layers; Nl=len(layers); LAYERS=[5,Nl//2,Nl-6]; hs=[]
    def mk(store,l):
        def h(m,i,o): store[l].append(o.reshape(-1,o.shape[-1]).detach().float().cpu())
        return h
    for l in LAYERS:
        Q[l]=[];K[l]=[];V[l]=[]
        hs.append(layers[l].self_attn.q_proj.register_forward_hook(mk(Q,l)))
        hs.append(layers[l].self_attn.k_proj.register_forward_hook(mk(K,l)))
        hs.append(layers[l].self_attn.v_proj.register_forward_hook(mk(V,l)))
    raw=open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt").read(400000)
    ids=tok(raw,return_tensors="pt").input_ids[0][:48*256].view(48,256).to(device)
    with torch.no_grad():
        for s in range(0,48,8): model(ids[s:s+8])
    for h in hs: h.remove()
del model; torch.cuda.empty_cache()

def recon_err(X,rec): return float((X-rec).norm()/X.norm())
def run(Xtr,Xte,k):
    C=Xtr.shape[1]
    _,_,Vh=torch.linalg.svd(Xtr,full_matrices=False); Vk=Vh[:k].t().contiguous()
    pca=recon_err(Xte,Xte@Vk@Vk.t())
    h=min(max(2*k,512),4096)
    We=nn.Parameter(Vk.clone()); Wd=nn.Parameter(Vk.t().contiguous().clone())
    corr=nn.Sequential(nn.Linear(k,h),nn.GELU(),nn.Linear(h,C)).to(device)
    nn.init.zeros_(corr[-1].weight); nn.init.zeros_(corr[-1].bias)
    opt=torch.optim.Adam([We,Wd]+list(corr.parameters()),lr=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS); ntr=Xtr.shape[0]
    for ep in range(EPOCHS):
        perm=torch.randperm(ntr,device=device)
        for b in range(0,ntr,BS):
            xb=Xtr[perm[b:b+BS]]; z=xb@We; rec=z@Wd+corr(z)
            loss=((rec-xb)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    with torch.no_grad(): z=Xte@We; ae=recon_err(Xte,z@Wd+corr(z))
    del We,Wd,corr; torch.cuda.empty_cache(); return pca,ae

lines=[]; lg=lambda s:(lines.append(s),print(s))
lg(f"Q/K/V nonlinear redundancy (PCA-init residual AE)  MODEL={MODEL}  layers={LAYERS}")
lg(f"{'proj':>4} {'layer':>5} {'C':>6} {'k':>6} {'PCA':>8} {'AE':>8} {'gap%':>7}")
lg("-"*52)
allg={}
for name,store in [("Q",Q),("K",K),("V",V)]:
    allg[name]=[]
    for l in LAYERS:
        X=torch.cat(store[l],0); store[l]=None
        X=X[torch.randperm(X.shape[0])[:min(MAX_TOK,X.shape[0])]]
        mu=X.mean(0,keepdim=True); sd=X.std(0,keepdim=True)+1e-6; Xn=((X-mu)/sd).to(device)
        n=Xn.shape[0]; ntr=int(0.7*n); pm=torch.randperm(n); Xtr=Xn[pm[:ntr]]; Xte=Xn[pm[ntr:]]; C=Xn.shape[1]
        for f in FRACS:
            k=max(2,int(f*C)); pca,ae=run(Xtr,Xte,k); gp=(pca-ae)/pca*100 if pca>0 else 0
            allg[name].append(gp)
            lg(f"{name:>4} {l:>5} {C:>6} {k:>6} {pca:>8.4f} {ae:>8.4f} {gp:>+6.1f}%")
        del Xn,Xtr,Xte; torch.cuda.empty_cache()
lg("")
for name in ("Q","K","V"): lg(f"MEAN gap% {name} = {np.mean(allg[name]):+.1f}%")
lg("gap% ~ 0 -> Q/K/V redundancy LINEAR -> ASVD/EigenAttention/MLA optimal.")
lg("gap% >>0 -> nonlinear Q/K/V redundancy beyond linear rank.")
open(OUT,"w").write("\n".join(lines)+"\n"); print("Saved:",OUT)
