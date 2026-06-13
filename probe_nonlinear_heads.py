"""
HEAD-OUTPUT nonlinear redundancy probe (robust).   python probe_nonlinear2.py <model>

v1 failed on Llama: the autoencoder didn't converge (error > 1.0 = worse than the
mean), so a negative gap was an artifact, not a signal. v2 fixes this by design:

  The nonlinear model is a RESIDUAL on top of PCA, INITIALIZED AT the PCA solution:
      x_hat = z @ Wd  +  corr(z),   z = x @ We
      init  We = V_k,  Wd = V_k^T,  corr(last layer) = 0   ->  x_hat = PCA recon
  So at init it reconstructs EXACTLY like rank-k PCA. Training (minibatch) can only
  IMPROVE on PCA -> AE error <= PCA error ALWAYS -> gap >= 0, no undertraining artifact.

  gap% = (PCA_err - AE_err)/PCA_err, on held-out, matched bottleneck k.
    gap% ~ 0  -> nonlinear refinement buys nothing -> redundancy is LINEAR (= FLAT-LLM).
    gap% >> 0 -> channels are nonlinear functions of the k linear components ->
                 real nonlinear redundancy beyond linear rank.
"""
import os, sys, math, torch, numpy as np
import torch.nn as nn

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
IS_VIT = ("deit" in MODEL) or ("vit" in MODEL)
MAX_TOK=8000; FRACS=[0.05,0.10]; EPOCHS=500; BS=512
tag=MODEL.split("/")[-1]; OUT=f"/work/hdd/bdjd/hypergraph_pruning/probe_nonlinear_heads_{tag}.txt"
device=torch.device("cuda")

# ---------- capture MLP-hidden activations ----------
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
    loader=DataLoader(Subset(ds,idx),batch_size=32,num_workers=4)
    hs=[]
    for l in LAYERS:
        acts[l]=[]
        def mk(l):
            def h(m,i): acts[l].append(i[0].detach().reshape(-1,i[0].shape[-1]).float().cpu())
            return h
        hs.append(model.blocks[l].attn.proj.register_forward_pre_hook(mk(l)))
    with torch.no_grad():
        for x,_ in loader: model(x.to(device))
    for h in hs: h.remove()
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16).eval().to(device)
    layers=model.model.layers; Nl=len(layers); LAYERS=[5,Nl//2,Nl-6]
    raw=open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt").read(400000)
    ids=tok(raw,return_tensors="pt").input_ids[0][:48*256].view(48,256).to(device)
    hs=[]
    for l in LAYERS:
        acts[l]=[]
        def mk(l):
            def h(m,i): acts[l].append(i[0].detach().reshape(-1,i[0].shape[-1]).float().cpu())
            return h
        hs.append(layers[l].self_attn.o_proj.register_forward_pre_hook(mk(l)))
    with torch.no_grad():
        for s in range(0,48,8): model(ids[s:s+8])
    for h in hs: h.remove()
del model; torch.cuda.empty_cache()

def recon_err(X, rec): return float((X-rec).norm()/X.norm())

def run(Xtr,Xte,k):
    C=Xtr.shape[1]
    # PCA basis (top-k right singular vectors of train)
    _,_,Vh=torch.linalg.svd(Xtr,full_matrices=False); Vk=Vh[:k].t().contiguous()  # [C,k]
    pca=recon_err(Xte, Xte@Vk@Vk.t())
    # residual AE, initialized AT PCA
    h=min(max(2*k,512),4096)
    We=nn.Parameter(Vk.clone()); Wd=nn.Parameter(Vk.t().contiguous().clone())
    corr=nn.Sequential(nn.Linear(k,h),nn.GELU(),nn.Linear(h,C)).to(device)
    nn.init.zeros_(corr[-1].weight); nn.init.zeros_(corr[-1].bias)
    opt=torch.optim.Adam([We,Wd]+list(corr.parameters()),lr=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    ntr=Xtr.shape[0]; l0=None
    for ep in range(EPOCHS):
        perm=torch.randperm(ntr,device=device)
        for b in range(0,ntr,BS):
            xb=Xtr[perm[b:b+BS]]; z=xb@We; rec=z@Wd+corr(z)
            loss=((rec-xb)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if l0 is None: l0=float(loss)
        sch.step()
    lf=float(loss)
    with torch.no_grad():
        z=Xte@We; ae=recon_err(Xte, z@Wd+corr(z))
    del We,Wd,corr; torch.cuda.empty_cache()
    return pca, ae, l0, lf

lines=[]; lg=lambda s:(lines.append(s),print(s))
lg(f"HEAD-OUTPUT nonlinear redundancy (PCA-init residual AE)  MODEL={MODEL}  layers={LAYERS}")
lg(f"{'layer':>5} {'C':>6} {'k':>6} {'PCA':>8} {'AE':>8} {'gap%':>7}  {'AEloss:start->end':>20}")
lg("-"*70)
allg=[]
for l in LAYERS:
    X=torch.cat(acts[l],0); acts[l]=None
    X=X[torch.randperm(X.shape[0])[:min(MAX_TOK,X.shape[0])]]
    mu=X.mean(0,keepdim=True); sd=X.std(0,keepdim=True)+1e-6; Xn=((X-mu)/sd).to(device)
    n=Xn.shape[0]; ntr=int(0.7*n); pm=torch.randperm(n); Xtr=Xn[pm[:ntr]]; Xte=Xn[pm[ntr:]]; C=Xn.shape[1]
    for f in FRACS:
        k=max(2,int(f*C)); pca,ae,l0,lf=run(Xtr,Xte,k)
        gp=(pca-ae)/pca*100 if pca>0 else 0; allg.append(gp)
        lg(f"{l:>5} {C:>6} {k:>6} {pca:>8.4f} {ae:>8.4f} {gp:>+6.1f}%  {l0:>8.4f} -> {lf:<8.4f}")
    del Xn,Xtr,Xte; torch.cuda.empty_cache()
lg("")
lg(f"MEAN gap% = {np.mean(allg):+.1f}%   (AE is init'd AT PCA, so gap >= 0 by construction)")
lg("gap% ~ 0  -> nonlinear refinement buys nothing -> redundancy LINEAR -> = FLAT-LLM.")
lg("gap% >>0  -> nonlinear redundancy beyond linear rank -> real foundation (with the")
lg("            caveat that exploiting it needs a nonlinear decoder = extra params).")
lg("(AEloss start->end should DROP a lot; if it didn't, training stalled.)")
open(OUT,"w").write("\n".join(lines)+"\n"); print("Saved:",OUT)
