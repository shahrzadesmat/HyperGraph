"""
Nonlinear set-redundancy probe.   python probe_nonlinear.py <model>

THE decisive test for the hypergraph idea. Effective rank measures only LINEAR
redundancy (= what FLAT-LLM/ASVD exploit). This asks: do MLP channels have
NONLINEAR redundancy beyond their linear rank — i.e., can a nonlinear predictor
reconstruct them where low-rank cannot?

Per MLP layer, at matched bottleneck dim k, on HELD-OUT activations:
  LINEAR  : PCA rank-k reconstruction error   (what FLAT-LLM achieves)
  NONLIN  : nonlinear autoencoder, bottleneck k, reconstruction error

  gap = err_LINEAR - err_NONLIN  (relative, same k, same standardized data)

  gap ~ 0       -> redundancy is all LINEAR -> FLAT-LLM is optimal -> idea collapses.
  gap >> 0      -> nonlinear redundancy beyond linear rank -> a foundation low-rank
                   cannot touch -> the (nonlinear) hypergraph idea has real ground.
"""
import os, sys, math, torch, numpy as np
import torch.nn as nn

MODEL = sys.argv[1] if len(sys.argv) > 1 else "deit_small_patch16_224"
IS_VIT = ("deit" in MODEL) or ("vit" in MODEL)
MAX_TOK = 8000; FRACS = [0.05, 0.10]; EPOCHS = 400
tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_nonlinear_{tag}.txt"
device = torch.device("cuda")

# ---------- capture MLP-hidden activations for a few representative layers ----------
acts = {}
if IS_VIT:
    import timm
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    model = timm.create_model(MODEL, pretrained=True).eval().to(device)
    Nb = len(model.blocks); LAYERS = [1, Nb//2, Nb-3]
    tf = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds = ImageFolder("/work/hdd/bdjd/imagenet_10pct/val", transform=tf)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:80].tolist()
    loader = DataLoader(Subset(ds, idx), batch_size=32, num_workers=4)
    hs=[]
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
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
    layers = model.model.layers; Nl=len(layers); LAYERS=[5, Nl//2, Nl-6]
    raw = open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt").read(400000)
    ids = tok(raw, return_tensors="pt").input_ids[0][:48*256].view(48,256).to(device)
    hs=[]
    for l in LAYERS:
        acts[l]=[]
        def mk(l):
            def h(m,i): acts[l].append(i[0].detach().reshape(-1,i[0].shape[-1]).float().cpu())
            return h
        hs.append(layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)))
    with torch.no_grad():
        for s in range(0,48,8): model(ids[s:s+8])
    for h in hs: h.remove()

# ---------- reconstruction comparisons ----------
def pca_err(Xtr, Xte, k):                              # best LINEAR rank-k reconstruction
    U,S,V = torch.linalg.svd(Xtr, full_matrices=False)
    Vk = V[:k].t()                                     # [C,k]
    rec = Xte @ Vk @ Vk.t()
    return float((Xte-rec).norm()/Xte.norm())

def ae_err(Xtr, Xte, k, epochs=EPOCHS):                # best NONLINEAR bottleneck-k reconstruction
    C = Xtr.shape[1]; h = min(max(2*k,512), 2048)
    enc = nn.Sequential(nn.Linear(C,h), nn.GELU(), nn.Linear(h,k))
    dec = nn.Sequential(nn.Linear(k,h), nn.GELU(), nn.Linear(h,C))
    ae = nn.Sequential(enc,dec).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=2e-3, weight_decay=0)
    Xtr=Xtr.to(device); Xte=Xte.to(device)
    for ep in range(epochs):
        opt.zero_grad(); rec=ae(Xtr); loss=((rec-Xtr)**2).mean(); loss.backward(); opt.step()
    with torch.no_grad(): rec=ae(Xte)
    e=float((Xte-rec).norm()/Xte.norm()); del ae,Xtr,Xte; torch.cuda.empty_cache(); return e

lines=[]; lg=lambda s:(lines.append(s),print(s))
lg(f"NONLINEAR set-redundancy probe   MODEL={MODEL}  layers={LAYERS}")
lg(f"{'layer':>5} {'C':>6} {'k':>6} {'PCA(lin)':>9} {'AE(nonlin)':>11} {'gap':>8} {'gap%':>7}")
lg("-"*60)
allgap=[]
for l in LAYERS:
    X = torch.cat(acts[l],0); acts[l]=None
    sel = torch.randperm(X.shape[0])[:min(MAX_TOK, X.shape[0])]
    X = X[sel]
    mu=X.mean(0,keepdim=True); sd=X.std(0,keepdim=True)+1e-6
    Xn=(X-mu)/sd                                       # per-channel standardize (fair to both)
    n=Xn.shape[0]; ntr=int(0.7*n); pm=torch.randperm(n)
    Xtr=Xn[pm[:ntr]]; Xte=Xn[pm[ntr:]]; C=Xn.shape[1]
    for f in FRACS:
        k=max(2,int(f*C))
        el=pca_err(Xtr,Xte,k); ea=ae_err(Xtr.clone(),Xte.clone(),k)
        gap=el-ea; gp=gap/el*100 if el>0 else 0
        allgap.append(gp)
        lg(f"{l:>5} {C:>6} {k:>6} {el:>9.4f} {ea:>11.4f} {gap:>+8.4f} {gp:>+6.1f}%")
lg("")
lg(f"MEAN gap% (PCA-AE)/PCA = {np.mean(allgap):+.1f}%")
lg("READ: gap% ~ 0 -> nonlinear AE no better than linear PCA at same k -> redundancy is")
lg("ALL LINEAR -> FLAT-LLM/low-rank already captures it -> hypergraph idea = low-rank.")
lg("gap% >> 0 (e.g. >20%) -> real nonlinear redundancy low-rank cannot reach -> foundation.")
lg("(Caveat: a nonlinear gap proves the redundancy EXISTS; exploiting it needs a nonlinear")
lg(" decoder = extra params/compute, so it isn't automatically a cheaper compressor.)")
open(OUT,"w").write("\n".join(lines)+"\n"); print("Saved:",OUT)
