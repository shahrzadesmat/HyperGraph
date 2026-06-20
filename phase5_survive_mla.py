"""
phase5_survive_mla.py — does the temporal redundancy SURVIVE MLA compression?
phase4 found keys ~30-50% temporally predictable. But that predictable part may be the HIGH-variance
part MLA KEEPS -> MLA could already absorb it -> temporal would NOT stack. This tests whether the MLA
LATENT z (joint [K;V] PCA code at dim d_c) is STILL temporally predictable.
  z_t = ([K;V]_t - mu)/sd @ Vk[:, :d_c]     (Vk = top eigvecs of joint [K;V] cov = MLA's shared code)
For each layer: temporal FVE (z_t from z_{t-1}, w=1, null-calibrated) of the raw key AND z at several d_c.
  FVE(z) stays ~ FVE(raw) as d_c shrinks -> temporal is ORTHOGONAL to MLA -> STACKS  (Path B alive)
  FVE(z) -> ~0           as d_c shrinks -> MLA already absorbed the temporal redundancy (Path A)
env: SM_MODEL  SM_CALIB=60000  SM_LAYERS=0,8,16,24,31  SM_DCS=2048,1024,512,256
"""
import os, math, time, torch
import torch.nn as nn
device = torch.device("cuda")
MODEL = os.environ.get("SM_MODEL", "meta-llama/Llama-2-7b-hf")
N_CALIB = int(os.environ.get("SM_CALIB", "60000"))
CHUNK = 2048
LAYERS_ENV = os.environ.get("SM_LAYERS", "0,8,16,24,31")
DCS = [int(x) for x in os.environ.get("SM_DCS", "2048,1024,512,256").split(",")]
H = 512; LR = 1e-3; MAX_EPOCHS = 120; PATIENCE = 10; BS = 8192; WD = 1e-2
OUT = "/work/hdd/bdjd/hypergraph_pruning/phase5_survive_mla.txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); C = model.config.hidden_size; C2 = 2 * C
LAYERS = [l for l in (int(x) for x in LAYERS_ENV.split(",")) if l < NL]
DCS = [d for d in DCS if d <= C2]
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
lg(f"phase5_survive_mla  MODEL={MODEL}  C={C}  C2={C2}  layers={LAYERS}  d_c={DCS}  calib={N_CALIB}")

# ---------- Stage A: capture K and V in sequence (chunk boundaries preserved) ----------
Kseq = {l: [] for l in LAYERS}; Vseq = {l: [] for l in LAYERS}; hooks = []
def mk(store, l):
    def h(m, i, o): store[l].append(o.detach()[0].to(torch.float16).cpu())
    return h
for l in LAYERS:
    hooks.append(attn[l].k_proj.register_forward_hook(mk(Kseq, l)))
    hooks.append(attn[l].v_proj.register_forward_hook(mk(Vseq, l)))
nchunks = max(1, min(N_CALIB // CHUNK, (train_ids.shape[0] - 1) // CHUNK))
with torch.no_grad():
    for c in range(nchunks):
        ids = train_ids[c * CHUNK:(c + 1) * CHUNK]
        if ids.shape[0] < 4: break
        model(ids.unsqueeze(0).to(device))
for h in hooks: h.remove()
del model, attn; torch.cuda.empty_cache()
lg(f"capture: {time.time()-t0:.0f}s  chunks={len(Kseq[LAYERS[0]])}")

# ---------- helpers (null-calibrated temporal FVE; same machinery as phase4) ----------
def make_pairs(chs):                         # w=1: X = k_{t-1}, Y = k_t  (within chunk)
    Xs, Ys = [], []
    for ch in chs:
        if ch.shape[0] < 2: continue
        Ys.append(ch[1:]); Xs.append(ch[:-1])
    return torch.cat(Xs).float(), torch.cat(Ys).float()
def fve(Yva, Yhat, my):
    return float(1.0 - ((Yva - Yhat) ** 2).sum() / ((Yva - my) ** 2).sum())
def fit_eval(Xtr, Ytr, Xva, Yva):
    Cout = Ytr.shape[1]
    mx = Xtr.mean(0, keepdim=True); sx = Xtr.std(0, keepdim=True) + 1e-6; my = Ytr.mean(0, keepdim=True)
    Xc = Xtr - mx; Yc = Ytr - my
    lam = 1e-2 * (Xc * Xc).mean() * Xc.shape[1]
    XtX = Xc.t() @ Xc + lam * torch.eye(Xc.shape[1], device=device)
    W = torch.linalg.solve(XtX, Xc.t() @ Yc)
    lin_va = (Xva - mx) @ W + my; fl = fve(Yva, lin_va, my)
    lin_tr = Xc @ W + my; Rtr = (Ytr - lin_tr).detach()
    del Xc, XtX, lin_tr, Yc; torch.cuda.empty_cache()
    Xn_va = (Xva - mx) / sx
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], H), nn.GELU(), nn.Linear(H, Cout)).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=WD)
    best = math.inf; best_sd = None; bad = 0; n = Xtr.shape[0]
    for ep in range(MAX_EPOCHS):
        mlp.train(); perm = torch.randperm(n, device=device)
        for b in range(0, n, BS):
            idx = perm[b:b + BS]; xb = (Xtr[idx] - mx) / sx
            loss = ((mlp(xb) - Rtr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            ve = float(((Yva - (lin_va + mlp(Xn_va))) ** 2).mean())
        if ve < best * (1 - 1e-4): best = ve; best_sd = {k: v.detach().clone() for k, v in mlp.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    mlp.load_state_dict(best_sd); mlp.eval()
    with torch.no_grad(): fn = fve(Yva, lin_va + mlp(Xn_va), my)
    return fl, fn
def temporal_fve(chunks):                    # -> (FVE_lin_real, FVE_lin_null, FVE_nl_real)
    X, Y = make_pairs(chunks); ntr = int(0.85 * X.shape[0])
    Xtr, Ytr, Xva, Yva = X[:ntr].to(device), Y[:ntr].to(device), X[ntr:].to(device), Y[ntr:].to(device)
    fl, fn = fit_eval(Xtr, Ytr, Xva, Yva)
    ptr = torch.randperm(Xtr.shape[0], device=device); pva = torch.randperm(Xva.shape[0], device=device)
    fln, _ = fit_eval(Xtr[ptr], Ytr, Xva[pva], Yva)
    del X, Y, Xtr, Ytr, Xva, Yva; torch.cuda.empty_cache()
    return fl, fln, fn

# ---------- Stage B: per layer — raw key FVE, then MLA latent FVE at each d_c ----------
lg("=" * 104)
lg("temporal FVE (key_t from key_{t-1}, w=1, null-calibrated). 'raw'=key; z@dc = MLA joint latent at d_c.")
lg(f"{'layer':>5} | {'raw key real(null)':>20} | " + " ".join(f"{'z@'+str(d)+' real(null)':>18}" for d in DCS))
lg("-" * 104)
summary = {dc: [] for dc in DCS}; sraw = []
for l in LAYERS:
    Kc = Kseq[l]; Vc = Vseq[l]
    flr, flnr, fnr = temporal_fve(Kc); sraw.append(flr - flnr)
    row = f"{l:>5} | {flr:>9.3f} ({flnr:>+6.3f}) | "
    Kall = torch.cat(Kc).float().to(device); Vall = torch.cat(Vc).float().to(device)
    KV = torch.cat([Kall, Vall], 1); mu = KV.mean(0, keepdim=True); sd = KV.std(0, keepdim=True) + 1e-6
    KVs = (KV - mu) / sd; cov = KVs.t() @ KVs / KV.shape[0]
    lamv, Q = torch.linalg.eigh(cov); Vk = Q.flip(1)
    del Kall, Vall, KV, KVs, cov, Q; torch.cuda.empty_cache()
    cells = []
    for dc in DCS:
        Vkc = Vk[:, :dc]; zc = []
        for ck, cv in zip(Kc, Vc):
            kv = torch.cat([ck.float(), cv.float()], 1).to(device)
            zc.append((((kv - mu) / sd) @ Vkc).cpu())
        flz, flnz, fnz = temporal_fve(zc); summary[dc].append(flz - flnz)
        cells.append(f"{flz:>9.3f} ({flnz:>+6.3f})")
        del zc; torch.cuda.empty_cache()
    lg(row + " ".join(f"{c:>18}" for c in cells)); flush()
    del Vk, mu, sd; torch.cuda.empty_cache()
lg("-" * 104)
mraw = sum(sraw) / len(sraw)
lg(f"mean temporal headroom (real-null):  raw key={mraw:+.3f}   " +
   "   ".join(f"z@{dc}={sum(summary[dc])/len(summary[dc]):+.3f}" for dc in DCS))
lg("")
lg("VERDICT: z@dc headroom stays ~ raw-key as d_c shrinks -> temporal SURVIVES MLA -> STACKS (Path B alive).")
lg("         z@dc headroom -> 0 as d_c shrinks -> MLA already absorbed the temporal redundancy (Path A).")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
