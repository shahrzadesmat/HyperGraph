"""
phase2b_mla_joint.py — Phase 2b: nonlinear MLA vs real (linear) MLA, head to head.
(MEMORY-FIXED: decoders stored on CPU, only one tau's worth moved to GPU per perplexity pass,
so it no longer OOMs accumulating all layers x taus on device.)

MLA's defining trick: cache ONE shared latent c (dim d_c) per token, derived from the hidden
state x, and reconstruct BOTH K and V from it. The optimal rank-d_c joint code of [K;V] is a
linear function of x (K=x*M_K, V=x*M_V), so z = x*W_z - bz with W_z folded from the model's own
k_proj/v_proj weights + the joint activation statistics. Then:
   LINEAR  (= MLA):    [K;V]_hat = z*Vk^T
   NONLINEAR (ours):   [K;V]_hat = z*Vk^T + corr(z)
both from the SAME shared latent z (dim d_c = the cache). Per-layer rank by joint variance
retention (tau). Surgery: k_proj/v_proj hooks compute z from x and emit Khat/Vhat. Compare
WikiText-2 perplexity at matched cache budget d_c.  cacheX = 2C / d_c.

CORRECTNESS CHECK (early): z-from-x via W_z must equal z-from-[K;V]; AND linear@mildest-tau must
be near baseline perplexity. Trust the numbers only if both hold.

WIN: nonlinear ppl < linear(=MLA) ppl at matched d_c -> nonlinear MLA beats MLA.
env: KV_TAUS=0.99,0.98,0.97,0.95  KV_CALIB=60000  KV_PPL_CHUNKS=60
"""
import os, math, time, torch
import torch.nn as nn

MODEL = "meta-llama/Llama-2-7b-hf"
TAUS = [float(x) for x in os.environ.get("KV_TAUS", "0.99,0.98,0.97,0.95").split(",")]
N_CALIB = int(os.environ.get("KV_CALIB", "60000"))
PPL_CHUNKS = int(os.environ.get("KV_PPL_CHUNKS", "60"))
CHUNK = 2048
H_CAP = 1024; DROPOUT = 0.1; WD = 1e-2; LR = 1e-3; MAX_EPOCHS = 200; PATIENCE = 15; MIN_DREL = 1e-4; BS = 8192
device = torch.device("cuda")
OUT = "/work/hdd/bdjd/hypergraph_pruning/phase2b_mla_joint.txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); C = model.config.hidden_size; C2 = 2 * C
LAYERS = list(range(NL))
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
test_ids = tok(test_txt, return_tensors="pt").input_ids[0]
fwd_flops_tok = 2 * sum(p.numel() for p in model.parameters())
lg(f"phase2b_mla_joint (mem-fixed)  MODEL={MODEL}  C={C}  layers={NL}  taus={TAUS}")

# ---------- Stage A: capture x, K, V (CPU) ----------
capX = {l: [] for l in LAYERS}; capK = {l: [] for l in LAYERS}; capV = {l: [] for l in LAYERS}; hooks = []
def mkK(l):
    def h(m, i, o):
        capX[l].append(i[0].detach().reshape(-1, i[0].shape[-1]).half().cpu())
        capK[l].append(o.detach().reshape(-1, o.shape[-1]).half().cpu())
    return h
def mkV(l):
    def h(m, i, o): capV[l].append(o.detach().reshape(-1, o.shape[-1]).half().cpu())
    return h
for l in LAYERS:
    hooks.append(attn[l].k_proj.register_forward_hook(mkK(l)))
    hooks.append(attn[l].v_proj.register_forward_hook(mkV(l)))
cb = min(N_CALIB, train_ids.shape[0])
with torch.no_grad():
    for s in range(0, cb - 1, CHUNK):
        model(train_ids[s:s + CHUNK].unsqueeze(0).to(device))
for h in hooks: h.remove()
Xact = {l: torch.cat(capX[l], 0) for l in LAYERS}; capX = None
Kact = {l: torch.cat(capK[l], 0) for l in LAYERS}; capK = None
Vact = {l: torch.cat(capV[l], 0) for l in LAYERS}; capV = None
lg(f"capture: {time.time()-t0:.0f}s   rows/layer={Kact[LAYERS[0]].shape[0]}")

class Corr(nn.Module):
    def __init__(s, k, h, Co):
        super().__init__(); s.l1 = nn.Linear(k, h); s.act = nn.GELU(); s.do = nn.Dropout(DROPOUT); s.l2 = nn.Linear(h, Co)
        nn.init.zeros_(s.l2.weight); nn.init.zeros_(s.l2.bias)
    def forward(s, z): return s.l2(s.do(s.act(s.l1(z))))

def fit_corr(Xtr, Xva, Vk, Co):
    k = Vk.shape[1]; Wd = Vk.t().contiguous(); h = min(2 * k, H_CAP)
    corr = Corr(k, h, Co).to(device); opt = torch.optim.AdamW(corr.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    best = math.inf; best_sd = None; bad = 0; n = Xtr.shape[0]
    for ep in range(MAX_EPOCHS):
        corr.train(); perm = torch.randperm(n, device=device)
        for b in range(0, n, BS):
            xb = Xtr[perm[b:b + BS]]; z = xb @ Vk; rec = z @ Wd + corr(z)
            loss = ((rec - xb) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); corr.eval()
        with torch.no_grad():
            zv = Xva @ Vk; ve = float(((zv @ Wd + corr(zv) - Xva) ** 2).mean())
        if ve < best * (1 - MIN_DREL): best = ve; best_sd = {kk: v.detach().clone() for kk, v in corr.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    if best_sd is not None: corr.load_state_dict(best_sd)
    corr.eval(); return corr, h

# ---------- Stage B: per-layer joint latent + decoders; STORE ON CPU ----------
DEC = {l: {} for l in LAYERS}; DC = {tau: {} for tau in TAUS}; sanity = []
for i, l in enumerate(LAYERS):
    K = Kact[l].float().to(device); V = Vact[l].float().to(device); X = Xact[l].float().to(device)
    KV = torch.cat([K, V], 1)                                   # [N, 2C]
    N = KV.shape[0]; ntr = int(0.85 * N); pm = torch.randperm(N)
    mu = KV[pm[:ntr]].mean(0, keepdim=True); sd = KV[pm[:ntr]].std(0, keepdim=True) + 1e-6
    KVs = (KV - mu) / sd
    cov = KVs[pm[:ntr]].t() @ KVs[pm[:ntr]] / ntr
    lam, Q = torch.linalg.eigh(cov); lam = lam.flip(0).clamp_min(0); Q = Q.flip(1)
    cumfrac = torch.cumsum(lam, 0) / lam.sum()
    M_K = attn[l].k_proj.weight.detach().float().t(); M_V = attn[l].v_proj.weight.detach().float().t()
    W_KV = torch.cat([M_K, M_V], 1)                              # [C, 2C], x@W_KV = [K;V]
    Xtr = KVs[pm[:ntr]]; Xva = KVs[pm[ntr:]]
    for tau in TAUS:
        dc = int(torch.searchsorted(cumfrac, torch.tensor(tau, device=device)).item()) + 1
        dc = max(8, min(C2, dc)); Vk = Q[:, :dc].contiguous()
        corr, h = fit_corr(Xtr, Xva, Vk, C2)
        A = Vk / sd.squeeze(0).unsqueeze(1)                     # [2C, dc]
        W_z = (W_KV @ A).contiguous(); bz = (mu @ A).squeeze(0).contiguous()
        if i == 0:                                              # CORRECTNESS CHECK on GPU before offload
            z_x = X @ W_z - bz; z_kv = KVs @ Vk
            sanity.append((tau, float((z_x - z_kv).abs().max()), float(z_kv.abs().mean())))
        # OFFLOAD decoder to CPU (this is the OOM fix)
        DEC[l][tau] = (W_z.cpu(), bz.cpu(), Vk.cpu(), mu.cpu(), sd.cpu(), corr.cpu().eval(), dc)
        DC[tau][l] = dc
        del Vk, corr, W_z, bz, A; torch.cuda.empty_cache()
    if i == 0:
        lg("CORRECTNESS CHECK (layer 0): max|z_from_x - z_from_KV| should be ~0 (vs |z|):")
        for tau, mx, mag in sanity:
            lg(f"   tau={tau}: max_abs_err={mx:.3e}   |z|~{mag:.3e}   {'OK' if mx < 1e-2*max(mag,1e-6) else 'FAIL!!'}")
        flush()
    del K, V, X, KV, KVs, cov, Q, Xtr, Xva, mu, sd, M_K, M_V, W_KV
    Kact[l] = None; Vact[l] = None; Xact[l] = None; torch.cuda.empty_cache()
    if (i + 1) % 8 == 0: lg(f"  fit {i+1}/{NL} layers   ({time.time()-t0:.0f}s)")
del Kact, Vact, Xact; torch.cuda.empty_cache()
lg(f"fit done: {time.time()-t0:.0f}s")

# ---------- Stage C: surgery + perplexity, ONE tau's decoders on GPU at a time ----------
GDEC = {}; MODE = ["none"]
def surg(l, half):                                             # half: 0=K (first C), 1=V (last C)
    sl = slice(0, C) if half == 0 else slice(C, C2)
    def h(m, i, o):
        if MODE[0] == "none" or l not in GDEC: return o
        W_z, bz, Vk, mu, sd, corr, dc = GDEC[l]
        sh = o.shape; x = i[0].reshape(-1, sh[-1]).float()
        z = x @ W_z - bz
        rec = z @ Vk[sl].t()
        if MODE[0] == "nonlinear": rec = rec + corr(z)[:, sl]
        return (rec * sd[:, sl] + mu[:, sl]).to(o.dtype).reshape(sh)
    return h
for l in LAYERS:
    attn[l].k_proj.register_forward_hook(surg(l, 0))
    attn[l].v_proj.register_forward_hook(surg(l, 1))

def perplexity():
    nll = 0.0; ntok = 0
    with torch.no_grad():
        for c in range(PPL_CHUNKS):
            ids = test_ids[c * CHUNK:(c + 1) * CHUNK]
            if ids.shape[0] < 2: break
            ids = ids.unsqueeze(0).to(device)
            nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1); ntok += ids.shape[1] - 1
    return math.exp(nll / ntok)

MODE[0] = "none"; ppl_base = perplexity()
lg(""); lg(f"baseline (full K+V) perplexity = {ppl_base:.3f}   (linear@mildest-tau should be near this)")
lg("-" * 104)
lg(f"{'tau':>5} {'mean_dc':>8} {'cacheX':>7} {'dFLOP/tok':>10} {'dFLOP%':>7} | {'ppl_MLA(lin)':>12} {'ppl_nonlin':>11} | {'Δppl':>8}  verdict")
res = []
for tau in TAUS:
    GDEC.clear()                                               # move only THIS tau's decoders to GPU
    for l in LAYERS:
        W_z, bz, Vk, mu, sd, corr, dc = DEC[l][tau]
        GDEC[l] = (W_z.to(device), bz.to(device), Vk.to(device), mu.to(device), sd.to(device), corr.to(device).eval(), dc)
    dcs = list(DC[tau].values()); mean_dc = sum(dcs) / len(dcs); cacheX = (C2 * NL) / sum(dcs)
    dflop = 0.0
    for l, dc in DC[tau].items():
        hh = min(2 * dc, H_CAP); dflop += 2 * (dc * hh + hh * C2)
    MODE[0] = "linear"; pl = perplexity()
    MODE[0] = "nonlinear"; pn = perplexity()
    GDEC.clear(); torch.cuda.empty_cache()
    d = pl - pn; res.append((tau, mean_dc, cacheX, dflop, pl, pn, d))
    lg(f"{tau:>5.2f} {mean_dc:>8.0f} {cacheX:>6.2f}x {dflop:>10.2e} {100*dflop/fwd_flops_tok:>6.2f}% | "
       f"{pl:>12.3f} {pn:>11.3f} | {d:>+8.3f}  {'NONLINEAR' if d>0 else 'no gain'}")
    flush()
lg("-" * 104)
nwin = sum(1 for *_, d in res if d > 0)
lg(f"baseline ppl={ppl_base:.3f}.  ppl_MLA(lin) = faithful linear MLA at the SAME shared latent.")
lg(f"VERDICT: nonlinear MLA beats linear MLA at {nwin}/{len(TAUS)} operating points.")
lg("Trust only if the correctness check passed AND linear@mildest-tau is near baseline.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
