"""
phase1_kv_surgery.py — THE exploitability gate for nonlinear KV-cache compression.

Question: at a matched per-token KV-cache budget k, does reconstructing K and V with a
NONLINEAR decoder preserve the model's function (WikiText-2 perplexity) better than a
LINEAR (PCA rank-k) reconstruction — the linear case being the MLA/ASVD/EigenAttention
family? If yes, the nonlinear-redundancy finding TRANSLATES downstream → it's a method.
If no, it's a finding only.

Per layer, for K (k_proj output) and V (v_proj output), fit on calibration activations
(the probe's machinery): frozen PCA encoder Vk (top-k eigvecs of standardized train cov)
+ nonlinear residual decoder
    x_hat = z @ Vk^T + corr(z),   z = ((x-mu)/sd) @ Vk,   corr = Lin(k,h)->GELU->Lin(h,C)
with corr's last layer zero-init (so it STARTS exactly at PCA and can only improve).

SURGERY: forward hooks on every layer's k_proj/v_proj replace the output (pre-RoPE) with
the rank-k reconstruction. Evaluate perplexity on held-out WikiText-2 TEST for:
  baseline (full K/V) | linear (z@Vk^T) | nonlinear (z@Vk^T + corr(z))   at matched k.

WIN  : ppl_nonlinear < ppl_linear at matched k  (and the gap grows as k shrinks).
LOSE : ppl_nonlinear >= ppl_linear              (gap doesn't translate; finding only).

  python phase1_kv_surgery.py            # default sweep KV_FRACS, all layers
env: KV_FRACS=0.5,0.25  KV_CALIB=60000  KV_PPL_CHUNKS=60  KV_LAYERS=all
"""
import os, math, time, torch, numpy as np
import torch.nn as nn

MODEL = "meta-llama/Llama-2-7b-hf"
FRACS = [float(x) for x in os.environ.get("KV_FRACS", "0.5,0.25").split(",")]  # cache budget k/C
N_CALIB = int(os.environ.get("KV_CALIB", "60000"))      # calibration tokens
PPL_CHUNKS = int(os.environ.get("KV_PPL_CHUNKS", "60"))  # test chunks of length CHUNK
CHUNK = 2048
H_CAP = 1024; DROPOUT = 0.1; WD = 1e-2; LR = 1e-3; MAX_EPOCHS = 200; PATIENCE = 15; MIN_DREL = 1e-4; BS = 8192
device = torch.device("cuda")
OUT = "/work/hdd/bdjd/hypergraph_pruning/phase1_kv_surgery.txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); C = model.config.hidden_size
LAYERS = list(range(NL)) if os.environ.get("KV_LAYERS", "all") == "all" else [int(x) for x in os.environ["KV_LAYERS"].split(",")]

train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
test_ids = tok(test_txt, return_tensors="pt").input_ids[0]

lg(f"phase1_kv_surgery  MODEL={MODEL}  C={C}  layers={len(LAYERS)}  fracs={FRACS}")
lg(f"calib={N_CALIB} tok  ppl_chunks={PPL_CHUNKS}x{CHUNK}  test_tokens={test_ids.shape[0]}")

# ===================== Stage A: capture calibration K,V (one pass) =====================
capK = {l: [] for l in LAYERS}; capV = {l: [] for l in LAYERS}; hooks = []
def mk(store, l):
    def h(m, i, o): store[l].append(o.detach().reshape(-1, o.shape[-1]).half().cpu())
    return h
for l in LAYERS:
    hooks.append(attn[l].k_proj.register_forward_hook(mk(capK, l)))
    hooks.append(attn[l].v_proj.register_forward_hook(mk(capV, l)))
cb = min(N_CALIB, train_ids.shape[0])
with torch.no_grad():
    for s in range(0, cb - 1, CHUNK):
        model(train_ids[s:s + CHUNK].unsqueeze(0).to(device))
for h in hooks: h.remove()
Kact = {l: torch.cat(capK[l], 0) for l in LAYERS}; capK = None
Vact = {l: torch.cat(capV[l], 0) for l in LAYERS}; capV = None
lg(f"capture: {time.time()-t0:.0f}s   calib rows/layer={Kact[LAYERS[0]].shape[0]}")

# ===================== Stage B: fit Vk + nonlinear decoder per (layer,site,frac) =====================
class Corr(nn.Module):
    def __init__(s, k, h, C):
        super().__init__(); s.l1 = nn.Linear(k, h); s.act = nn.GELU(); s.do = nn.Dropout(DROPOUT); s.l2 = nn.Linear(h, C)
        nn.init.zeros_(s.l2.weight); nn.init.zeros_(s.l2.bias)
    def forward(s, z): return s.l2(s.do(s.act(s.l1(z))))

def fit_site(X):
    """X [N,C] cpu fp16 -> {frac: (mu,sd,Vk,corr)} with shared mu/sd/eigh."""
    N = X.shape[0]; ntr = int(0.85 * N); pm = torch.randperm(N)
    Xtr = X[pm[:ntr]].float().to(device); Xva = X[pm[ntr:]].float().to(device)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    cov = Xtr.t() @ Xtr / Xtr.shape[0]
    lam, Q = torch.linalg.eigh(cov); Q = Q.flip(1)          # eigvecs desc
    out = {}
    for fr in FRACS:
        k = max(8, int(fr * C)); Vk = Q[:, :k].contiguous(); Wd = Vk.t().contiguous()
        corr = Corr(k, min(2 * k, H_CAP), C).to(device)
        opt = torch.optim.AdamW(corr.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
        best = math.inf; best_sd = None; bad = 0; n = Xtr.shape[0]
        for ep in range(MAX_EPOCHS):
            corr.train(); perm = torch.randperm(n, device=device)
            for b in range(0, n, BS):
                xb = Xtr[perm[b:b + BS]]; z = xb @ Vk; rec = z @ Wd + corr(z)
                loss = ((rec - xb) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            corr.eval()
            with torch.no_grad():
                zv = Xva @ Vk; ve = float(((zv @ Wd + corr(zv) - Xva) ** 2).mean())
            if ve < best * (1 - MIN_DREL): best = ve; best_sd = {kk: v.detach().clone() for kk, v in corr.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= PATIENCE: break
        if best_sd is not None: corr.load_state_dict(best_sd)
        corr.eval()
        out[fr] = (mu.detach(), sd.detach(), Vk.detach(), corr)
    del Xtr, Xva, cov, Q; torch.cuda.empty_cache()
    return out

DEC = {}  # (l,site) -> {frac:(mu,sd,Vk,corr)}
for i, l in enumerate(LAYERS):
    DEC[(l, "K")] = fit_site(Kact[l]); Kact[l] = None
    DEC[(l, "V")] = fit_site(Vact[l]); Vact[l] = None
    if (i + 1) % 8 == 0: lg(f"  fit {i+1}/{len(LAYERS)} layers   ({time.time()-t0:.0f}s)")
del Kact, Vact; torch.cuda.empty_cache()
lg(f"fit done: {time.time()-t0:.0f}s")

# ===================== Stage C: surgery + perplexity =====================
MODE = ["none"]; CUR = [FRACS[0]]   # MODE in {none,linear,nonlinear}; CUR = active frac
def surg(l, site):
    def h(m, i, o):
        if MODE[0] == "none": return o
        mu, sd, Vk, corr = DEC[(l, site)][CUR[0]]
        sh = o.shape; x = o.reshape(-1, sh[-1]).float()
        z = ((x - mu) / sd) @ Vk; rec = z @ Vk.t()
        if MODE[0] == "nonlinear": rec = rec + corr(z)
        return (rec * sd + mu).to(o.dtype).reshape(sh)
    return h
for l in LAYERS:
    attn[l].k_proj.register_forward_hook(surg(l, "K"))
    attn[l].v_proj.register_forward_hook(surg(l, "V"))

def perplexity():
    nll = 0.0; ntok = 0
    with torch.no_grad():
        for c in range(PPL_CHUNKS):
            ids = test_ids[c * CHUNK:(c + 1) * CHUNK]
            if ids.shape[0] < 2: break
            ids = ids.unsqueeze(0).to(device)
            loss = model(ids, labels=ids).loss.item()
            nll += loss * (ids.shape[1] - 1); ntok += ids.shape[1] - 1
    return math.exp(nll / ntok)

MODE[0] = "none"; ppl_base = perplexity()
lg(""); lg(f"baseline (full K+V) perplexity = {ppl_base:.3f}")
lg("-" * 78)
lg(f"{'k/C':>6} {'k':>6} {'cacheX':>7} | {'ppl_linear':>11} {'ppl_nonlin':>11} | {'Δppl(lin-nl)':>13}  verdict")
results = []
for fr in FRACS:
    CUR[0] = fr; k = max(8, int(fr * C)); cacheX = (2 * C) / (2 * k)
    MODE[0] = "linear"; pl = perplexity()
    MODE[0] = "nonlinear"; pn = perplexity()
    d = pl - pn; win = d > 0
    results.append((fr, k, cacheX, pl, pn, d, win))
    lg(f"{fr:>6.3f} {k:>6} {cacheX:>6.2f}x | {pl:>11.3f} {pn:>11.3f} | {d:>+13.3f}  {'NONLINEAR WINS' if win else 'no gain'}")
    flush()
lg("-" * 78)
nwin = sum(1 for *_, w in results if w)
lg(f"baseline ppl={ppl_base:.3f}.  Lower perplexity = better.  cacheX = KV-cache compression factor.")
lg(f"VERDICT: nonlinear beats linear at {nwin}/{len(FRACS)} budgets tested.")
if nwin == len(FRACS):
    lg("=> the activation-reconstruction gap TRANSLATES downstream -> nonlinear KV compression is a METHOD.")
    lg("   next: MLA-faithful joint latent + matched-budget curve vs MLA + decode-FLOP accounting.")
elif nwin > 0:
    lg("=> partial: nonlinear helps at some budgets -> investigate which regime; method candidate.")
else:
    lg("=> nonlinear does NOT beat linear downstream -> finding only, not a compressor (this way).")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
