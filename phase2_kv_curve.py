"""
phase2_kv_curve.py — Phase 2a: quality-vs-cache-budget CURVE for nonlinear KV-cache
compression, with PER-LAYER RANK ALLOCATION, vs a strong linear (PCA) baseline.

Phase 1 showed nonlinear > linear at matched UNIFORM budget (near-lossless at 2x, BOTH
broken at 4x). The 4x cliff was uniform allocation: compressing every layer to the same
rank wastes budget on robust layers and starves sensitive ones. Phase 2a fixes that:

  PER-(layer,site) ALLOCATION by variance retention — each K/V keeps the rank k_l needed to
  retain a fraction tau of its activation variance (from the eigenspectrum). Sensitive
  layers (slow spectral decay) auto-get more rank. Sweep tau -> a curve of (compression, ppl).

For each tau, for each layer's K and V:
  k_l = smallest rank with cumulative variance >= tau   (clamped [8, C])
  fit frozen-PCA encoder Vk=top-k_l eigvecs + nonlinear residual decoder corr (PCA-init)
SURGERY: hooks replace K/V with rank-k_l reconstruction; WikiText-2 test perplexity for
  baseline (full) / linear (z@Vk^T, = MLA/ASVD family) / nonlinear (z@Vk^T + corr(z)).
Also report the nonlinear decoder's added decode-FLOPs/token (the cost MLA's *linear*,
absorbable up-projection avoids — the honest price of the nonlinearity).

WIN: nonlinear ppl-vs-compression curve sits below linear across the range AND pushes the
usable compression well past the 2x cliff.   Linear baseline here = per-layer activation-PCA
(strong; ASVD/EigenAttention-grade). A true joint-latent MLA is Phase 2b.

env: KV_TAUS=0.96,0.92,0.88  KV_CALIB=60000  KV_PPL_CHUNKS=60  KV_LAYERS=all
"""
import os, math, time, torch
import torch.nn as nn

MODEL = "meta-llama/Llama-2-7b-hf"
TAUS = [float(x) for x in os.environ.get("KV_TAUS", "0.96,0.92,0.88").split(",")]
N_CALIB = int(os.environ.get("KV_CALIB", "60000"))
PPL_CHUNKS = int(os.environ.get("KV_PPL_CHUNKS", "60"))
CHUNK = 2048
H_CAP = 1024; DROPOUT = 0.1; WD = 1e-2; LR = 1e-3; MAX_EPOCHS = 200; PATIENCE = 15; MIN_DREL = 1e-4; BS = 8192
device = torch.device("cuda")
OUT = "/work/hdd/bdjd/hypergraph_pruning/phase2_kv_curve.txt"
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
fwd_flops_tok = 2 * sum(p.numel() for p in model.parameters())   # ~ full forward FLOPs/token

lg(f"phase2_kv_curve  MODEL={MODEL}  C={C}  layers={len(LAYERS)}  taus={TAUS}")
lg(f"calib={N_CALIB} tok  ppl_chunks={PPL_CHUNKS}x{CHUNK}  fwd_flops/tok~={fwd_flops_tok:.2e}")

# ---------- Stage A: capture calibration K,V ----------
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

# ---------- Stage B: per (layer,site) eigh; per tau allocate k_l and fit decoder ----------
class Corr(nn.Module):
    def __init__(s, k, h, C):
        super().__init__(); s.l1 = nn.Linear(k, h); s.act = nn.GELU(); s.do = nn.Dropout(DROPOUT); s.l2 = nn.Linear(h, C)
        nn.init.zeros_(s.l2.weight); nn.init.zeros_(s.l2.bias)
    def forward(s, z): return s.l2(s.do(s.act(s.l1(z))))

def prep(X):
    """-> mu,sd,Q(desc eigvecs [C,C]),cumfrac(desc cumulative variance frac), Xtr_s,Xva_s on GPU."""
    N = X.shape[0]; ntr = int(0.85 * N); pm = torch.randperm(N)
    Xtr = X[pm[:ntr]].float().to(device); Xva = X[pm[ntr:]].float().to(device)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    cov = Xtr.t() @ Xtr / Xtr.shape[0]
    lam, Q = torch.linalg.eigh(cov); lam = lam.flip(0).clamp_min(0); Q = Q.flip(1)
    cumfrac = torch.cumsum(lam, 0) / lam.sum()
    return mu.detach(), sd.detach(), Q.detach(), cumfrac.detach(), Xtr, Xva

def fit_corr(Xtr, Xva, Vk):
    k = Vk.shape[1]; Wd = Vk.t().contiguous(); h = min(2 * k, H_CAP)
    corr = Corr(k, h, C).to(device)
    opt = torch.optim.AdamW(corr.parameters(), lr=LR, weight_decay=WD)
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
    corr.eval()
    return corr, h

DEC = {}                       # (l,site) -> {tau:(mu,sd,Vk,corr,k)}
KL = {tau: {} for tau in TAUS}  # tau -> {(l,site):k}
for i, l in enumerate(LAYERS):
    for site, A in (("K", Kact), ("V", Vact)):
        mu, sd, Q, cumfrac, Xtr, Xva = prep(A[l]); A[l] = None
        DEC[(l, site)] = {}
        for tau in TAUS:
            k = int(torch.searchsorted(cumfrac, torch.tensor(tau, device=device)).item()) + 1
            k = max(8, min(C, k)); Vk = Q[:, :k].contiguous()
            corr, h = fit_corr(Xtr, Xva, Vk)
            DEC[(l, site)][tau] = (mu, sd, Vk, corr, k); KL[tau][(l, site)] = k
        del Xtr, Xva, Q; torch.cuda.empty_cache()
    if (i + 1) % 8 == 0: lg(f"  fit {i+1}/{len(LAYERS)} layers   ({time.time()-t0:.0f}s)")
del Kact, Vact; torch.cuda.empty_cache()
lg(f"fit done: {time.time()-t0:.0f}s")

# ---------- Stage C: surgery + perplexity per tau ----------
MODE = ["none"]; CUR = [TAUS[0]]
def surg(l, site):
    def h(m, i, o):
        if MODE[0] == "none": return o
        mu, sd, Vk, corr, k = DEC[(l, site)][CUR[0]]
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
            nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1); ntok += ids.shape[1] - 1
    return math.exp(nll / ntok)

MODE[0] = "none"; ppl_base = perplexity()
lg(""); lg(f"baseline (full K+V) perplexity = {ppl_base:.3f}   full cache = {2*C} floats/token")
lg("-" * 104)
lg(f"{'tau':>5} {'mean_k':>7} {'cacheX':>7} {'dFLOP/tok':>10} {'dFLOP%fwd':>9} | {'ppl_linear':>11} {'ppl_nonlin':>11} | {'Δppl':>8}  verdict")
res = []
for tau in TAUS:
    CUR[0] = tau
    ks = list(KL[tau].values()); mean_k = sum(ks) / len(ks)
    comp_bytes = sum(ks)                         # compressed floats/token summed over layers,sites
    full_bytes = 2 * C * len(LAYERS)             # full K+V floats/token
    cacheX = full_bytes / comp_bytes
    # nonlinear decoder added FLOPs/token (corr over all cached layers,sites)
    dflop = 0.0
    for (l, site), k in KL[tau].items():
        h = min(2 * k, H_CAP); dflop += 2 * (k * h + h * C)
    MODE[0] = "linear"; pl = perplexity()
    MODE[0] = "nonlinear"; pn = perplexity()
    d = pl - pn
    res.append((tau, mean_k, cacheX, dflop, pl, pn, d))
    lg(f"{tau:>5.2f} {mean_k:>7.0f} {cacheX:>6.2f}x {dflop:>10.2e} {100*dflop/fwd_flops_tok:>8.2f}% | "
       f"{pl:>11.3f} {pn:>11.3f} | {d:>+8.3f}  {'NONLINEAR' if d>0 else 'no gain'}")
    flush()
lg("-" * 104)
lg(f"baseline ppl={ppl_base:.3f}.  cacheX = KV-cache compression factor.  dFLOP/tok = nonlinear")
lg("decoder's added decode FLOPs per cached token (the linear up-proj is absorbable -> ~0).")
nwin = sum(1 for *_, d in res if d > 0)
lg(f"VERDICT: nonlinear beats linear at {nwin}/{len(TAUS)} operating points.")
lg("Read the curve: at iso-perplexity, the compression factor where nonlinear stays acceptable")
lg("vs where linear breaks = the headline win.  Next (2b): joint K+V latent vs REAL MLA.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
