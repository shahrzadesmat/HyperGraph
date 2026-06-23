"""
phase10_nonlinear_attn.py — TIER-2: the empty cell of the 2x2.

  { linear , nonlinear }  x  { Euclidean metric ||Δk||² , attention metric Δkᵀ Σ_q Δk }

  linear + Euclidean    = PCA / MLA / key-PCA
  linear + attention    = KQ-SVD  (= our phase9 attnAware)            [arXiv 2512.05916]
  nonlinear + Euclidean = KV-CAR / phase1-2b "ours"                   (trained on ||Δk||²)
  nonlinear + attention = ??? NOBODY                                  <-- THIS PROBE

Question: once the codec is already in the RIGHT inner product (Σ_q), does a NONLINEAR residual
on the latent buy anything the linear attention-aware codec (KQ-SVD/phase9) doesn't?  The prior
"nonlinear curvature is too small" verdict was measured under EUCLIDEAN MSE — the wrong metric.
Re-measure curvature in the Σ_q geometry the model actually reads.

Per head (K only, V left full — clean first read), at matched rank r:
  base (linear):  k̂ = D z,        z = Eᵀ k,   E = Σ_q^{1/2} W,  D = Σ_q^{-1/2} W,  W=eigvecs(Σq^½ Ck Σq^½)
  nonlinear:      k̂ = D z + corr(z)   (per-head Linear(r→h)->GELU->Linear(h→dh), zero-init -> starts at base)
  corrW trained on the WHITENED loss  ||Σ_q^{1/2}(k-k̂)||²   (= logit error, the new cell)
  corrE trained on the EUCLIDEAN loss ||k-k̂||²              (contrast: nonlinearity under the wrong metric)

TWO read-outs:
  (1) DIAGNOSTIC (recon space, with Gaussian null): held-out relative LOGIT error
        keyPCA(lin) | attnAware(lin) | +corrE | +corrW | null-floor(+corrW on covariance-matched Gaussian)
      WIN for the cell = corrW << attnAware AND the gain survives the Gaussian null (Δgap>0).
  (2) DOWNSTREAM: WikiText-2 perplexity (real RoPE+softmax) for
        keyPCA | attnAware | attnAware+corrW | attnAware+corrE
      WIN = attnAware+corrW ppl < attnAware ppl  -> nonlinearity pays IN the right metric.

If BOTH say no -> the empty cell is empty for a reason: the objective is the entire lever, nonlinearity
is a red herring even in the Σ_q metric (a clean negative that closes the 2x2). If yes -> novel positive.

env: P10_CALIB=60000 P10_PPL_CHUNKS=60 P10_RANKS=16,32 KV_RCOND=1e-3 P10_NULL=1
  python phase10_nonlinear_attn.py
"""
import os, math, time, torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")
MODEL = os.environ.get("KV_MODEL", "meta-llama/Llama-2-7b-hf")
N_CALIB = int(os.environ.get("P10_CALIB", "60000"))
PPL_CHUNKS = int(os.environ.get("P10_PPL_CHUNKS", "60")); CHUNK = 2048
RANKS = [int(x) for x in os.environ.get("P10_RANKS", "16,32").split(",")]
RCOND = float(os.environ.get("KV_RCOND", "1e-3"))
DO_NULL = os.environ.get("P10_NULL", "1") == "1"
# corr training
H_CAP = 256; LR = 1e-3; WD = 1e-2; BS = 16384; MAX_EPOCHS = 60; PATIENCE = 8; MIN_DREL = 1e-4
OUT = os.environ.get("KV_OUT", "phase10_nonlinear_attn.txt")
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); cfg = model.config
nh = cfg.num_attention_heads; nkv = getattr(cfg, "num_key_value_heads", nh); dh = cfg.hidden_size // nh
LAYERS = list(range(NL))
assert nkv == nh, f"MHA assumed (nkv==nh); got nkv={nkv}, nh={nh}"
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
test_ids = tok(test_txt, return_tensors="pt").input_ids[0]
lg(f"phase10_nonlinear_attn  MODEL={MODEL}  heads={nh}  head_dim={dh}  layers={NL}  ranks={RANKS}  rcond={RCOND}  null={DO_NULL}")

# ---------- Stage A: capture raw per-head K (CPU) + streaming Cq ----------
capK = {l: [] for l in LAYERS}
Cq = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}; cnt = {l: 0 for l in LAYERS}; hooks = []
def mk_k(l):
    def h(m, i, o): capK[l].append(o.detach().reshape(-1, nkv, dh).half().cpu())
    return h
def mk_q(l):
    def h(m, i, o):
        q = o.detach().reshape(-1, nh, dh).float(); Cq[l] += torch.einsum('nhd,nhe->hde', q, q); cnt[l] += q.shape[0]
    return h
for l in LAYERS:
    hooks.append(attn[l].k_proj.register_forward_hook(mk_k(l)))
    hooks.append(attn[l].q_proj.register_forward_hook(mk_q(l)))
cb = min(N_CALIB, train_ids.shape[0])
with torch.no_grad():
    for s in range(0, cb - 1, CHUNK):
        model(train_ids[s:s + CHUNK].unsqueeze(0).to(device))
for h in hooks: h.remove()
for l in LAYERS: Cq[l] /= cnt[l]
Kact = {l: torch.cat(capK[l], 0) for l in LAYERS}; capK = None
lg(f"capture: {time.time()-t0:.0f}s   tokens/layer={Kact[LAYERS[0]].shape[0]}")

# ---------- helpers ----------
def eigvecs_desc(C): lam, U = torch.linalg.eigh(C); return U.flip(-1)
def whiten_pair(Cql):
    lam, V = torch.linalg.eigh(Cql); lam = lam.clamp_min(0)
    floor = RCOND * lam.amax(-1, keepdim=True).clamp_min(1e-12); lam = lam.clamp_min(floor)
    sq = lam.sqrt(); isq = sq.reciprocal()
    A = torch.einsum('hdj,hj,hfj->hdf', V, sq, V); Ainv = torch.einsum('hdj,hj,hfj->hdf', V, isq, V)
    return A, Ainv

class HeadCorr(nn.Module):                                   # independent tiny MLP per head, batched
    def __init__(s, nh, r, h, dh):
        super().__init__()
        s.W1 = nn.Parameter(torch.randn(nh, r, h) / max(1, r) ** 0.5); s.b1 = nn.Parameter(torch.zeros(nh, h))
        s.W2 = nn.Parameter(torch.zeros(nh, h, dh)); s.b2 = nn.Parameter(torch.zeros(nh, dh))   # zero-init -> starts at base
    def forward(s, z):                                       # z [N,nh,r] -> [N,nh,dh]
        a = F.gelu(torch.einsum('nhr,hrm->nhm', z, s.W1) + s.b1)
        return torch.einsum('nhm,hmd->nhd', a, s.W2) + s.b2

def relW(K, base_corr, A):                                   # held-out relative LOGIT error sqrt(Σ||A Δk||²/Σ||A k||²)
    num = den = 0.0
    with torch.no_grad():
        for b in range(0, K.shape[0], BS):
            k = K[b:b+BS].to(device).float(); kh = base_corr(k)
            num += (torch.einsum('hde,nhe->nhd', A, k - kh) ** 2).sum().item()
            den += (torch.einsum('hde,nhe->nhd', A, k) ** 2).sum().item()
    return math.sqrt(num / den)

def fit_corr(Ktr, Kva, E, D, A, r, metric):                 # metric: 'W' (whitened) or 'E' (euclidean)
    h = min(2 * r, H_CAP); corr = HeadCorr(nh, r, h, dh).to(device)
    opt = torch.optim.AdamW(corr.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    n = Ktr.shape[0]; best = math.inf; best_sd = None; bad = 0
    def loss_of(k):
        z = torch.einsum('nhd,hdr->nhr', k, E); kh = torch.einsum('nhr,hdr->nhd', z, D) + corr(z)
        r_ = k - kh
        return ((torch.einsum('hde,nhe->nhd', A, r_) ** 2).mean() if metric == 'W' else (r_ ** 2).mean())
    for ep in range(MAX_EPOCHS):
        corr.train(); perm = torch.randperm(n)              # CPU perm: Ktr is CPU (indexed on CPU, batch moved to GPU)
        for b in range(0, n, BS):
            k = Ktr[perm[b:b+BS]].to(device).float()
            loss = loss_of(k); opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); corr.eval()
        with torch.no_grad():
            ve = 0.0
            for b in range(0, Kva.shape[0], BS):
                ve += loss_of(Kva[b:b+BS].to(device).float()).item() * min(BS, Kva.shape[0]-b)
            ve /= Kva.shape[0]
        if ve < best * (1 - MIN_DREL): best = ve; best_sd = {k: v.detach().clone() for k, v in corr.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    if best_sd is not None: corr.load_state_dict(best_sd)
    corr.eval(); return corr

def gauss_null(Ck, n):                                       # per-head Gaussian with 2nd moment ~Ck via eigh
    lam, V = torch.linalg.eigh(Ck)                           # robust to PSD/rank-deficiency (no Cholesky)
    sl = lam.clamp_min(0).sqrt()                             # [nh,dh]
    out = torch.empty(n, nh, dh, dtype=torch.float16)
    for b in range(0, n, 20000):
        m = min(20000, n - b); Z = torch.randn(m, nh, dh, device=device)
        out[b:b+m] = torch.einsum('nhj,hj,hdj->nhd', Z, sl, V).half().cpu()   # Cov = V diag(lam) V^T = Ck
    return out

# ---------- Stage B: per-layer bases + corr fits + diagnostic ----------
DEC = {l: {} for l in LAYERS}                                 # l -> {'Uk','E','D','A', r:{'corrW','corrE'}}
diag = {r: {"keyPCA": [], "attn": [], "corrE": [], "corrW": [], "null_lin": [], "null_corrW": []} for r in RANKS}
for i, l in enumerate(LAYERS):
    K = Kact[l].float().to(device); N = K.shape[0]; ntr = int(0.85 * N); pm = torch.randperm(N, device=device)
    Ck = torch.einsum('nhd,nhe->hde', K[pm[:ntr]], K[pm[:ntr]]) / ntr
    Uk = eigvecs_desc(Ck); A, Ainv = whiten_pair(Cq[l]); W = eigvecs_desc(A @ Ck @ A)
    Efull = (A @ W).contiguous(); Dfull = (Ainv @ W).contiguous()
    DEC[l] = {"Uk": Uk.cpu(), "E": Efull.cpu(), "D": Dfull.cpu(), "A": A.cpu()}
    Ktr = K[pm[:ntr]].half().cpu(); Kva = K[pm[ntr:]].half().cpu()
    if DO_NULL: Knull = gauss_null(Ck, N); Ntr2 = int(0.85 * N); Ntr_n = Knull[:Ntr2]; Nva_n = Knull[Ntr2:]
    for r in RANKS:
        E = Efull[:, :, :r].contiguous(); D = Dfull[:, :, :r].contiguous(); Uk_r = Uk[:, :, :r].contiguous()
        corrW = fit_corr(Ktr, Kva, E, D, A, r, 'W'); corrE = fit_corr(Ktr, Kva, E, D, A, r, 'E')
        # diagnostic (held-out val), all in the LOGIT metric -- BEFORE moving corr to CPU (.cpu() is in-place for modules)
        lin = lambda k: torch.einsum('nhr,hdr->nhd', torch.einsum('nhd,hdr->nhr', k, E), D)
        kp  = lambda k: torch.einsum('nhr,hdr->nhd', torch.einsum('nhd,hdr->nhr', k, Uk_r), Uk_r)
        diag[r]["keyPCA"].append(relW(Kva, kp, A))
        diag[r]["attn"].append(relW(Kva, lin, A))
        diag[r]["corrE"].append(relW(Kva, lambda k: lin(k) + corrE(torch.einsum('nhd,hdr->nhr', k, E)), A))
        diag[r]["corrW"].append(relW(Kva, lambda k: lin(k) + corrW(torch.einsum('nhd,hdr->nhr', k, E)), A))
        if DO_NULL:
            cW0 = fit_corr(Ntr_n, Nva_n, E, D, A, r, 'W')
            diag[r]["null_lin"].append(relW(Nva_n, lin, A))
            diag[r]["null_corrW"].append(relW(Nva_n, lambda k: lin(k) + cW0(torch.einsum('nhd,hdr->nhr', k, E)), A))
            del cW0
        DEC[l][r] = {"corrW": corrW.cpu(), "corrE": corrE.cpu()}   # store CPU copies AFTER the GPU diagnostic
    Kact[l] = None; del K, Ck, A, Ainv, W, Efull, Dfull, Uk, Ktr, Kva
    if DO_NULL: del Knull
    torch.cuda.empty_cache()
    if (i + 1) % 8 == 0: lg(f"  fit {i+1}/{NL} layers   ({time.time()-t0:.0f}s)")
del Kact; torch.cuda.empty_cache()
lg(f"fit done: {time.time()-t0:.0f}s")

lg("=" * 104)
lg("DIAGNOSTIC — held-out relative LOGIT error (lower=better), mean over layers.  corrW is the NEW CELL.")
lg(f"{'rank':>5} | {'keyPCA(lin)':>11} {'attn(lin)':>10} {'+corrE':>8} {'+corrW':>8} | {'nl-gain':>8} {'Δgap-null':>10}  verdict")
lg("-" * 104)
for r in RANKS:
    m = {k: (sum(v)/len(v) if v else float('nan')) for k, v in diag[r].items()}
    nlgain = m["attn"] - m["corrW"]                          # how much the whitened residual cuts logit error
    dgap = float('nan')
    if DO_NULL: dgap = nlgain - (m["null_lin"] - m["null_corrW"])   # real nl-gain minus overfitting floor
    ok = (DO_NULL and dgap > 0.002) or (not DO_NULL and nlgain > 0.002)
    lg(f"{r:>5} | {m['keyPCA']:>11.4f} {m['attn']:>10.4f} {m['corrE']:>8.4f} {m['corrW']:>8.4f} | "
       f"{nlgain:>+8.4f} {dgap:>+10.4f}  {'NONLINEAR HELPS' if ok else 'no real gain'}")
    flush()
lg("-" * 104)
lg("nl-gain = attn(lin) − +corrW (logit-error cut by the whitened residual).  Δgap-null subtracts the")
lg("Gaussian-null floor (residual gain on covariance-matched Gaussian = pure overfitting).  Δgap>0 => real.")

# ---------- Stage C: perplexity surgery (real RoPE+softmax) ----------
GDEC = {}; MODE = ["none"]; R = [RANKS[0]]
def surg(l):
    def h(m, i, o):
        if MODE[0] == "none" or l not in GDEC: return o
        sh = o.shape; k = o.reshape(-1, nkv, dh).float(); g = GDEC[l]; r = R[0]
        if MODE[0] == "keyPCA":
            U = g["Uk"][:, :, :r]; z = torch.einsum('nhd,hdr->nhr', k, U); kh = torch.einsum('nhr,hdr->nhd', z, U)
        else:
            E = g["E"][:, :, :r]; D = g["D"][:, :, :r]; z = torch.einsum('nhd,hdr->nhr', k, E)
            kh = torch.einsum('nhr,hdr->nhd', z, D)
            if MODE[0] == "corrW": kh = kh + g[r]["corrW"](z)
            elif MODE[0] == "corrE": kh = kh + g[r]["corrE"](z)
        return kh.reshape(sh).to(o.dtype)
    return h
for l in LAYERS: attn[l].k_proj.register_forward_hook(surg(l))
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
lg(""); lg(f"baseline (full K) perplexity = {ppl_base:.3f}")
lg(f"{'rank':>5} {'cacheX':>7} | {'keyPCA':>8} {'attnAware':>10} {'+corrW(new)':>12} {'+corrE':>8} | {'Δ(attn-corrW)':>13}  verdict")
lg("-" * 104)
for r in RANKS:
    R[0] = r; GDEC.clear()
    for l in LAYERS:
        g = DEC[l]; GDEC[l] = {"Uk": g["Uk"].to(device), "E": g["E"].to(device), "D": g["D"].to(device),
                                r: {"corrW": g[r]["corrW"].to(device), "corrE": g[r]["corrE"].to(device)}}
    MODE[0] = "keyPCA"; pk = perplexity()
    MODE[0] = "attn"; pa = perplexity()
    MODE[0] = "corrW"; pw = perplexity()
    MODE[0] = "corrE"; pe = perplexity()
    GDEC.clear(); torch.cuda.empty_cache()
    d = pa - pw
    lg(f"{r:>5} {dh/r:>6.1f}x | {pk:>8.3f} {pa:>10.3f} {pw:>12.3f} {pe:>8.3f} | {d:>+13.3f}  "
       f"{'NL HELPS IN Σq' if d > 1e-3 else 'no gain'}")
    flush()
lg("-" * 104)
lg("WIN for the empty cell = +corrW < attnAware (nonlinearity pays once the metric is Σq).")
lg("If +corrW ≈ attnAware AND diagnostic Δgap≈0 -> the objective is the whole lever; nonlinearity is a")
lg("red herring even in the right metric -> a clean negative that closes the 2x2 (linear×nonlinear)×(Euclid×Σq).")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
