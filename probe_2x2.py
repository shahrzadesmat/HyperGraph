"""
probe_2x2.py — WHERE does nonlinearity help in KV compression?
The full 2x2: {linear, nonlinear} encoder  x  {linear, nonlinear} decoder, matched k.

Unified residual-on-PCA autoencoder so all four cells are perfectly comparable and every
cell STARTS EXACTLY AT PCA (frozen Vk = top-k PCA eigenvectors is the shared reference):

    encode:  z   = x @ Vk        + (enc_corr(x) if encoder is nonlinear)     # x standardized
    decode:  xhat = z @ Vk^T     + (dec_corr(z) if decoder is nonlinear)
    corr(.) = Linear(.,h) -> GELU -> Dropout -> Linear(h,.)   with LAST LAYER ZERO-INIT.

Vk is FROZEN (a buffer, never trained). Only the active corr(s) are trained. At init the
corrs output 0, so every cell == PCA exactly; training can only improve held-out error.

Cells:
  (1) lin-enc , lin-dec   = PCA                         (no trainable params; the baseline)
  (2) lin-enc , nonlin-dec= OUR method                  (frozen PCA code, nonlinear decode)
  (3) nonlin-enc, lin-dec = CONTROL                     (provably <= PCA: linear decode is
                                                         confined to the k-dim PCA subspace)
  (4) nonlin-enc, nonlin-dec = FULL autoencoder         (the exploitation upgrade)

For each (site,layer,k) we report held-out reconstruction error of each cell, its gap vs PCA,
the Gaussian-null floor of each cell (covariance-matched surrogate = pure overfitting), the
NULL-CORRECTED gap, and the ENCODER MARGINAL = (err2 - err4)/err2 (what freeing the encoder
adds on top of the nonlinear decoder).

Expected: gap3 ~ 0 (encoder alone cannot beat PCA);  gap2 > 0 (decoder helps);
gap4 >= gap2 (encoder adds a bonus) -> quantifies the encoder-side redundancy you asked about.

  python probe_2x2.py [meta-llama/Llama-2-7b-hf]
"""
import os, sys, math, time, torch
import torch.nn as nn

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
SITES = os.environ.get("P2_SITES", "K,V").split(",")
N_BLOCKS = int(os.environ.get("P2_BLOCKS", "1200")); BLOCK_TOK = 256
FRACS = [float(x) for x in os.environ.get("P2_FRACS", "0.05,0.10").split(",")]
SPLIT = (0.70, 0.15, 0.15)
H_CAP = 1024; DROPOUT = 0.1; WD = 1e-2; LR = 1e-3; MAX_EPOCHS = 200; PATIENCE = 15; MIN_DREL = 1e-4; BS = 4096
SEED = 0
device = torch.device("cuda")
tag = MODEL.split("/")[-1]; OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_2x2_{tag}.txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time(); torch.manual_seed(SEED)

# ---------------- capture K/V (document-tagged), one forward pass ----------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
layers = model.model.layers; Nl = len(layers); LAYERS = [5, Nl // 2, Nl - 6]
C = layers[0].self_attn.k_proj.out_features          # K/V activation dim (handles MHA and GQA)
raw = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
all_ids = tok(raw, return_tensors="pt").input_ids[0]
nb = min(N_BLOCKS, all_ids.shape[0] // BLOCK_TOK)
ids = all_ids[:nb * BLOCK_TOK].view(nb, BLOCK_TOK)
cap = {s: {l: [] for l in LAYERS} for s in SITES}; bids = []; hs = []
def mk(store, l):
    def h(m, i, o): store[l].append(o.detach().reshape(-1, o.shape[-1]).half().cpu())
    return h
for l in LAYERS:
    if "K" in SITES: hs.append(layers[l].self_attn.k_proj.register_forward_hook(mk(cap["K"], l)))
    if "V" in SITES: hs.append(layers[l].self_attn.v_proj.register_forward_hook(mk(cap["V"], l)))
B = 8; b0 = 0
with torch.no_grad():
    for s in range(0, nb, B):
        blk = ids[s:s + B].to(device); bb = blk.shape[0]; model(blk)
        bids.append(torch.arange(b0, b0 + bb).repeat_interleave(BLOCK_TOK)); b0 += bb
for h in hs: h.remove()
BID = torch.cat(bids)
ACTS = {s: {l: torch.cat(cap[s][l], 0) for l in LAYERS} for s in SITES}; cap = None
del model; torch.cuda.empty_cache()
lg(f"probe_2x2  MODEL={MODEL}  C(KV dim)={C}  layers={LAYERS}  sites={SITES}  fracs={FRACS}")
lg(f"capture: {time.time()-t0:.0f}s  rows={BID.shape[0]}  blocks={int(BID.max())+1}")

# ---------------- model pieces ----------------
class Corr(nn.Module):
    def __init__(s, din, h, dout):
        super().__init__(); s.l1 = nn.Linear(din, h); s.act = nn.GELU(); s.do = nn.Dropout(DROPOUT); s.l2 = nn.Linear(h, dout)
        nn.init.zeros_(s.l2.weight); nn.init.zeros_(s.l2.bias)
    def forward(s, x): return s.l2(s.do(s.act(s.l1(x))))

class AE(nn.Module):
    def __init__(s, Vk, h, enc_nl, dec_nl):
        super().__init__()
        Cc, k = Vk.shape
        s.register_buffer("Vk", Vk); s.register_buffer("Wd", Vk.t().contiguous())  # FROZEN
        s.enc = Corr(Cc, h, k) if enc_nl else None
        s.dec = Corr(k, h, Cc) if dec_nl else None
    def forward(s, x):
        z = x @ s.Vk
        if s.enc is not None: z = z + s.enc(x)
        xh = z @ s.Wd
        if s.dec is not None: xh = xh + s.dec(z)
        return xh

def rerr(X, Xh): return float((X - Xh).norm() / X.norm())

def train_eval(Vk, h, enc_nl, dec_nl, Xtr, Xval, Xte):
    ae = AE(Vk, h, enc_nl, dec_nl).to(device)
    params = [p for p in ae.parameters() if p.requires_grad]
    if not params:                                   # PCA cell — no training
        ae.eval()
        with torch.no_grad(): return rerr(Xte, ae(Xte))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    best = math.inf; best_state = None; bad = 0; n = Xtr.shape[0]
    for ep in range(MAX_EPOCHS):
        ae.train(); perm = torch.randperm(n, device=device)
        for b in range(0, n, BS):
            xb = Xtr[perm[b:b + BS]]; loss = ((ae(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); ae.eval()
        with torch.no_grad(): ve = float(((ae(Xval) - Xval) ** 2).mean())
        if ve < best * (1 - MIN_DREL): best = ve; best_state = {kk: v.detach().clone() for kk, v in ae.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    if best_state is not None: ae.load_state_dict(best_state)
    ae.eval()
    with torch.no_grad(): e = rerr(Xte, ae(Xte))
    del ae; torch.cuda.empty_cache(); return e

@torch.no_grad()
def gen_gauss(lam, Q, n):                            # standardized-space Gaussian, cov = Q diag(lam) Q^T
    sl = lam.clamp_min(0).sqrt(); QT = Q.t().contiguous(); out = torch.empty(n, Q.shape[0], device=device)
    for b in range(0, n, 50000):
        m = min(50000, n - b); Z = torch.randn(m, Q.shape[0], device=device); out[b:b + m] = (Z * sl) @ QT
    return out

@torch.no_grad()
def pca_eigh(Xstd):
    cov = Xstd.t() @ Xstd / Xstd.shape[0]
    lam, Q = torch.linalg.eigh(cov); return lam.flip(0).clamp_min(0), Q.flip(1)

# ---------------- run the 2x2 ----------------
lg("=" * 116)
lg(f"{'site':>4} {'lyr':>3} {'k':>5} {'h':>5} | {'PCA':>7} {'lin-nl(2)':>9} {'nl-lin(3)':>9} {'nl-nl(4)':>9} | "
   f"{'Δgap2':>6} {'Δgap3':>6} {'Δgap4':>6} | {'encMARG':>7}")
lg("-" * 116)
for site in SITES:
    for l in LAYERS:
        X = ACTS[site][l]; ACTS[site][l] = None
        # document-level split
        units = torch.unique(BID); g = torch.Generator().manual_seed(12345)
        up = units[torch.randperm(len(units), generator=g)]
        n1 = int(SPLIT[0] * len(up)); n2 = int((SPLIT[0] + SPLIT[1]) * len(up))
        trm = torch.isin(BID, up[:n1]); vam = torch.isin(BID, up[n1:n2]); tem = torch.isin(BID, up[n2:])
        Xtr = X[trm].float().to(device); Xva = X[vam].float().to(device); Xte = X[tem].float().to(device)
        mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
        Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd
        lam, Q = pca_eigh(Xtr)
        # covariance-matched Gaussian null (generated once per site,layer)
        Gtr = gen_gauss(lam, Q, Xtr.shape[0]); Gva = gen_gauss(lam, Q, Xva.shape[0]); Gte = gen_gauss(lam, Q, Xte.shape[0])
        lamG, QG = pca_eigh(Gtr)
        for f in FRACS:
            k = max(4, int(f * C)); h = min(2 * k, H_CAP)
            Vk = Q[:, :k].contiguous(); VkG = QG[:, :k].contiguous()
            # real
            e1 = train_eval(Vk, h, False, False, Xtr, Xva, Xte)
            e2 = train_eval(Vk, h, False, True,  Xtr, Xva, Xte)
            e3 = train_eval(Vk, h, True,  False, Xtr, Xva, Xte)
            e4 = train_eval(Vk, h, True,  True,  Xtr, Xva, Xte)
            # null floor
            n1_ = train_eval(VkG, h, False, False, Gtr, Gva, Gte)
            n2_ = train_eval(VkG, h, False, True,  Gtr, Gva, Gte)
            n3_ = train_eval(VkG, h, True,  False, Gtr, Gva, Gte)
            n4_ = train_eval(VkG, h, True,  True,  Gtr, Gva, Gte)
            gap = lambda e, ep: (ep - e) / ep * 100
            d2 = gap(e2, e1) - gap(n2_, n1_); d3 = gap(e3, e1) - gap(n3_, n1_); d4 = gap(e4, e1) - gap(n4_, n1_)
            encm = (e2 - e4) / e2 * 100
            lg(f"{site:>4} {l:>3} {k:>5} {h:>5} | {e1:>7.4f} {e2:>9.4f} {e3:>9.4f} {e4:>9.4f} | "
               f"{d2:>+5.1f}% {d3:>+5.1f}% {d4:>+5.1f}% | {encm:>+6.1f}%")
            flush()
        del Xtr, Xva, Xte, Gtr, Gva, Gte; torch.cuda.empty_cache()
lg("-" * 116)
lg("Δgap = (gap vs PCA on REAL) - (gap vs PCA on Gaussian null)  [null-corrected, in %].")
lg("encMARG = (err_cell2 - err_cell4)/err_cell2 = reconstruction the NONLINEAR ENCODER adds on")
lg("top of the nonlinear decoder.  Expect: Δgap3~0 (encoder alone can't beat PCA),")
lg("Δgap2>0 (decoder helps), Δgap4>=Δgap2 & encMARG>0 (freeing the encoder adds redundancy).")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
