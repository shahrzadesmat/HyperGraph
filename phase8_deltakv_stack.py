"""
phase8_deltakv_stack.py — EXPERIMENT (b): does TEMPORAL INNOVATION coding (DeltaKV) STACK on top
of the nonlinear-MLA latent? i.e. at a matched bits/token budget, does caching the innovation
e_t = z_t - predict(z_{t-1}) beat caching the per-token latent z_t directly?

Setup (per layer, reuses phase2b's joint-latent machinery):
  joint latent  z = x·W_z - bz   (d_c dims, folded from k_proj/v_proj + joint PCA of [K;V])
  decode        [K;V] = z·Vkᵀ + corr(z)        (the SAME nonlinear decoder, untouched)
  temporal AR   ẑ_t = z_{t-1}·P + c            (LINEAR ridge on calibration latent sequences;
                                                phase4 said the temporal predictor is linear)
Codec at matched B bits/dim:
  PER-TOKEN : cache Q_B(z_t)                       -> z_dec = Q_B(z_t)
  DELTA     : cache Q_B(e_t), e_t = z_t - ẑ_t      -> z_dec = ẑ_t + Q_B(e_t)   (open-loop v1)
Because Var(e) < Var(z) (z is ~35% temporally predictable, phase5), the innovation quantizes
finer at the same B -> lower distortion -> lower perplexity at equal bits.

METRICS
  1) diagnostic: latent temporal FVE (real vs shuffled-NULL) + predicted bits/dim saved.
  2) downstream: WikiText-2 ppl vs bits/token for PER-TOKEN vs DELTA vs DELTA-NULL.
WIN: DELTA ppl < PER-TOKEN ppl at matched bits, AND the gain VANISHES under the null
     (shuffled predictor) -> temporal stacking is a real free lever on nonlinear MLA.
CAVEAT (v1): open-loop (predicts from TRUE z_{t-1}); if it wins, build the closed-loop codec.

env: KV_TAU=0.9  KV_CALIB=60000  KV_PPL_CHUNKS=40  KV_BITS=2,3,4,6,16  KV_CLIP=4.0  KV_MODEL=...
"""
import os, math, time, torch
import torch.nn as nn

device = torch.device("cuda")
MODEL = os.environ.get("KV_MODEL", "meta-llama/Llama-2-7b-hf")
TAU = float(os.environ.get("KV_TAU", "0.9"))
N_CALIB = int(os.environ.get("KV_CALIB", "60000"))
PPL_CHUNKS = int(os.environ.get("KV_PPL_CHUNKS", "40")); CHUNK = 2048
BITS = [int(x) for x in os.environ.get("KV_BITS", "2,3,4,6,16").split(",")]
CLIP = float(os.environ.get("KV_CLIP", "4.0"))          # quantizer clips at ±CLIP·std per dim
H_CAP = 1024; DROPOUT = 0.1; WD = 1e-2; LR = 1e-3; MAX_EPOCHS = 200; PATIENCE = 15; MIN_DREL = 1e-4; BS = 8192
RIDGE = 1e-2                                             # temporal-predictor ridge (relative)
OUT = "/work/hdd/bfxa/dshah13/HyperGraph/results/phase8_deltakv_stack" + os.environ.get("KV_TAG", "") + ".txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); C = model.config.hidden_size; C2 = 2 * C; LAYERS = list(range(NL))
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
test_ids = tok(test_txt, return_tensors="pt").input_ids[0]
lg(f"phase8_deltakv_stack  MODEL={MODEL}  C={C}  layers={NL}  tau={TAU}  bits={BITS}  clip={CLIP}")

# ---------- Stage A: capture x, K, V PER CHUNK (order preserved for the temporal predictor) ----------
capX = {l: [] for l in LAYERS}; capK = {l: [] for l in LAYERS}; capV = {l: [] for l in LAYERS}; hooks = []
def mkK(l):
    def h(m, i, o):
        capX[l].append(i[0].detach()[0].half().cpu())          # [T, C] hidden state (k_proj input)
        capK[l].append(o.detach()[0].half().cpu())              # [T, C] key
    return h
def mkV(l):
    def h(m, i, o): capV[l].append(o.detach()[0].half().cpu())  # [T, C] value
    return h
for l in LAYERS:
    hooks.append(attn[l].k_proj.register_forward_hook(mkK(l)))
    hooks.append(attn[l].v_proj.register_forward_hook(mkV(l)))
nch = max(1, min(N_CALIB // CHUNK, (train_ids.shape[0] - 1) // CHUNK))
with torch.no_grad():
    for c in range(nch):
        ids = train_ids[c * CHUNK:(c + 1) * CHUNK]
        if ids.shape[0] < 4: break
        model(ids.unsqueeze(0).to(device))
for h in hooks: h.remove()
lg(f"capture: {time.time()-t0:.0f}s  chunks={len(capX[LAYERS[0]])}")

# ---------- nonlinear decoder ----------
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
    corr.eval(); return corr

# ---------- temporal predictor (linear ridge, per-chunk pairs) ----------
def fit_predictor(zchunks, shuffle=False):
    """zchunks: list of [T,dc] latent sequences. Fit ẑ_t = z_{t-1}P + c. Return P,c,sd_e,FVE_real."""
    Xs = [z[:-1] for z in zchunks if z.shape[0] > 1]
    Ys = [z[1:] for z in zchunks if z.shape[0] > 1]
    X = torch.cat(Xs).to(device); Y = torch.cat(Ys).to(device)
    n = X.shape[0]; ntr = int(0.85 * n)
    if shuffle:                                             # NULL: break the temporal pairing
        X = X[torch.randperm(n, device=device)]
    Xtr, Ytr, Xva, Yva = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    mx = Xtr.mean(0, keepdim=True); my = Ytr.mean(0, keepdim=True)
    Xc = Xtr - mx; Yc = Ytr - my
    lam = RIDGE * (Xc * Xc).mean() * Xc.shape[1]
    P = torch.linalg.solve(Xc.t() @ Xc + lam * torch.eye(Xc.shape[1], device=device), Xc.t() @ Yc)
    c = (my - mx @ P).squeeze(0)
    predva = Xva @ P + c
    fve = float(1 - ((Yva - predva) ** 2).sum() / ((Yva - my) ** 2).sum())
    e = Y - (X @ P + c); sd_e = e.std(0) + 1e-6           # innovation std per dim (on real pairing)
    del X, Y, Xc, Yc; torch.cuda.empty_cache()
    return P.contiguous(), c.contiguous(), sd_e, fve

# ---------- Stage B: per-layer codec + predictor ----------
DEC = {}; diag = []
for l in LAYERS:
    Kc = torch.cat(capK[l]).float().to(device); Vc = torch.cat(capV[l]).float().to(device)
    KV = torch.cat([Kc, Vc], 1)                            # [N, 2C]
    N = KV.shape[0]; ntr = int(0.85 * N); pm = torch.randperm(N, device=device)
    mu = KV[pm[:ntr]].mean(0, keepdim=True); sd = KV[pm[:ntr]].std(0, keepdim=True) + 1e-6
    KVs = (KV - mu) / sd
    cov = KVs[pm[:ntr]].t() @ KVs[pm[:ntr]] / ntr
    lam, Q = torch.linalg.eigh(cov); lam = lam.flip(0).clamp_min(0); Q = Q.flip(1)
    cumfrac = torch.cumsum(lam, 0) / lam.sum()
    dc = int(torch.searchsorted(cumfrac, torch.tensor(TAU, device=device)).item()) + 1
    dc = max(8, min(C2, dc)); Vk = Q[:, :dc].contiguous()
    corr = fit_corr(KVs[pm[:ntr]], KVs[pm[ntr:]], Vk, C2)
    # fold z = x·W_z - bz
    M_K = attn[l].k_proj.weight.detach().float().t(); M_V = attn[l].v_proj.weight.detach().float().t()
    W_KV = torch.cat([M_K, M_V], 1)                        # [C, 2C]
    A = Vk / sd.squeeze(0).unsqueeze(1)                    # [2C, dc]
    W_z = (W_KV @ A).contiguous(); bz = (mu @ A).squeeze(0).contiguous()
    # latent sequences per chunk (from the captured hidden state x), for the predictor
    zchunks = [ (xc.float().to(device) @ W_z - bz) for xc in capX[l] ]
    sd_z = torch.cat(zchunks).std(0) + 1e-6               # per-dim latent std (for per-token quantizer)
    P, cpred, sd_e, fve_real = fit_predictor(zchunks)
    Pn, cn, sd_en, fve_null = fit_predictor(zchunks, shuffle=True)
    # predicted bits/dim saved at matched distortion (Gaussian): ½ log2(var_z/var_e)
    bits_saved = float((0.5 * torch.log2((sd_z ** 2) / (sd_e ** 2)).clamp_min(0)).mean())
    DEC[l] = dict(Vk=Vk, corr=corr, W_z=W_z, bz=bz, mu=mu, sd=sd, dc=dc,
                  P=P, c=cpred, sd_z=sd_z, sd_e=sd_e,
                  P_null=Pn, c_null=cn, sd_e_null=sd_en)
    diag.append((l, dc, fve_real, fve_null, fve_real - fve_null, bits_saved))
    capK[l] = capV[l] = None  # keep capX[l] for nothing else; free
    del Kc, Vc, KV, KVs, cov, Q, zchunks; torch.cuda.empty_cache()
    if (l + 1) % 8 == 0: lg(f"  fit {l+1}/{NL}  ({time.time()-t0:.0f}s)")
del capK, capV, capX; torch.cuda.empty_cache()
lg(f"fit done: {time.time()-t0:.0f}s")

# ---------- diagnostic table ----------
lg("=" * 88)
lg("DIAGNOSTIC: temporal predictability OF THE LATENT z (real vs shuffled-null), per layer")
lg(f"{'layer':>5} {'d_c':>5} | {'FVE_real':>9} {'FVE_null':>9} {'headroom':>9} | {'bits/dim saved':>14}")
lg("-" * 88)
for l, dc, fr, fn, hd, bs in diag:
    lg(f"{l:>5} {dc:>5} | {fr:>9.4f} {fn:>9.4f} {hd:>+9.4f} | {bs:>14.3f}")
mh = sum(h for *_, h, _ in diag) / len(diag); mb = sum(b for *_, b in diag) / len(diag)
lg("-" * 88)
lg(f"mean temporal headroom (real-null) = {mh:+.4f}   |   mean bits/dim saved = {mb:.3f}")
lg("headroom>>0 => latent IS predictable -> DeltaKV has room.  bits/dim saved = the free budget.")
flush()

# ---------- quantizer ----------
def quant(v, std, B):                                     # uniform mid-rise, clip ±CLIP·std per dim
    if B >= 16: return v
    step = (2 * CLIP * std) / (2 ** B)
    q = torch.round(v / step).clamp(-(2 ** (B - 1)), 2 ** (B - 1) - 1)
    return q * step

# ---------- Stage C: surgery (open-loop) + perplexity ----------
MODE = ["none", 16]                                       # (scheme, bits); scheme in none/pertoken/delta/deltanull
def decode_latent(l, z):
    d = DEC[l]; scheme, B = MODE
    if scheme == "pertoken":
        return quant(z, d["sd_z"], B)
    P, c, sde = (d["P_null"], d["c_null"], d["sd_e_null"]) if scheme == "deltanull" else (d["P"], d["c"], d["sd_e"])
    # delta / deltanull: predict from previous token (open loop = true z_{t-1})
    zprev = torch.roll(z, 1, dims=0); zprev[0] = 0
    pred = zprev @ P + c; pred[0] = 0                      # first token: no predictor, e=z
    e = z - pred
    return pred + quant(e, sde, B)
def surg(l, half):
    sl = slice(0, C) if half == 0 else slice(C, C2)
    d = DEC[l]
    def h(m, i, o):
        if MODE[0] == "none": return o
        sh = o.shape; x = i[0].reshape(-1, C).float()
        z = x @ d["W_z"] - d["bz"]
        zdec = decode_latent(l, z)
        rec = zdec @ d["Vk"][sl].t() + d["corr"](zdec)[:, sl]
        return (rec * d["sd"][:, sl] + d["mu"][:, sl]).to(o.dtype).reshape(sh)
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
MODE[:] = ["pertoken", 16]; ppl_lossless = perplexity()   # wiring: B=16 ~ unquantized nonlinear MLA
lg(""); lg(f"baseline (full K+V) ppl = {ppl_base:.3f}   |   nonlinear-MLA @ ~lossless(B16) ppl = {ppl_lossless:.3f}")
dc_mean = sum(d['dc'] for d in DEC.values()) / NL
lg(f"mean d_c = {dc_mean:.0f}   (cache = d_c·B bits/token; bytesX vs fp16 full = {2*C2}/(d_c·B/16))")
lg("=" * 96)
lg(f"{'bits':>4} {'bits/tok':>9} | {'ppl per-token':>13} {'ppl DELTA':>10} {'ppl DELTA-null':>14} | {'Δ(pt-delta)':>11}  verdict")
lg("-" * 96)
# build a null predictor set once (shuffled-fit) by swapping P,c,sd_e -> approximated by re-marking scheme
res = []
for B in BITS:
    MODE[:] = ["pertoken", B]; ppt = perplexity()
    MODE[:] = ["delta", B]; pdl = perplexity()
    MODE[:] = ["deltanull", B]; pdn = perplexity()        # uses same P; see note below
    bits_tok = dc_mean * B
    res.append((B, ppt, pdl, pdn)); d = ppt - pdl
    lg(f"{B:>4} {bits_tok:>9.0f} | {ppt:>13.3f} {pdl:>10.3f} {pdn:>14.3f} | {d:>+11.3f}  "
       f"{'DELTA WINS' if d > 1e-3 else 'no gain'}")
    flush()
lg("-" * 96)
nwin = sum(1 for B, ppt, pdl, pdn in res if ppt - pdl > 1e-3 and B < 16)
lg(f"VERDICT: DELTA beats per-token at {nwin}/{sum(1 for B in BITS if B<16)} quantized budgets.")
lg("WIN needs: DELTA < per-token at matched bits AND the gain absent under DELTA-null (shuffled predictor).")
lg("If yes -> temporal stacking is a real free lever on nonlinear MLA -> build closed-loop codec + RD curve.")
lg("If no  -> per-token nonlinear MLA already captures the temporal redundancy (like gate1's block result).")
lg("v1 caveat: OPEN-LOOP (predicts from true z_{t-1}); closed-loop is the faithful follow-up.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
