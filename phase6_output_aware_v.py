"""
phase6_output_aware_v.py — DIRECTION 1: OUTPUT-SPACE value compression.
Values reach the residual only as Y = sum_h (A_h v_h) M_h, where M_h = o_proj's head-h block (W_O).
So a value direction matters in proportion to how much W_O AMPLIFIES it, not how much the value VARIES.
MLA compresses V by value-variance; if W_O nulls some high-variance value directions, that's wasted budget.

In the value-covariance eigenbasis {e_j} (orthonormal), with variance lam_j and output-gain g_j=e_j^T G e_j
(G=M_h M_h^T), the OUTPUT energy of direction j is exactly lam_j*g_j (Cv is diagonal here, so the trace is exact).
  value-PCA (=MLA): keep top-d by  lam_j        (minimizes value reconstruction error)
  output-aware     : keep top-d by  lam_j * g_j  (minimizes OUTPUT error -- optimal in this basis)
Same orthonormal basis, different ranking. Metric1: output-error = sum_dropped(lam*g)/sum(lam*g).
Metric2: WikiText-2 perplexity with V replaced by the rank-d v_hat (K left full). WIN: output-aware < value-PCA.
env: P6_CALIB=60000  P6_DS=32,64,128,256  P6_PPL_CHUNKS=50
"""
import os, math, time, torch
device = torch.device("cuda")
MODEL = os.environ.get("P6_MODEL", "meta-llama/Llama-2-7b-hf")
N_CALIB = int(os.environ.get("P6_CALIB", "60000")); CHUNK = 2048
DS = sorted(int(x) for x in os.environ.get("P6_DS", "32,64,128,256").split(","))
PPL_CHUNKS = int(os.environ.get("P6_PPL_CHUNKS", "50"))
OUT = "/work/hdd/bdjd/hypergraph_pruning/phase6_output_aware_v.txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); cfg = model.config
nh = cfg.num_attention_heads; nkv = getattr(cfg, "num_key_value_heads", nh); dh = cfg.hidden_size // nh
hid = cfg.hidden_size; LAYERS = list(range(NL))
assert nkv == nh, f"MHA assumed (nkv==nh); got {nkv},{nh}"
DS = [d for d in DS if d <= dh]
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]; test_ids = tok(test_txt, return_tensors="pt").input_ids[0]
lg(f"phase6_output_aware_v  MODEL={MODEL}  heads={nh}  head_dim={dh}  layers={NL}  d={DS} (of {dh})")

# ---------- Stage A: per-head value covariance (streaming) ----------
Cv = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}; cnt = {l: 0 for l in LAYERS}; hooks = []
def mkv(l):
    def h(m, i, o):
        v = o.detach().reshape(-1, nh, dh).float()
        Cv[l] += torch.einsum('nhd,nhe->hde', v, v); cnt[l] += v.shape[0]
    return h
for l in LAYERS: hooks.append(attn[l].v_proj.register_forward_hook(mkv(l)))
cb = min(N_CALIB, train_ids.shape[0])
with torch.no_grad():
    for s in range(0, cb - 1, CHUNK): model(train_ids[s:s + CHUNK].unsqueeze(0).to(device))
for h in hooks: h.remove()
for l in LAYERS: Cv[l] /= cnt[l]
lg(f"capture+cov: {time.time()-t0:.0f}s  tokens/layer={cnt[LAYERS[0]]}")

# ---------- Stage B: per head: Cv eigh (value basis) + output-gain g from W_O ----------
EV = {}; LAM = {}; GAIN = {}; gspread = []
for l in LAYERS:
    lam, E = torch.linalg.eigh(Cv[l]); lam = lam.flip(-1).clamp_min(0); E = E.flip(-1)   # [nh,dh],[nh,dh,dh] desc
    Wo = attn[l].o_proj.weight.detach().float()                    # [hid, nh*dh]
    WoT = Wo.t().reshape(nh, dh, hid)                              # [nh, dh, hid] head-major
    G = torch.einsum('hdc,hec->hde', WoT, WoT)                     # G_h = M_h M_h^T  [nh,dh,dh]
    g = torch.einsum('hdj,hde,hej->hj', E, G, E)                   # diag(E^T G E)  [nh,dh] (output gain per dir)
    EV[l] = E; LAM[l] = lam; GAIN[l] = g.clamp_min(0)
    gn = g.clamp_min(1e-9); gspread.append(float((gn.max(1).values / gn.median(1).values).mean()))
    del Wo, WoT, G
lg(f"output-gain spread (max/median of g per head, mean over heads&layers): {sum(gspread)/len(gspread):.2f} "
   f"(>>1 => W_O strongly non-isotropic -> output-aware can differ from value-PCA)")

# ---------- Stage C1: output-error (exact in the value eigenbasis) ----------
def keep_idx(l, scheme, d):                                       # indices of kept Cv-eigendirections per head
    score = LAM[l] if scheme == "value" else LAM[l] * GAIN[l]     # [nh,dh]
    return torch.topk(score, d, dim=1).indices                    # [nh,d]
def out_err(l, scheme, d):                                        # mean over heads of dropped output-energy fraction
    oe = LAM[l] * GAIN[l]; idx = keep_idx(l, scheme, d)
    keep = torch.zeros_like(oe); keep.scatter_(1, idx, 1.0)
    return float(((oe * (1 - keep)).sum(1) / (oe.sum(1) + 1e-9)).mean())
lg("=" * 84)
lg("OUTPUT-ERROR (fraction of W_O-output energy lost; lower=better), mean over heads")
lg(f"{'d':>5} {'comp':>6} | {'value-PCA(=MLA)':>16} {'output-aware':>13} | {'reduction':>10}")
lg("-" * 84)
for d in DS:
    ea = sum(out_err(l, "value", d) for l in LAYERS) / NL
    eb = sum(out_err(l, "output", d) for l in LAYERS) / NL
    red = (ea - eb) / ea * 100 if ea > 0 else 0
    lg(f"{d:>5} {dh/d:>5.1f}x | {ea:>16.4f} {eb:>13.4f} | {red:>+9.1f}%"); flush()
lg("-" * 84)

# ---------- Stage C2: perplexity surgery on v_proj ----------
CURP = {}; MODE = ["none"]
def setP(scheme, d):
    for l in LAYERS:
        idx = keep_idx(l, scheme, d); Ek = torch.gather(EV[l], 2, idx.unsqueeze(1).expand(nh, dh, d))  # [nh,dh,d]
        CURP[l] = Ek @ Ek.transpose(1, 2)                         # projection [nh,dh,dh]
def surg(l):
    def h(m, i, o):
        if MODE[0] == "none" or l not in CURP: return o
        sh = o.shape; v = o.reshape(-1, nh, dh).float()
        vh = torch.einsum('nhd,hde->nhe', v, CURP[l])
        return vh.reshape(sh).to(o.dtype)
    return h
for l in LAYERS: attn[l].v_proj.register_forward_hook(surg(l))
def perplexity():
    nll = 0.0; ntok = 0
    with torch.no_grad():
        for c in range(PPL_CHUNKS):
            ids = test_ids[c * CHUNK:(c + 1) * CHUNK]
            if ids.shape[0] < 2: break
            ids = ids.unsqueeze(0).to(device)
            nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1); ntok += ids.shape[1] - 1
    return math.exp(nll / ntok)
MODE[0] = "none"; pbase = perplexity()
setP("value", dh); MODE[0] = "go"; pfull = perplexity()                # wiring: d=dh => identity => baseline
lg(f"\nbaseline ppl={pbase:.3f}   WIRING: value@d={dh} ppl={pfull:.3f} -> {'OK' if abs(pfull-pbase)<0.03 else 'MISMATCH!!'}")
lg(f"{'d':>5} {'comp':>6} | {'ppl value-PCA':>14} {'ppl output-aware':>16} | {'Δppl(val-out)':>13}  verdict")
lg("-" * 84)
res = []
for d in DS:
    setP("value", d); pa = perplexity()
    setP("output", d); pb = perplexity()
    res.append((d, pa, pb)); lg(f"{d:>5} {dh/d:>5.1f}x | {pa:>14.3f} {pb:>16.3f} | {pa-pb:>+13.3f}  {'OUTPUT-AWARE WINS' if pb<pa-1e-3 else 'no gain'}"); flush()
lg("-" * 84)
nwin = sum(1 for d, pa, pb in res if pb < pa - 1e-3)
lg(f"VERDICT: output-aware beats value-PCA(=MLA) on perplexity at {nwin}/{len(DS)} budgets.")
lg("If yes -> W_O makes value-variance the wrong objective; output-space V compression is a real lever MLA misses.")
lg("Scoped to V (K full); pre-attention surgery; output-gain ignores attention averaging (perplexity is exact).")
lg(f"total: {time.time()-t0:.0f}s"); flush(); print("Saved:", OUT)
