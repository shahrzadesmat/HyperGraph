"""
phase12_joint_kv.py — make the Fisher result USABLE: compress BOTH K and V at a matched TOTAL budget.

phase11 (K-only) showed the 2nd-order output metric M̄ beats KQ-SVD's Σ_q for the KEY basis, but
"16x on K" is only ~1.9x on the total cache (V left full). Here we compress K AND V per head at rank r
(total cacheX = head_dim/r counting both), and compare three full stacks:

  mla    : keyPCA-K  + valuePCA-V       (Euclidean both;  MLA / Palu-style)
  kqsvd  : attnAware-K(Σ_q) + outV-V(G) (1st-order logit K + output-aware V; = KQ-SVD's K & V)
  fisher : fisher-K(M̄)     + outV-V(G) (2nd-order output K + output-aware V; THIS WORK)

K metrics (per head):  Σ_q = Σ_m q qᵀ ;  M̄ = Σ_m c_m q qᵀ ,  c_m = Σ_n a_mn²‖W_O(v_n−ō_m)‖².
V metric  (per head):  G = W_OᵀW_O  (a value error reaches the output through W_O).
All via the SAME whitened generalized eigenproblem (whiten by metric^{1/2}, PCA, un-whiten).

WIN: the fisher stack stays usable (near-baseline ppl) at a HIGHER total cacheX than mla/kqsvd
-> the 2nd-order objective extends the usable-compression frontier on the FULL cache.

Needs real attention weights -> attn_implementation="eager", output_attentions=True (calibration only).
env: P12_CALIB_SEQS=80 P12_SEQLEN=512 P12_BATCH=4 P12_PPL_CHUNKS=60 P12_RANKS=8,16,32,64 KV_RCOND=1e-3
  python phase12_joint_kv.py
"""
import os, math, time, torch
device = torch.device("cuda")
MODEL = os.environ.get("KV_MODEL", "meta-llama/Llama-2-7b-hf")
N_SEQS = int(os.environ.get("P12_CALIB_SEQS", "80"))
SEQLEN = int(os.environ.get("P12_SEQLEN", "512"))
BATCH = int(os.environ.get("P12_BATCH", "4"))
PPL_CHUNKS = int(os.environ.get("P12_PPL_CHUNKS", "60")); CHUNK = 2048
RANKS = [int(x) for x in os.environ.get("P12_RANKS", "8,16,32,64").split(",")]
RCOND = float(os.environ.get("KV_RCOND", "1e-3"))
OUT = os.environ.get("KV_OUT", "phase12_joint_kv.txt")
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             attn_implementation="eager").eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); cfg = model.config
nh = cfg.num_attention_heads; nkv = getattr(cfg, "num_key_value_heads", nh); dh = cfg.hidden_size // nh
hid = cfg.hidden_size; LAYERS = list(range(NL))
assert nkv == nh, f"MHA assumed (nkv==nh); got {nkv},{nh}  (GQA needs head-grouping)"
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]; test_ids = tok(test_txt, return_tensors="pt").input_ids[0]
lg(f"phase12_joint_kv  MODEL={MODEL}  heads={nh}  head_dim={dh}  layers={NL}  ranks={RANKS}")
lg(f"calib: {N_SEQS} seqs x {SEQLEN} tok (batch {BATCH});  total cacheX = {dh}/r (BOTH K and V)")

# ---- per-head G = W_OᵀW_O and G^{1/2} ----
Gm = {}; Ghalf = {}
for l in LAYERS:
    Wo = attn[l].o_proj.weight.detach().float(); WoT = Wo.t().reshape(nh, dh, hid)
    G = torch.einsum('hdc,hec->hde', WoT, WoT); Gm[l] = G.contiguous()
    lam, U = torch.linalg.eigh(G); Ghalf[l] = torch.einsum('hdj,hj,hej->hde', U, lam.clamp_min(0).sqrt(), U).contiguous()
    del Wo, WoT

# ---- Stage A: accumulate M̄, Σ_q, C_k, C_v per head ----
Mbar = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}
Sq   = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}
Ck   = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}
Cv   = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}
ntok = 0; qcap = {}; kcap = {}; vcap = {}; hooks = []
def mk(store, l):
    def h(m, i, o): store[l] = o.detach()
    return h
for l in LAYERS:
    hooks.append(attn[l].q_proj.register_forward_hook(mk(qcap, l)))
    hooks.append(attn[l].k_proj.register_forward_hook(mk(kcap, l)))
    hooks.append(attn[l].v_proj.register_forward_hook(mk(vcap, l)))
seq_ids = train_ids[:N_SEQS * SEQLEN].view(N_SEQS, SEQLEN)
with torch.no_grad():
    for s in range(0, N_SEQS, BATCH):
        ids = seq_ids[s:s + BATCH].to(device); B = ids.shape[0]
        out = model(ids, output_attentions=True)
        for l in LAYERS:
            q = qcap[l].reshape(B, SEQLEN, nh, dh).float()
            k = kcap[l].reshape(B, SEQLEN, nh, dh).float()
            v = vcap[l].reshape(B, SEQLEN, nh, dh).float()
            a = out.attentions[l].float()
            u = torch.einsum('blhd,hed->blhe', v, Ghalf[l])
            A2 = a * a; unorm = (u * u).sum(-1)
            s0 = A2.sum(-1).transpose(1, 2)
            s1 = torch.einsum('bhmn,bnhd->bmhd', A2, u)
            s2 = torch.einsum('bhmn,bnh->bmh', A2, unorm)
            ubar = torch.einsum('bhmn,bnhd->bmhd', a, u); ubarn = (ubar * ubar).sum(-1)
            c = (s2 - 2 * (ubar * s1).sum(-1) + ubarn * s0).clamp_min(0)
            Mbar[l] += torch.einsum('blh,blhd,blhe->hde', c, q, q)
            Sq[l]   += torch.einsum('blhd,blhe->hde', q, q)
            Ck[l]   += torch.einsum('blhd,blhe->hde', k, k)
            Cv[l]   += torch.einsum('blhd,blhe->hde', v, v)
        ntok += B * SEQLEN; del out
        if (s // BATCH + 1) % 5 == 0: lg(f"  calib {s+B}/{N_SEQS} seqs  ({time.time()-t0:.0f}s)")
for h in hooks: h.remove()
del qcap, kcap, vcap; torch.cuda.empty_cache()
lg(f"capture: {time.time()-t0:.0f}s   tokens={ntok}")

# ---- Stage B: per-head codecs ----
def eigvecs_desc(C): lam, U = torch.linalg.eigh(C); return U.flip(-1)
def whiten_pair(C):
    lam, V = torch.linalg.eigh(C); lam = lam.clamp_min(0)
    fl = RCOND * lam.amax(-1, keepdim=True).clamp_min(1e-12); lam = lam.clamp_min(fl)
    return (torch.einsum('hdj,hj,hfj->hdf', V, lam.sqrt(), V),
            torch.einsum('hdj,hj,hfj->hdf', V, lam.rsqrt(), V))
def codec(metric, C):
    A, Ai = whiten_pair(metric); W = eigvecs_desc(A @ C @ A); return (A @ W).contiguous(), (Ai @ W).contiguous()
EK = {m: {} for m in ("keyPCA", "attnAware", "fisher")}; DK = {m: {} for m in ("keyPCA", "attnAware", "fisher")}
EV = {m: {} for m in ("valuePCA", "outV")};             DV = {m: {} for m in ("valuePCA", "outV")}
for l in LAYERS:
    Uk = eigvecs_desc(Ck[l]); EK["keyPCA"][l] = Uk; DK["keyPCA"][l] = Uk
    EK["attnAware"][l], DK["attnAware"][l] = codec(Sq[l], Ck[l])
    EK["fisher"][l],    DK["fisher"][l]    = codec(Mbar[l], Ck[l])
    Uv = eigvecs_desc(Cv[l]); EV["valuePCA"][l] = Uv; DV["valuePCA"][l] = Uv
    EV["outV"][l], DV["outV"][l] = codec(Gm[l], Cv[l])
lg(f"codecs built: {time.time()-t0:.0f}s")

# ---- Stage C: joint surgery (k_proj + v_proj) + perplexity ----
STACK = {"mla": ("keyPCA", "valuePCA"), "kqsvd": ("attnAware", "outV"), "fisher": ("fisher", "outV")}
MODE = ["none"]; R = [RANKS[0]]
def mk_surg(EmapK, DmapK, EmapV, DmapV):
    def surg_k(l):
        def h(m, i, o):
            if MODE[0] == "none": return o
            km = STACK[MODE[0]][0]; sh = o.shape; k = o.reshape(-1, nkv, dh).float()
            E = EmapK[km][l][:, :, :R[0]]; D = DmapK[km][l][:, :, :R[0]]
            z = torch.einsum('nhd,hdr->nhr', k, E); return torch.einsum('nhr,hdr->nhd', z, D).reshape(sh).to(o.dtype)
        return h
    def surg_v(l):
        def h(m, i, o):
            if MODE[0] == "none": return o
            vm = STACK[MODE[0]][1]; sh = o.shape; v = o.reshape(-1, nh, dh).float()
            E = EmapV[vm][l][:, :, :R[0]]; D = DmapV[vm][l][:, :, :R[0]]
            z = torch.einsum('nhd,hdr->nhr', v, E); return torch.einsum('nhr,hdr->nhd', z, D).reshape(sh).to(o.dtype)
        return h
    for l in LAYERS:
        attn[l].k_proj.register_forward_hook(surg_k(l)); attn[l].v_proj.register_forward_hook(surg_v(l))
mk_surg(EK, DK, EV, DV)
def perplexity():
    nll = 0.0; nt = 0
    with torch.no_grad():
        for c in range(PPL_CHUNKS):
            ids = test_ids[c * CHUNK:(c + 1) * CHUNK]
            if ids.shape[0] < 2: break
            ids = ids.unsqueeze(0).to(device)
            nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1); nt += ids.shape[1] - 1
    return math.exp(nll / nt)

MODE[0] = "none"; ppl_base = perplexity()
R[0] = dh; MODE[0] = "fisher"; ppl_full = perplexity()       # wiring: r=dh => K&V identity => baseline
lg(""); lg(f"WIRING: fisher@r={dh} (K&V full rank) ppl={ppl_full:.3f} vs baseline {ppl_base:.3f} -> "
   f"{'OK' if abs(ppl_full-ppl_base)<0.05 else 'MISMATCH (raise KV_RCOND)'}")
lg(f"baseline (full K+V) perplexity = {ppl_base:.3f}")
lg(f"{'rank':>5} {'cacheX':>7} | {'mla':>8} {'kqsvd':>8} {'fisher':>8} | {'Δ(kqsvd-fisher)':>15} {'Δ(mla-fisher)':>13}  verdict")
lg("-" * 96)
res = []
for r in RANKS:
    R[0] = r
    MODE[0] = "mla"; pm = perplexity()
    MODE[0] = "kqsvd"; pq = perplexity()
    MODE[0] = "fisher"; pf = perplexity()
    res.append((r, pm, pq, pf))
    lg(f"{r:>5} {dh/r:>6.1f}x | {pm:>8.3f} {pq:>8.3f} {pf:>8.3f} | {pq-pf:>+15.3f} {pm-pf:>+13.3f}  "
       f"{'FISHER BEST' if pf <= min(pm, pq) + 1e-3 else 'no'}")
    flush()
lg("-" * 96)
nwin = sum(1 for r, pm, pq, pf in res if pf <= min(pm, pq) + 1e-3)
lg(f"VERDICT: fisher stack is best at {nwin}/{len(RANKS)} TOTAL-cache budgets (both K & V compressed).")
lg("WIN = fisher stays usable at higher total cacheX than mla/kqsvd -> 2nd-order objective extends the")
lg("usable-compression frontier on the FULL cache. cacheX counts BOTH K and V (rank r of head_dim each).")
lg("V uses the output metric G=W_OᵀW_O in both kqsvd & fisher; the K metric (Σq vs M̄) is the differentiator.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
