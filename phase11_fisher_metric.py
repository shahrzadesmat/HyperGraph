"""
phase11_fisher_metric.py — go SECOND-ORDER on the objective: the output/Fisher metric.

phase9 (= KQ-SVD) preserves the LOGIT qᵀk, weighting key error by Σ_q = E[q qᵀ]. But the model
reads the attention OUTPUT  o_m = W_O Σ_n a_mn v_n, after softmax. Propagating a key error Δk_n
through softmax + value mixing (keys independent, the OBS/GPTQ approximation):

  Δo_m = Σ_n a_mn (q_mᵀΔk_n/√d) · W_O(v_n − ō_m),   ō_m = Σ_n a_mn v_n

E_m‖Δo_m‖²  ->  a REWEIGHTED query 2nd moment (per head):

  M̄ = Σ_m c_m q_m q_mᵀ ,   c_m = Σ_n a_mn² · ‖W_O(v_n − ō_m)‖²        (G = W_OᵀW_O)

c_m = output-sensitivity of query m = (attention concentration a²) × (value spread it sees, through W_O).
Plain Σ_q (KQ-SVD) is the special case c_m ≡ 1. Then the SAME whitened generalized eigenproblem:

  W = top-r eigvecs of M̄^{1/2} C_k M̄^{1/2} ,  encode z = Wᵀ M̄^{1/2} k ,  decode k̂ = M̄^{-1/2} W z

THREE K codecs at matched rank r (per head), V left full:
  keyPCA    : eigvecs(C_k)              -- MLA / Euclidean
  attnAware : whiten by Σ_q             -- KQ-SVD / phase9 (logit, 1st order)
  fisher    : whiten by M̄              -- THIS WORK (output, 2nd order)

WIN: fisher ppl < attnAware ppl at matched r  -> the 2nd-order output metric beats the 1st-order logit
metric. That is the one direction phase10 left open (deeper on the OBJECTIVE, not the architecture).

Needs the real attention weights -> attn_implementation="eager", output_attentions=True (calibration only).
env: P11_CALIB_SEQS=80 P11_SEQLEN=512 P11_BATCH=4 P11_PPL_CHUNKS=60 P11_RANKS=8,16,32,64 KV_RCOND=1e-3
  python phase11_fisher_metric.py
"""
import os, math, time, torch
device = torch.device("cuda")
MODEL = os.environ.get("KV_MODEL", "meta-llama/Llama-2-7b-hf")
N_SEQS = int(os.environ.get("P11_CALIB_SEQS", "80"))
SEQLEN = int(os.environ.get("P11_SEQLEN", "512"))
BATCH = int(os.environ.get("P11_BATCH", "4"))
PPL_CHUNKS = int(os.environ.get("P11_PPL_CHUNKS", "60")); CHUNK = 2048
RANKS = [int(x) for x in os.environ.get("P11_RANKS", "8,16,32,64").split(",")]
RCOND = float(os.environ.get("KV_RCOND", "1e-3"))
OUT = os.environ.get("KV_OUT", "phase11_fisher_metric.txt")
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
assert nkv == nh, f"MHA assumed (nkv==nh); got {nkv},{nh}"
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
test_ids = tok(test_txt, return_tensors="pt").input_ids[0]
lg(f"phase11_fisher_metric  MODEL={MODEL}  heads={nh}  head_dim={dh}  layers={NL}  ranks={RANKS}")
lg(f"calib: {N_SEQS} seqs x {SEQLEN} tok (batch {BATCH});  rcond={RCOND}")

# ---- per-head G^{1/2} from W_O (for the value-distinctiveness weighting) ----
Ghalf = {}
for l in LAYERS:
    Wo = attn[l].o_proj.weight.detach().float()              # [hid, nh*dh]
    WoT = Wo.t().reshape(nh, dh, hid)                         # [nh, dh, hid] head-major
    G = torch.einsum('hdc,hec->hde', WoT, WoT)               # G_h = W_O,hᵀ W_O,h  [nh,dh,dh]
    lam, U = torch.linalg.eigh(G); lam = lam.clamp_min(0)
    Ghalf[l] = torch.einsum('hdj,hj,hej->hde', U, lam.sqrt(), U).contiguous()   # G^{1/2}
    del Wo, WoT, G

# ---- Stage A: accumulate M̄ (fisher), Σ_q, C_k per head, streaming over calibration ----
Mbar = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}
Sq   = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}
Ck   = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}
ntok = 0
qcap = {}; kcap = {}; vcap = {}; hooks = []
def mk(store, l):
    def h(m, i, o): store[l] = o.detach()                    # [B*L, nh*dh]
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
            a = out.attentions[l].float()                    # [B,nh,L,L]  (real softmax, post-RoPE)
            u = torch.einsum('blhd,hed->blhe', v, Ghalf[l])  # whitened values G^{1/2} v  [B,L,nh,dh]
            A2 = a * a
            unorm = (u * u).sum(-1)                           # [B,L,nh]
            s0 = A2.sum(-1).transpose(1, 2)                   # Σ_n a²            [B,L,nh]
            s1 = torch.einsum('bhmn,bnhd->bmhd', A2, u)       # Σ_n a² u_n        [B,L,nh,dh]
            s2 = torch.einsum('bhmn,bnh->bmh', A2, unorm)     # Σ_n a² ‖u_n‖²     [B,L,nh]
            ubar = torch.einsum('bhmn,bnhd->bmhd', a, u)      # G^{1/2} ō_m       [B,L,nh,dh]
            ubarn = (ubar * ubar).sum(-1)                     # ‖ō_m‖²_G          [B,L,nh]
            c = (s2 - 2 * (ubar * s1).sum(-1) + ubarn * s0).clamp_min(0)   # c_m  [B,L,nh]
            Mbar[l] += torch.einsum('blh,blhd,blhe->hde', c, q, q)
            Sq[l]   += torch.einsum('blhd,blhe->hde', q, q)
            Ck[l]   += torch.einsum('blhd,blhe->hde', k, k)
        ntok += B * SEQLEN
        del out
        if (s // BATCH + 1) % 5 == 0: lg(f"  calib {s+B}/{N_SEQS} seqs  ({time.time()-t0:.0f}s)")
for h in hooks: h.remove()
del qcap, kcap, vcap; torch.cuda.empty_cache()
lg(f"capture: {time.time()-t0:.0f}s   tokens={ntok}")

# ---- Stage B: per-head codecs for the 3 metrics ----
def eigvecs_desc(C): lam, U = torch.linalg.eigh(C); return U.flip(-1)
def whiten_pair(C):
    lam, V = torch.linalg.eigh(C); lam = lam.clamp_min(0)
    floor = RCOND * lam.amax(-1, keepdim=True).clamp_min(1e-12); lam = lam.clamp_min(floor)
    sq = lam.sqrt(); isq = sq.reciprocal()
    A = torch.einsum('hdj,hj,hfj->hdf', V, sq, V); Ainv = torch.einsum('hdj,hj,hfj->hdf', V, isq, V)
    return A, Ainv
def whitened_codec(metric, Ckl):                             # E,D for k̂ = D (Eᵀ k), full rank (slice later)
    A, Ainv = whiten_pair(metric); W = eigvecs_desc(A @ Ckl @ A)
    return (A @ W).contiguous(), (Ainv @ W).contiguous()
EE = {m: {} for m in ("keyPCA", "attnAware", "fisher")}; DD = {m: {} for m in ("keyPCA", "attnAware", "fisher")}
for l in LAYERS:
    Uk = eigvecs_desc(Ck[l]); EE["keyPCA"][l] = Uk; DD["keyPCA"][l] = Uk
    EE["attnAware"][l], DD["attnAware"][l] = whitened_codec(Sq[l], Ck[l])
    EE["fisher"][l],    DD["fisher"][l]    = whitened_codec(Mbar[l], Ck[l])
lg(f"codecs built: {time.time()-t0:.0f}s")
# how different is M̄ from Σ_q? (subspace overlap of top-r) -- if ~identical, fisher can't differ from KQ-SVD
def topr_overlap(Ea, Eb, r):                                  # proper subspace overlap in [0,1]
    Qa = torch.linalg.qr(Ea[:, :, :r]).Q                      # orthonormalize the oblique decode cols first
    Qb = torch.linalg.qr(Eb[:, :, :r]).Q
    return float((torch.einsum('hdr,hds->hrs', Qa, Qb) ** 2).sum(dim=(-2, -1)).mean() / r)
for r in (8, 32):
    ov = sum(topr_overlap(DD["attnAware"][l], DD["fisher"][l], r) for l in LAYERS) / NL
    lg(f"  top-{r} decode-subspace overlap  attnAware vs fisher = {ov:.3f}  (1.0 => identical basis)")

# ---- Stage C: surgery + perplexity (real RoPE+softmax) ----
MODE = ["none"]; R = [RANKS[0]]
def surg(l):
    def h(m, i, o):
        if MODE[0] == "none": return o
        sh = o.shape; k = o.reshape(-1, nkv, dh).float()
        E = EE[MODE[0]][l][:, :, :R[0]]; D = DD[MODE[0]][l][:, :, :R[0]]
        z = torch.einsum('nhd,hdr->nhr', k, E); khat = torch.einsum('nhr,hdr->nhd', z, D)
        return khat.reshape(sh).to(o.dtype)
    return h
for l in LAYERS: attn[l].k_proj.register_forward_hook(surg(l))
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
R[0] = dh; MODE[0] = "fisher"; ppl_full = perplexity()      # wiring: r=dh => P=I => baseline
lg(""); lg(f"WIRING: fisher@r={dh} ppl={ppl_full:.3f} vs baseline {ppl_base:.3f} -> "
   f"{'OK' if abs(ppl_full-ppl_base)<0.05 else 'MISMATCH (raise KV_RCOND)'}")
lg(f"baseline (full K) perplexity = {ppl_base:.3f}")
lg(f"{'rank':>5} {'cacheX':>7} | {'keyPCA':>8} {'attnAware':>10} {'fisher(new)':>12} | {'Δ(attn-fisher)':>14}  verdict")
lg("-" * 92)
res = []
for r in RANKS:
    R[0] = r
    MODE[0] = "keyPCA"; pk = perplexity()
    MODE[0] = "attnAware"; pa = perplexity()
    MODE[0] = "fisher"; pf = perplexity()
    d = pa - pf; res.append((r, pk, pa, pf, d))
    lg(f"{r:>5} {dh/r:>6.1f}x | {pk:>8.3f} {pa:>10.3f} {pf:>12.3f} | {d:>+14.3f}  "
       f"{'FISHER WINS' if d > 1e-3 else 'no gain over KQ-SVD'}")
    flush()
lg("-" * 92)
nwin = sum(1 for *_, d in res if d > 1e-3)
lg(f"VERDICT: fisher (2nd-order output metric) beats attnAware (=KQ-SVD, 1st-order logit) at {nwin}/{len(RANKS)} ranks.")
lg("If fisher << attnAware -> the output/Fisher metric is a real, novel improvement over KQ-SVD.")
lg("If fisher ≈ attnAware (and subspace overlap ~1) -> 1st order already captures it; KQ-SVD is enough.")
lg("K-only (V full); Σq/M̄ use pre-RoPE q (proxy) with REAL attention weights; ppl exact downstream.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
