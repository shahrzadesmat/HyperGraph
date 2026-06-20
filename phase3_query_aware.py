"""
phase3_query_aware.py — OUTPUT-AWARE (query-subspace) vs RECONSTRUCTION-AWARE (key-PCA = MLA)
key compression, PER HEAD, at matched rank. Tests the "attention only needs the query subspace"
headroom that MLA / KV-CAR ignore (they minimize key RECONSTRUCTION, i.e. key variance).

Per head, the attention score is  s = q . k  (q = x.W_Q, k = x.W_K, both head-dim d).
Compress each head's key to rank r two ways:
  (a) key-PCA   : keep top-r eigenvectors of  Ck = E[k k^T]   (key variance)  -> what PCA/MLA do.
  (b) query-aware: keep top-r eigenvectors of Cq = E[q q^T]   (query energy)  -> aims to keep q.k.

Metric 1 (cheap, exact-in-expectation over the query distribution): relative SCORE-ERROR energy
   rel = tr( (I-P) Cq (I-P) Ck ) / tr( Cq Ck ),   P = projection onto the kept subspace.
   (= E[(k-Pk)^T Cq (k-Pk)] / E[k^T Cq k], the expected (q.k) error from compressing k.)
Metric 2 (downstream, REAL softmax + RoPE): WikiText-2 perplexity with each layer's K replaced by
   the rank-r per-head k_hat. Compares baseline / key-PCA / query-aware.

WIN: query-aware gives LOWER score-error AND LOWER perplexity at matched r  -> the output-aware
headroom is real, and an output-aware (nonlinear) encoder is worth building.
CAVEAT: the query subspace is computed on PRE-RoPE q (a proxy; RoPE rotates it by position). The
surgery applies the REAL RoPE + softmax downstream, so Metric 2 is exact regardless.
  python phase3_query_aware.py    # Llama-2-7b (MHA)
"""
import os, math, time, torch

device = torch.device("cuda")
MODEL = "meta-llama/Llama-2-7b-hf"
N_CALIB = int(os.environ.get("QA_CALIB", "60000"))
PPL_CHUNKS = int(os.environ.get("QA_PPL_CHUNKS", "60")); CHUNK = 2048
RANKS = [int(x) for x in os.environ.get("QA_RANKS", "8,16,32,48,64,96").split(",")]
PPL_RANKS = [int(x) for x in os.environ.get("QA_PPL_RANKS", "32,64").split(",")]
OUT = "/work/hdd/bdjd/hypergraph_pruning/phase3_query_aware.txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); cfg = model.config
nh = cfg.num_attention_heads
nkv = getattr(cfg, "num_key_value_heads", nh)
dh = cfg.hidden_size // nh
LAYERS = list(range(NL))
assert nkv == nh, f"this diagnostic assumes MHA (nkv==nh); got nkv={nkv}, nh={nh} (GQA needs head-grouping)"
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
test_ids = tok(test_txt, return_tensors="pt").input_ids[0]
lg(f"phase3_query_aware  MODEL={MODEL}  heads={nh}  head_dim={dh}  layers={NL}")
lg(f"ranks(score)={RANKS}  ppl_ranks={PPL_RANKS} (of {dh})  calib={N_CALIB}")

# ---------- Stage A: streaming per-head covariances Ck (key), Cq (query) ----------
Ck = {l: torch.zeros(nkv, dh, dh, device=device) for l in LAYERS}
Cq = {l: torch.zeros(nh, dh, dh, device=device) for l in LAYERS}
cnt = {l: 0 for l in LAYERS}; hooks = []
def mk_k(l):
    def h(m, i, o):
        k = o.detach().reshape(-1, nkv, dh).float()
        Ck[l] += torch.einsum('nhd,nhe->hde', k, k); cnt[l] += k.shape[0]
    return h
def mk_q(l):
    def h(m, i, o):
        q = o.detach().reshape(-1, nh, dh).float()
        Cq[l] += torch.einsum('nhd,nhe->hde', q, q)
    return h
for l in LAYERS:
    hooks.append(attn[l].k_proj.register_forward_hook(mk_k(l)))
    hooks.append(attn[l].q_proj.register_forward_hook(mk_q(l)))
cb = min(N_CALIB, train_ids.shape[0])
with torch.no_grad():
    for s in range(0, cb - 1, CHUNK):
        model(train_ids[s:s + CHUNK].unsqueeze(0).to(device))
for h in hooks: h.remove()
for l in LAYERS: Ck[l] /= cnt[l]; Cq[l] /= cnt[l]
lg(f"capture+cov: {time.time()-t0:.0f}s   tokens/layer={cnt[LAYERS[0]]}")

# ---------- Stage B: per-head eigenvectors (descending) ----------
def eigvecs_desc(C):                       # C [nh,dh,dh] -> eigenvectors as columns, top..bottom
    lam, U = torch.linalg.eigh(C); return U.flip(-1)
Uk = {l: eigvecs_desc(Ck[l]) for l in LAYERS}     # key-PCA dirs
Uq = {l: eigvecs_desc(Cq[l]) for l in LAYERS}     # query dirs

# ---------- Stage C1: SCORE-ERROR diagnostic (cheap) ----------
def proj(U, r):                            # P = U[:,:,:r] U[:,:,:r]^T  per head
    Ur = U[:, :, :r]; return Ur @ Ur.transpose(-2, -1)
I_dh = torch.eye(dh, device=device)
def score_err(P, Ckl, Cql):                # per-head rel score-error -> mean over heads
    IP = I_dh - P
    num = ((IP @ Cql @ IP) * Ckl).sum(dim=(-2, -1))     # tr((I-P)Cq(I-P) Ck)
    den = (Cql * Ckl).sum(dim=(-2, -1)) + 1e-9          # tr(Cq Ck)
    return (num / den).mean().item()
lg("=" * 88)
lg("SCORE-ERROR (relative; lower=better key compression for the q.k score), averaged over heads")
lg(f"{'rank':>5} {'cacheX':>7} | {'key-PCA(=MLA)':>14} {'query-aware':>12} | {'reduction':>10}")
lg("-" * 88)
sa_all = {}; sb_all = {}
for r in RANKS:
    eas = [score_err(proj(Uk[l], r), Ck[l], Cq[l]) for l in LAYERS]
    ebs = [score_err(proj(Uq[l], r), Ck[l], Cq[l]) for l in LAYERS]
    ea = sum(eas) / len(eas); eb = sum(ebs) / len(ebs); sa_all[r] = ea; sb_all[r] = eb
    red = (ea - eb) / ea * 100 if ea > 0 else 0
    lg(f"{r:>5} {dh/r:>6.1f}x | {ea:>14.4f} {eb:>12.4f} | {red:>+9.1f}%")
    flush()
lg("-" * 88)
lg("If query-aware << key-PCA, the keys carry score-irrelevant variance that MLA wastes budget on.")

# ---------- Stage C2: perplexity surgery (real softmax + RoPE) ----------
MODE = ["none"]; R = [PPL_RANKS[0]]
def surg(l):
    def h(m, i, o):
        if MODE[0] == "none": return o
        sh = o.shape; k = o.reshape(-1, nkv, dh).float()
        U = (Uk[l] if MODE[0] == "keyPCA" else Uq[l])[:, :, :R[0]]    # [nh,dh,r]
        z = torch.einsum('nhd,hdr->nhr', k, U)
        khat = torch.einsum('nhr,hdr->nhd', z, U)
        return khat.reshape(sh).to(o.dtype)
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
R[0] = dh; MODE[0] = "keyPCA"; ppl_full = perplexity()      # WIRING CHECK: r=dh => P=I => k_hat=k => ppl==baseline
lg(""); lg(f"WIRING CHECK: keyPCA at r={dh} (full rank) ppl={ppl_full:.3f} vs baseline {ppl_base:.3f}  "
   f"-> {'OK' if abs(ppl_full - ppl_base) < 0.05 else 'MISMATCH!! surgery reshape/einsum is wrong'}")
lg(""); lg(f"baseline (full K) perplexity = {ppl_base:.3f}")
lg(f"{'rank':>5} {'cacheX(K)':>9} | {'ppl key-PCA(=MLA)':>17} {'ppl query-aware':>15} | {'Δppl(MLA-QA)':>12}  verdict")
lg("-" * 88)
res = []
for r in PPL_RANKS:
    R[0] = r
    MODE[0] = "keyPCA"; pa = perplexity()
    MODE[0] = "queryAware"; pb = perplexity()
    d = pa - pb; res.append((r, pa, pb, d))
    lg(f"{r:>5} {dh/r:>8.1f}x | {pa:>17.3f} {pb:>15.3f} | {d:>+12.3f}  {'QUERY-AWARE WINS' if d > 0 else 'no gain'}")
    flush()
lg("-" * 88)
nwin = sum(1 for *_, d in res if d > 0)
lg(f"VERDICT: query-aware beats key-PCA(=MLA) on perplexity at {nwin}/{len(PPL_RANKS)} ranks.")
lg("If yes on BOTH metrics -> output-aware compression has real headroom over reconstruction (MLA);")
lg("the lever to beat linear is the OBJECTIVE (preserve the score/attention), not encoder nonlinearity.")
lg("Diagnostic isolates K (V left full); query subspace is pre-RoPE (proxy); ppl is exact downstream.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
