"""
phase9_attn_aware.py — ATTENTION-AWARE (logit-metric) KEY compression, done CORRECTLY.

This is the FIX for phase3_query_aware.py. phase3 tested the right *instinct* ("preserve
the q.k score, not the key vector") but implemented the wrong *math*: it projected the key
onto the top-r eigenvectors of the QUERY covariance Cq alone (k_hat = Uq Uq^T k), which
throws away all key energy outside the query subspace and ignores Ck entirely. Result:
catastrophic (ppl 1631 / nan). The correct objective is a GENERALIZED (whitened)
eigenproblem that uses BOTH Cq and Ck.

THE OBJECTIVE (per head).  Logit s = q.k. Store r numbers per key, reconstruct k_hat.
  J = E_{q,k}[ (q^T k - q^T k_hat)^2 ] = E_k[ Δk^T Σ_q Δk ] = E_k[ || Σ_q^{1/2} Δk ||^2 ]
where Σ_q = E[q q^T], Δk = k - k_hat.  (Plain MLA/key-PCA minimize E||Δk||^2 = the Σ_q = I
special case -- the WRONG inner product: the model never reads ||Δk||, it reads q^T Δk.)

THE SOLUTION (whiten -> PCA -> un-whiten).  With A = Σ_q^{1/2}, g = A k:
  J = E|| g - g_hat ||^2  -> ordinary PCA on g -> W = top-r eigvecs of  M_K = A Ck A.
  ENCODE  z   = W^T A k            (the r cached numbers)
  DECODE  k_hat = A^{-1} W z       (oblique projector P = A^{-1} W W^T A, P^2=P, rank r)
Both fold into matmuls -> ZERO inference overhead vs MLA.  Provably J(attn-aware) <= J(key-PCA)
because key-PCA is just a feasible (suboptimal, Σ_q=I) point of the SAME problem.

THREE methods compared at matched rank r (per head):
  keyPCA     : E=D=eigvecs(Ck)         -- plain MLA / Palu / phase3 key-PCA   (the baseline)
  queryWrong : E=D=eigvecs(Cq)         -- phase3's BUGGED query-aware          (the negative)
  attnAware  : E=A W, D=A^{-1} W        -- THIS WORK (whitened generalized eig)

Metric 1 (cheap, exact-in-expectation): relative SCORE-ERROR  tr((I-P)^T Σ_q (I-P) Ck)/tr(Σ_q Ck).
  attnAware MUST be lowest here, by construction -- a built-in correctness check.
Metric 2 (downstream, real RoPE+softmax): WikiText-2 perplexity with each layer's K replaced
  by the rank-r k_hat. WIN: attnAware ppl < keyPCA ppl at matched r.

env: QA_CALIB=60000  QA_PPL_CHUNKS=60  QA_RANKS=8,16,32,48,64,96  QA_PPL_RANKS=32,64  KV_RCOND=1e-3
  python phase9_attn_aware.py    # Llama-2-7b (MHA)
"""
import os, math, time, torch

device = torch.device("cuda")
MODEL = os.environ.get("KV_MODEL", "meta-llama/Llama-2-7b-hf")
N_CALIB = int(os.environ.get("QA_CALIB", "60000"))
PPL_CHUNKS = int(os.environ.get("QA_PPL_CHUNKS", "60")); CHUNK = 2048
RANKS = [int(x) for x in os.environ.get("QA_RANKS", "8,16,32,48,64,96").split(",")]
PPL_RANKS = [int(x) for x in os.environ.get("QA_PPL_RANKS", "32,64").split(",")]
RCOND = float(os.environ.get("KV_RCOND", "1e-3"))     # eigenvalue floor for Σ_q^{-1/2} stability
OUT = os.environ.get("KV_OUT", "phase9_attn_aware.txt")
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
lg(f"phase9_attn_aware  MODEL={MODEL}  heads={nh}  head_dim={dh}  layers={NL}  rcond={RCOND}")
lg(f"ranks(score)={RANKS}  ppl_ranks={PPL_RANKS} (of {dh})  calib={N_CALIB}")

# ---------- Stage A: streaming per-head 2nd moments Ck (key), Cq (query) ----------
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

# ---------- Stage B: per-head codecs (E encoder, D decoder; columns = directions, desc) ----------
def eigvecs_desc(C):                              # C [nh,dh,dh] -> eigenvectors as columns, top..bottom
    lam, U = torch.linalg.eigh(C); return U.flip(-1)

def whiten_pair(Cql):                             # A = Σq^{1/2}, Ainv = Σq^{-1/2}  [nh,dh,dh] (clamped)
    lam, V = torch.linalg.eigh(Cql)               # ascending, V columns eigvecs
    lam = lam.clamp_min(0)
    floor = RCOND * lam.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    lam = lam.clamp_min(floor)                    # floor tiny query eigenvalues -> bounded inverse
    sq = lam.sqrt(); isq = sq.reciprocal()
    A    = torch.einsum('hdj,hj,hfj->hdf', V, sq,  V)
    Ainv = torch.einsum('hdj,hj,hfj->hdf', V, isq, V)
    return A, Ainv

# E[method][l] = encoder columns [nh,dh,dh] (z = E[:, :r]^T k);  D[method][l] = decoder [nh,dh,dh] (k_hat = D[:, :r] z)
EE = {m: {} for m in ("keyPCA", "queryWrong", "attnAware")}
DD = {m: {} for m in ("keyPCA", "queryWrong", "attnAware")}
for l in LAYERS:
    Uk = eigvecs_desc(Ck[l]); Uq = eigvecs_desc(Cq[l])
    EE["keyPCA"][l] = Uk;     DD["keyPCA"][l] = Uk          # orthonormal: E=D=U  -> P=UU^T
    EE["queryWrong"][l] = Uq; DD["queryWrong"][l] = Uq      # phase3's bug: project onto query subspace
    A, Ainv = whiten_pair(Cq[l])
    W = eigvecs_desc(A @ Ck[l] @ A)                          # top eigvecs of M_K = A Ck A
    EE["attnAware"][l] = (A @ W).contiguous()               # encoder  E = A W      (z = W^T A k)
    DD["attnAware"][l] = (Ainv @ W).contiguous()            # decoder  D = A^{-1} W (k_hat = A^{-1} W z)
lg(f"codecs built: {time.time()-t0:.0f}s")

# ---------- Stage C1: SCORE-ERROR diagnostic (general oblique-projector formula) ----------
I_dh = torch.eye(dh, device=device)
def score_err(method, l, r):                      # rel score-error tr((I-P)^T Σq (I-P) Ck) / tr(Σq Ck), mean over heads
    E = EE[method][l][:, :, :r]; D = DD[method][l][:, :, :r]
    P = D @ E.transpose(-2, -1)                    # [nh,dh,dh] oblique projector D E^T
    IP = I_dh - P
    num = ((IP.transpose(-2, -1) @ Cq[l] @ IP) * Ck[l]).sum(dim=(-2, -1))   # tr((I-P)^T Σq (I-P) Ck) per head
    den = (Cq[l] * Ck[l]).sum(dim=(-2, -1)) + 1e-9                          # tr(Σq Ck) per head
    return (num / den).mean().item()
lg("=" * 100)
lg("SCORE-ERROR (relative; lower=better).  attnAware is the provable min -> MUST be <= keyPCA & queryWrong")
lg(f"{'rank':>5} {'cacheX':>7} | {'keyPCA(=MLA)':>13} {'queryWrong(p3)':>14} {'attnAware(ours)':>15} | {'ours vs MLA':>11}")
lg("-" * 100)
for r in RANKS:
    ea = sum(score_err("keyPCA", l, r) for l in LAYERS) / NL
    eb = sum(score_err("queryWrong", l, r) for l in LAYERS) / NL
    ec = sum(score_err("attnAware", l, r) for l in LAYERS) / NL
    red = (ea - ec) / ea * 100 if ea > 0 else 0
    flag = "" if ec <= ea + 1e-9 else "  <-- BUG: ours should never lose here"
    lg(f"{r:>5} {dh/r:>6.1f}x | {ea:>13.4f} {eb:>14.4f} {ec:>15.4f} | {red:>+10.1f}%{flag}")
    flush()
lg("-" * 100)
lg("attnAware << keyPCA on score-error => keys carry logit-irrelevant variance MLA wastes budget on.")
lg("queryWrong > keyPCA reproduces phase3's failure: projecting onto the query subspace discards key energy.")

# ---------- Stage C2: perplexity surgery (real RoPE + softmax) ----------
MODE = ["none"]; R = [PPL_RANKS[0]]
def surg(l):
    def h(m, i, o):
        if MODE[0] == "none": return o
        sh = o.shape; k = o.reshape(-1, nkv, dh).float()
        E = EE[MODE[0]][l][:, :, :R[0]]; D = DD[MODE[0]][l][:, :, :R[0]]   # [nh,dh,r]
        z = torch.einsum('nhd,hdr->nhr', k, E)        # encode: z = E^T k
        khat = torch.einsum('nhr,hdr->nhd', z, D)     # decode: k_hat = D z
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
R[0] = dh; MODE[0] = "keyPCA"; ppl_full = perplexity()   # WIRING: r=dh => P=I => k_hat=k => ppl==baseline
lg(""); lg(f"WIRING CHECK: keyPCA at r={dh} (full rank) ppl={ppl_full:.3f} vs baseline {ppl_base:.3f}  "
   f"-> {'OK' if abs(ppl_full - ppl_base) < 0.05 else 'MISMATCH!! surgery reshape/einsum is wrong'}")
R[0] = dh; MODE[0] = "attnAware"; ppl_full_aa = perplexity()  # attnAware@full rank: P=I too (WW^T=I)
lg(f"WIRING CHECK: attnAware at r={dh} (full rank) ppl={ppl_full_aa:.3f} vs baseline {ppl_base:.3f}  "
   f"-> {'OK' if abs(ppl_full_aa - ppl_base) < 0.05 else 'MISMATCH!! whitening inverse is off (raise KV_RCOND)'}")
lg(""); lg(f"baseline (full K) perplexity = {ppl_base:.3f}")
lg(f"{'rank':>5} {'cacheX(K)':>9} | {'ppl keyPCA(=MLA)':>16} {'ppl queryWrong':>14} {'ppl attnAware':>13} | {'Δppl(MLA-ours)':>14}  verdict")
lg("-" * 100)
res = []
for r in PPL_RANKS:
    R[0] = r
    MODE[0] = "keyPCA"; pa = perplexity()
    MODE[0] = "queryWrong"; pb = perplexity()
    MODE[0] = "attnAware"; pc = perplexity()
    d = pa - pc; res.append((r, pa, pb, pc, d))
    lg(f"{r:>5} {dh/r:>8.1f}x | {pa:>16.3f} {pb:>14.3f} {pc:>13.3f} | {d:>+14.3f}  {'ATTN-AWARE WINS' if d > 0 else 'no gain'}")
    flush()
lg("-" * 100)
nwin = sum(1 for *_, d in res if d > 0)
lg(f"VERDICT: attnAware beats keyPCA(=MLA) on perplexity at {nwin}/{len(PPL_RANKS)} ranks.")
lg("If yes on BOTH metrics -> the lever to beat linear KV compression is the OBJECTIVE (preserve the")
lg("attention logit via the Σq-whitened generalized eigenproblem), not encoder nonlinearity.")
lg("Diagnostic isolates K (V full); Σq is pre-RoPE (proxy); ppl is exact (real RoPE+softmax downstream).")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
