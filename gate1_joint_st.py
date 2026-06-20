"""
gate1_joint_st.py — GATE #1: is the dimensional x temporal COUPLING EXPLOITABLE?
Does a JOINT spatio-temporal transform of the keys beat the SEPARABLE (dimensional-then-temporal)
one at a MATCHED coefficient budget? Scoped to KEYS (V left full) to isolate the principle.

Per layer: standardize K; SPATIAL KLT Vk (C x C = PCA of K); TEMPORAL KLT Ut (Tb x Tb, KLT over
token-blocks). 2-D coeffs of a block X[Tb,C]:  A = Ut^T @ X @ Vk.  Energy E(i,j)=mean_block A[i,j]^2.
At budget r (avg coeffs/token; N = Tb*r kept cells), three SUPPORTS over the (temporal i x spatial j) grid:
  (A) per-token MLA : all Tb temporal x top-r spatial          (full-temporal rectangle = plain MLA)
  (B) best separable: top-Kt temporal x top-dc spatial, Kt*dc=N (strongest SEQUENTIAL/independent combo)
  (C) joint optimal : top-N cells of E by energy               (non-rectangular -> exploits the coupling)
By construction kept-energy (A)<=(B)<=(C); the question is the PERPLEXITY gap and whether (C)-(B) WIDENS
as r shrinks. Surgery replaces K with the block-reconstructed K_hat; WikiText-2 perplexity.
WIN: ppl(C) < ppl(B) and the (C)-(B) gain GROWS at aggressive r  -> coupling is exploitable (gate #1 PASS).
SANITY: full-budget reconstruction == baseline (orthonormal transforms).
env: G1_MODEL  G1_CALIB=50000  G1_TB=16  G1_RS=64,128,256,512  G1_PPL_CHUNKS=50
"""
import os, math, time, torch
device = torch.device("cuda")
MODEL = os.environ.get("G1_MODEL", "meta-llama/Llama-2-7b-hf")
N_CALIB = int(os.environ.get("G1_CALIB", "50000"))
TB = int(os.environ.get("G1_TB", "16")); CHUNK = 2048
RS = sorted(int(x) for x in os.environ.get("G1_RS", "64,128,256,512").split(","))
PPL_CHUNKS = int(os.environ.get("G1_PPL_CHUNKS", "50"))
OUT = "/work/hdd/bdjd/hypergraph_pruning/gate1_joint_st.txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); C = model.config.hidden_size; LAYERS = list(range(NL))
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
test_ids = tok(test_txt, return_tensors="pt").input_ids[0]
lg(f"gate1_joint_st  MODEL={MODEL}  C={C}  layers={NL}  Tb={TB}  budgets r={RS} (comp={[C//r for r in RS]}x)")

# ---------- Stage A: capture K per chunk (blocks of TB tokens) ----------
Kseq = {l: [] for l in LAYERS}; hooks = []
def mk(l):
    def h(m, i, o): Kseq[l].append(o.detach()[0].to(torch.float16).cpu())   # [T, C] pre-RoPE
    return h
for l in LAYERS: hooks.append(attn[l].k_proj.register_forward_hook(mk(l)))
nch = max(1, min(N_CALIB // CHUNK, (train_ids.shape[0] - 1) // CHUNK))
with torch.no_grad():
    for c in range(nch):
        ids = train_ids[c * CHUNK:(c + 1) * CHUNK]
        if ids.shape[0] < TB: break
        model(ids.unsqueeze(0).to(device))
for h in hooks: h.remove()
lg(f"capture: {time.time()-t0:.0f}s  chunks={len(Kseq[LAYERS[0]])}")

# ---------- Stage B: per-layer spatial Vk, temporal Ut, 2-D energy E ----------
def eig_desc(Cmat):
    lam, U = torch.linalg.eigh(Cmat); return U.flip(-1)
PAR = {}   # l -> (Vk, Ut, mu, sd, E)
for l in LAYERS:
    K = torch.cat(Kseq[l]).float().to(device)                 # [Ntok, C]
    nb = K.shape[0] // TB; K = K[:nb * TB]
    mu = K.mean(0, keepdim=True); sd = K.std(0, keepdim=True) + 1e-6
    Ks = (K - mu) / sd
    Cs = Ks.t() @ Ks / Ks.shape[0]; Vk = eig_desc(Cs)         # spatial KLT [C,C]
    Xb = Ks.reshape(nb, TB, C)                                # blocks
    Ct = torch.einsum('nac,nbc->ab', Xb, Xb) / (nb * C)       # temporal cov [Tb,Tb]
    Ut = eig_desc(Ct)                                         # temporal KLT [Tb,Tb]
    A = torch.einsum('ab,nbc->nac', Ut.t(), Xb) @ Vk          # 2-D coeffs [nb,Tb,C]
    E = (A * A).mean(0)                                       # energy [Tb,C]
    PAR[l] = (Vk, Ut.contiguous(), mu, sd, E)
    Kseq[l] = None
    del K, Ks, Cs, Xb, Ct, A; torch.cuda.empty_cache()
    if (l + 1) % 8 == 0: lg(f"  transforms {l+1}/{NL}  ({time.time()-t0:.0f}s)")
del Kseq; torch.cuda.empty_cache()

# ---------- support masks from energy E ----------
def divisors_leq(N, cap):
    return [k for k in range(1, cap + 1) if N % k == 0]
def build_mask(E, scheme, r):
    Tb_, C_ = E.shape; N = Tb_ * r; M = torch.zeros_like(E)
    sp = E.sum(0); tp = E.sum(1)                              # spatial / temporal marginals
    sj = torch.argsort(sp, descending=True); si = torch.argsort(tp, descending=True)
    if scheme == "A":                                        # all temporal x top-r spatial
        M[:, sj[:r]] = 1.0
    elif scheme == "B":                                      # best separable rectangle Kt x dc
        best = -1.0; bm = (Tb_, r)
        for Kt in divisors_leq(N, Tb_):
            dc = N // Kt
            if dc > C_: continue
            e = E[si[:Kt]][:, sj[:dc]].sum().item()
            if e > best: best = e; bm = (Kt, dc)
        Kt, dc = bm; M[si[:Kt].unsqueeze(1), sj[:dc].unsqueeze(0)] = 1.0
    elif scheme == "S":                                      # separable SURROGATE: top-N of marginal outer-product
        Esep = torch.outer(tp, sp) / (E.sum() + 1e-9)        # optimal support IF energy were separable
        idx = torch.topk(Esep.flatten(), N).indices
        Mf = torch.zeros(Tb_ * C_, device=E.device); Mf[idx] = 1.0; M = Mf.reshape(Tb_, C_)
    else:                                                    # C: top-N cells of the ACTUAL energy
        idx = torch.topk(E.flatten(), N).indices
        Mf = torch.zeros(Tb_ * C_, device=E.device); Mf[idx] = 1.0; M = Mf.reshape(Tb_, C_)
    return M

# ---------- Stage C: surgery + perplexity ----------
CUR = {}; MODE = ["none"]
def surg(l):
    def h(m, i, o):
        if MODE[0] == "none" or l not in CUR: return o
        Vk, Ut, mu, sd, _ = PAR[l]; M = CUR[l]
        sh = o.shape; K = o.reshape(-1, C).float()
        T = K.shape[0]; nb = T // TB
        if nb == 0: return o
        used = nb * TB
        Xb = ((K[:used] - mu) / sd).reshape(nb, TB, C)
        A = (torch.einsum('ab,nbc->nac', Ut.t(), Xb) @ Vk) * M           # transform + truncate
        Xh = torch.einsum('ab,nbc->nac', Ut, (A @ Vk.t()))               # inverse
        Kh = (Xh.reshape(used, C) * sd + mu)
        out = K.clone(); out[:used] = Kh
        return out.reshape(sh).to(o.dtype)
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
def set_masks(scheme, r):
    for l in LAYERS: CUR[l] = build_mask(PAR[l][4], scheme, r)

MODE[0] = "none"; ppl_base = perplexity()
# wiring sanity: full budget (r=C) under scheme C must reproduce baseline (exact inverse)
set_masks("C", C); MODE[0] = "go"; ppl_full = perplexity()
lg(f"\nbaseline ppl={ppl_base:.3f}   |   WIRING: full-budget(r={C}) ppl={ppl_full:.3f} -> "
   f"{'OK' if abs(ppl_full-ppl_base)<0.03 else 'MISMATCH!!'}")
lg("=" * 104)
lg("ppl (lower=better). tmp:B-A=value of temporal compression; shp:S-B=staircase-vs-rectangle (NOT coupling);")
lg("CPL:S-C = PURE coupling (joint energy-optimal vs separable-surrogate). Gate#1 hinges on CPL.")
lg(f"{'r':>5} {'comp':>6} | {'(A)MLA':>8} {'(B)rect':>8} {'(S)sep':>8} {'(C)joint':>8} | {'tmp:B-A':>8} {'shp:S-B':>8} {'CPL:S-C':>9}")
lg("-" * 104)
res = []
for r in RS:
    set_masks("A", r); MODE[0] = "go"; pa = perplexity()
    set_masks("B", r); pb = perplexity()
    set_masks("S", r); ps = perplexity()
    set_masks("C", r); pc = perplexity()
    res.append((r, pa, pb, ps, pc))
    lg(f"{r:>5} {C//r:>5}x | {pa:>8.3f} {pb:>8.3f} {ps:>8.3f} {pc:>8.3f} | {pa-pb:>+8.3f} {pb-ps:>+8.3f} {ps-pc:>+9.3f}")
    flush()
lg("-" * 104)
cpl = [ps - pc for r, pa, pb, ps, pc in res]                 # RS ascending -> cpl[0]=most aggressive
widen = all(cpl[i] >= cpl[i + 1] - 1e-6 for i in range(len(cpl) - 1))   # coupling gain largest at aggressive r
nwin = sum(1 for c in cpl if c > 1e-3)
lg(f"VERDICT (pure coupling): joint(C) beats separable-surrogate(S) at {nwin}/{len(RS)} budgets.")
lg(f"  CPL gain (S-C) by budget r={RS}: {[round(c,3) for c in cpl]}  | widens toward aggressive r: {'YES' if widen else 'no'}")
lg(f"GATE #1 = {'PASS' if (nwin >= len(RS)-1 and widen) else 'FAIL'}  "
   f"(PASS needs C<S on nearly all budgets AND the coupling gain growing as r shrinks).")
lg("Scoped to K (V full); block codec (Tb-token latency); pre-RoPE. If PASS -> joint-MLA + DeltaKV head-to-head.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
