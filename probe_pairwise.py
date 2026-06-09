"""
Pairwise error-interaction probe.  Usage: python probe_pairwise.py <hf_model>

Decides: do we have a GRAPH or just a CURVE?

The error-prop probe showed a layer's compression error amplifies to the output
(depth-varying). But that is a 1-D curve. A GRAPH only exists if different layers'
compression errors are COUPLED — i.e. how layer i's error affects the output
depends on which OTHER layers are also compressed, in a structured (pairwise) way.

For each layer i we form delta_i = (final hidden with ONLY layer i rank-truncated)
- (clean final hidden), a vector in output space. Then:

  OVERLAP   cos_ij = <delta_i, delta_j> / (||delta_i|| ||delta_j||)
            -> do layers i and j push the output in the SAME directions?
               (errors landing on a shared downstream subspace = coupling)

  INTERACT  iota_ij = || delta_ij - (delta_i + delta_j) || / (||delta_i||+||delta_j||)
            where delta_ij = compress BOTH i and j together.
            -> is the joint effect non-additive? (nonlinear coupling)

READ:
  off-diagonal ~ 0 (cos~0, iota~0)         -> errors INDEPENDENT -> NO graph, just a
                                              per-layer curve (heuristic, not structural)
  off-diagonal STRUCTURED (blocks/bands)   -> errors COUPLED in specific pairs -> that
                                              matrix IS the adjacency -> joint rank
                                              allocation needed -> real GRAPH novelty
  off-diagonal uniformly high              -> globally coupled (dense, not a sparse graph)
"""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
SEQ_LEN = 256
N_TOKENS = 1536
FRAC = 0.30
tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_pairwise_{tag}.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)

if hasattr(model.model, "decoder"):
    layers = model.model.decoder.layers; ffn_out = lambda L: L.fc2
else:
    layers = model.model.layers; ffn_out = lambda L: L.mlp.down_proj
N = len(layers)
Hdim = ffn_out(layers[0]).in_features
R = max(1, int(FRAC*Hdim))

with open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt") as f:
    text = f.read(N_TOKENS*8)
ids = tok(text, return_tensors="pt").input_ids[0][: ((N_TOKENS//SEQ_LEN)+1)*SEQ_LEN]
nseq = len(ids)//SEQ_LEN
seqs = ids[:nseq*SEQ_LEN].view(nseq, SEQ_LEN).to(device)

# ---- pass 1: per-layer FFN-hidden covariance + clean final ----
G = {b: torch.zeros(Hdim, Hdim, dtype=torch.float32, device=device) for b in range(N)}
def cap(b):
    def hook(m, i):
        x = i[0].reshape(-1, i[0].shape[-1]).float(); G[b] += x.t()@x
    return hook
hs = [ffn_out(L).register_forward_pre_hook(cap(b)) for b, L in enumerate(layers)]
with torch.no_grad():
    clean = model(seqs, output_hidden_states=True).hidden_states[-1].detach().float().reshape(-1).cpu()
for h in hs: h.remove()
Vr = {}
for b in range(N):
    _, evecs = torch.linalg.eigh(G[b].double()); Vr[b] = evecs[:, -R:].float().cpu(); G[b]=None

def run(compress):                                   # compress: set of layer idxs to rank-truncate
    hh=[]
    for i in compress:
        Vi = Vr[i].to(device)
        def mk(Vi):
            def ph(m, inp):
                x=inp[0]; xp=((x.float()@Vi)@Vi.t()).to(x.dtype); return (xp,)+inp[1:]
            return ph
        hh.append(ffn_out(layers[i]).register_forward_pre_hook(mk(Vi)))
    with torch.no_grad():
        f = model(seqs, output_hidden_states=True).hidden_states[-1].detach().float().reshape(-1).cpu()
    for h in hh: h.remove()
    torch.cuda.empty_cache()
    return f - clean                                 # delta vector

# ---- single-layer deltas ----
D = [run({i}) for i in range(N)]
nrm = torch.tensor([d.norm() for d in D])

# ---- overlap (cosine) matrix ----
COS = torch.zeros(N, N)
for i in range(N):
    for j in range(N):
        COS[i,j] = float(torch.dot(D[i], D[j])/(nrm[i]*nrm[j]+1e-8))

# ---- nonlinear interaction matrix (upper triangle) ----
IOTA = torch.zeros(N, N)
for i in range(N):
    for j in range(i+1, N):
        dij = run({i, j})
        IOTA[i,j] = IOTA[j,i] = float((dij-(D[i]+D[j])).norm()/(nrm[i]+nrm[j]+1e-8))

lines=[]
def log(s): lines.append(s); print(s)
log(f"PAIRWISE error interaction — graph or curve?")
log(f"MODEL={MODEL}  layers={N}  rank={R}/{Hdim}={FRAC:.0%}")

def offdiag(M):
    m=M.clone(); m.fill_diagonal_(0); idx=~torch.eye(N,dtype=torch.bool)
    return m[idx]
oc = offdiag(COS).abs(); oi = offdiag(IOTA)
log("")
log(f"OVERLAP |cos| off-diagonal:  mean={oc.mean():.3f}  median={oc.median():.3f}  max={oc.max():.3f}")
log(f"INTERACT iota off-diagonal:  mean={oi.mean():.3f}  median={oi.median():.3f}  max={oi.max():.3f}")
# adjacent vs distant (locality test)
adj_c=[abs(float(COS[i,i+1])) for i in range(N-1)]
dist_c=[abs(float(COS[i,j])) for i in range(N) for j in range(N) if abs(i-j)>=8]
adj_i=[float(IOTA[i,i+1]) for i in range(N-1)]
dist_i=[float(IOTA[i,j]) for i in range(N) for j in range(N) if abs(i-j)>=8]
log("")
log(f"locality |cos|:  adjacent(|i-j|=1)={sum(adj_c)/len(adj_c):.3f}   distant(|i-j|>=8)={sum(dist_c)/len(dist_c):.3f}")
log(f"locality iota :  adjacent(|i-j|=1)={sum(adj_i)/len(adj_i):.3f}   distant(|i-j|>=8)={sum(dist_i)/len(dist_i):.3f}")

# 8x8 block-averaged |cos| to reveal structure
B=8; step=max(1,N//B)
log("")
log(f"block-averaged |cos| ({step}-layer blocks) — look for off-diagonal structure:")
bl=list(range(0,N,step))
for a in bl:
    row=[]
    for c in bl:
        sub=COS[a:a+step, c:c+step].abs().mean(); row.append(f"{float(sub):.2f}")
    log("  "+" ".join(row))

log("")
log("VERDICT GUIDE:")
log("  off-diag ~0 & adjacent~distant  -> CURVE only (independent layers, heuristic)")
log("  adjacent >> distant             -> LOCAL/banded graph (neighbour coupling)")
log("  block structure off-diagonal    -> COMMUNITY graph (real structural novelty)")
log("  uniformly high everywhere       -> dense global coupling (not a sparse graph)")
torch.save({'COS':COS,'IOTA':IOTA,'nrm':nrm}, OUT.replace('.txt','.pt'))
open(OUT,"w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT} (+ .pt with full matrices)")
