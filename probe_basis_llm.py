"""
Basis-vs-unit probe for LLMs.  Usage: python probe_basis_llm.py <hf_model>

Same test as probe_basis.py but on an LLM's FFN hidden neurons (the units LLM
structured pruning removes). For each layer, at kept-dim budgets k, compare
held-out reconstruction error of the FFN-hidden activations under:
  (a) CHANNEL subset  : keep best k neurons (pivoted Cholesky), LS-reconstruct
  (b) LOW-RANK        : keep top-k PCA directions, project

err_channel >> err_lowrank  =>  FFN redundancy is dense/basis-reachable, not
unit-aligned -> the basis-coupling-graph direction applies to LLMs too.
"""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hyperedge_prune import pivoted_cholesky

MODEL = sys.argv[1]
SEQ_LEN = 256
MAX_TOKENS = 12000
TRAIN_FRAC = 0.7
RIDGE = 1e-3
FRACS = [0.05, 0.10, 0.20, 0.30]
tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_basis_llm_{tag}.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)

if hasattr(model.model, "decoder"):                 # OPT: hidden = fc2 input
    layers = model.model.decoder.layers
    hidden_mod = lambda L: L.fc2
else:                                                # Llama/Mistral: hidden = down_proj input
    layers = model.model.layers
    hidden_mod = lambda L: L.mlp.down_proj
N = len(layers)
Hdim = hidden_mod(layers[0]).in_features

with open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt") as f:
    text = f.read(int(MAX_TOKENS * 8))
ids = tok(text, return_tensors="pt").input_ids[0][: ((MAX_TOKENS//SEQ_LEN)+2)*SEQ_LEN]
nseq = len(ids)//SEQ_LEN
seqs = ids[:nseq*SEQ_LEN].view(nseq, SEQ_LEN)

hid = {b: [] for b in range(N)}
handles = []
def mk(b):
    def hook(m, i):                                  # forward_pre_hook: capture INPUT (FFN hidden)
        x = i[0]
        hid[b].append(x.detach().reshape(-1, x.shape[-1]).half().cpu())
    return hook
for b, L in enumerate(layers):
    handles.append(hidden_mod(L).register_forward_pre_hook(mk(b)))
with torch.no_grad():
    for s in range(0, nseq, 8):
        model(seqs[s:s+8].to(device))
for h in handles: h.remove()
del model; torch.cuda.empty_cache()

ntok = torch.cat(hid[0],0).shape[0]
sel = torch.randperm(ntok)[:min(MAX_TOKENS, ntok)]
n = sel.shape[0]; ntr = int(TRAIN_FRAC*n)
pm = torch.randperm(n); trI, teI = pm[:ntr], pm[ntr:]
ks = [max(1, int(f*Hdim)) for f in FRACS]
kmax = max(ks)

lines=[]
def log(s): lines.append(s); print(s)
log(f"BASIS vs UNIT (LLM FFN hidden) — held-out reconstruction error")
log(f"MODEL={MODEL}  layers={N}  ffn_hidden={Hdim}  n_train={ntr} n_test={n-ntr}")
log(f"{'lyr':>4} | " + " ".join(f"k={k}".center(15) for k in ks))
log(f"{'':>4} | " + " ".join("chan / lowrank".center(15) for _ in ks))
log("-"*(7+16*len(ks)))

gap_acc = {k:[] for k in ks}
for b in range(N):
    Hall = torch.cat(hid[b],0)[sel].float().to(device); hid[b]=None
    mu = Hall[trI].mean(0, keepdim=True)
    Htr = Hall[trI]-mu; Hte = Hall[teI]-mu
    cov = (Htr.t()@Htr).double()
    perm, _ = pivoted_cholesky(cov, maxk=kmax)
    evals, evecs = torch.linalg.eigh(cov)
    nrm = Hte.norm(); cells=[]
    for k in ks:
        K = torch.tensor(perm[:k], device=device)
        Xk = Htr[:, K]
        W = torch.linalg.solve(Xk.t()@Xk + RIDGE*torch.eye(k,device=device)*(Xk.t()@Xk).diagonal().mean(),
                               Xk.t()@Htr)
        err_ch = float((Hte - Hte[:,K]@W).norm()/nrm)
        Vk = evecs[:, -k:].float()
        err_lr = float((Hte - (Hte@Vk)@Vk.t()).norm()/nrm)
        gap_acc[k].append(err_ch - err_lr)
        cells.append(f"{err_ch:.3f}/{err_lr:.3f}".center(15))
    del Hall, Htr, Hte, cov, evecs; torch.cuda.empty_cache()
    log(f"{b:>4} | " + " ".join(cells))

log("")
log("mean error gap (channel - lowrank), per k:")
for k in ks:
    g = sum(gap_acc[k])/len(gap_acc[k])
    log(f"  k={k:5d} ({k/Hdim*100:.0f}% of ffn):  gap = {g:+.3f}")
log("")
log("READ: large positive gap = low-rank reconstructs FFN hidden far better than")
log("channel subset at same budget -> redundancy DENSE, unreachable by neuron")
log("removal -> basis-coupling pruning has headroom in LLMs too.")
open(OUT,"w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
