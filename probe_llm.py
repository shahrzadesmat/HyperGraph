"""
Redundancy probe for LLMs.  Usage: python probe_llm.py <hf_model>

Measures the same redundancy map as the ViT probes, on an LLM:
  CROSS-TYPE  : reconstruct each layer's MLP/FFN contribution from its ATTENTION
                contribution (self + local), held-out, centered.
  CROSS-BLOCK : reconstruct each layer's MLP contribution from preceding layers'
                MLP contributions, held-out, centered.
Control = shuffled dictionary (held-out) -> should be ~1.0.

Handles OPT (model.model.decoder.layers, self_attn/fc2) and
Llama/Mistral (model.model.layers, self_attn/mlp.down_proj).
"""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
SEQ_LEN = 256
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 14000))
TRAIN_FRAC = 0.7
RIDGE = 1e-2
PRECEDING = 6
tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_llm_{tag}.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)

# locate layers + hook points per architecture
if hasattr(model.model, "decoder"):                 # OPT
    layers = model.model.decoder.layers
    attn_mod = lambda L: L.self_attn
    mlp_mod  = lambda L: L.fc2
else:                                                # Llama / Mistral / Qwen
    layers = model.model.layers
    attn_mod = lambda L: L.self_attn
    mlp_mod  = lambda L: L.mlp.down_proj
N = len(layers)

# text
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join(t for t in ds["text"] if len(t) > 50)
ncap = int(MAX_TOKENS * 1.4)
ids = tok(text, return_tensors="pt").input_ids[0][: ((ncap//SEQ_LEN)+1)*SEQ_LEN]
nseq = len(ids)//SEQ_LEN
seqs = ids[:nseq*SEQ_LEN].view(nseq, SEQ_LEN)

attn = {b: [] for b in range(N)}
mlp  = {b: [] for b in range(N)}
handles = []
def mk(store, b):
    def hook(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        store[b].append(out.detach().reshape(-1, out.shape[-1]).float().cpu())
    return hook
for b, L in enumerate(layers):
    handles.append(attn_mod(L).register_forward_hook(mk(attn, b)))
    handles.append(mlp_mod(L).register_forward_hook(mk(mlp, b)))

with torch.no_grad():
    for s in range(0, nseq, 8):
        batch = seqs[s:s+8].to(device)
        model(batch)
for h in handles: h.remove()
del model; torch.cuda.empty_cache()   # free model before the linear-algebra phase

ntok = torch.cat(attn[0],0).shape[0]
sel = torch.randperm(ntok)[:min(MAX_TOKENS, ntok)]
A  = {b: torch.cat(attn[b],0)[sel].to(device) for b in range(N)}
Mp = {b: torch.cat(mlp[b],0)[sel].to(device) for b in range(N)}
n = A[0].shape[0]; ntr = int(TRAIN_FRAC*n)
pm = torch.randperm(n, device=device); trI, teI = pm[:ntr], pm[ntr:]
def sc(X):
    mu = X[trI].mean(0, keepdim=True); return X[trI]-mu, X[teI]-mu
Atr,Ate,Mtr,Mte = {},{},{},{}
for b in range(N):
    Atr[b],Ate[b] = sc(A[b]); Mtr[b],Mte[b] = sc(Mp[b])
def fe(Ftr,Ytr,Fte,Yte,shuffle=False):
    if shuffle: Ftr = Ftr[torch.randperm(Ftr.shape[0], device=Ftr.device)]
    p = Ftr.shape[1]; G = Ftr.t()@Ftr; lam = RIDGE*torch.trace(G)/p
    M = torch.linalg.solve(G+lam*torch.eye(p, device=Ftr.device), Ftr.t()@Ytr)
    return float((Yte-Fte@M).norm()/Yte.norm())

lines=[];
def log(s): lines.append(s); print(s)
log(f"MODEL={MODEL}  layers={N}  hidden={A[0].shape[1]}  n_train={ntr} n_test={n-ntr}")

# CROSS-TYPE
log("\nCROSS-TYPE: mlp <- attention (held-out)")
log(f"{'lyr':>4} {'self':>7} {'local':>7} {'ctrl':>7}")
es, el = [], []
for i in range(N):
    e_self = fe(Atr[i], Mtr[i], Ate[i], Mte[i])
    loc=[j for j in (i-1,i,i+1) if 0<=j<N]
    Ftr=torch.cat([Atr[j] for j in loc],1); Fte=torch.cat([Ate[j] for j in loc],1)
    e_loc = fe(Ftr, Mtr[i], Fte, Mte[i])
    e_ctrl= fe(Atr[i], Mtr[i], Ate[i], Mte[i], shuffle=True)
    es.append(e_self); el.append(e_loc)
    log(f"{i:>4} {e_self:>7.3f} {e_loc:>7.3f} {e_ctrl:>7.3f}")
log(f"mean cross-type: self={sum(es)/len(es):.3f} local={sum(el)/len(el):.3f}")

# CROSS-BLOCK
log("\nCROSS-BLOCK: mlp <- preceding layers' mlp (held-out)")
ee=[]
for i in range(1,N):
    prev=[j for j in range(max(0,i-PRECEDING),i)]
    Ftr=torch.cat([Mtr[j] for j in prev],1); Fte=torch.cat([Mte[j] for j in prev],1)
    ee.append(fe(Ftr,Mtr[i],Fte,Mte[i]))
log(f"mean cross-block err = {sum(ee)/len(ee):.3f}  (control ~1.0)")
log("\nViT ref (DeiT-Small): cross-type local~0.72, cross-block~0.6-0.9 (weak).")
open(OUT,"w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
