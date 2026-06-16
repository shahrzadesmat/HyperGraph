"""
Cross-TYPE redundancy probe (attention <-> MLP) for a decoder LM. Held-out.
  python probe_crosstype_llm.py [hf_model]

LLM port of probe_crosstype.py. In a Llama block:
    h = x + self_attn(norm(x))         # attn CONTRIBUTION = self_attn output
    y = h + mlp(norm(h))               # mlp  CONTRIBUTION = mlp output
both live in the same residual space (hidden_size) -> directly comparable.

Question: can a layer's MLP contribution be reconstructed from the ATTENTION
contribution (same / neighbouring layers)?  Low held-out error with control ~1.0
=> attn and MLP do overlapping work => cross-type redundancy that isomorphic
pruning's within-type assumption is structurally blind to.

Memory: activations are kept on CPU (fp16); the model is freed after capture and
only per-layer slices are moved to the GPU for each regression.
"""
import os, sys, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
SEQ_LEN, N_SEQ = 256, 96
MAX_TOKENS = 24000
TRAIN_FRAC = 0.7
RIDGE = 1e-2
tag = MODEL.split("/")[-1]
OUT = f"probe_crosstype_llm_{tag}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    token=os.environ.get("HF_TOKEN")).eval().to(device)
layers = model.model.layers
N = len(layers)
print(f"Loaded {MODEL}: {N} layers, hidden={model.config.hidden_size}, device={device}")

def get_text():
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        txt = "\n".join(t for t in ds["text"] if t.strip())
        if len(txt) > 2000:
            return txt
    except Exception as e:
        print("datasets load failed, embedded text:", str(e)[:80])
    return ("The history of natural language processing began in the 1950s. " * 600)

ids = tok(get_text()[:2_000_000], return_tensors="pt").input_ids[0]
ids = ids[:N_SEQ * SEQ_LEN].view(N_SEQ, SEQ_LEN).to(device)

attn = {b: [] for b in range(N)}   # attention residual contribution (hidden)
mlp  = {b: [] for b in range(N)}   # mlp residual contribution (hidden)
handles = []
def mk(store, b):
    def hook(m, i, o):
        out = o[0] if isinstance(o, (tuple, list)) else o
        store[b].append(out.detach().reshape(-1, out.shape[-1]).half().cpu())
    return hook
for b in range(N):
    handles.append(layers[b].self_attn.register_forward_hook(mk(attn, b)))
    handles.append(layers[b].mlp.register_forward_hook(mk(mlp, b)))
with torch.no_grad():
    for s in range(0, N_SEQ, 4):
        model(ids[s:s+4])
for h in handles:
    h.remove()

# ---- subsample on CPU, then FREE the model (regression doesn't need it) ----
ntok = torch.cat(attn[0], 0).shape[0]
sel = torch.randperm(ntok)[:min(MAX_TOKENS, ntok)]
A  = {b: torch.cat(attn[b], 0)[sel] for b in range(N)}   # CPU fp16
Mp = {b: torch.cat(mlp[b],  0)[sel] for b in range(N)}   # CPU fp16
del attn, mlp, model, layers, ids
gc.collect(); torch.cuda.empty_cache()

n = A[0].shape[0]; ntr = int(TRAIN_FRAC * n)
perm = torch.randperm(n); trI, teI = perm[:ntr], perm[ntr:]   # CPU indices

def cen(store_b):
    """CPU fp16 (n,d) -> (Xtr, Xte) on GPU, fp32, centered with TRAIN mean."""
    X = store_b.float()
    mu = X[trI].mean(0, keepdim=True)
    return (X[trI] - mu).to(device), (X[teI] - mu).to(device)

def fit_eval(Ftr, Ytr, Fte, Yte, shuffle=False):
    if shuffle:
        Ftr = Ftr[torch.randperm(Ftr.shape[0], device=Ftr.device)]
    p = Ftr.shape[1]; G = Ftr.t() @ Ftr
    lam = RIDGE * torch.trace(G) / p
    M = torch.linalg.solve(G + lam * torch.eye(p, device=Ftr.device), Ftr.t() @ Ytr)
    return float((Yte - Fte @ M).norm() / Yte.norm())

lines = []
def log(s): lines.append(s); print(s)
log(f"CROSS-TYPE redundancy (LLM): reconstruct MLP contribution from ATTENTION (held-out)")
log(f"MODEL={MODEL}  layers={N}  n_train={ntr} n_test={n-ntr}")
log(f"{'lyr':>4} {'mlp<-attn_self':>15} {'mlp<-attn_local':>16} {'control':>9}")
log("-" * 48)
es, el = [], []
for i in range(N):
    Atr_i, Ate_i = cen(A[i])
    Mtr_i, Mte_i = cen(Mp[i])
    e_self = fit_eval(Atr_i, Mtr_i, Ate_i, Mte_i)
    e_ctrl = fit_eval(Atr_i, Mtr_i, Ate_i, Mte_i, shuffle=True)
    loc = [j for j in (i-1, i, i+1) if 0 <= j < N]
    ltr, lte = [], []
    for j in loc:
        if j == i:
            ltr.append(Atr_i); lte.append(Ate_i)
        else:
            tj, vj = cen(A[j]); ltr.append(tj); lte.append(vj)
    e_loc = fit_eval(torch.cat(ltr, 1), Mtr_i, torch.cat(lte, 1), Mte_i)
    es.append(e_self); el.append(e_loc)
    log(f"{i:>4} {e_self:>15.3f} {e_loc:>16.3f} {e_ctrl:>9.3f}")
    del Atr_i, Ate_i, Mtr_i, Mte_i, ltr, lte
    torch.cuda.empty_cache()
log(f"mean: self={sum(es)/len(es):.3f}  local={sum(el)/len(el):.3f}  (control ~1.0)")
log("")
log("READ: low held-out err + control ~1.0 -> MLP work is partly reproducible from")
log("ATTENTION -> cross-type redundancy exists -> isomorphic's within-type assumption")
log("leaves redundancy unexploited. If err ~ 1.0, attn & MLP are independent.")
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\nSaved: {OUT}")
