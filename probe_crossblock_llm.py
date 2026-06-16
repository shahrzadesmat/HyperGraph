"""
Cross-BLOCK redundancy probe for a decoder LM. Held-out, centered.
  python probe_crossblock_llm.py [hf_model]

LLM port of probe_crossblock.py. Two questions:
  Part A (block-level): can layer i's MLP contribution Δ_i be reconstructed from
                        OTHER layers' contributions Δ_j ?
  Part B (neuron-level): can a LATE layer's Δ_i be reconstructed from EARLIER
                        layers' actual hidden NEURONS (down_proj input)?

err near 0 AND control near 1.0 -> real, non-trivial cross-block redundancy.

SCALE NOTE: at 7B, Δ is hidden_size (4096) and hidden is intermediate (11008),
so concatenating raw signals over ~32 layers is intractable. We PCA-reduce each
layer's signal on the TRAIN split before regressing (delta -> K_D, hidden -> K_H).

Memory: activations stay on CPU (fp16); the model is freed after capture. Only the
small PCA-reduced features (K_D / K_H per layer) live on the GPU; raw targets are
moved per-layer.
"""
import os, sys, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
SEQ_LEN, N_SEQ = 256, 64
MAX_TOKENS = 12000
TRAIN_FRAC = 0.7
RIDGE = 1e-3
K_D = 128            # PCA comps kept per layer for Δ (block contribution)
K_H = 256            # PCA comps kept per layer for hidden neurons
tag = MODEL.split("/")[-1]
OUT = f"probe_crossblock_llm_{tag}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    token=os.environ.get("HF_TOKEN")).eval().to(device)
layers = model.model.layers
N = len(layers)
LATE = [N-4, N-3, N-2, N-1]
print(f"Loaded {MODEL}: {N} layers, hidden={model.config.hidden_size}, "
      f"intermediate={model.config.intermediate_size}, device={device}")

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

delta  = {b: [] for b in range(N)}   # mlp output (hidden)
hidden = {b: [] for b in range(N)}   # down_proj input = post-act MLP hidden (intermediate)
handles = []
def mk_out(store, b):
    def hook(m, i, o):
        store[b].append(o.detach().reshape(-1, o.shape[-1]).half().cpu())
    return hook
def mk_in(store, b):
    def hook(m, i):
        store[b].append(i[0].detach().reshape(-1, i[0].shape[-1]).half().cpu())
    return hook
for b in range(N):
    handles.append(layers[b].mlp.register_forward_hook(mk_out(delta, b)))
    handles.append(layers[b].mlp.down_proj.register_forward_pre_hook(mk_in(hidden, b)))
with torch.no_grad():
    for s in range(0, N_SEQ, 2):
        model(ids[s:s+2])
for h in handles:
    h.remove()

ntok = torch.cat(delta[0], 0).shape[0]
sel = torch.randperm(ntok)[:min(MAX_TOKENS, ntok)]
delta  = {b: torch.cat(delta[b],  0)[sel] for b in range(N)}   # CPU fp16
hidden = {b: torch.cat(hidden[b], 0)[sel] for b in range(N)}   # CPU fp16
del model, layers, ids
gc.collect(); torch.cuda.empty_cache()

n = sel.shape[0]; ntr = int(TRAIN_FRAC * n)
gperm = torch.randperm(n); trI, teI = gperm[:ntr], gperm[ntr:]

def center_cpu(store_b):
    """CPU fp16 (n,d) -> (Xtr, Xte) CPU fp16, centered with TRAIN mean."""
    X = store_b.float()
    mu = X[trI].mean(0, keepdim=True)
    return (X[trI] - mu).half(), (X[teI] - mu).half()

def reduce_block(tr_cpu, te_cpu, k):
    """PCA basis from TRAIN (on GPU); return reduced (n,k) on GPU fp32."""
    Xtr = tr_cpu.float().to(device)
    Vh = torch.linalg.svd(Xtr, full_matrices=False)[2]
    P = Vh[:k].t()
    Rtr = Xtr @ P
    Rte = te_cpu.float().to(device) @ P
    del Xtr
    return Rtr, Rte

def ls_err(Ftr, Ytr, Fte, Yte, shuffle=False):
    if shuffle:
        Ftr = Ftr[torch.randperm(Ftr.shape[0], device=Ftr.device)]
    p = Ftr.shape[1]; G = Ftr.t() @ Ftr
    lam = RIDGE * torch.trace(G) / p
    M = torch.linalg.solve(G + lam * torch.eye(p, device=Ftr.device), Ftr.t() @ Ytr)
    return float((Yte - Fte @ M).norm() / Yte.norm())

# ---- precompute centered Δ (CPU targets) + PCA-reduced Δ features (GPU) ----
Dtr_cpu, Dte_cpu, DR_tr, DR_te = {}, {}, {}, {}
for b in range(N):
    dtr, dte = center_cpu(delta[b])
    Dtr_cpu[b], Dte_cpu[b] = dtr, dte
    DR_tr[b], DR_te[b] = reduce_block(dtr, dte, K_D)   # small (n,K_D) on GPU
del delta
gc.collect(); torch.cuda.empty_cache()

def target(b, test=False):
    return (Dte_cpu[b] if test else Dtr_cpu[b]).float().to(device)

lines = []
def log(s): lines.append(s); print(s)

# ---- Part A: block-level ----
log(f"CROSS-BLOCK redundancy (LLM)  MODEL={MODEL}  layers={N}  n_train={ntr} n_test={n-ntr}")
log(f"(features PCA-reduced: Δ->{K_D} comps/layer, hidden->{K_H} comps/layer)")
log("")
log(f"PART A  block-level: reconstruct Δ_i from OTHER layers' Δ (PCA-{K_D})")
log(f"{'layer':>5} {'err_allOthers':>14} {'err_earlierOnly':>16} {'err_control':>12}")
log("-" * 52)
ea, ec = [], []
for i in range(N):
    Yt, Yv = target(i), target(i, test=True)
    others_tr = torch.cat([DR_tr[j] for j in range(N) if j != i], 1)
    others_te = torch.cat([DR_te[j] for j in range(N) if j != i], 1)
    e_all  = ls_err(others_tr, Yt, others_te, Yv)
    e_ctrl = ls_err(others_tr, Yt, others_te, Yv, shuffle=True)
    if i > 0:
        et = torch.cat([DR_tr[j] for j in range(i)], 1)
        ev = torch.cat([DR_te[j] for j in range(i)], 1)
        e_caus = ls_err(et, Yt, ev, Yv); ec.append(e_caus); cs = f"{e_caus:.3f}"
        del et, ev
    else:
        cs = "  n/a"
    ea.append(e_all)
    log(f"{i:>5} {e_all:>14.3f} {cs:>16} {e_ctrl:>12.3f}")
    del Yt, Yv, others_tr, others_te; torch.cuda.empty_cache()
log(f"mean: all-others={sum(ea)/len(ea):.3f}   earlier-only={sum(ec)/len(ec):.3f}   (control ~1.0)")

# ---- Part B: late layers from EARLIER hidden neurons (PCA-K_H per layer) ----
log("")
log(f"PART B  late layers reconstructed from EARLIER hidden NEURONS (causal, PCA-{K_H})")
log(f"{'layer':>5} {'#earlierLayers':>15} {'feat_dim':>9} {'err_neuron':>11} {'err_control':>12}")
log("-" * 56)
HR_cache = {}
def hidden_reduced(b):
    if b not in HR_cache:
        htr, hte = center_cpu(hidden[b])
        HR_cache[b] = reduce_block(htr, hte, K_H)
    return HR_cache[b]
for i in LATE:
    feats_tr = torch.cat([hidden_reduced(j)[0] for j in range(i)], 1)
    feats_te = torch.cat([hidden_reduced(j)[1] for j in range(i)], 1)
    Yt, Yv = target(i), target(i, test=True)
    e   = ls_err(feats_tr, Yt, feats_te, Yv)
    ctl = ls_err(feats_tr, Yt, feats_te, Yv, shuffle=True)
    log(f"{i:>5} {i:>15} {feats_tr.shape[1]:>9} {e:>11.3f} {ctl:>12.3f}")
    del feats_tr, feats_te, Yt, Yv; torch.cuda.empty_cache()

log("")
log("READ: centered err near 0 with control near 1.0 -> genuine cross-block redundancy.")
log("Part B shows whether late layers are reproducible from earlier NEURONS (the fold")
log("units) -> exploitable for cross-block pruning. PCA reduction keeps it tractable at 7B.")
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\nSaved: {OUT}")
