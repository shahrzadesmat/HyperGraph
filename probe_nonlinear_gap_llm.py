"""
NONLINEAR-vs-LINEAR reconstruction gap probe for a decoder LM.
  python probe_nonlinear_gap_llm.py [hf_model]

Premise under test: LLM activations live on a NONLINEAR manifold whose intrinsic
dimension is below their linear (PCA/SVD) rank. Every SOTA compressor (SVD-LLM,
ASVD, FLAT-LLM, ESPACE, SliceGPT) is LINEAR, so any nonlinear redundancy is
unexploited. BUT a nonlinear decoder cannot be fused into a matmul -> it costs
extra inference compute. So the honest question is not "is there a gap" (known)
but "does a nonlinear bottleneck win at MATCHED BUDGET?".

For each (layer, site) we compress activations to a bottleneck r and compare
held-out reconstruction error:
  PCA@r         : linear, the SVD/ESPACE baseline (fusable, ~free at inference)
  AE@r          : nonlinear autoencoder bottleneck r (NOT fusable -> costs params/FLOPs)
  PCA@r_eq      : linear at rank r_eq chosen so linear params == AE params (matched budget)
  linAE@r       : linear autoencoder (sanity: should ~= PCA@r)

WIN  : err(AE@r) < err(PCA@r_eq)   -> nonlinear structure beats linear AT EQUAL COST
       -> genuinely exploitable redundancy no linear method can capture.
NULL : err(AE@r) ~= err(PCA@r_eq)  -> the gain is just "more params", not nonlinearity
       -> SVD/ESPACE already near-optimal at equal budget; idea does not pay for itself.

Error is relative Frobenius on HELD-OUT tokens, in standardized space.
"""
import os, sys, gc, math, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
SEQ_LEN, N_SEQ = 256, 128            # 32768 tokens (>> activation dim -> PCA well-posed)
MAX_TOKENS = 24000
TRAIN_FRAC = 0.8
RANKS = [32, 128]                    # bottleneck dims to test
AE_HIDDEN = 1024                     # encoder/decoder hidden width m
AE_STEPS, AE_BATCH, AE_LR = 2000, 1024, 1e-3
tag = MODEL.split("/")[-1]
OUT = f"probe_nonlinear_gap_llm_{tag}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    token=os.environ.get("HF_TOKEN")).eval().to(device)
mlayers = model.model.layers
N = len(mlayers)
LAYERS = sorted(set([0, N//5, 2*N//5, 3*N//5, 4*N//5, N-1]))   # ~6 representative layers
print(f"Loaded {MODEL}: {N} layers; probing layers {LAYERS}; device={device}")

def get_text():
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        t = "\n".join(x for x in ds["text"] if x.strip())
        if len(t) > 2000: return t
    except Exception as e:
        print("datasets load failed, embedded text:", str(e)[:80])
    return ("The history of natural language processing began in the 1950s. " * 800)

ids = tok(get_text()[:3_000_000], return_tensors="pt").input_ids[0]
ids = ids[:N_SEQ * SEQ_LEN].view(N_SEQ, SEQ_LEN).to(device)

# ---- capture down_proj input (MLP hidden) and o_proj input (attn value output) ----
store = {(l, s): [] for l in LAYERS for s in ("mlp_hidden", "attn_out")}
def mk(key):
    def hook(m, i):
        store[key].append(i[0].detach().reshape(-1, i[0].shape[-1]).half().cpu())
    return hook
handles = []
for l in LAYERS:
    handles.append(mlayers[l].mlp.down_proj.register_forward_pre_hook(mk((l, "mlp_hidden"))))
    handles.append(mlayers[l].self_attn.o_proj.register_forward_pre_hook(mk((l, "attn_out"))))
with torch.no_grad():
    for s in range(0, N_SEQ, 4):
        model(ids[s:s+4])
for h in handles: h.remove()
del model, mlayers, ids
gc.collect(); torch.cuda.empty_cache()

# ---- helpers ----
def standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp(min=1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd

def rel_err(Y, Yhat):
    return float((Y - Yhat).norm() / Y.norm())

def pca_err(Xtr, Xte, r):
    Vh = torch.linalg.svd(Xtr, full_matrices=False)[2]        # (k,d)
    P = Vh[:r].t()                                            # (d,r)
    return rel_err(Xte, (Xte @ P) @ P.t())

class AE(nn.Module):
    def __init__(self, d, m, r, nonlinear=True):
        super().__init__()
        act = nn.GELU() if nonlinear else nn.Identity()
        self.enc = nn.Sequential(nn.Linear(d, m), act, nn.Linear(m, r))
        act2 = nn.GELU() if nonlinear else nn.Identity()
        self.dec = nn.Sequential(nn.Linear(r, m), act2, nn.Linear(m, d))
    def forward(self, x):
        return self.dec(self.enc(x))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def train_ae(Xtr, Xte, d, r, nonlinear):
    ae = AE(d, AE_HIDDEN, r, nonlinear).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=AE_LR)
    n = Xtr.shape[0]
    ae.train()
    for step in range(AE_STEPS):
        idx = torch.randint(0, n, (AE_BATCH,), device=device)
        xb = Xtr[idx]
        loss = ((ae(xb) - xb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        err = rel_err(Xte, ae(Xte))
    return err, ae.n_params()

lines = []
def log(s): lines.append(s); print(s)
log(f"NONLINEAR-vs-LINEAR reconstruction gap  MODEL={MODEL}")
log(f"AE hidden m={AE_HIDDEN}, steps={AE_STEPS}; held-out Frobenius rel-err (standardized)")
log(f"WIN = AE@r < PCA@r_eq (nonlinear beats linear at MATCHED param budget)")
log("")
log(f"{'layer':>5} {'site':>11} {'d':>6} {'r':>4} {'PCA@r':>7} {'linAE@r':>8} "
    f"{'AE@r':>7} {'r_eq':>5} {'PCA@r_eq':>9} {'gap(win>0)':>11}")
log("-" * 92)

summary = {}
for l in LAYERS:
    for site in ("mlp_hidden", "attn_out"):
        X = torch.cat(store[(l, site)], 0)
        sel = torch.randperm(X.shape[0])[:min(MAX_TOKENS, X.shape[0])]
        X = X[sel].float()
        ntr = int(TRAIN_FRAC * X.shape[0])
        Xtr, Xte = standardize(X[:ntr].to(device), X[ntr:].to(device))
        d = X.shape[1]
        store[(l, site)] = None
        for r in RANKS:
            e_pca = pca_err(Xtr, Xte, r)
            e_lin, _ = train_ae(Xtr, Xte, d, r, nonlinear=False)
            e_ae, p_ae = train_ae(Xtr, Xte, d, r, nonlinear=True)
            r_eq = max(r, round(p_ae / d))             # linear basis with same #params
            r_eq = min(r_eq, min(Xtr.shape) - 1)
            e_eq = pca_err(Xtr, Xte, r_eq)
            gap = e_eq - e_ae                          # >0 => nonlinear wins at matched budget
            summary[(l, site, r)] = gap
            log(f"{l:>5} {site:>11} {d:>6} {r:>4} {e_pca:>7.3f} {e_lin:>8.3f} "
                f"{e_ae:>7.3f} {r_eq:>5} {e_eq:>9.3f} {gap:>+11.3f}")
        del Xtr, Xte; torch.cuda.empty_cache()

log("")
wins = [g for g in summary.values() if g > 0.005]
log(f"matched-budget WINS (gap>0.005): {len(wins)}/{len(summary)}   mean gap={sum(summary.values())/len(summary):+.3f}")
log("")
log("READ:")
log("  PCA@r vs AE@r large  -> nonlinear gap EXISTS (known geometry).")
log("  AE@r < PCA@r_eq      -> nonlinear gap is EXPLOITABLE at equal cost (the real result).")
log("  linAE@r ~= PCA@r     -> AE training is sound (linear AE matches PCA).")
log("  If AE@r ~= PCA@r_eq  -> linear SVD/ESPACE already near-optimal per-budget; idea does not pay off.")
open(OUT, "w").write("\n".join(lines) + "\n")
print("Saved:", OUT)
