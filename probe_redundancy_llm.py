"""
Probe: does an LLM have HIGHER-ORDER redundancy that pairwise methods miss?
LLM port of probe_redundancy.py (DeiT version).   python probe_redundancy_llm.py [hf_model]

For each decoder layer we capture the activations of its prunable channels:
  MLP hidden  = input to down_proj  (post-SiLU(gate)*up ; dim = intermediate_size)
  Attn out    = input to o_proj     (concatenated head value outputs ; dim = hidden)
and ask two questions:

  (1) PAIRWISE redundancy  — # channels that are near-duplicates of ANOTHER single
      channel (max |corr| > 0.9). This is all a graph method (GOHSP/DepGraph) sees.
  (2) TOTAL redundancy     — effective rank (participation ratio of covariance
      eigenvalues). C channels but eff rank r << C => C - r wasted dimensions.

HIGHER-ORDER gap = (total redundant dims) - (pairwise-duplicate channels).
Large gap => redundancy only a SET-level (hypergraph) method can exploit.

Default model is TinyLlama-1.1B (ungated, Llama-2 architecture). Pass a model id
(e.g. meta-llama/Llama-2-7b-hf with HF_TOKEN set) to reproduce on the 7B.
"""
import os, sys, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL    = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SEQ_LEN  = 256
N_SEQ    = 16            # calibration sequences  -> N_SEQ*SEQ_LEN token-vectors
MAX_TOKENS = 8000        # subsample this many token-vectors for covariance
CORR_DUP = 0.9           # |corr| above this = pairwise near-duplicate
tag = MODEL.split("/")[-1]
OUT = f"probe_redundancy_llm_{tag}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).eval().to(device)
layers = model.model.layers
L = len(layers)
print(f"Loaded {MODEL}: {L} layers, hidden={model.config.hidden_size}, "
      f"intermediate={model.config.intermediate_size}, device={device}")

# ---- calibration text: wikitext-2-raw (datasets cache), fallback to embedded ----
def get_text():
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        txt = "\n".join(t for t in ds["text"] if t.strip())
        if len(txt) > 2000:
            return txt
    except Exception as e:
        print("datasets load failed, using embedded text:", str(e)[:80])
    return ("The history of natural language processing began in the 1950s. " * 400)

ids = tok(get_text()[:1_000_000], return_tensors="pt").input_ids[0]
ids = ids[:N_SEQ * SEQ_LEN].view(N_SEQ, SEQ_LEN).to(device)

# ---- hooks: MLP hidden (down_proj input) + attn value output (o_proj input) ----
acts = {}
def mk_prehook(name):
    def hook(m, i):
        acts.setdefault(name, []).append(i[0].detach().reshape(-1, i[0].shape[-1]).float().cpu())
    return hook

handles = []
for b in range(L):
    handles.append(layers[b].mlp.down_proj.register_forward_pre_hook(mk_prehook(f"L{b:02d}_mlp_hidden")))
    handles.append(layers[b].self_attn.o_proj.register_forward_pre_hook(mk_prehook(f"L{b:02d}_attn_out")))

with torch.no_grad():
    for s in range(0, N_SEQ, 4):
        model(ids[s:s+4])
for h in handles:
    h.remove()

# ---- analysis (identical math to the DeiT probe) ----
def effective_rank(eigs):
    eigs = eigs.clamp(min=0); s = eigs.sum()
    return 0.0 if s <= 0 else float((s * s) / (eigs * eigs).sum())

def analyze(X):
    if X.shape[0] > MAX_TOKENS:
        X = X[torch.randperm(X.shape[0])[:MAX_TOKENS]]
    X = X - X.mean(0, keepdim=True)
    C = X.shape[1]
    cov = (X.T @ X) / (X.shape[0] - 1)
    eigs = torch.linalg.eigvalsh(cov)
    eff = effective_rank(eigs)
    d = torch.sqrt(torch.diag(cov)).clamp(min=1e-8)
    corr = cov / (d[:, None] * d[None, :]); corr.fill_diagonal_(0)
    max_per_ch = corr.abs().max(dim=1).values
    n_dup = int((max_per_ch > CORR_DUP).sum())
    total_redundant = C - eff
    return dict(C=C, eff_rank=round(eff, 1), eff_frac=round(eff / C, 3),
                mean_max_corr=round(float(max_per_ch.mean()), 3), n_pairwise_dup=n_dup,
                total_redundant=round(total_redundant, 1),
                higher_order_gap=round(total_redundant - n_dup, 1))

lines = []
def log(s): lines.append(s); print(s)

log(f"HIGHER-ORDER REDUNDANCY PROBE  MODEL={MODEL}  layers={L}")
log(f"tokens/layer={min(N_SEQ*SEQ_LEN, MAX_TOKENS)}  CORR_DUP={CORR_DUP}")
log("")
log(f"{'layer':<16} {'C':>6} {'eff_rank':>9} {'eff/C':>6} {'meanMaxCorr':>12} "
    f"{'pairDup':>8} {'totRedund':>10} {'HO_gap':>9}")
log("-" * 86)
results = {}
for name in sorted(acts.keys()):
    r = analyze(torch.cat(acts[name], 0)); results[name] = r
    log(f"{name:<16} {r['C']:>6} {r['eff_rank']:>9} {r['eff_frac']:>6} "
        f"{r['mean_max_corr']:>12} {r['n_pairwise_dup']:>8} {r['total_redundant']:>10} "
        f"{r['higher_order_gap']:>9}")

mlp  = [r for n, r in results.items() if 'mlp'  in n]
attn = [r for n, r in results.items() if 'attn' in n]
log("")
log("INTERPRETATION:")
log(f"  MLP  hidden: mean eff/C = {sum(r['eff_frac'] for r in mlp)/len(mlp):.3f}  (lower = more redundant)")
log(f"  MLP  mean pairwise-duplicate channels = {sum(r['n_pairwise_dup'] for r in mlp)/len(mlp):.0f} / {mlp[0]['C']}")
log(f"  MLP  mean HIGHER-ORDER gap = {sum(r['higher_order_gap'] for r in mlp)/len(mlp):.0f} dims "
    f"(redundancy invisible to pairwise methods)")
log(f"  Attn out: mean eff/C = {sum(r['eff_frac'] for r in attn)/len(attn):.3f}")
log(f"  Attn mean pairwise-duplicate channels = {sum(r['n_pairwise_dup'] for r in attn)/len(attn):.0f} / {attn[0]['C']}")
log(f"  Attn mean HIGHER-ORDER gap = {sum(r['higher_order_gap'] for r in attn)/len(attn):.0f} dims")
log("")
log("If HO_gap >> pairDup, pairwise graph methods (GOHSP/DepGraph) leave most")
log("redundancy on the table -> a set-level (hypergraph) method has real headroom.")

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\nSaved: {OUT}")
