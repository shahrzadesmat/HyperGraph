"""
Error-propagation probe.  Usage: python probe_errorprop.py <hf_model>

Tests the weakness shared by ASVD/SVD-LLM/FLAT-LLM: they minimize PER-LAYER
reconstruction error and allocate ranks greedily, IGNORING how a layer's
compression error propagates through the residual stream to the output.

For each layer i, we low-rank-truncate ONLY that layer's FFN hidden (rank-r PCA
projection, the same primitive FLAT-LLM uses) and measure:
  delta_local_i : absolute change in layer i's FFN OUTPUT (the residual it injects)
                  -> this is what per-layer methods minimize
  delta_e2e_i   : absolute change in the model's FINAL hidden state
                  -> the actual end-to-end effect
  amplification = delta_e2e_i / delta_local_i

If amplification >> 1 and VARIES sharply with depth, error propagation is real
and structured -> a propagation graph + JOINT rank allocation beats per-layer
greedy (the SOTA's blind spot). If amplification ~ 1 for all layers, errors don't
propagate and the idea is dead.
"""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
SEQ_LEN = 256
N_TOKENS = 4000
FRAC = 0.30                                   # keep this fraction of FFN rank
tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_errorprop_{tag}.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)

if hasattr(model.model, "decoder"):                 # OPT
    layers = model.model.decoder.layers
    ffn_out = lambda L: L.fc2
else:                                                # Llama/Mistral
    layers = model.model.layers
    ffn_out = lambda L: L.mlp.down_proj
N = len(layers)
Hdim = ffn_out(layers[0]).in_features
R = max(1, int(FRAC * Hdim))

with open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt") as f:
    text = f.read(N_TOKENS * 8)
ids = tok(text, return_tensors="pt").input_ids[0][: ((N_TOKENS//SEQ_LEN)+1)*SEQ_LEN]
nseq = len(ids)//SEQ_LEN
seqs = ids[:nseq*SEQ_LEN].view(nseq, SEQ_LEN).to(device)

# ---- Pass 1 (clean): accumulate FFN-hidden covariance + capture clean FFN outputs ----
G = {b: torch.zeros(Hdim, Hdim, dtype=torch.float32, device=device) for b in range(N)}
clean_ffn = {}
handles = []
def cap_in(b):
    def hook(m, i):                                  # down_proj input = FFN hidden
        x = i[0].reshape(-1, i[0].shape[-1]).float()
        G[b] += x.t() @ x
    return hook
def cap_out(b):
    def hook(m, i, o):                               # down_proj output = FFN contribution
        clean_ffn[b] = o.detach().float().cpu()
    return hook
for b, L in enumerate(layers):
    handles.append(ffn_out(L).register_forward_pre_hook(cap_in(b)))
    handles.append(ffn_out(L).register_forward_hook(cap_out(b)))
with torch.no_grad():
    clean_final = model(seqs, output_hidden_states=True).hidden_states[-1].detach().float().cpu()
for h in handles: h.remove()

# ---- per-layer top-R PCA basis (CPU, moved to GPU when used) ----
Vr = {}
for b in range(N):
    evals, evecs = torch.linalg.eigh(G[b].double())
    Vr[b] = evecs[:, -R:].float().cpu()
    G[b] = None

# ---- per-layer: compress only layer i, measure local + end-to-end deltas ----
lines=[]
def log(s): lines.append(s); print(s)
log(f"ERROR PROPAGATION — compress ONLY layer i (rank {R}/{Hdim} = {FRAC:.0%} FFN)")
log(f"MODEL={MODEL}  layers={N}  ffn_hidden={Hdim}")
log(f"{'lyr':>4} {'delta_local':>12} {'delta_e2e':>11} {'amplif(e2e/loc)':>16} {'e2e_rel':>9}")
log("-"*58)
cfn_norm = {b: clean_ffn[b].norm() for b in range(N)}
final_norm = clean_final.norm()
amps=[]
for i in range(N):
    Vi = Vr[i].to(device)
    comp_ffn_holder = {}
    def proj_hook(m, inp):
        x = inp[0]
        xf = x.float()
        xp = (xf @ Vi) @ Vi.t()                      # rank-R projection of FFN hidden
        return (xp.to(x.dtype),) + inp[1:]
    def grab_out(m, i_, o):
        comp_ffn_holder['o'] = o.detach().float().cpu()
    h1 = ffn_out(layers[i]).register_forward_pre_hook(proj_hook)
    h2 = ffn_out(layers[i]).register_forward_hook(grab_out)
    with torch.no_grad():
        comp_final = model(seqs, output_hidden_states=True).hidden_states[-1].detach().float().cpu()
    h1.remove(); h2.remove(); Vi = None; torch.cuda.empty_cache()
    d_local = float((clean_ffn[i] - comp_ffn_holder['o']).norm())
    d_e2e   = float((clean_final - comp_final).norm())
    amp = d_e2e / (d_local + 1e-8)
    amps.append(amp)
    log(f"{i:>4} {d_local:>12.2f} {d_e2e:>11.2f} {amp:>16.2f} {d_e2e/float(final_norm):>9.4f}")

log("")
log(f"amplification (e2e/local): mean={sum(amps)/len(amps):.2f}  min={min(amps):.2f}  max={max(amps):.2f}")
log("READ: amplification >> 1 and VARYING with depth => a layer's compression error")
log("propagates/amplifies to the output differently per layer => per-layer greedy")
log("allocation (FLAT-LLM/ASVD/SVD-LLM) is suboptimal => error-propagation graph +")
log("JOINT rank allocation is the novel structural fix. If amplification ~1 flat, dead.")
open(OUT,"w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
