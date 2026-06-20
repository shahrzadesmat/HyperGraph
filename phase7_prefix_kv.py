"""
phase7_prefix_kv.py — DIRECTION 2: cross-request / shared-prefix KV compressibility.
Agentic/RAG/multi-turn serving reuses ONE prefix's KV across MANY requests with DIFFERENT continuations.
Question: how hard can we compress a prefix's KV (continuation-AGNOSTIC: calibrated on the prefix alone)
while preserving the model's BEHAVIOR on diverse query-probes? If very compressible & consistent across
probes -> compress the shared prefix once, reuse cheaply -- a cross-request lever single-seq methods ignore.

Metric = KL( full-prefix next-token dist || compressed-prefix next-token dist ) at the probe positions.
KL directly measures how much compressing the prefix CHANGES behavior (non-circular; captures exactly the
prefix's influence). Per layer the prefix K&V are compressed to rank-d via the prefix's OWN PCA (SVD).
Report mean and WORST-CASE KL over (prefix x probe) vs d. WIN: KL stays tiny to aggressive d (e.g. >=8-16x)
with low spread -> prefix KV is far more compressible than single-seq KV (~2-3x), and it transfers.
env: P7_L=1024  P7_CT=256  P7_NSCEN=4  P7_NPROBE=6  P7_DS=16,32,64,128,256
"""
import os, math, time, torch
device = torch.device("cuda")
MODEL = os.environ.get("P7_MODEL", "meta-llama/Llama-2-7b-hf")
L = int(os.environ.get("P7_L", "1024")); CT = int(os.environ.get("P7_CT", "256"))
SINK = int(os.environ.get("P7_SINK", "4"))      # keep first SINK prefix tokens FULL (attention sinks, incompressible)
NSCEN = int(os.environ.get("P7_NSCEN", "4")); NPROBE = int(os.environ.get("P7_NPROBE", "6"))
DS = sorted(int(x) for x in os.environ.get("P7_DS", "16,32,64,128,256").split(","))
OUT = "/work/hdd/bdjd/hypergraph_pruning/phase7_prefix_kv.txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); C = model.config.hidden_size; LAYERS = list(range(NL))
maxd = max(DS)
test_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(test_txt, return_tensors="pt").input_ids[0]
# prefixes (NSCEN non-overlapping L-windows) + a shared pool of NPROBE diverse probes (later CT-windows)
prefixes = [ids[i * L:(i + 1) * L] for i in range(NSCEN)]
base = NSCEN * L
probes = [ids[base + i * CT: base + (i + 1) * CT] for i in range(NPROBE)]
lg(f"phase7_prefix_kv  MODEL={MODEL}  C={C}  L={L} CT={CT}  scenarios={NSCEN} probes={NPROBE}  d={DS} (comp={[C//d for d in DS]}x)")

# ---------- single hook per proj: capture prefix K/V, or compress prefix positions ----------
CAP = {}; CURB = {}; DD = [0]; MODE = ["none"]          # MODE: capture / surg / none
def hook(l, which):
    key = (l, which)
    def h(m, i, o):
        if MODE[0] == "capture":
            CAP[key] = o.detach()[0, :L].float(); return o          # [L, C] prefix activations
        if MODE[0] == "surg":
            mu, Vk = CURB[key]; d = DD[0]; sh = o.shape; X = o.reshape(-1, C).float()
            Xp = ((X[SINK:L] - mu) @ Vk[:, :d]) @ Vk[:, :d].t() + mu  # compress prefix[SINK:L]; keep sinks & continuation full
            out = X.clone(); out[SINK:L] = Xp
            return out.reshape(sh).to(o.dtype)
        return o
    return h
for l in LAYERS:
    attn[l].k_proj.register_forward_hook(hook(l, "k"))
    attn[l].v_proj.register_forward_hook(hook(l, "v"))

def probe_logprobs(seq):                                            # log-softmax at positions predicting probe tokens
    with torch.no_grad():
        lo = model(seq.unsqueeze(0).to(device)).logits[0, L - 1:seq.shape[0] - 1].float()   # [CT, V]
    return torch.log_softmax(lo, -1)

# ---------- per scenario: build prefix basis (SVD), then KL(full || compressed) across probes ----------
KL = {d: [] for d in DS}
for s in range(NSCEN):
    pref = prefixes[s]
    MODE[0] = "capture"; _ = model(pref.unsqueeze(0).to(device))    # fills CAP for this prefix
    for l in LAYERS:
        for which in ("k", "v"):
            X = CAP[(l, which)][SINK:]; mu = X.mean(0, keepdim=True)   # basis from NON-sink prefix tokens
            Vh = torch.linalg.svd(X - mu, full_matrices=False).Vh    # [min(L-SINK,C), C]
            CURB[(l, which)] = (mu, Vh.t()[:, :maxd].contiguous())    # (mu[1,C], Vk[C,maxd])
    for p in range(NPROBE):
        seq = torch.cat([pref, probes[p]])
        MODE[0] = "none"; lpf = probe_logprobs(seq); pf = lpf.exp()
        for d in DS:
            MODE[0] = "surg"; DD[0] = d; lpc = probe_logprobs(seq)
            KL[d].append(float((pf * (lpf - lpc)).sum(-1).mean()))
    lg(f"  scenario {s+1}/{NSCEN} done  ({time.time()-t0:.0f}s)")
    CAP.clear()

# ---------- report ----------
lg("=" * 80)
lg("KL( full-prefix || compressed-prefix ) at probe positions, over prefix x probe (nats; lower=better).")
lg(f"{'d':>5} {'comp':>6} | {'mean KL':>10} {'worst KL':>10} {'median KL':>10}")
lg("-" * 80)
for d in DS:
    v = torch.tensor(KL[d]); lg(f"{d:>5} {C//d:>5}x | {v.mean():>10.4f} {v.max():>10.4f} {v.median():>10.4f}"); flush()
lg("-" * 80)
mild = max(DS); aggr = min(DS)
lg(f"READ: KL near 0 means compressing the shared prefix barely changes behavior -> reuse it cheaply.")
lg(f"  prefix tolerates up to {C//aggr}x with mean KL={torch.tensor(KL[aggr]).mean():.4f}; "
   f"single-seq KV is ~2-3x near-lossless. If KL stays tiny here at >>3x -> cross-request lever is REAL.")
lg(f"  low worst-vs-mean spread => the SAME prefix compression transfers across diverse probes (the agentic win).")
lg(f"total: {time.time()-t0:.0f}s"); flush(); print("Saved:", OUT)
