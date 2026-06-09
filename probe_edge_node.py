"""
Edge-vs-Node head importance probe.   python probe_edge_node.py <hf_model>

Tests the weakness HeadKV/RazorAttention ADMIT: they score each head's
importance INDEPENDENTLY (node-level), but functional circuits are COMPOSITIONAL
(an induction head = two heads composing). If a head's COMPOSITION centrality
(graph edge structure) predicts its true criticality BEYOND its independent
node score, then circuit/edge importance carries signal node-level methods miss.

Per head h we compute:
  NODE[h]  : retrieval score — attention from the answer position to the needle
             token (the standard retrieval-head signal HeadKV uses).
  EDGE[h]  : composition centrality — sum over downstream heads B and X in {Q,K,V}
             of ||W_X^B @ W_O^h||_F^2  (how strongly downstream heads READ head h's
             output). Weight-based, banded to the next BAND layers (our coupling
             finding says coupling is local).
  CRIT[h]  : ground-truth criticality — increase in needle-answer NLL when head h
             is ablated.

DECISIVE NUMBER: partial Spearman corr( EDGE, CRIT | NODE ).
  >> 0  -> composition graph predicts criticality beyond node score -> real signal,
           the circuit-graph angle has headroom over HeadKV.
  ~ 0  -> edge adds nothing over node -> ties HeadKV, circuit graph inert (stop).
"""
import os, sys, math, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
BAND  = 6
N_EX  = 24
OUT   = f"/work/hdd/bdjd/hypergraph_pruning/probe_edge_node_{MODEL.split('/')[-1]}.txt"
device = torch.device("cuda")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             attn_implementation="eager").eval().to(device)
cfg = model.config
L, H = cfg.num_hidden_layers, cfg.num_attention_heads
Dh = cfg.hidden_size // H
layers = model.model.layers

# ---------- build needle examples (single-token answers) ----------
filler = ("The hallway was long and the lamps flickered softly in the evening air. "
          "People walked by without noticing the quiet hum of the old building. ")
vals = [str(v) for v in [42,73,15,88,31,64,97,26,53,19,77,38,61,45,92,14,
                         57,83,29,66,71,48,35,90]][:N_EX]
exs = []
for i, v in enumerate(vals):
    pre = filler * (2 + (i % 3))
    post = filler * (2 + ((i+1) % 3))
    needle = f"The secret number is {v}. "
    prompt = pre + needle + post + "The secret number is"
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    tgt = tok(f" {v}", add_special_tokens=False).input_ids[0]   # first answer token
    # locate the needle value-token position: tokenize prefix up to the value
    pos_ids = tok(pre + "The secret number is", return_tensors="pt").input_ids[0]
    needle_val_pos = len(pos_ids)            # the value token follows " is"
    exs.append((ids, tgt, needle_val_pos))
maxlen = max(len(ids) for ids,_,_ in exs)
B = len(exs)
batch = torch.full((B, maxlen), tok.eos_token_id or 0, dtype=torch.long)
attn_mask = torch.zeros((B, maxlen), dtype=torch.long)
last_pos, tgts, needle_pos = [], [], []
for i,(ids,tgt,nv) in enumerate(exs):
    batch[i,:len(ids)] = ids; attn_mask[i,:len(ids)] = 1
    last_pos.append(len(ids)-1); tgts.append(tgt); needle_pos.append(nv)
batch = batch.to(device); attn_mask = attn_mask.to(device)
last_pos = torch.tensor(last_pos); tgts = torch.tensor(tgts); needle_pos = torch.tensor(needle_pos)

def answer_nll(logits):           # logits [B, S, V] -> mean NLL of answer token
    lp = torch.log_softmax(logits.float(), -1)
    nll = [-float(lp[i, last_pos[i], tgts[i]]) for i in range(B)]
    return np.array(nll)

# ---------- baseline + NODE score (attention to needle), one example at a time ----------
NODE = np.zeros((L, H))
base_nll = np.zeros(B)
with torch.no_grad():
    for i in range(B):
        ids = batch[i:i+1, :last_pos[i]+1]
        out = model(ids, output_attentions=True)
        base_nll[i] = -float(torch.log_softmax(out.logits[0, last_pos[i]].float(), -1)[tgts[i]])
        for l in range(L):
            a = out.attentions[l][0, :, last_pos[i], needle_pos[i]]   # [H] attn last->needle
            NODE[l] += a.float().cpu().numpy()
        del out; torch.cuda.empty_cache()
NODE /= B

# ---------- EDGE score: composition centrality (weights, banded) ----------
EDGE = np.zeros((L, H))
with torch.no_grad():
    for la in range(L):
        Wo = layers[la].self_attn.o_proj.weight.float()        # [d_model, H*Dh]
        for lb in range(la+1, min(L, la+1+BAND)):
            for proj in ("q_proj","k_proj","v_proj"):
                Wx = getattr(layers[lb].self_attn, proj).weight.float()  # [Hkv*Dh, d_model]
                R = Wx @ Wo                                    # [Hkv*Dh, H*Dh]
                R2 = (R*R)
                # column-block a (head a of source) Frobenius^2 = sum over all downstream rows
                for a in range(H):
                    EDGE[la, a] += float(R2[:, a*Dh:(a+1)*Dh].sum())
        torch.cuda.empty_cache()

# ---------- CRIT: ablate each head, measure answer-NLL increase ----------
def ablate_hook(h):
    def pre(m, inp):
        x = inp[0].clone(); x[..., h*Dh:(h+1)*Dh] = 0; return (x,)+inp[1:]
    return pre
CRIT = np.zeros((L, H))
with torch.no_grad():
    for l in range(L):
        for h in range(H):
            hk = layers[l].self_attn.o_proj.register_forward_pre_hook(ablate_hook(h))
            out = model(batch, attention_mask=attn_mask)
            hk.remove()
            CRIT[l, h] = (answer_nll(out.logits) - base_nll).mean()
            del out
        torch.cuda.empty_cache()

# ---------- correlations ----------
def rank(x):
    o = x.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(x)); return r
def pear(a,b):
    a=a-a.mean(); b=b-b.mean()
    d=(np.sqrt((a*a).sum())*np.sqrt((b*b).sum()))+1e-12
    return float((a*b).sum()/d)
def spear(a,b): return pear(rank(a), rank(b))
n = NODE.flatten(); e = EDGE.flatten(); c = CRIT.flatten()
r_nc, r_ec, r_ne = spear(n,c), spear(e,c), spear(n,e)
partial = (r_ec - r_ne*r_nc)/math.sqrt(max(1e-12,(1-r_ne**2)*(1-r_nc**2)))   # partial spearman(e,c|n)

lines=[]
def log(s): lines.append(s); print(s)
log(f"EDGE-vs-NODE head importance   MODEL={MODEL}  L={L} H={H}  band={BAND}  n_ex={B}")
log(f"baseline answer NLL mean={base_nll.mean():.3f}")
log("")
log(f"Spearman( NODE , CRIT )            = {r_nc:+.3f}   (HeadKV-style node score vs true criticality)")
log(f"Spearman( EDGE , CRIT )            = {r_ec:+.3f}   (composition centrality vs criticality)")
log(f"Spearman( NODE , EDGE )            = {r_ne:+.3f}   (do the two signals overlap?)")
log(f"PARTIAL Spearman( EDGE, CRIT|NODE) = {partial:+.3f}   <-- DECISIVE")
log("")
log("READ: partial >> 0  -> edge/composition predicts criticality BEYOND node score")
log("      -> circuit-graph importance carries signal HeadKV misses (headroom).")
log("      partial ~ 0   -> edge adds nothing over node -> ties HeadKV (stop).")
# top heads by each signal (sanity)
def top(x,k=8):
    idx=np.argsort(-x.flatten())[:k]; return [(int(i//H),int(i%H)) for i in idx]
log("")
log(f"top CRIT heads : {top(CRIT)}")
log(f"top NODE heads : {top(NODE)}")
log(f"top EDGE heads : {top(EDGE)}")
open(OUT,"w").write("\n".join(lines)+"\n")
print("Saved:", OUT)
