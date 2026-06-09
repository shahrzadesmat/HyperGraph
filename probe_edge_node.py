"""
Edge-vs-Node head importance probe (v2, validity-gated).  python probe_edge_node.py <hf_model>

Tests whether a head's COMPOSITION centrality (edge/graph structure) predicts its
true criticality BEYOND its independent NODE score (HeadKV-style).

VALIDITY GATES (v1 failed these -> was uninterpretable):
  - the model must actually SOLVE the retrieval task (greedy answer correct, low NLL)
  - NODE must predict CRIT (known-good signal) -> if not, criticality is noise, abort read

Per head:
  NODE = retrieval attention from answer position to the needle digit (HeadKV signal)
  EDGE = composition centrality: sum_{downstream B, X in QKV} ||W_X^B @ W_O^h||^2 (banded)
  CRIT = increase in answer NLL when head ablated, over SOLVED examples only

DECISIVE: partial Spearman(EDGE, CRIT | NODE).  >>0 = edge has signal node misses.
"""
import os, sys, math, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
BAND, N_EX = 6, 48
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_edge_node_{MODEL.split('/')[-1]}.txt"
device = torch.device("cuda")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             attn_implementation="eager").eval().to(device)
cfg = model.config
L, H = cfg.num_hidden_layers, cfg.num_attention_heads
Dh = cfg.hidden_size // H
layers = model.model.layers

# ---------- needle examples: single-digit secret (one-token answer) ----------
import random
rng = random.Random(0)
filler = "The lamps flickered softly in the long quiet hall that evening. "
Q = "The secret digit is"
def line(d, k):
    return f"{Q} {d}. " + filler*k + f"{Q} {d}.\n"
bpref = tok(Q, add_special_tokens=False).input_ids                 # tokens for "...is"
exs = []
for i in range(N_EX):
    v = str(i % 10)
    demos = [str(x) for x in rng.sample([n for n in range(10) if n != i % 10], 2)]
    # 1 short demo (format) + LONG single-mention query -> stresses sparse retrieval heads
    text = line(demos[0], 2) + f"{Q} {v}. " + filler*(10+i%4) + Q
    pid = tok(text, return_tensors="pt").input_ids[0].tolist()
    ctoks = tok(f"{Q} {v}", add_special_tokens=False).input_ids[len(bpref):]  # tokens for " v": [space,digit] or [digit]
    tid = int(ctoks[-1])                                            # the DIGIT token
    ids = pid + ctoks[:-1]                                          # append leading space so next-token target = digit
    occ = [p for p,t in enumerate(ids) if t == tid]                # demos use other digits -> first occ = query needle
    if not occ:
        continue
    exs.append((torch.tensor(ids), tid, int(occ[0])))

# ---------- baseline + solve-check + NODE (one example at a time, with attentions) ----------
NODE = np.zeros((L, H)); base_nll = {}; solved = []
with torch.no_grad():
    for i, (ids, tid, nvp) in enumerate(exs):
        out = model(ids.unsqueeze(0).to(device), output_attentions=True)
        logit = out.logits[0, -1].float()
        nll = -float(torch.log_softmax(logit, -1)[tid])
        pred = int(logit.argmax())
        base_nll[i] = nll
        if pred == tid:
            solved.append(i)
            for l in range(L):
                NODE[l] += out.attentions[l][0, :, -1, nvp].float().cpu().numpy()
        del out; torch.cuda.empty_cache()
if solved: NODE /= len(solved)

# ---------- EDGE: composition centrality (weights, banded) ----------
EDGE = np.zeros((L, H))
with torch.no_grad():
    for la in range(L):
        Wo = layers[la].self_attn.o_proj.weight.float()             # [d, H*Dh]
        for lb in range(la + 1, min(L, la + 1 + BAND)):
            for proj in ("q_proj", "k_proj", "v_proj"):
                Wx = getattr(layers[lb].self_attn, proj).weight.float()
                R2 = (Wx @ Wo) ** 2                                  # [*, H*Dh]
                for a in range(H):
                    EDGE[la, a] += float(R2[:, a*Dh:(a+1)*Dh].sum())
        torch.cuda.empty_cache()

# ---------- CRIT: ablate each head, measure NLL increase over SOLVED examples ----------
sol = [exs[i] for i in solved]
CRIT = np.zeros((L, H))
def ablate(h):
    def pre(m, inp):
        x = inp[0].clone(); x[..., h*Dh:(h+1)*Dh] = 0; return (x,) + inp[1:]
    return pre
if sol:
    # pad solved into a batch
    ml = max(len(ids) for ids,_,_ in sol); Bs = len(sol)
    bt = torch.full((Bs, ml), tok.eos_token_id or 0, dtype=torch.long)
    am = torch.zeros((Bs, ml), dtype=torch.long); lp_i = []; tg = []
    for j,(ids,tid,_) in enumerate(sol):
        bt[j,:len(ids)] = ids; am[j,:len(ids)] = 1; lp_i.append(len(ids)-1); tg.append(tid)
    bt = bt.to(device); am = am.to(device)
    base_sol = np.array([base_nll[i] for i in solved])
    with torch.no_grad():
        for l in range(L):
            for h in range(H):
                hk = layers[l].self_attn.o_proj.register_forward_pre_hook(ablate(h))
                out = model(bt, attention_mask=am)
                hk.remove()
                lg = out.logits.float()
                ab = np.array([-float(torch.log_softmax(lg[j, lp_i[j]], -1)[tg[j]]) for j in range(Bs)])
                CRIT[l, h] = (ab - base_sol).mean()
                del out
            torch.cuda.empty_cache()

# ---------- correlations ----------
def rank(x):
    o = x.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(x)); return r
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); d=np.sqrt((a*a).sum())*np.sqrt((b*b).sum())+1e-12
    return float((a*b).sum()/d)
def spear(a,b): return pear(rank(a), rank(b))
n, e, c = NODE.flatten(), EDGE.flatten(), CRIT.flatten()
r_nc, r_ec, r_ne = spear(n,c), spear(e,c), spear(n,e)
partial = (r_ec - r_ne*r_nc)/math.sqrt(max(1e-12,(1-r_ne**2)*(1-r_nc**2)))

lines=[]
def log(s): lines.append(s); print(s)
log(f"EDGE-vs-NODE (v2)  MODEL={MODEL}  L={L} H={H} band={BAND}")
log(f"VALIDITY:  examples={len(exs)}  SOLVED(greedy correct)={len(solved)}/{len(exs)}"
    f"   mean baseline NLL(solved)={np.mean([base_nll[i] for i in solved]) if solved else float('nan'):.3f}")
log(f"           CRIT range=[{CRIT.min():+.3f},{CRIT.max():+.3f}]  (need spread for signal)")
log("")
log(f"GATE  Spearman(NODE,CRIT)        = {r_nc:+.3f}   (must be clearly >0 for a VALID test)")
log(f"      Spearman(EDGE,CRIT)        = {r_ec:+.3f}")
log(f"      Spearman(NODE,EDGE)        = {r_ne:+.3f}")
log(f"DECISIVE  partial(EDGE,CRIT|NODE)= {partial:+.3f}")
log("")
if len(solved) < 0.5*len(exs) or abs(r_nc) < 0.15:
    log(">> INVALID: task not solved or NODE doesn't predict CRIT -> criticality is noise,")
    log("   the partial number is NOT interpretable. Need a stronger task/ablation.")
else:
    log(">> VALID test. partial>>0 -> edge adds signal beyond node (headroom);")
    log("   partial~0 -> edge ties node (HeadKV is enough, stop).")
def top(x,k=8):
    idx=np.argsort(-x.flatten())[:k]; return [(int(i//H),int(i%H)) for i in idx]
log(f"\ntop CRIT heads: {top(CRIT)}")
log(f"top NODE heads: {top(NODE)}")
log(f"top EDGE heads: {top(EDGE)}")
open(OUT,"w").write("\n".join(lines)+"\n"); print("Saved:", OUT)
