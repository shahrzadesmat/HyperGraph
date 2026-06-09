"""
Edge-vs-Node head importance probe (v6, definitive).  python probe_edge_node.py <hf_model>

Decisive question: does EDGE (composition centrality) predict head criticality
BEYOND the real head-importance baselines — gradient/Taylor importance (what
SlimGPT/HIS use) AND depth (amplification) — ?

Signals per head:
  NODE_attn : retrieval attention from answer pos to needle (HeadKV-style)
  NODE_grad : gradient/Taylor importance  sum_tok (sum_{d in head} g_d a_d)^2
              (g = dL/d(head output), a = head output) -> the SlimGPT/HIS signal
  EDGE      : composition centrality sum_{downstream B, X in QKV} ||W_X^B @ W_O^h||^2
  DEPTH     : layer index (amplification proxy: early layers critical)
  CRIT      : ground-truth = answer-NLL increase when head ablated (solved ex only)

VALID if criticality is predictable by SOME known signal (|corr|>0.15).
DECISIVE: partial(EDGE, CRIT | NODE_grad, DEPTH).  >>0 -> edge has UNIQUE signal
beyond gradient-importance and depth -> novel. ~0 -> edge is subsumed (stop).
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
cfg = model.config; L, H = cfg.num_hidden_layers, cfg.num_attention_heads
Dh = cfg.hidden_size // H; layers = model.model.layers

# ---------- long single-mention needle examples (v5: solvable + criticality spread) ----------
import random
rng = random.Random(0)
filler = "The lamps flickered softly in the long quiet hall that evening. "
Q = "The secret digit is"
def line(d, k): return f"{Q} {d}. " + filler*k + f"{Q} {d}.\n"
bpref = tok(Q, add_special_tokens=False).input_ids
exs = []
for i in range(N_EX):
    v = str(i % 10)
    demos = [str(x) for x in rng.sample([n for n in range(10) if n != i % 10], 2)]
    text = line(demos[0], 2) + f"{Q} {v}. " + filler*(10+i%4) + Q
    pid = tok(text, return_tensors="pt").input_ids[0].tolist()
    ctoks = tok(f"{Q} {v}", add_special_tokens=False).input_ids[len(bpref):]
    tid = int(ctoks[-1]); ids = pid + ctoks[:-1]
    occ = [p for p, t in enumerate(ids) if t == tid]
    if occ: exs.append((torch.tensor(ids), tid, int(occ[0])))

# ---------- baseline + solve-check + NODE_attn ----------
NODE_attn = np.zeros((L, H)); base_nll = {}; solved = []
with torch.no_grad():
    for i, (ids, tid, nvp) in enumerate(exs):
        out = model(ids.unsqueeze(0).to(device), output_attentions=True)
        logit = out.logits[0, -1].float()
        base_nll[i] = -float(torch.log_softmax(logit, -1)[tid])
        if int(logit.argmax()) == tid:
            solved.append(i)
            for l in range(L):
                NODE_attn[l] += out.attentions[l][0, :, -1, nvp].float().cpu().numpy()
        del out; torch.cuda.empty_cache()
if solved: NODE_attn /= len(solved)

# ---------- NODE_grad: gradient/Taylor head importance (SlimGPT/HIS-style) ----------
for p in model.parameters(): p.requires_grad_(False)
acts, grads = {}, {}
def seed(m, i, o): o.requires_grad_(True); return o
def cap(l):
    def pre(m, inp):
        a = inp[0]
        if a.requires_grad:
            acts[l] = a
            a.register_hook(lambda g, l=l: grads.__setitem__(l, g.detach()))
    return pre
eh = model.model.embed_tokens.register_forward_hook(seed)
chs = [layers[l].self_attn.o_proj.register_forward_pre_hook(cap(l)) for l in range(L)]
NODE_grad = np.zeros((L, H))
torch.set_grad_enabled(True)
for i in solved:
    ids, tid, _ = exs[i]
    out = model(ids.unsqueeze(0).to(device))
    loss = -torch.log_softmax(out.logits[0, -1].float(), -1)[tid]
    acts.clear(); grads.clear(); loss.backward()
    for l in range(L):
        a = acts[l].detach().float(); g = grads[l].float()
        ga = (g * a).reshape(-1, H, Dh).sum(-1)              # [tok, H] per-head Taylor
        NODE_grad[l] += (ga ** 2).sum(0).cpu().numpy()
    model.zero_grad(set_to_none=True); del out, loss; torch.cuda.empty_cache()
torch.set_grad_enabled(False)
eh.remove(); [h.remove() for h in chs]
if solved: NODE_grad /= len(solved)

# ---------- EDGE: composition centrality ----------
EDGE = np.zeros((L, H))
with torch.no_grad():
    for la in range(L):
        Wo = layers[la].self_attn.o_proj.weight.float()
        for lb in range(la+1, min(L, la+1+BAND)):
            for proj in ("q_proj", "k_proj", "v_proj"):
                R2 = (getattr(layers[lb].self_attn, proj).weight.float() @ Wo) ** 2
                for a in range(H): EDGE[la, a] += float(R2[:, a*Dh:(a+1)*Dh].sum())
        torch.cuda.empty_cache()

# ---------- CRIT: ablation on solved examples ----------
sol = [exs[i] for i in solved]; CRIT = np.zeros((L, H))
def ablate(h):
    def pre(m, inp):
        x = inp[0].clone(); x[..., h*Dh:(h+1)*Dh] = 0; return (x,) + inp[1:]
    return pre
if sol:
    ml = max(len(ids) for ids,_,_ in sol); Bs = len(sol)
    bt = torch.full((Bs, ml), tok.eos_token_id or 0, dtype=torch.long); am = torch.zeros((Bs, ml), dtype=torch.long)
    lp_i, tg = [], []
    for j,(ids,tid,_) in enumerate(sol):
        bt[j,:len(ids)] = ids; am[j,:len(ids)] = 1; lp_i.append(len(ids)-1); tg.append(tid)
    bt = bt.to(device); am = am.to(device); base_sol = np.array([base_nll[i] for i in solved])
    with torch.no_grad():
        for l in range(L):
            for h in range(H):
                hk = layers[l].self_attn.o_proj.register_forward_pre_hook(ablate(h))
                lg = model(bt, attention_mask=am).logits.float(); hk.remove()
                ab = np.array([-float(torch.log_softmax(lg[j, lp_i[j]], -1)[tg[j]]) for j in range(Bs)])
                CRIT[l, h] = (ab - base_sol).mean()
            torch.cuda.empty_cache()

# ---------- correlations + multi-partial ----------
DEPTH = np.repeat(np.arange(L), H).astype(float).reshape(L, H)   # layer index
def rank(x):
    o = x.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(x)); return r
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/(np.sqrt((a*a).sum())*np.sqrt((b*b).sum())+1e-12))
def spear(a,b): return pear(rank(a), rank(b))
def resid(y, Xs):
    A = np.column_stack([np.ones(len(y))]+Xs); beta,_,_,_ = np.linalg.lstsq(A, y, rcond=None); return y - A@beta
def partial(e, c, conds):
    return pear(resid(rank(e), [rank(x) for x in conds]), resid(rank(c), [rank(x) for x in conds]))
na, ng, e, c, d = NODE_attn.flatten(), NODE_grad.flatten(), EDGE.flatten(), CRIT.flatten(), DEPTH.flatten()

lines=[]; lg=lambda s:(lines.append(s),print(s))
lg(f"EDGE-vs-NODE (v6 definitive)  MODEL={MODEL}  L={L} H={H} band={BAND}")
lg(f"VALIDITY: solved={len(solved)}/{len(exs)}  baseNLL={np.mean([base_nll[i] for i in solved]) if solved else float('nan'):.3f}"
   f"  CRIT=[{c.min():+.3f},{c.max():+.3f}]")
lg("")
lg(f"Spearman( NODE_attn , CRIT ) = {spear(na,c):+.3f}")
lg(f"Spearman( NODE_grad , CRIT ) = {spear(ng,c):+.3f}   (gradient/Taylor = SlimGPT/HIS signal)")
lg(f"Spearman( DEPTH     , CRIT ) = {spear(d,c):+.3f}   (amplification proxy; neg = early critical)")
lg(f"Spearman( EDGE      , CRIT ) = {spear(e,c):+.3f}")
lg("")
lg(f"partial( EDGE, CRIT | NODE_grad )         = {partial(e,c,[ng]):+.3f}")
lg(f"partial( EDGE, CRIT | NODE_grad, DEPTH )  = {partial(e,c,[ng,d]):+.3f}   <-- DECISIVE")
lg("")
best = max(abs(spear(ng,c)), abs(spear(na,c)), abs(spear(d,c)))
if len(solved) < 0.5*len(exs) or best < 0.15:
    lg(f">> INVALID: best known signal corr={best:.3f} <0.15 -> criticality not predictable, test inconclusive.")
else:
    lg(f">> VALID (best known signal corr={best:.3f}).  partial(EDGE|grad,depth)>>0 -> edge has UNIQUE")
    lg("   signal beyond gradient-importance & depth -> novel. ~0 -> subsumed by known signals (stop).")
def top(x,k=6): idx=np.argsort(-x.flatten())[:k]; return [(int(i//H),int(i%H)) for i in idx]
lg(f"\ntop CRIT: {top(CRIT)}\ntop NODE_grad: {top(NODE_grad)}\ntop EDGE: {top(EDGE)}")
open(OUT,"w").write("\n".join(lines)+"\n"); print("Saved:", OUT)
