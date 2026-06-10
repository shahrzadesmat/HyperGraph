"""
Circuit-asymmetric vs symmetric allocation — the METHOD-WIN experiment.
  python probe_circuit_alloc.py <hf_model>

The QK/OV probe showed QK and OV circuits compress DIFFERENTLY (OV more, depth-
structured). This tests whether exploiting that asymmetry actually WINS: at a
MATCHED total rank budget, does allocating rank per-circuit by its measured
effective rank beat splitting the budget symmetrically?

We low-rank-project (activation-aware) each head's Q, K (QK circuit, rank r_qk)
and value output / o_proj input (OV circuit, rank r_ov), then measure WikiText
perplexity. Three allocations at each budget R (avg rank per circuit-head):
  SYMMETRIC : r_qk = r_ov = R                       (MLA/FLAT-LLM-style uniform)
  ASYMMETRIC: r_qk,r_ov ∝ measured effrank, mean R  (ours — spend rank where needed)
  (matched total budget by construction)

WIN: ppl(ASYM) < ppl(SYM) at the same budget -> the circuit asymmetry is exploitable.
"""
import os, sys, math, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
SEQ_LEN, N_CAL, EVAL_SEQ, EVAL_LEN = 256, 16, 12, 512
BUDGETS = [16, 24, 32]                              # avg rank per circuit-head (out of d_head)
tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_circuit_alloc_{tag}.txt"
device = torch.device("cuda")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
cfg = model.config; L, Hq = cfg.num_hidden_layers, cfg.num_attention_heads
Hkv = getattr(cfg, "num_key_value_heads", Hq); Dh = cfg.hidden_size // Hq; layers = model.model.layers
assert Hkv == Hq, "this version assumes MHA (Llama-2); GQA needs per-kv-head handling"

raw = open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt").read()
allids = tok(raw[:3_000_000], return_tensors="pt").input_ids[0]
cal = allids[:N_CAL*SEQ_LEN].view(N_CAL, SEQ_LEN).to(device)
ev  = allids[N_CAL*SEQ_LEN: N_CAL*SEQ_LEN + EVAL_SEQ*EVAL_LEN].view(EVAL_SEQ, EVAL_LEN).to(device)

# ---- capture Q, K, AO(value output = o_proj input) on calibration ----
Q={l:[] for l in range(L)}; K={l:[] for l in range(L)}; AO={l:[] for l in range(L)}
hs=[]
def cap_out(s,l):
    def h(m,i,o): s[l].append(o.detach().reshape(-1,o.shape[-1]).float().cpu())
    return h
def cap_in(s,l):
    def h(m,i): s[l].append(i[0].detach().reshape(-1,i[0].shape[-1]).float().cpu())
    return h
for l in range(L):
    a=layers[l].self_attn
    hs.append(a.q_proj.register_forward_hook(cap_out(Q,l)))
    hs.append(a.k_proj.register_forward_hook(cap_out(K,l)))
    hs.append(a.o_proj.register_forward_pre_hook(cap_in(AO,l)))
with torch.no_grad():
    for s in range(0,N_CAL,4): model(cal[s:s+4])
for h in hs: h.remove()

# ---- per-head bases (desc eigvecs) + effrank ----
Vq=torch.zeros(L,Hq,Dh,Dh,dtype=torch.float16); Vk=Vq.clone(); Vov=Vq.clone()
QKeff=np.zeros((L,Hq)); OVeff=np.zeros((L,Hq))
def basis_eff(M):                                  # M [tok,Dh] -> (Vdesc [Dh,Dh], effrank@90)
    M=(M-M.mean(0,keepdim=True)).to(device)
    cov=(M.t()@M).double()
    evl,evec=torch.linalg.eigh(cov)
    Vd=evec.flip(1).float()
    e=evl.flip(0).clamp(min=0); c=torch.cumsum(e,0)/e.sum().clamp(min=1e-12)
    eff=int((c<0.90).sum().item())+1
    return Vd.half().cpu(), eff
for l in range(L):
    Qc=torch.cat(Q[l],0); Kc=torch.cat(K[l],0); Ac=torch.cat(AO[l],0); Q[l]=K[l]=AO[l]=None
    for hh in range(Hq):
        vq,eq=basis_eff(Qc[:,hh*Dh:(hh+1)*Dh]); vk,ek=basis_eff(Kc[:,hh*Dh:(hh+1)*Dh])
        vo,eo=basis_eff(Ac[:,hh*Dh:(hh+1)*Dh])
        Vq[l,hh]=vq; Vk[l,hh]=vk; Vov[l,hh]=vo
        QKeff[l,hh]=0.5*(eq+ek); OVeff[l,hh]=eo
    del Qc,Kc,Ac; torch.cuda.empty_cache()
Vq=Vq.to(device); Vk=Vk.to(device); Vov=Vov.to(device)
gmean=(QKeff.sum()+OVeff.sum())/(2*L*Hq)

# ---- allocations (matched total budget: global mean rank = R for both) ----
def alloc_sym(R):  return np.full((L,Hq),R,int), np.full((L,Hq),R,int)
def alloc_asym(R):
    c=R/gmean
    rqk=np.clip(np.round(QKeff*c),1,Dh).astype(int)
    rov=np.clip(np.round(OVeff*c),1,Dh).astype(int)
    return rqk,rov

def build_P(Vbasis,r):                              # -> P [L,Hq,Dh,Dh]
    P=torch.empty(L,Hq,Dh,Dh,device=device,dtype=torch.float16)
    for l in range(L):
        for h in range(Hq):
            V=Vbasis[l,h,:,:int(r[l,h])].float()
            P[l,h]=(V@V.t()).half()
    return P

def mk_proj_out(P,l):                               # project q_proj/k_proj OUTPUT per head
    def h(m,i,o):
        B,S,_=o.shape
        x=o.view(B,S,Hq,Dh).float()
        x=torch.einsum('bshd,hde->bshe',x,P[l].float())
        return x.reshape(B,S,Hq*Dh).to(o.dtype)
    return h
def mk_proj_in(P,l):                                # project o_proj INPUT (value output) per head
    def pre(m,inp):
        x=inp[0]; B,S,_=x.shape
        xx=x.view(B,S,Hq,Dh).float()
        xx=torch.einsum('bshd,hde->bshe',xx,P[l].float())
        return (xx.reshape(B,S,Hq*Dh).to(x.dtype),)+inp[1:]
    return pre

def perplexity(rqk,rov):
    Pq=build_P(Vq,rqk); Pk=build_P(Vk,rqk); Pov=build_P(Vov,rov)
    hh=[]
    for l in range(L):
        a=layers[l].self_attn
        hh.append(a.q_proj.register_forward_hook(mk_proj_out(Pq,l)))
        hh.append(a.k_proj.register_forward_hook(mk_proj_out(Pk,l)))
        hh.append(a.o_proj.register_forward_pre_hook(mk_proj_in(Pov,l)))
    nll=0.0; ntok=0
    with torch.no_grad():
        for s in range(EVAL_SEQ):
            x=ev[s:s+1]; out=model(x,labels=x)
            nll+=float(out.loss)*(EVAL_LEN-1); ntok+=EVAL_LEN-1
    for h in hh: h.remove()
    del Pq,Pk,Pov; torch.cuda.empty_cache()
    return math.exp(nll/ntok)

def ppl_base():
    nll=0.0; ntok=0
    with torch.no_grad():
        for s in range(EVAL_SEQ):
            x=ev[s:s+1]; out=model(x,labels=x); nll+=float(out.loss)*(EVAL_LEN-1); ntok+=EVAL_LEN-1
    return math.exp(nll/ntok)

lines=[]; lg=lambda s:(lines.append(s),print(s))
lg(f"CIRCUIT-ASYMMETRIC vs SYMMETRIC allocation  MODEL={MODEL}  L={L} Hq={Hq} d_head={Dh}")
lg(f"QK_effrank mean={QKeff.mean():.1f}  OV_effrank mean={OVeff.mean():.1f}  global_mean={gmean:.1f}")
base=ppl_base(); lg(f"\nNO-COMPRESSION perplexity = {base:.3f}")
lg(f"\n{'budget R':>9} {'sym_ppl':>9} {'asym_ppl':>9} {'asym vs sym':>12}")
lg("-"*44)
res=[]
for R in BUDGETS:
    rqk_s,rov_s=alloc_sym(R); rqk_a,rov_a=alloc_asym(R)
    ps=perplexity(rqk_s,rov_s); pa=perplexity(rqk_a,rov_a)
    gain=(ps-pa)/ps*100
    res.append((R,ps,pa,gain))
    lg(f"{R:>9} {ps:>9.3f} {pa:>9.3f} {gain:>+11.2f}%")
lg("")
lg("WIN if asym_ppl < sym_ppl (positive %): circuit-asymmetric allocation beats symmetric")
lg("at matched budget -> the QK/OV compressibility asymmetry is exploitable -> the method works.")
# show the asym schedule at the middle budget (per-layer means) for the figure
R=BUDGETS[len(BUDGETS)//2]; rqk_a,rov_a=alloc_asym(R)
lg(f"\nasym schedule at R={R} (per-layer mean kept rank):")
lg(f"{'layer':>5} {'r_qk':>6} {'r_ov':>6}")
for l in range(0,L,max(1,L//16)):
    lg(f"{l:>5} {rqk_a[l].mean():>6.1f} {rov_a[l].mean():>6.1f}")
open(OUT,"w").write("\n".join(lines)+"\n"); print("Saved:",OUT)
