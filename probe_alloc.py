"""
Allocation showdown.  Usage: python probe_alloc.py <hf_model>

THE payoff test. Compress every FFN layer with activation-aware low-rank (the
FLAT-LLM primitive), to a FIXED total rank budget, allocated three ways, and
compare held-out WikiText perplexity:

  (a) UNIFORM   : every layer gets the same rank
  (b) LOCAL     : greedy allocation minimizing sum of PER-LAYER reconstruction
                  error  -> faithful proxy for FLAT-LLM/ASVD (local importance)
  (c) JOINT     : greedy allocation minimizing sum of AMPLIFICATION-WEIGHTED error,
                  where amp_i = (end-to-end output error / local error) measured by
                  the error-prop probe -> OUR method (protect high-amplification
                  early layers, compress damping late layers)

WIN CONDITION: ppl(JOINT) < ppl(LOCAL) at the same total budget -> downstream
error amplification is the signal local methods miss -> real, novel contribution.
If ppl(JOINT) ~ ppl(LOCAL) -> amplification doesn't pay off; honest negative.
"""
import os, sys, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
SEQ_LEN = 256
CAL_TOKENS = 4000
EVAL_SEQS, EVAL_LEN = 12, 1024
AVG_FRAC = 0.50                                  # keep 50% of FFN rank on average (sane ppl regime)
RMIN_F, RMAX_F = 0.10, 0.90
tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_alloc_{tag}.txt"

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)

if hasattr(model.model, "decoder"):
    layers = model.model.decoder.layers; ffn = lambda L: L.fc2
else:
    layers = model.model.layers; ffn = lambda L: L.mlp.down_proj
N = len(layers)
Hdim = ffn(layers[0]).in_features
Wd = [ffn(L).weight.detach().float() for L in layers]    # (d_model, Hdim) on GPU

# ---- text: calibration (early) + eval (held-out, later) ----
with open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt") as f:
    raw = f.read()
ids_all = tok(raw[:2_000_000], return_tensors="pt").input_ids[0]
cal_n = ((CAL_TOKENS//SEQ_LEN)+1)*SEQ_LEN
cal = ids_all[:cal_n].view(-1, SEQ_LEN).to(device)
eval_ids = ids_all[cal_n: cal_n + EVAL_SEQS*EVAL_LEN].view(EVAL_SEQS, EVAL_LEN).to(device)

# ---- pass 1: capture FFN hidden per layer on calibration ----
hid = {b: [] for b in range(N)}
def cap(b):
    def h(m, i): hid[b].append(i[0].reshape(-1, i[0].shape[-1]).float().cpu())
    return h
hs=[ffn(L).register_forward_pre_hook(cap(b)) for b,L in enumerate(layers)]
with torch.no_grad():
    clean_final = model(cal, output_hidden_states=True).hidden_states[-1].detach().float().reshape(-1).cpu()
for h in hs: h.remove()

# ---- per-layer eigvecs (desc) + local-output error curve on a rank grid ----
rmin, rmax = int(RMIN_F*Hdim), int(RMAX_F*Hdim)
step = Hdim//40
grid = list(range(rmin, rmax+1, step))
V = [None]*N
elocal = [dict() for _ in range(N)]              # elocal[b][r] = relative local FFN-output error
clean_local = {}
for b in range(N):
    Hc = torch.cat(hid[b],0).to(device); hid[b]=None
    mu = Hc.mean(0, keepdim=True); Hc = Hc-mu
    cov = (Hc.t()@Hc).double()
    ev, evec = torch.linalg.eigh(cov)
    Vb = evec.flip(1).float()                    # descending
    V[b] = Vb.half().cpu()
    out = Hc @ Wd[b].t(); base = out.norm()
    clean_local[b] = base
    for r in grid:
        Vr = Vb[:, :r]
        Hp = (Hc @ Vr) @ Vr.t()
        elocal[b][r] = float(((Hc-Hp) @ Wd[b].t()).norm()/base)
    del Hc, cov, evec; torch.cuda.empty_cache()

# ---- amplification amp_i (error-prop) at reference rank = AVG_FRAC ----
rref = int(AVG_FRAC*Hdim)
def proj_hook_factory(Vr):
    Vr = Vr.float()
    def ph(m, inp):
        x=inp[0]; xp=((x.float()@Vr)@Vr.t()).to(x.dtype); return (xp,)+inp[1:]
    return ph
amp=[]
for i in range(N):
    Vr = V[i][:, :rref].to(device)
    grab={}
    def go(m,i_,o): grab['o']=o.detach().float().cpu()
    h1=ffn(layers[i]).register_forward_pre_hook(proj_hook_factory(Vr))
    h2=ffn(layers[i]).register_forward_hook(go)
    with torch.no_grad():
        cf = model(cal, output_hidden_states=True).hidden_states[-1].detach().float().reshape(-1).cpu()
    h1.remove(); h2.remove()
    # clean local output for layer i at this calibration:
    dloc = elocal[i][min(grid, key=lambda r:abs(r-rref))]*float(clean_local[i])
    de2e = float((clean_final-cf).norm())
    amp.append(de2e/(dloc+1e-6)); Vr=None; torch.cuda.empty_cache()

# ---- allocators ----
TOTAL = int(AVG_FRAC*N*Hdim)
def greedy(weight):                              # weight[b] multiplies local error reduction
    r=[rmin]*N; rem=TOTAL-sum(r)
    while rem>=step:
        best,bg=-1,-1e18
        for b in range(N):
            if r[b]+step>rmax: continue
            g=weight[b]*(elocal[b][r[b]]-elocal[b][r[b]+step])
            if g>bg: bg,best=g,b
        if best<0: break
        r[best]+=step; rem-=step
    return r
alloc = {
 "uniform": [min(grid, key=lambda r:abs(r-int(AVG_FRAC*Hdim)))]*N,
 "local":   greedy([1.0]*N),
 "joint":   greedy(amp),
}

# ---- perplexity under a rank allocation ----
def perplexity(rks):
    Vg=[V[b][:, :rks[b]].to(device) for b in range(N)]
    hh=[ffn(layers[b]).register_forward_pre_hook(proj_hook_factory(Vg[b])) for b in range(N)]
    nll=0.0; ntok=0
    with torch.no_grad():
        for s in range(EVAL_SEQS):
            x=eval_ids[s:s+1]
            out=model(x, labels=x)
            nll+=float(out.loss)*(EVAL_LEN-1); ntok+=EVAL_LEN-1
    for h in hh: h.remove()
    for v in Vg: v=None
    torch.cuda.empty_cache()
    return math.exp(nll/ntok)

# baseline (no compression)
def ppl_base():
    nll=0.0; ntok=0
    with torch.no_grad():
        for s in range(EVAL_SEQS):
            x=eval_ids[s:s+1]; out=model(x, labels=x)
            nll+=float(out.loss)*(EVAL_LEN-1); ntok+=EVAL_LEN-1
    return math.exp(nll/ntok)

lines=[]
def log(s): lines.append(s); print(s)
log(f"ALLOCATION SHOWDOWN  MODEL={MODEL}  layers={N}  ffn={Hdim}")
log(f"avg keep={AVG_FRAC:.0%}  total_rank_budget={TOTAL}  (rank bounds [{rmin},{rmax}])")
log(f"amp_i (downstream amplification): min={min(amp):.2f} max={max(amp):.2f} mean={sum(amp)/N:.2f}")
log("")
log(f"{'method':>8}  {'sum_rank':>9}  {'perplexity':>11}")
log("-"*34)
base = ppl_base()
log(f"{'NO-COMP':>8}  {N*Hdim:>9}  {base:>11.3f}")
res={}
for name,rks in alloc.items():
    p=perplexity(rks); res[name]=p
    log(f"{name:>8}  {sum(rks):>9}  {p:>11.3f}")
log("")
log("per-layer ranks (uniform / local / joint):")
log(" lyr  amp   uniform  local  joint")
for b in range(N):
    log(f" {b:>3} {amp[b]:>5.2f}   {alloc['uniform'][b]:>6} {alloc['local'][b]:>6} {alloc['joint'][b]:>6}")
log("")
dl = (res['local']-res['joint'])/res['local']*100
du = (res['uniform']-res['joint'])/res['uniform']*100
log(f"JOINT vs LOCAL  : {dl:+.2f}%  perplexity reduction (positive = JOINT better)")
log(f"JOINT vs UNIFORM: {du:+.2f}%  perplexity reduction (positive = JOINT better)")
log("WIN if JOINT < LOCAL: amplification-aware allocation beats local-greedy (FLAT-LLM proxy).")
open(OUT,"w").write("\n".join(lines)+"\n")
print(f"\nSaved: {OUT}")
