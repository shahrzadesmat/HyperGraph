"""
QK-circuit vs OV-circuit compressibility asymmetry.   python probe_qkov.py <hf_model>

THE anchor experiment for Circuit-Asymmetric Pruning. Tests whether the two
circuits inside each attention head compress DIFFERENTLY:
  QK circuit  (where to attend): governed by Q = X W_q, K = X W_k
  OV circuit  (what to write):   governed by the value output attn_out (-> W_o)

We measure, ACTIVATION-AWARE (real data, not weight-only), per head per layer:
  effrank@90%   : # of dims to capture 90% energy of the activation (out of d_head)
                  -> low effrank = compressible
  energy@r      : fraction of energy in the top-r dims, for r in {8,16,32,64,96}
                  -> the actionable "truncate to rank r costs what" curve

QK side = mean over {Q, K};  OV side = the head's value output (pre-W_o).

ASYMMETRY = (QK effrank) - (OV effrank), per head/layer.
  Large & systematic  -> circuits compress differently -> asymmetric per-circuit
                         rank allocation is justified (the method).
  ~ 0 everywhere      -> no asymmetry, idea's premise fails.
"""
import os, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
SEQ_LEN, N_SEQ = 256, 16
RANKS = [8, 16, 32, 64, 96]
tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_qkov_{tag}.txt"
device = torch.device("cuda")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
cfg = model.config
L, Hq = cfg.num_hidden_layers, cfg.num_attention_heads
Hkv = getattr(cfg, "num_key_value_heads", Hq)
Dh = cfg.hidden_size // Hq
layers = model.model.layers

with open("/work/hdd/bdjd/hypergraph_pruning/wikitext_train.txt") as f:
    text = f.read(N_SEQ*SEQ_LEN*8)
ids = tok(text, return_tensors="pt").input_ids[0][:N_SEQ*SEQ_LEN].view(N_SEQ, SEQ_LEN).to(device)

# ---- capture Q (q_proj out), K (k_proj out), AO (o_proj in = value output) ----
Q = {l: [] for l in range(L)}; K = {l: [] for l in range(L)}; AO = {l: [] for l in range(L)}
hs = []
def mk_out(store, l):
    def h(m, i, o): store[l].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
    return h
def mk_in(store, l):
    def h(m, i): store[l].append(i[0].detach().reshape(-1, i[0].shape[-1]).float().cpu())
    return h
for l in range(L):
    a = layers[l].self_attn
    hs.append(a.q_proj.register_forward_hook(mk_out(Q, l)))
    hs.append(a.k_proj.register_forward_hook(mk_out(K, l)))
    hs.append(a.o_proj.register_forward_pre_hook(mk_in(AO, l)))
with torch.no_grad():
    for s in range(0, N_SEQ, 4):
        model(ids[s:s+4])
for h in hs: h.remove()

# ---- per-head effective rank + energy curves ----
def spectrum(M):                                   # M [tok, dh] -> singular values (desc)
    M = M - M.mean(0, keepdim=True)
    s = torch.linalg.svdvals(M.to(device)).cpu()
    return s
def effrank(s, thr=0.90):
    e = (s**2); c = torch.cumsum(e, 0)/e.sum().clamp(min=1e-12)
    return int((c < thr).sum().item()) + 1
def energy_at(s, r):
    e = (s**2); return float(e[:r].sum()/e.sum().clamp(min=1e-12))

QKrank = np.zeros((L, Hq)); OVrank = np.zeros((L, Hq))
QKen = {r: np.zeros((L, Hq)) for r in RANKS}; OVen = {r: np.zeros((L, Hq)) for r in RANKS}
for l in range(L):
    Qc = torch.cat(Q[l], 0); Kc = torch.cat(K[l], 0); Ac = torch.cat(AO[l], 0)
    Q[l]=K[l]=AO[l]=None
    for hh in range(Hq):
        qh = Qc[:, hh*Dh:(hh+1)*Dh]
        kvh = hh % Hkv if Hkv < Hq else hh            # GQA: map q-head to its kv-head
        kh = Kc[:, kvh*Dh:(kvh+1)*Dh]
        ah = Ac[:, hh*Dh:(hh+1)*Dh]                    # value output for this q-head
        sq, sk, sa = spectrum(qh), spectrum(kh), spectrum(ah)
        QKrank[l, hh] = 0.5*(effrank(sq)+effrank(sk))
        OVrank[l, hh] = effrank(sa)
        for r in RANKS:
            QKen[r][l, hh] = 0.5*(energy_at(sq, r)+energy_at(sk, r))
            OVen[r][l, hh] = energy_at(sa, r)
    del Qc, Kc, Ac; torch.cuda.empty_cache()

# ---- report ----
lines=[]; lg=lambda s:(lines.append(s),print(s))
lg(f"QK vs OV circuit compressibility  MODEL={MODEL}  L={L} Hq={Hq} Hkv={Hkv} d_head={Dh}")
lg(f"tokens={ids.numel()}   (effrank out of {Dh}; lower = more compressible)")
lg("")
lg(f"OVERALL  QK_effrank={QKrank.mean():.1f}±{QKrank.std():.1f}   OV_effrank={OVrank.mean():.1f}±{OVrank.std():.1f}"
   f"   asymmetry(QK-OV)={QKrank.mean()-OVrank.mean():+.1f}")
lg("energy retained at matched rank r (mean over all heads):")
lg(f"  {'r':>4} {'QK_energy':>10} {'OV_energy':>10} {'gap':>8}")
for r in RANKS:
    lg(f"  {r:>4} {QKen[r].mean():>10.3f} {OVen[r].mean():>10.3f} {QKen[r].mean()-OVen[r].mean():>+8.3f}")
lg("")
lg(f"{'layer':>5} {'QK_effrank':>11} {'OV_effrank':>11} {'asym(QK-OV)':>12}")
for l in range(L):
    lg(f"{l:>5} {QKrank[l].mean():>11.1f} {OVrank[l].mean():>11.1f} {QKrank[l].mean()-OVrank[l].mean():>+12.1f}")
lg("")
# consistency: fraction of heads where the SAME circuit is more compressible
frac_ov_lower = float((OVrank < QKrank).mean())
lg(f"fraction of heads with OV more compressible than QK: {frac_ov_lower:.2%}")
lg("READ: a large, CONSISTENT asymmetry (one circuit systematically lower effrank,")
lg("frac near 0 or 1, energy gap large) => the two circuits should get DIFFERENT ranks")
lg("=> asymmetric per-circuit allocation is the method. If asym~0 & frac~50%, premise fails.")
torch.save({'QKrank':QKrank,'OVrank':OVrank,
            'QKen':{r:QKen[r] for r in RANKS},'OVen':{r:OVen[r] for r in RANKS}}, OUT.replace('.txt','.pt'))
open(OUT,"w").write("\n".join(lines)+"\n"); print("Saved:", OUT, "(+ .pt)")
