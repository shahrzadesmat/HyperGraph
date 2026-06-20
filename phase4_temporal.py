"""
phase4_temporal.py — LEVER B headroom: temporal (across-token) predictability of keys.

Per-token methods (MLA, KV-CAR) compress every key INDEPENDENTLY. But keys arrive as a SEQUENCE;
if key_t is predictable from recent keys, you could cache only the INNOVATION (the surprise) — a
compression axis ORTHOGONAL to MLA (it could stack on top). This measures whether that headroom
exists, null-calibrated so we don't mistake overfitting for signal.

Within each contiguous CHUNK (a real token sequence), form pairs ([k_{t-1..t-w}] -> k_t) and fit:
  LINEAR    : ridge       k_hat = my + (X-mx) W                       -> held-out FVE_lin
  NONLINEAR : k_hat = linear + MLP(X)   [linear FROZEN]               -> held-out FVE_nl (>= FVE_lin)
  NULL      : SAME fits but pairs temporally SHUFFLED (k_t vs random key) -> overfit floor (~0)
FVE = 1 - ||k_t - k_hat||^2 / ||k_t - mean||^2  (fraction of key variance explained by history).
Split is BY CHUNK (no temporal leakage). Keys are pre-RoPE (what you'd actually cache).

READ: FVE_lin(real) - FVE_lin(null) = genuine temporal headroom MLA leaves on the table.
      FVE_nl - FVE_lin             = nonlinear temporal structure (is a nonlinear predictor worth it).
env: TMP_MODEL  TMP_CALIB=80000  TMP_WINDOWS=1,4  TMP_LAYERS=0,8,16,24,31
"""
import os, math, time, torch
import torch.nn as nn

device = torch.device("cuda")
MODEL = os.environ.get("TMP_MODEL", "meta-llama/Llama-2-7b-hf")
N_CALIB = int(os.environ.get("TMP_CALIB", "80000"))
CHUNK = 2048
WINDOWS = [int(x) for x in os.environ.get("TMP_WINDOWS", "1,4").split(",")]
LAYERS_ENV = os.environ.get("TMP_LAYERS", "0,8,16,24,31")
H = 512; LR = 1e-3; MAX_EPOCHS = 150; PATIENCE = 12; BS = 8192; WD = 1e-2
OUT = "/work/hdd/bdjd/hypergraph_pruning/phase4_temporal" + os.environ.get("TMP_TAG", "") + ".txt"
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True)); flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
attn = [lyr.self_attn for lyr in model.model.layers]
NL = len(attn); C = model.config.hidden_size
LAYERS = [l for l in (int(x) for x in LAYERS_ENV.split(",")) if l < NL]
train_txt = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
train_ids = tok(train_txt, return_tensors="pt").input_ids[0]
lg(f"phase4_temporal  MODEL={MODEL}  C={C}  layers={LAYERS}  windows={WINDOWS}  calib={N_CALIB}")

# ---------- Stage A: capture per-chunk key sequences (order preserved) ----------
Kseq = {l: [] for l in LAYERS}; hooks = []
def mk(l):
    def h(m, i, o): Kseq[l].append(o.detach()[0].to(torch.float16).cpu())   # [chunk_len, C], pre-RoPE
    return h
for l in LAYERS: hooks.append(attn[l].k_proj.register_forward_hook(mk(l)))
nchunks = max(1, min(N_CALIB // CHUNK, (train_ids.shape[0] - 1) // CHUNK))
with torch.no_grad():
    for c in range(nchunks):
        ids = train_ids[c * CHUNK:(c + 1) * CHUNK]
        if ids.shape[0] < max(WINDOWS) + 2: break
        model(ids.unsqueeze(0).to(device))
for h in hooks: h.remove()
lg(f"capture: {time.time()-t0:.0f}s   chunks={len(Kseq[LAYERS[0]])}  chunk_len~{CHUNK}")
del model, attn; torch.cuda.empty_cache()       # free the 13GB model — Stage B only fits on captured keys

# ---------- helpers ----------
def make_pairs(chs, w):                      # chs: list of [T,C] -> X [n, w*C], Y [n, C]
    Xs, Ys = [], []
    for ch in chs:
        T = ch.shape[0]
        if T <= w: continue
        Ys.append(ch[w:])
        Xs.append(torch.cat([ch[w - j - 1:T - j - 1] for j in range(w)], dim=1))
    return torch.cat(Xs).float(), torch.cat(Ys).float()

def fve(Yva, Yhat, my):
    num = ((Yva - Yhat) ** 2).sum(); den = ((Yva - my) ** 2).sum()
    return float(1.0 - num / den)

def fit_eval(Xtr, Ytr, Xva, Yva):            # returns (FVE_lin, FVE_nl)
    mx = Xtr.mean(0, keepdim=True); sx = Xtr.std(0, keepdim=True) + 1e-6
    my = Ytr.mean(0, keepdim=True)
    Xc = Xtr - mx; Yc = Ytr - my
    lam = 1e-2 * (Xc * Xc).mean() * Xc.shape[1]
    XtX = Xc.t() @ Xc + lam * torch.eye(Xc.shape[1], device=device)
    W = torch.linalg.solve(XtX, Xc.t() @ Yc)                       # [w*C, C]
    lin_va = (Xva - mx) @ W + my
    fl = fve(Yva, lin_va, my)
    # nonlinear residual on top of the FROZEN linear
    lin_tr = Xc @ W + my
    Rtr = (Ytr - lin_tr).detach()
    del Xc, XtX, lin_tr, Yc; torch.cuda.empty_cache()          # free heavy intermediates (w=4 OOM fix)
    Xn_va = (Xva - mx) / sx                                     # val standardized once; train done per-batch
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], H), nn.GELU(), nn.Linear(H, C)).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=WD)
    best = math.inf; best_sd = None; bad = 0; n = Xtr.shape[0]
    for ep in range(MAX_EPOCHS):
        mlp.train(); perm = torch.randperm(n, device=device)
        for b in range(0, n, BS):
            idx = perm[b:b + BS]
            xb = (Xtr[idx] - mx) / sx
            loss = ((mlp(xb) - Rtr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            nl_va = lin_va + mlp(Xn_va); ve = float(((Yva - nl_va) ** 2).mean())
        if ve < best * (1 - 1e-4): best = ve; best_sd = {k: v.detach().clone() for k, v in mlp.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    mlp.load_state_dict(best_sd); mlp.eval()
    with torch.no_grad():
        fn = fve(Yva, lin_va + mlp(Xn_va), my)
    return fl, fn

# ---------- Stage B: per layer x window, real vs null ----------
lg("=" * 96)
lg("FVE = fraction of key_t variance predicted from recent keys (held out). real vs NULL (shuffled).")
lg(f"{'layer':>5} {'w':>2} | {'FVE_lin real':>12} {'FVE_lin null':>12} | {'FVE_nl real':>11} {'nl-lin':>7} | headroom")
lg("-" * 96)
summ = {}
for w in WINDOWS:
    for l in LAYERS:
        X, Y = make_pairs(Kseq[l], w)
        # split BY CHUNK proxy: contiguous 85/15 of the pair stream keeps temporal blocks together
        ntr = int(0.85 * X.shape[0])
        Xtr, Ytr = X[:ntr].to(device), Y[:ntr].to(device)
        Xva, Yva = X[ntr:].to(device), Y[ntr:].to(device)
        fl, fn = fit_eval(Xtr, Ytr, Xva, Yva)
        # NULL: destroy the temporal pairing (shuffle X rows vs Y) in BOTH train and val
        ptr = torch.randperm(Xtr.shape[0], device=device); pva = torch.randperm(Xva.shape[0], device=device)
        fln, fnn = fit_eval(Xtr[ptr], Ytr, Xva[pva], Yva)
        head = fl - fln
        lg(f"{l:>5} {w:>2} | {fl:>12.4f} {fln:>12.4f} | {fn:>11.4f} {fn-fl:>+7.4f} | {head:>+.4f}")
        flush()
        summ.setdefault(w, []).append((fl, fln, fn))
        del X, Y, Xtr, Ytr, Xva, Yva; torch.cuda.empty_cache()
lg("-" * 96)
for w in WINDOWS:
    fls = [a for a, _, _ in summ[w]]; flns = [b for _, b, _ in summ[w]]; fns = [c for _, _, c in summ[w]]
    al = sum(fls) / len(fls); an = sum(flns) / len(flns); af = sum(fns) / len(fns)
    lg(f"window w={w}:  mean FVE_lin real={al:.3f}  null={an:.3f}  -> temporal headroom={al-an:+.3f} | "
       f"mean nonlinear gain={af-al:+.3f}")
lg("")
lg("VERDICT GUIDE: headroom>>0 => keys ARE temporally predictable (cache the innovation; MLA misses this).")
lg("nonlinear gain>0 => the temporal predictor should be nonlinear. Both small => Lever B is also a dead end.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
