"""
probe_nonlinear3.py  —  confound-free nonlinear-vs-linear redundancy probe.

Synthesized from the 3 surviving design-lens agents (overfitting-power,
null-controls, claim-exploitability), which independently converged on the same
fixes after the v2/qkv/heads/mult probes were shown to be pure OVERFITTING:
Llama gaps were all NEGATIVE (held-out worse than PCA) and the gap SIGN tracked
C/n, not redundancy. No raw number from those probes licenses any claim.

FIXES vs v2 (each removes a named confound):
  1. FROZEN encoder/decoder  We=Vk, Wd=Vk^T (buffers, no grad); ONLY the
     nonlinear residual corr() trains.  -> removes ~24M confound params, isolates
     "nonlinear decode of the FIXED PCA code" from "a better linear code", AND
     makes the Gaussian null's population gap provably 0 (for jointly-Gaussian x
     the optimal decoder E[x|z] is linear).
  2. DOCUMENT/IMAGE-level split (70/15/15) BEFORE flattening to rows -> no
     pooled-token leakage; capture far more data.
  3. TRAIN-ONLY standardization + PCA (mu/sd/Vk from train rows only).
  4. Capacity cap (N_train/h >= 50) + Dropout + AdamW weight_decay + EARLY STOP
     on val MSE (restore best). gap>=0 is no longer "by construction": the model
     must EARN a positive held-out gap.
  5. THE control: covariance-matched GAUSSIAN NULL run through the byte-identical
     pipeline. Report Delta_gap = gap_real - gap_null. The null is linear by
     construction, so any positive null gap IS the estimator's overfitting floor.
  6. BLOCK BOOTSTRAP over whole test documents/images -> 95% CI on the gap.

DECISION (per site,k): NONLINEAR REDUNDANCY EXISTS iff Delta_gap lower-bound
> +2pp and stable across seeds. Else ALL LINEAR -> collapses into FLAT-LLM/
ASVD/MLA (a clean, publishable negative).

  python probe_nonlinear3.py <model> [mlp,heads]
"""
import os, sys, math, time, torch, numpy as np
import torch.nn as nn

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-hf"
SITES = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["mlp", "heads"])
IS_VIT = ("deit" in MODEL) or ("vit" in MODEL)

# ---- volumes / hyperparams (consensus of the 3 lenses) ----
N_BLOCKS   = 1400      # Llama: 256-tok blocks (doc-level split units)
BLOCK_TOK  = 256
N_IMG      = 1800      # DeiT: images (image-level split units)
SPLIT      = (0.70, 0.15, 0.15)
FRACS      = [0.05, 0.10]
SEEDS      = [0, 1, 2]
H_CAP      = 1024      # corr hidden cap; also enforce N_train/h >= 50
NH_MIN     = 50
DROPOUT    = 0.1
WD         = 1e-2
LR         = 1e-3
MAX_EPOCHS = 300
PATIENCE   = 20
MIN_DREL   = 1e-4      # relative min improvement on val
BS         = 2048
BOOT       = 2000
EIG_SUB    = 60000     # subsample rows for the covariance/eigh (stable, cheap)

tag = MODEL.split("/")[-1]
OUT = f"/work/hdd/bdjd/hypergraph_pruning/probe_nonlinear3_{tag}.txt"
device = torch.device("cuda")
lines = []; lg = lambda s: (lines.append(str(s)), print(s, flush=True))
flush = lambda: open(OUT, "w").write("\n".join(lines) + "\n")


# ======================== capture (document-tagged) ========================
def capture():
    """Return acts[site][layer] = CPU fp16 [Nrows, C], and a shared block-id
    array bid [Nrows] (same row order for every site/layer). One forward pass."""
    acts = {s: {} for s in SITES}
    bids = []
    if IS_VIT:
        import timm
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, Subset
        model = timm.create_model(MODEL, pretrained=True).eval().to(device)
        Nb = len(model.blocks); LAYERS = [1, Nb // 2, Nb - 3]
        tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        ds = ImageFolder("/work/hdd/bdjd/imagenet_10pct/val", transform=tf)
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:N_IMG].tolist()
        loader = DataLoader(Subset(ds, idx), batch_size=32, num_workers=8)
        hs = []
        for l in LAYERS:
            for s in SITES: acts[s][l] = []
            if "mlp" in SITES:
                def mk(l):
                    def h(m, i, o): acts["mlp"][l].append(o.detach().reshape(-1, o.shape[-1]).half().cpu())
                    return h
                hs.append(model.blocks[l].mlp.act.register_forward_hook(mk(l)))
            if "heads" in SITES:
                def mkh(l):
                    def h(m, i): acts["heads"][l].append(i[0].detach().reshape(-1, i[0].shape[-1]).half().cpu())
                    return h
                hs.append(model.blocks[l].attn.proj.register_forward_pre_hook(mkh(l)))
            qkv_sites = [s for s in ("q", "k", "v") if s in SITES]
            if qkv_sites:
                dim = model.blocks[0].attn.qkv.out_features // 3
                SL = {"q": slice(0, dim), "k": slice(dim, 2 * dim), "v": slice(2 * dim, 3 * dim)}
                def mkq(l, qs=qkv_sites, SL=SL):
                    def h(m, i, o):
                        o = o.detach().reshape(-1, o.shape[-1]).half().cpu()
                        for s in qs: acts[s][l].append(o[:, SL[s]])
                    return h
                hs.append(model.blocks[l].attn.qkv.register_forward_hook(mkq(l)))
        img0 = 0
        with torch.no_grad():
            for x, _ in loader:
                b = x.shape[0]; model(x.to(device))
                tpb = acts[SITES[0]][LAYERS[0]][-1].shape[0] // b
                bids.append(torch.arange(img0, img0 + b).repeat_interleave(tpb))
                img0 += b
        for h in hs: h.remove()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).eval().to(device)
        layers = model.model.layers; Nl = len(layers); LAYERS = [5, Nl // 2, Nl - 6]
        from datasets import load_dataset                      # canonical, reproducible source
        raw = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
        all_ids = tok(raw, return_tensors="pt").input_ids[0]
        nb = min(N_BLOCKS, all_ids.shape[0] // BLOCK_TOK)
        ids = all_ids[:nb * BLOCK_TOK].view(nb, BLOCK_TOK)
        hs = []
        for l in LAYERS:
            for s in SITES: acts[s][l] = []
            if "mlp" in SITES:
                def mk(l):
                    def h(m, i): acts["mlp"][l].append(i[0].detach().reshape(-1, i[0].shape[-1]).half().cpu())
                    return h
                hs.append(layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)))
            if "heads" in SITES:
                def mkh(l):
                    def h(m, i): acts["heads"][l].append(i[0].detach().reshape(-1, i[0].shape[-1]).half().cpu())
                    return h
                hs.append(layers[l].self_attn.o_proj.register_forward_pre_hook(mkh(l)))
            for s in ("q", "k", "v"):
                if s in SITES:
                    pname = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[s]
                    def mkq(store, l):
                        def h(m, i, o): store[l].append(o.detach().reshape(-1, o.shape[-1]).half().cpu())
                        return h
                    hs.append(getattr(layers[l].self_attn, pname).register_forward_hook(mkq(acts[s], l)))
        b0 = 0; B = 8
        with torch.no_grad():
            for s in range(0, nb, B):
                blk = ids[s:s + B].to(device); b = blk.shape[0]; model(blk)
                bids.append(torch.arange(b0, b0 + b).repeat_interleave(BLOCK_TOK))
                b0 += b
        for h in hs: h.remove()
    del model; torch.cuda.empty_cache()
    bid = torch.cat(bids)
    out = {}
    for s in SITES:
        out[s] = {l: torch.cat(acts[s][l], 0) for l in LAYERS}
        acts[s] = None
    return out, bid, LAYERS


# ======================== streaming linear algebra ========================
@torch.no_grad()
def train_stats(X):  # mu, sd from TRAIN rows only (streaming)
    n, C = X.shape; s = torch.zeros(C, device=device); s2 = torch.zeros(C, device=device)
    for b in range(0, n, 100000):
        xb = X[b:b + 100000].to(device).float(); s += xb.sum(0); s2 += (xb * xb).sum(0)
    mu = s / n; var = s2 / n - mu * mu
    return mu, var.clamp_min(1e-8).sqrt() + 1e-6

@torch.no_grad()
def cov_eigh(X, mu, sd, sub):  # covariance of standardized TRAIN rows -> eigh
    n, C = X.shape
    if n > sub:
        sel = torch.randperm(n)[:sub]; Xs = X[sel]
    else:
        Xs = X
    ns = Xs.shape[0]; cov = torch.zeros(C, C, device=device)
    for b in range(0, ns, 20000):
        xb = ((Xs[b:b + 20000].to(device).float() - mu) / sd); cov += xb.t() @ xb
    cov /= ns
    lam, Q = torch.linalg.eigh(cov)              # ascending
    lam = lam.flip(0); Q = Q.flip(1)             # descending
    return lam.clamp_min(0), Q

@torch.no_grad()
def gauss_null(lam, Q, n, gen):  # CPU fp16 surrogate with covariance = Q diag(lam) Q^T
    C = Q.shape[0]; sl = lam.sqrt(); out = torch.empty(n, C, dtype=torch.float16)
    QT = Q.t().contiguous()
    for b in range(0, n, 20000):
        m = min(20000, n - b)
        Z = torch.randn(m, C, generator=gen, device=device)
        out[b:b + m] = ((Z * sl) @ QT).half().cpu()
    return out


# ======================== probe core (frozen encoder + early stop) ========================
class Corr(nn.Module):
    def __init__(s, k, h, C):
        super().__init__()
        s.l1 = nn.Linear(k, h); s.act = nn.GELU(); s.do = nn.Dropout(DROPOUT); s.l2 = nn.Linear(h, C)
        nn.init.zeros_(s.l2.weight); nn.init.zeros_(s.l2.bias)
    def forward(s, z): return s.l2(s.do(s.act(s.l1(z))))

@torch.no_grad()
def eval_rows(X, mu, sd, We, Wd, corr, want_rows=False):
    """relative-Frobenius PCA & AE error on X; optionally per-row (a,b,c)."""
    n = X.shape[0]; A = B = Cn = 0.0; ra = []; rb = []; rc = []
    corr.eval()
    for s in range(0, n, 20000):
        xb = (X[s:s + 20000].to(device).float() - mu) / sd
        z = xb @ We
        pca = z @ Wd
        ae = pca + corr(z)
        a = ((xb - pca) ** 2).sum(1); b = ((xb - ae) ** 2).sum(1); c = (xb * xb).sum(1)
        A += a.sum().item(); B += b.sum().item(); Cn += c.sum().item()
        if want_rows: ra.append(a.cpu()); rb.append(b.cpu()); rc.append(c.cpu())
    pca_err = math.sqrt(A / Cn); ae_err = math.sqrt(B / Cn)
    if want_rows:
        return pca_err, ae_err, (torch.cat(ra), torch.cat(rb), torch.cat(rc))
    return pca_err, ae_err

def fit(Xtr, Xval, Xte, k, h, seed, want_rows=False):
    torch.manual_seed(seed)
    mu, sd = train_stats(Xtr)
    lam, Q = cov_eigh(Xtr, mu, sd, EIG_SUB)
    Vk = Q[:, :k].contiguous()
    We = Vk.detach(); Wd = Vk.t().contiguous().detach()         # FROZEN
    C = Xtr.shape[1]; corr = Corr(k, h, C).to(device)
    opt = torch.optim.AdamW(corr.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    ntr = Xtr.shape[0]; best = math.inf; best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        corr.train(); perm = torch.randperm(ntr)
        for b in range(0, ntr, BS):
            xb = (Xtr[perm[b:b + BS]].to(device).float() - mu) / sd
            z = xb @ We; rec = z @ Wd + corr(z)
            loss = ((rec - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        _, vae = eval_rows(Xval, mu, sd, We, Wd, corr)            # val MSE proxy via rel-err
        if vae < best * (1 - MIN_DREL):
            best = vae; best_state = {kk: v.detach().clone() for kk, v in corr.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    if best_state is not None: corr.load_state_dict(best_state)
    res = eval_rows(Xte, mu, sd, We, Wd, corr, want_rows=want_rows)
    del corr; torch.cuda.empty_cache()
    return res  # (pca,ae) or (pca,ae,(a,b,c))


def block_boot(a, b, c, bid_te):
    """paired block bootstrap of gap over test documents/images."""
    blocks = torch.unique(bid_te)
    # per-block sums
    order = torch.argsort(bid_te); bid_s = bid_te[order]; a_s = a[order]; b_s = b[order]; c_s = c[order]
    uniq, counts = torch.unique_consecutive(bid_s, return_counts=True)
    idx = torch.cumsum(torch.cat([torch.tensor([0]), counts]), 0)
    bsum = torch.stack([torch.stack([a_s[idx[i]:idx[i + 1]].sum(),
                                     b_s[idx[i]:idx[i + 1]].sum(),
                                     c_s[idx[i]:idx[i + 1]].sum()]) for i in range(len(uniq))])
    nb = bsum.shape[0]; g = torch.empty(BOOT)
    gen = torch.Generator().manual_seed(0)
    for r in range(BOOT):
        sel = torch.randint(0, nb, (nb,), generator=gen)
        S = bsum[sel].sum(0)
        pca = (S[0] / S[2]).sqrt(); ae = (S[1] / S[2]).sqrt()
        g[r] = (pca - ae) / pca * 100
    return torch.quantile(g, 0.025).item(), torch.quantile(g, 0.975).item()


# ======================== run ========================
t0 = time.time()
ACTS, BID, LAYERS = capture()
lg(f"probe_nonlinear3  MODEL={MODEL}  sites={SITES}  layers={LAYERS}")
lg(f"blocks/imgs split {SPLIT}  seeds={SEEDS}  k={FRACS}  h_cap={H_CAP}  early-stop p={PATIENCE}")
lg(f"capture: {time.time()-t0:.0f}s   rows={BID.shape[0]}  units={int(BID.max())+1}")
lg("=" * 96)
hdr = f"{'site':>6} {'layer':>5} {'C':>6} {'k':>5} {'h':>5} | {'gap_real':>9} {'gap_null':>9} {'Δgap':>7} {'Δlo':>7} {'Δhi':>7}  verdict"
verdicts = []

for site in SITES:
    for l in LAYERS:
        X = ACTS[site][l]; C = X.shape[1]
        # document/image-level split (shared across k, fixed by a master seed)
        units = torch.unique(BID); g = torch.Generator().manual_seed(12345)
        up = units[torch.randperm(len(units), generator=g)]
        n1 = int(SPLIT[0] * len(up)); n2 = int((SPLIT[0] + SPLIT[1]) * len(up))
        tr_u, va_u, te_u = up[:n1], up[n1:n2], up[n2:]
        tr_m = torch.isin(BID, tr_u); va_m = torch.isin(BID, va_u); te_m = torch.isin(BID, te_u)
        Xtr, Xval, Xte = X[tr_m], X[va_m], X[te_m]; bid_te = BID[te_m]
        for f in FRACS:
            k = max(2, int(f * C)); h = min(2 * k, H_CAP)
            h = min(h, max(8, Xtr.shape[0] // NH_MIN))           # enforce N_train/h >= 50
            gr = []; gn = []; lo = hi = float("nan")
            for si, seed in enumerate(SEEDS):
                want = (si == 0)
                if want:
                    pca, ae, (ra, rb, rc) = fit(Xtr, Xval, Xte, k, h, seed, want_rows=True)
                    lo, hi = block_boot(ra, rb, rc, bid_te)
                else:
                    pca, ae = fit(Xtr, Xval, Xte, k, h, seed)
                gr.append((pca - ae) / pca * 100)
                # matched-covariance Gaussian null (byte-identical pipeline)
                mu, sd = train_stats(Xtr); lam, Q = cov_eigh(Xtr, mu, sd, EIG_SUB)
                gen = torch.Generator(device=device).manual_seed(1000 + seed)
                Gtr = gauss_null(lam, Q, Xtr.shape[0], gen)
                Gva = gauss_null(lam, Q, Xval.shape[0], gen)
                Gte = gauss_null(lam, Q, Xte.shape[0], gen)
                pca0, ae0 = fit(Gtr, Gva, Gte, k, h, seed)
                gn.append((pca0 - ae0) / pca0 * 100)
                del Gtr, Gva, Gte
            gr = np.array(gr); gn = np.array(gn)
            dgap = gr.mean() - gn.mean()
            dlo = lo - gn.mean(); dhi = hi - gn.mean()             # conservative: real CI minus null floor
            ok = (dlo > 2.0)
            verdicts.append((site, l, k, dgap, dlo, ok))
            lg(f"{site:>6} {l:>5} {C:>6} {k:>5} {h:>5} | "
               f"{gr.mean():>+8.1f}% {gn.mean():>+8.1f}% {dgap:>+6.1f}% {dlo:>+6.1f}% {dhi:>+6.1f}%  "
               f"{'NONLINEAR' if ok else 'linear'}")
            flush()
        ACTS[site][l] = None

lg("=" * 96)
nnl = sum(1 for *_, ok in verdicts if ok)
lg(f"Δgap = gap_real - gap_null (matched-cov Gaussian).  Δlo/Δhi = test-doc block-bootstrap 95% CI minus null floor.")
lg(f"VERDICT: {nnl}/{len(verdicts)} (site,layer,k) cells show NONLINEAR redundancy (Δgap lower-bound > +2pp).")
if nnl == 0:
    lg("=> redundancy is statistically indistinguishable from LINEAR low-rank")
    lg("   -> the hypergraph/effective-rank idea collapses into FLAT-LLM/ASVD/MLA.")
    lg("   (This is a clean, publishable NEGATIVE — and it kills the wrong direction early.)")
else:
    lg("=> nonlinear redundancy SURVIVES the null on some cells -> candidate real foundation;")
    lg("   next: h-sweep + N-sweep plateau + downstream perplexity/top-1 at matched budget.")
lg(f"total: {time.time()-t0:.0f}s")
flush(); print("Saved:", OUT)
