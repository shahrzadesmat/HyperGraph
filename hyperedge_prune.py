"""
Hyperedge-based MLP pruning (v1) — exploits HIGHER-ORDER redundancy.

For each block's MLP hidden layer we:
  1. collect activation second-moment statistics (covariance + mean) on calib data
  2. greedily select a spanning subset of hidden neurons via PIVOTED CHOLESKY on
     the covariance (= column subset selection; picks neurons that add the most
     NEW variance — set-level, not pairwise)
  3. for the redundant neurons, solve least-squares reconstruction from the kept
     ones and FOLD their weight into the kept neurons' fc2 columns (+ bias).
     Removal is lossless up to the reconstruction residual.
  4. a single tolerance tau (variance retained per layer) is binary-searched to
     hit the global MAC budget -> budget-conditioned construction.

Exact algebra for the fold:
    hidden h_t (post-GELU) is consumed only by fc2 (linear): out = Σ_j fc2[:,j] h_j
    if h_t ≈ Σ_{s∈K} a_s h_s + β_t   (LS fit, β_t = constant from the mean), then
        fc2[:,s]  += a_s β?  ->  fc2[:,s] += a_s fc2[:,t]   for s∈K
        fc2.bias  += β_t fc2[:,t]
    and we drop fc2 col t and fc1 row t.

Only MLP is pruned in v1 (attention's Q·K path is nonlinear -> not exact-foldable).
"""

import copy
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. collect covariance + mean of post-GELU MLP hidden activations
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_mlp_stats(model, loader, device, max_batches=None):
    """Returns {block_idx: (cov[H,H] float64, mu[H] float64)} on CPU."""
    H = {b: blk.mlp.fc1.out_features for b, blk in enumerate(model.blocks)}
    G   = {b: torch.zeros(H[b], H[b], dtype=torch.float64, device=device) for b in H}
    S   = {b: torch.zeros(H[b], dtype=torch.float64, device=device) for b in H}
    n   = 0

    cap = {}
    handles = []
    def mk(b):
        def hook(m, i, o):
            cap[b] = o.detach().reshape(-1, o.shape[-1]).double()   # (tokens, H)
        return hook
    for b, blk in enumerate(model.blocks):
        handles.append(blk.mlp.act.register_forward_hook(mk(b)))

    model.eval()
    for bi, (x, _) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        cap.clear()
        model(x.to(device))
        for b in H:
            Xb = cap[b]
            G[b] += Xb.t() @ Xb
            S[b] += Xb.sum(0)
            if b == 0:
                n += Xb.shape[0]

    for h in handles:
        h.remove()

    stats = {}
    for b in H:
        mu = S[b] / n
        cov = G[b] / n - torch.outer(mu, mu)          # covariance (PSD)
        cov = 0.5 * (cov + cov.t())                   # symmetrize
        stats[b] = (cov.cpu(), mu.cpu())
    return stats


# ---------------------------------------------------------------------------
# 2. pivoted Cholesky = greedy spanning column selection
# ---------------------------------------------------------------------------

def pivoted_cholesky(cov, eps=1e-10, maxk=None):
    """
    Greedy column subset selection on a PSD matrix.
    maxk: stop after selecting this many pivots (default: full rank).
    Returns:
      perm           : list of channel indices in selection order (most info first)
      var_retained   : tensor, var_retained[k] = fraction of variance captured by
                       the first k pivots  (len = len(perm)+1, starts at 0.0)
    """
    cov = cov.clone().double()
    C = cov.shape[0]
    kmax = C if maxk is None else min(C, maxk)
    d = torch.diag(cov).clone()
    total = d.sum().clamp(min=1e-30)
    L = torch.zeros(C, kmax, dtype=torch.float64, device=cov.device)
    perm = []
    var = [0.0]
    for k in range(kmax):
        p = int(torch.argmax(d).item())
        if d[p] <= eps * total:
            break
        perm.append(p)
        Lk = (cov[:, p] - L[:, :k] @ L[p, :k]) / torch.sqrt(d[p])
        L[:, k] = Lk
        d = (d - Lk * Lk).clamp(min=0.0)
        var.append(float((total - d.sum()) / total))
    return perm, torch.tensor(var)


# ---------------------------------------------------------------------------
# 2b. selection variants (for the novelty ablation) — all return a keep-list
#     of length k.  The FOLD is identical afterwards; only WHICH channels are
#     kept differs, isolating the value of set-level vs pairwise vs magnitude.
# ---------------------------------------------------------------------------

def select_cholesky(cov, k):
    """Set-level: greedy spanning subset (orthogonalizes against the whole kept
    set each step) — our method. Captures higher-order redundancy."""
    perm, _ = pivoted_cholesky(cov)
    if len(perm) < k:                      # pad with remaining (highest residual)
        rest = [i for i in range(cov.shape[0]) if i not in set(perm)]
        perm = perm + rest
    return perm[:k]


def select_magnitude(cov, k):
    """No coupling at all: keep the k highest-variance channels."""
    var = torch.diag(cov)
    return torch.argsort(var, descending=True)[:k].tolist()


def select_pairwise(cov, k):
    """Pairwise (graph-style): max-min correlation greedy. Each step adds the
    channel least correlated with the kept set, using ONLY pairwise correlation
    (never the joint span). This is what a pairwise graph method can see."""
    C = cov.shape[0]
    d = torch.sqrt(torch.diag(cov).clamp(min=1e-12))
    corr = (cov / (d[:, None] * d[None, :])).abs()
    var = torch.diag(cov)
    first = int(torch.argmax(var).item())
    kept = [first]
    maxcorr = corr[first].clone()
    maxcorr[first] = 2.0
    while len(kept) < k:
        nxt = int(torch.argmin(maxcorr).item())
        kept.append(nxt)
        maxcorr = torch.maximum(maxcorr, corr[nxt])
        maxcorr[nxt] = 2.0
    return kept


_SELECTORS = {"cholesky": select_cholesky,
              "magnitude": select_magnitude,
              "pairwise": select_pairwise}


def prune_with_selection(model, stats, k_per_block, selection):
    """Prune every MLP block to its keep-count using the chosen selection
    method; fold is identical (least-squares reconstruction into fc2)."""
    selfn = _SELECTORS[selection]
    for b, blk in enumerate(model.blocks):
        cov, mu = stats[b]
        k = k_per_block[b]
        keep = selfn(cov, k)
        fold_block_mlp(blk, cov, mu, keep, k)


# ---------------------------------------------------------------------------
# 3. fold + prune one MLP block to keep-count k
# ---------------------------------------------------------------------------

def fold_block_mlp(block, cov, mu, perm, k, ridge=1e-6):
    """Keep the first k pivots; reconstruct & fold the rest into fc2 (+ bias)."""
    H = block.mlp.fc1.out_features
    k = max(1, min(k, len(perm)))
    K = perm[:k]
    keepset = set(K)
    D = [i for i in range(H) if i not in keepset]

    fc1, fc2 = block.mlp.fc1, block.mlp.fc2
    dev = fc2.weight.device
    wdtype = fc2.weight.dtype
    Kt = torch.tensor(K, dtype=torch.long, device=dev)

    if D:
        Dt = torch.tensor(D, dtype=torch.long, device=dev)
        cov = cov.to(dev).double()
        mu = mu.to(dev).double()
        GKK = cov[Kt][:, Kt]
        GKD = cov[Kt][:, Dt]
        lam = ridge * (torch.trace(cov) / H)
        A = torch.linalg.solve(GKK + lam * torch.eye(k, dtype=torch.float64, device=dev), GKD)  # (k,|D|)
        muK, muD = mu[Kt], mu[Dt]
        beta = muD - A.t() @ muK                          # (|D|,) intercepts

        W2 = fc2.weight.data.double()                     # (embed, H), on dev
        W2K = W2[:, Kt] + W2[:, Dt] @ A.t()               # fold coeffs
        new_bias = (fc2.bias.data.double() if fc2.bias is not None
                    else torch.zeros(W2.shape[0], dtype=torch.float64, device=dev))
        new_bias = new_bias + W2[:, Dt] @ beta            # fold intercepts
        fc2.weight = nn.Parameter(W2K.to(wdtype))
        fc2.bias = nn.Parameter(new_bias.to(wdtype))
    fc2.in_features = k

    # fc1: keep rows K (same order as fc2 cols)
    fc1.weight = nn.Parameter(fc1.weight.data[Kt, :])
    if fc1.bias is not None:
        fc1.bias = nn.Parameter(fc1.bias.data[Kt])
    fc1.out_features = k


# ---------------------------------------------------------------------------
# 4. budget-conditioned allocation: one tau -> per-layer keep count
# ---------------------------------------------------------------------------

def k_for_tau(var_retained, tau, min_keep=16):
    """Smallest k such that var_retained[k] >= tau."""
    idx = torch.searchsorted(var_retained, torch.tensor(float(tau)))
    return max(min_keep, int(idx.item()))


def prune_mlp_hyperedge(model, stats, perms, var_curves, tau, min_keep=16):
    """Fold+prune every block's MLP to the keep-count implied by tau."""
    kmap = {}
    for b, blk in enumerate(model.blocks):
        k = k_for_tau(var_curves[b], tau, min_keep)
        k = min(k, len(perms[b]))
        kmap[b] = k
        cov, mu = stats[b]
        fold_block_mlp(blk, cov, mu, perms[b], k)
    return kmap


def calibrate_tau(model, stats, perms, var_curves, target_macs_g, device,
                  count_fn, crop=224, min_keep=16, iters=16):
    """Binary-search tau in [0,1] so pruned MACs hit the target."""
    target = target_macs_g * 1e9
    lo, hi = 0.0, 1.0
    best_tau = 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        trial = copy.deepcopy(model).to(device)
        prune_mlp_hyperedge(trial, stats, perms, var_curves, mid, min_keep)
        macs, _ = count_fn(trial, device, crop)
        del trial
        if macs <= target:        # under budget -> can keep MORE -> raise tau
            best_tau = mid
            lo = mid
        else:                      # over budget -> keep less -> lower tau
            hi = mid
    return best_tau
