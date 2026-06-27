"""
spectral_cka_vs_shesha.py
=========================
Standalone script: CKA (from distinction_encoder_test.py) vs Shesha under
spectral deletion — same experimental setup as
distinction-spectral-test-extended.py but stripped down to exactly two metrics.

Experimental setup
------------------
* Synthetic representation with controlled 1/k spectral decay  (n=200, d=256)
* Progressive deletion of top-k principal components  (k = 0 … 50)
* Two metrics computed at every k:
    - CKA       : standard (biased) linear CKA, matching compute_cka() in
                  distinction_encoder_test.py (GPU via PyTorch when available,
                  pure-numpy fallback otherwise)
    - Shesha    : split-half RDM correlation  (split_half_shesha from the
                  extended test, 50 random splits per k)

Output
------
* Console table + collapse-point summary
* CSV: ./shesha-distinction/spectral_cka_vs_shesha.csv
"""

import csv
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES   = 200
N_FEATURES  = 256
N_SPLITS    = 50          # Split-half repetitions for Shesha
RANDOM_SEED = 320
K_MAX       = 50          # Remove PCs 0 … K_MAX (inclusive)
EPS         = 1e-9

OUTPUT_DIR  = Path(__file__).resolve().parent / "shesha-distinction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL GPU IMPORT
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    _TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    print("[device] CUDA available — using PyTorch GPU for CKA")
else:
    print("[device] No CUDA / no PyTorch — using numpy CPU for CKA")


# ─────────────────────────────────────────────────────────────────────────────
# CKA  (exact match to compute_cka in distinction_encoder_test.py)
# Falls back to numpy when GPU is unavailable.
# ─────────────────────────────────────────────────────────────────────────────

def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Standard (biased) linear CKA.

    Replicates compute_cka() from distinction_encoder_test.py:
      1. Center both representations (subtract column means)
      2. Compute linear Gram matrices  K = X Xᵀ,  L = Y Yᵀ
      3. Double-center via  H K H,  H L H
      4. Return  tr(Kc Lc) / sqrt( tr(Kc²) · tr(Lc²) )
    """
    if _TORCH_AVAILABLE:
        Xt = torch.tensor(X, dtype=torch.float64, device='cuda')
        Yt = torch.tensor(Y, dtype=torch.float64, device='cuda')

        n = Xt.shape[0]
        Xt = Xt - Xt.mean(0)
        Yt = Yt - Yt.mean(0)

        K = Xt @ Xt.T
        L = Yt @ Yt.T

        H = (torch.eye(n, dtype=torch.float64, device='cuda')
             - torch.ones((n, n), dtype=torch.float64, device='cuda') / n)
        K = H @ K @ H
        L = H @ L @ H

        num = (K * L).sum()
        den = torch.sqrt((K * K).sum() * (L * L).sum()) + 1e-12
        return float(torch.clamp(num / den, 0.0, 1.0).item())

    else:
        # Pure-numpy equivalent
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        n = X.shape[0]
        K = X @ X.T
        L = Y @ Y.T

        H = np.eye(n) - np.ones((n, n)) / n
        K = H @ K @ H
        L = H @ L @ H

        num = np.sum(K * L)
        den = np.sqrt(np.sum(K * K) * np.sum(L * L)) + 1e-12
        return float(np.clip(num / den, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# SHESHA  (identical to split_half_shesha in the extended test)
# ─────────────────────────────────────────────────────────────────────────────

def split_half_shesha(X: np.ndarray, n_splits: int = N_SPLITS,
                      random_state=None) -> float:
    """
    Within-representation stability: split features into two random halves,
    compute RDM (correlation distance) for each, return mean Spearman rho.
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)
    correlations = []

    for _ in range(n_splits):
        perm = rng.permutation(n_features)
        half = n_features // 2

        X1 = X[:, perm[:half]]
        X2 = X[:, perm[half:]]

        rdm1 = pdist(X1, metric='correlation')
        rdm2 = pdist(X2, metric='correlation')

        if not (np.all(np.isfinite(rdm1)) and np.all(np.isfinite(rdm2))):
            continue
        if np.nanstd(rdm1) < EPS or np.nanstd(rdm2) < EPS:
            continue

        r, _ = spearmanr(rdm1, rdm2)
        if np.isfinite(r):
            correlations.append(r)

    return float(np.mean(correlations)) if correlations else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION  (same as the extended test)
# ─────────────────────────────────────────────────────────────────────────────

def make_spectral_representation(n=N_SAMPLES, d=N_FEATURES, seed=RANDOM_SEED):
    """
    Controlled 1/k spectral decay:  singular values = 100, 50, 33.3, 25, …
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    S_mat = np.zeros((n, d))
    np.fill_diagonal(S_mat, [100.0 / (i + 1) for i in range(min(n, d))])
    return U @ S_mat @ V.T


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 60)
    print("  Spectral Deletion: CKA  vs  Shesha")
    print("=" * 60)
    print(f"  n={N_SAMPLES}, d={N_FEATURES}, k=0…{K_MAX}, seed={RANDOM_SEED}")
    print()

    # --- build representation & fit PCA ---
    X_orig = make_spectral_representation()
    pca = PCA(n_components=min(N_SAMPLES, N_FEATURES), random_state=RANDOM_SEED)
    pca.fit(X_orig)
    X_pca_full = pca.transform(X_orig)

    removal_levels = list(range(0, K_MAX + 1))

    header = f"{'k':>5}  {'CKA':>9}  {'Shesha':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for k in removal_levels:
        X_pca = X_pca_full.copy()
        X_pca[:, :k] = 0.0
        X_mod = pca.inverse_transform(X_pca)

        cka_val    = compute_cka(X_orig, X_mod)
        shesha_val = split_half_shesha(X_mod, random_state=RANDOM_SEED + k)

        results.append({'pcs_removed': k, 'cka': cka_val, 'shesha': shesha_val})
        print(f"{k:>5}  {cka_val:>9.4f}  {shesha_val:>9.4f}")

    # --- save CSV ---
    out_csv = OUTPUT_DIR / 'spectral_cka_vs_shesha.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pcs_removed', 'cka', 'shesha'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[saved] {out_csv}")

    # --- collapse analysis ---
    print("\n" + "-" * 40)
    print("First k where metric drops below threshold:")
    print("-" * 40)
    for metric, threshold in [('cka', 0.5), ('shesha', 0.5), ('cka', 0.1)]:
        for r in results:
            if r[metric] < threshold:
                print(f"  {metric:<8} < {threshold}: k = {r['pcs_removed']}")
                break
        else:
            print(f"  {metric:<8} < {threshold}: never")

    print()
    print("Values at k = 30:")
    r30 = next(r for r in results if r['pcs_removed'] == 30)
    print(f"  CKA    = {r30['cka']:.4f}")
    print(f"  Shesha = {r30['shesha']:.4f}")
    print(f"  Gap    = {r30['shesha'] - r30['cka']:+.4f}")

    print("\n" + "=" * 60)
    return results


if __name__ == '__main__':
    run()
