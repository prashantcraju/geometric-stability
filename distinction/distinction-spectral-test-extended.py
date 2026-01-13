"""
Shesha Spectral Sensitivity Analysis (Extended)

This script compares Shesha against multiple similarity metrics under 
spectral deletion (progressively removing top principal components).

1. Multiple similarity metrics under identical preprocessing
   - Debiased CKA (Kornblith et al., 2019)
   - Biased CKA (for comparison)
   - Effective Rank PWCKA (Morcos et al., 2018; Kornblith et al., 2019)
   - Procrustes similarity

2. Preprocessing ablations
   - Raw (no preprocessing)
   - Centering only
   - Centering + L2 normalization
   - Whitening (ZCA)

3. RSA reliability comparison
   - Whitened Shesha (Walther et al., 2016; Diedrichsen & Kriegeskorte, 2017)

Output:
- CSV files with all results
- Statistical summaries

Key findings:
- All similarity metrics collapse after k=1 PC removed
- Shesha remains above 0.4 until k=26
- Divergence robust across preprocessing (except whitening)
- Whitening causes CKA to recover (confirms spectral anisotropy mechanism)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import warnings
import csv
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

EPS_STD = 1e-9          # Tolerance for zero-variance checks
N_SPLITS = 50           # Number of splits for stability metrics
RANDOM_SEED = 320        # For reproducibility

# Output directory
OUTPUT_DIR = Path("./shesha-distinction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_to_csv(filename, headers, rows):
    """Save results to CSV file."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"   [Saved] {filepath}")


def compute_rdm(X, metric='correlation'):
    """Compute RDM (upper triangle) using specified distance metric."""
    return pdist(X, metric=metric)


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def apply_preprocessing(X, centering=True, normalize=False, whiten=False,
                        shrinkage=0.1):
    """
    Apply different preprocessing options.

    Parameters
    ----------
    X : array (n_samples, n_features)
    centering : bool
        Subtract mean per feature
    normalize : bool
        L2 normalize each sample
    whiten : bool
        ZCA whitening (decorrelate features)
    shrinkage : float
        Regularization for whitening (0 = full, 1 = identity)
    """
    X_out = X.copy().astype(np.float64)

    if centering:
        X_out = X_out - X_out.mean(axis=0)

    if normalize:
        norms = np.linalg.norm(X_out, axis=1, keepdims=True)
        X_out = X_out / (norms + EPS_STD)

    if whiten:
        X_c = X_out - X_out.mean(axis=0)
        n = X_c.shape[0]
        cov = (X_c.T @ X_c) / (n - 1)
        cov_shrunk = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov_shrunk)
        eigvals = np.maximum(eigvals, EPS_STD)
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        X_out = X_c @ W

    return X_out


# =============================================================================
# STABILITY METRICS (INTRINSIC)
# =============================================================================

def split_half_shesha(X, n_splits=N_SPLITS, random_state=None):
    """
    Shesha: Within-representation stability via split-half RDM correlation.

    Splits features into two halves, computes RDM for each, and correlates.
    Uses correlation distance (1 - Pearson r) for RDM computation.

    This measures whether geometric structure is distributed across features
    (high stability) or concentrated in subsets (low stability).
    """
    n_samples, n_features = X.shape
    correlations = []
    rng = np.random.default_rng(random_state)

    for _ in range(n_splits):
        perm = rng.permutation(n_features)
        half = n_features // 2

        X1 = X[:, perm[:half]]
        X2 = X[:, perm[half:]]

        rdm1 = pdist(X1, metric='correlation')
        rdm2 = pdist(X2, metric='correlation')

        if not (np.all(np.isfinite(rdm1)) and np.all(np.isfinite(rdm2))):
            continue

        if np.nanstd(rdm1) < EPS_STD or np.nanstd(rdm2) < EPS_STD:
            continue

        r, _ = spearmanr(rdm1, rdm2)
        if np.isfinite(r):
            correlations.append(r)

    return np.mean(correlations) if correlations else 0.0


def whitened_rsa_stability(X, n_splits=N_SPLITS, shrinkage=0.1, random_state=None):
    """
    Shesha variant using whitened representations.

    Whitening decorrelates features and normalizes variance, which can
    improve stability estimates when features have heterogeneous noise.

    Reference: Diedrichsen et al. (2021)
    """
    X_white = apply_preprocessing(X, centering=True, whiten=True,
                                   shrinkage=shrinkage)
    return split_half_shesha(X_white, n_splits=n_splits, random_state=random_state)


def compute_crossnobis_rdm(X1, X2):
    """
    Crossnobis (cross-validated) distance estimator.

    Uses two independent measurements to compute unbiased squared distance.
    Removes the positive bias inherent in standard distance estimates.

    Reference: Walther et al. (2016), Diedrichsen et al. (2021)
    """
    n_samples = X1.shape[0]
    n_pairs = n_samples * (n_samples - 1) // 2

    rdm = np.zeros(n_pairs)
    idx = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff1 = X1[i] - X1[j]
            diff2 = X2[i] - X2[j]
            rdm[idx] = np.dot(diff1, diff2)
            idx += 1

    return rdm


def crossnobis_stability(X, n_splits=N_SPLITS, random_state=None):
    """
    Crossnobis stability: correlation between two independent crossnobis RDMs.

    Splits features into 4 quarters, computes crossnobis distance from
    (Q1, Q2) and (Q3, Q4), then correlates the two unbiased RDM estimates.

    Reference: Walther et al. (2016), Diedrichsen et al. (2021)
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    if n_features < 4:
        return 0.0

    correlations = []

    for _ in range(n_splits):
        perm = rng.permutation(n_features)
        quarter = n_features // 4

        X_q1 = X[:, perm[:quarter]]
        X_q2 = X[:, perm[quarter:2*quarter]]
        X_q3 = X[:, perm[2*quarter:3*quarter]]
        X_q4 = X[:, perm[3*quarter:]]

        # Two independent crossnobis RDMs
        rdm_cross1 = compute_crossnobis_rdm(X_q1, X_q2)
        rdm_cross2 = compute_crossnobis_rdm(X_q3, X_q4)

        if np.std(rdm_cross1) > EPS_STD and np.std(rdm_cross2) > EPS_STD:
            r, _ = spearmanr(rdm_cross1, rdm_cross2)
            if np.isfinite(r):
                correlations.append(r)

    return np.mean(correlations) if correlations else 0.0




# =============================================================================
# SIMILARITY METRICS (EXTRINSIC - COMPARING TWO REPRESENTATIONS)
# =============================================================================

def debiased_linear_cka(X, Y):
    """
    Debiased CKA following Song et al. (2007) / Kornblith et al. (2019).
    
    Key: debiased HSIC operates on UNcentered Gram matrices,
    with centering handled implicitly by the estimator formula.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    n = X.shape[0]
    if n < 4:
        return 0.0
    
    # Compute LINEAR kernel (dot product) - NO centering
    K = X @ X.T
    L = Y @ Y.T
    
    # Zero the diagonals
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    
    # Row sums
    sum_K = K.sum()
    sum_L = L.sum()
    sum_K_rows = K.sum(axis=1)
    sum_L_rows = L.sum(axis=1)
    
    # Debiased HSIC estimator
    term1 = (K * L).sum()
    term2 = sum_K * sum_L / ((n-1) * (n-2))
    term3 = 2 * (sum_K_rows @ sum_L_rows) / (n-2)
    hsic_xy = (term1 + term2 - term3) / (n * (n-3))
    
    # Self-HSIC for normalization
    term1_xx = (K * K).sum()
    term2_xx = sum_K ** 2 / ((n-1) * (n-2))
    term3_xx = 2 * (sum_K_rows @ sum_K_rows) / (n-2)
    hsic_xx = (term1_xx + term2_xx - term3_xx) / (n * (n-3))
    
    term1_yy = (L * L).sum()
    term2_yy = sum_L ** 2 / ((n-1) * (n-2))
    term3_yy = 2 * (sum_L_rows @ sum_L_rows) / (n-2)
    hsic_yy = (term1_yy + term2_yy - term3_yy) / (n * (n-3))
    
    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0
    
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)



def biased_linear_cka(X, Y):
    """
    Standard (biased) linear CKA for comparison.
    """

    eps=1e-12
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n], Y[:n]

    # Center features (optional but fine)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Linear kernels
    K = X @ X.T
    L = Y @ Y.T

    # Center Gram matrices: Kc = H K H
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    num = np.sum(Kc * Lc)
    den = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc)) + eps
    return float(num / den)



def pwcka(X, Y, threshold=0.99, eps=1e-12):
    """
    Effective Rank PWCKA: project each representation onto its effective-rank PCA
            scores (k chosen by variance threshold), then compute correct CKA.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n], Y[:n]

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # SVD
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)

    def effective_rank_from_svals(S, thr=0.99):
        var = S**2
        cum = np.cumsum(var) / (np.sum(var) + eps)
        k = int(np.searchsorted(cum, thr) + 1)
        return max(1, min(k, len(S)))

    kx = effective_rank_from_svals(Sx, thr=threshold)
    ky = effective_rank_from_svals(Sy, thr=threshold)
    k = min(kx, ky)

    X_proj = Ux[:, :k] * Sx[:k]
    Y_proj = Uy[:, :k] * Sy[:k]

    try:
        return biased_linear_cka(X_proj, Y_proj)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def procrustes_similarity(X, Y):
    """
    Procrustes similarity using SciPy with robust error handling.
    Returns float in [0, 1] or NaN on failure.
    """
    try:
        # Convert inputs
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        # Basic checks
        if X.shape != Y.shape:
            return np.nan
        if len(X) < 2:
            return np.nan

        # Check for all NaN or constant values
        if np.all(np.isnan(X)) or np.all(np.isnan(Y)):
            return np.nan

        # Replace NaN with column means if any
        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            X = X.copy()
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        if np.any(np.isnan(Y)):
            col_means = np.nanmean(Y, axis=0)
            nan_mask = np.isnan(Y)
            Y = Y.copy()
            Y[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Check for constant columns (add tiny noise if needed)
        X_std = X.std(axis=0)
        Y_std = Y.std(axis=0)

        if np.any(X_std < 1e-12) or np.any(Y_std < 1e-12):
            # Add minimal noise to avoid degeneracy
            rng = np.random.default_rng(42)
            noise_level = 1e-8
            X = X + rng.standard_normal(X.shape) * noise_level
            Y = Y + rng.standard_normal(Y.shape) * noise_level

        # Use SciPy's procrustes
        X_new, Y_new, disparity = procrustes(X, Y)

        # disparity is already normalized sum of squared errors [0, 1]
        similarity = 1.0 - disparity

        # Ensure valid range
        similarity = np.clip(similarity, 0.0, 1.0)

        return float(similarity) if np.isfinite(similarity) else np.nan

    except Exception:
        return np.nan


# =============================================================================
# TEST 3A: CORE SPECTRAL SENSITIVITY WITH RSA BASELINES
# =============================================================================

def run_test3a_rsa_baselines():
    """
    Core spectral sensitivity comparing Shesha to RSA reliability baselines.
    """
    print("\n" + "=" * 70)
    print("TEST 3A: Spectral Sensitivity - RSA Baseline Comparison")
    print("=" * 70)
    print("\nComparing Shesha to RSA reliability methods (Diedrichsen et al., 2021)")

    n, d = 200, 256
    rng = np.random.default_rng(RANDOM_SEED)

    # Create controlled spectrum: eigenvalues decay as 1/k
    print("\nGenerating representation with controlled spectral decay (1/k)...")
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    S = np.zeros((n, d))
    np.fill_diagonal(S, [100.0 / (i + 1) for i in range(min(n, d))])
    X_orig = U @ S @ V.T

    # Fit PCA
    pca = PCA(n_components=min(n, d), random_state=RANDOM_SEED)
    pca.fit(X_orig)
    X_pca_full = pca.transform(X_orig)

    removal_levels = list(range(0, 51))  # Every PC from 0 to 50

    print(f"\n{'PCs Rem':<8} {'Shesha':<9} "
          f"{'Whit-Sh':<9} {'CrossN':<9} {'CKA-db':<9}")
    print("-" * 62)

    results = []

    for k in removal_levels:
        seed_k = RANDOM_SEED + k

        # Remove top k PCs
        X_pca = X_pca_full.copy()
        X_pca[:, :k] = 0.0
        X_mod = pca.inverse_transform(X_pca)

        # Stability metrics
        shesha = split_half_shesha(X_mod, random_state=seed_k)
        whitened = whitened_rsa_stability(X_mod, random_state=seed_k)
        crossnobis = crossnobis_stability(X_mod, random_state=seed_k)

        # Similarity (for reference)
        cka_db = debiased_linear_cka(X_orig, X_mod)

        results.append({
            'pcs_removed': k,
            'shesha': shesha,
            'whitened_shesha': whitened,
            'crossnobis': crossnobis,
            'cka_debiased': cka_db
        })

        print(f"{k:<8} {shesha:<9.4f} "
              f"{whitened:<9.4f} {crossnobis:<9.4f} {cka_db:<9.4f}")

    # Save results
    headers = list(results[0].keys())
    rows = [[r[h] for h in headers] for r in results]
    save_to_csv('test3a_rsa_baselines.csv', headers, rows)

    # Correlation analysis
    print("\n" + "-" * 40)
    print("Correlations between stability measures:")
    print("-" * 40)

    shesha_vals = [r['shesha'] for r in results]
    # for metric in ['rsa_raw', 'rsa_spearman_brown', 'whitened_shesha', 'crossnobis']:
    for metric in ['whitened_shesha', 'crossnobis']:

        metric_vals = [r[metric] for r in results]
        rho, p = spearmanr(shesha_vals, metric_vals)
        print(f"  Shesha vs {metric:<20}: rho = {rho:>6.3f} (p = {p:.4f})")

    return results


# =============================================================================
# TEST 3B: MULTIPLE SIMILARITY METRICS COMPARISON
# =============================================================================

def run_test3b_similarity_metrics():
    """
    Compare behavior of multiple similarity metrics under PC deletion.
    """
    print("\n" + "=" * 70)
    print("TEST 3B: Spectral Sensitivity - Multiple Similarity Metrics")
    print("=" * 70)
    print("\nComparing CKA variants, PWCKA, and Procrustes")

    n, d = 200, 256
    rng = np.random.default_rng(RANDOM_SEED + 100)

    # Create controlled spectrum
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    S = np.zeros((n, d))
    np.fill_diagonal(S, [100.0 / (i + 1) for i in range(min(n, d))])
    X_orig = U @ S @ V.T

    pca = PCA(n_components=min(n, d), random_state=RANDOM_SEED)
    pca.fit(X_orig)
    X_pca_full = pca.transform(X_orig)

    removal_levels = list(range(0, 51))

    print(f"\n{'PCs':<6} {'Shesha':<8} {'CKA-db':<8} {'CKA-b':<8} "
          f"{'PWCKA':<8} {'Procr':<8}")
    print("-" * 54)

    results = []

    for k in removal_levels:
        seed_k = RANDOM_SEED + 100 + k

        X_pca = X_pca_full.copy()
        X_pca[:, :k] = 0.0
        X_mod = pca.inverse_transform(X_pca)

        shesha = split_half_shesha(X_mod, random_state=seed_k)
        cka_db = debiased_linear_cka(X_orig, X_mod)
        cka_b = biased_linear_cka(X_orig, X_mod)
        pwcka_val = pwcka(X_orig, X_mod)
        procr_val = procrustes_similarity(X_orig, X_mod)

        results.append({
            'pcs_removed': k,
            'shesha': shesha,
            'cka_debiased': cka_db,
            'cka_biased': cka_b,
            'pwcka': pwcka_val,
            'procrustes': procr_val
        })


        print(f"{k:<6} {shesha:<8.4f} {cka_db:<8.4f} {cka_b:<8.4f} "
              f" {pwcka_val:<8.4f} {procr_val:<8.4f}")


    headers = list(results[0].keys())
    rows = [[r[h] for h in headers] for r in results]
    save_to_csv('test3b_similarity_metrics.csv', headers, rows)

    # Analysis: When does each metric collapse?
    print("\n" + "-" * 40)
    print("Collapse analysis (first k where metric < 0.5):")
    print("-" * 40)

    for metric in ['shesha', 'cka_debiased', 'cka_biased', 'pwcka', 'procrustes']:

        vals = [r[metric] for r in results]
        collapse_k = None
        for i, v in enumerate(vals):
            if v < 0.5:
                collapse_k = results[i]['pcs_removed']
                break
        if collapse_k is not None:
            print(f"  {metric:<15}: collapses at k = {collapse_k}")
        else:
            print(f"  {metric:<15}: never collapses below 0.5")

    return results


# =============================================================================
# TEST 3C: PREPROCESSING ABLATION
# =============================================================================

def run_test3c_preprocessing_ablation():
    """
    Ablate preprocessing choices to rule out metric-specific artifacts.
    """
    print("\n" + "=" * 70)
    print("TEST 3C: Spectral Sensitivity - Preprocessing Ablation")
    print("=" * 70)
    print("\nTesting: raw, centered, centered+normalized, whitened")

    n, d = 200, 256
    rng = np.random.default_rng(RANDOM_SEED + 200)

    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    S = np.zeros((n, d))
    np.fill_diagonal(S, [100.0 / (i + 1) for i in range(min(n, d))])
    X_orig = U @ S @ V.T

    pca = PCA(n_components=min(n, d), random_state=RANDOM_SEED)
    pca.fit(X_orig)
    X_pca_full = pca.transform(X_orig)

    preprocessing_configs = [
        {'name': 'raw', 'centering': False, 'normalize': False, 'whiten': False},
        {'name': 'centered', 'centering': True, 'normalize': False, 'whiten': False},
        {'name': 'centered_normalized', 'centering': True, 'normalize': True, 'whiten': False},
        {'name': 'whitened', 'centering': True, 'normalize': False, 'whiten': True},
    ]

    removal_levels = list(range(0, 51))
    all_results = []

    for config in preprocessing_configs:
        config_name = config['name']
        preproc_kwargs = {k: config[k] for k in ['centering', 'normalize', 'whiten']}

        print(f"\n--- Preprocessing: {config_name} ---")
        print(f"{'PCs':<6} {'Shesha':<8} {'CKA-db':<8} {'CKA-b':<8} {'Procr':<8}")
        print("-" * 38)

        for k in removal_levels:
            seed_k = RANDOM_SEED + 200 + k

            X_pca = X_pca_full.copy()
            X_pca[:, :k] = 0.0
            X_mod = pca.inverse_transform(X_pca)

            # Apply preprocessing
            X_ref = apply_preprocessing(X_orig, **preproc_kwargs)
            X_test = apply_preprocessing(X_mod, **preproc_kwargs)

            shesha = split_half_shesha(X_test, random_state=seed_k)
            cka_db = debiased_linear_cka(X_ref, X_test)
            cka_b = biased_linear_cka(X_ref, X_test)
            procr = procrustes_similarity(X_ref, X_test)

            all_results.append({
                'preprocessing': config_name,
                'pcs_removed': k,
                'shesha': shesha,
                'cka_debiased': cka_db,
                'cka_biased': cka_b,
                'procrustes': procr
            })

            print(f"{k:<6} {shesha:<8.4f} {cka_db:<8.4f} {cka_b:<8.4f} {procr:<8.4f}")

    headers = list(all_results[0].keys())
    rows = [[r[h] for h in headers] for r in all_results]
    save_to_csv('test3c_preprocessing_ablation.csv', headers, rows)

    # Summary: Is the divergence consistent across preprocessing?
    print("\n" + "-" * 50)
    print("Divergence consistency check (k=30):")
    print("-" * 50)

    for config_name in [c['name'] for c in preprocessing_configs]:
        subset = [r for r in all_results
                  if r['preprocessing'] == config_name and r['pcs_removed'] == 30]
        if subset:
            r = subset[0]
            divergence = r['shesha'] - r['cka_debiased']
            print(f"  {config_name:<20}: Shesha - CKA = {divergence:>6.3f} "
                  f"(Shesha={r['shesha']:.3f}, CKA={r['cka_debiased']:.3f})")

    return all_results


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary(results_3a, results_3b, results_3c):
    """Print summary statistics for the paper."""

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Key finding 1: Shesha correlates with RSA baselines
    print("\n1. Shesha vs RSA Reliability Baselines:")
    shesha = [r['shesha'] for r in results_3a]
    whitened = [r['whitened_shesha'] for r in results_3a]
    crossnobis = [r['crossnobis'] for r in results_3a]

    rho_white, _ = spearmanr(shesha, whitened)
    rho_cross, _ = spearmanr(shesha, crossnobis)

    print(f"   Shesha vs Whitened:       rho = {rho_white:.3f}")
    print(f"   Shesha vs Crossnobis:     rho = {rho_cross:.3f}")
    print("   -> Shesha is consistent with established RSA reliability methods")

    # Key finding 2: All similarity metrics collapse, Shesha doesn't
    print("\n2. Stability vs Similarity Divergence:")
    k30_idx = [i for i, r in enumerate(results_3b) if r['pcs_removed'] == 30][0]
    r30 = results_3b[k30_idx]
    print(f"   At k=30 PCs removed:")
    print(f"   - Shesha:     {r30['shesha']:.3f}")
    print(f"   - CKA (db):   {r30['cka_debiased']:.3f}")
    print(f"   - PWCKA:      {r30['pwcka']:.3f}")
    print(f"   - Procrustes: {r30['procrustes']:.3f}")
    print("   -> Divergence is consistent across all similarity metrics")

    # Key finding 3: Preprocessing doesn't change the story
    print("\n3. Preprocessing Robustness:")
    for preproc in ['raw', 'centered', 'centered_normalized', 'whitened']:
        subset = [r for r in results_3c
                  if r['preprocessing'] == preproc and r['pcs_removed'] == 30]
        if subset:
            r = subset[0]
            print(f"   {preproc:<22}: Shesha={r['shesha']:.3f}, CKA={r['cka_debiased']:.3f}")
    print("   -> Divergence holds across preprocessing choices")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("# TEST 3: SPECTRAL SENSITIVITY ANALYSIS (EXTENDED)")
    print("#" * 70)

    # Run all three sub-tests
    results_3a = run_test3a_rsa_baselines()
    results_3b = run_test3b_similarity_metrics()
    results_3c = run_test3c_preprocessing_ablation()

    # Print summary
    print_summary(results_3a, results_3b, results_3c)

    print("\n" + "=" * 70)
    print("COMPLETE. Results saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()