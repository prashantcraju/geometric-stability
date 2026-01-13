"""
Shesha Distinction Ground Truth Tests
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import warnings
from pathlib import Path
import csv

warnings.filterwarnings('ignore')

OUTDIR = Path("./shesha-distinction")
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPLICIT CONSTANTS
# =============================================================================

EPS_STD = 1e-9          # Tolerance for zero-variance checks
N_SPLITS = 50           # Number of splits for Shesha
MIN_SAMPLES_CKA = 4     # Minimum samples for debiased CKA (denominators use n-3, n-2)
QUADRANT_THRESHOLD = 0.4  # Threshold for high/low classification (matches paper guidelines)
RANDOM_SEED = 320        # For reproducibility

# =============================================================================
# 1. METRICS (Debiased & Standardized)
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
    
    # Set up RNG for reproducibility
    rng = np.random.default_rng(random_state)

    def get_rdm_vector(M):
        """Compute RDM as correlation distance (consistent with GPU version)."""
        if M.shape[0] < 2:
            return np.array([0])
        return pdist(M, metric='correlation')

    for _ in range(n_splits):
        # Random split of features
        perm = rng.permutation(n_features)
        half = n_features // 2

        # True split: use all features (no truncation)
        X1 = X[:, perm[:half]]
        X2 = X[:, perm[half:]]

        # Get the two distance vectors
        rdm1 = get_rdm_vector(X1)
        rdm2 = get_rdm_vector(X2)

        # FIX: Check for any non-finite values
        if not (np.all(np.isfinite(rdm1)) and np.all(np.isfinite(rdm2))):
            continue

        # FIX: Use tolerance with nanstd for belt-and-suspenders safety
        if np.nanstd(rdm1) < EPS_STD or np.nanstd(rdm2) < EPS_STD:
            continue

        r, _ = spearmanr(rdm1, rdm2)

        # Use isfinite to catch both nan and inf edge cases
        if np.isfinite(r):
            correlations.append(r)

    return np.mean(correlations) if correlations else 0.0


# =============================================================================
# 2. GENERATORS
# =============================================================================

def generate_controlled_stability(n_samples=200, n_features=256,
                                   signal_dims=50, signal_weight=0.5,
                                   random_state=None):
    """
    Generate representation with controlled stability.
    
    Parameters:
    -----------
    signal_weight : float
        Mixing coefficient alpha in [0, 1]. Higher = more stable structure.
        Output = signal_weight * signal + (1 - signal_weight) * noise
    """
    rng = np.random.default_rng(random_state)
    
    latent = rng.standard_normal((n_samples, signal_dims))
    projection = rng.standard_normal((signal_dims, n_features))
    signal = latent @ projection
    signal = signal / (np.std(signal) + EPS_STD)
    noise = rng.standard_normal((n_samples, n_features))
    return signal_weight * signal + (1 - signal_weight) * noise


def save_to_csv(filename, headers, rows):
    """Save results to CSV file."""
    filepath = OUTDIR / filename
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"   [Saved data to {filename}]")


def print_quadrant_stats(sample_shesha, sample_cka, quadrant_labels):
    """Print per-quadrant summary statistics including n, mean, std, and range."""
    print("\n   Per-quadrant summary:")
    print(f"   {'Quadrant':<12} {'n':<4} {'Shesha (mean +/- std)':<22} {'[min, max]':<14} "
          f"{'CKA (mean +/- std)':<22} {'[min, max]':<14}")
    print("   " + "-" * 90)
    
    for quadrant in ['High/High', 'High/Low', 'Low/Low', 'Low/High']:
        idx = [i for i, x in enumerate(quadrant_labels) if x == quadrant]
        if idx:
            shesha_vals = [sample_shesha[i] for i in idx]
            cka_vals = [sample_cka[i] for i in idx]
            print(f"   {quadrant:<12} {len(idx):<4} "
                  f"{np.mean(shesha_vals):>6.3f} +/- {np.std(shesha_vals):<6.3f}       "
                  f"[{np.min(shesha_vals):>5.3f}, {np.max(shesha_vals):>5.3f}]  "
                  f"{np.mean(cka_vals):>6.3f} +/- {np.std(cka_vals):<6.3f}       "
                  f"[{np.min(cka_vals):>5.3f}, {np.max(cka_vals):>5.3f}]")


# =============================================================================
# 3. THE VALIDATION SUITE
# =============================================================================

def run_comprehensive_suite():
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUITE: SHESHA vs. SIMILARITY METRICS")
    print("=" * 70)
    print(f"\nUsing constants: EPS_STD={EPS_STD}, N_SPLITS={N_SPLITS}, "
          f"QUADRANT_THRESHOLD={QUADRANT_THRESHOLD}")

    # Single RNG for entire suite - no mixing of RNG systems
    rng = np.random.default_rng(RANDOM_SEED)

    # --- TEST 1: FINE-GRAINED GROUND TRUTH RECOVERY ---
    print("\n[TEST 1] Ground Truth Recovery (Sensitivity)")
    print("-" * 50)
    signal_weights = np.linspace(0, 1, 21)
    results_t1 = []
    shesha_scores = []

    for i, alpha in enumerate(signal_weights):
        seed = RANDOM_SEED + i
        X = generate_controlled_stability(signal_weight=alpha, random_state=seed)
        score = split_half_shesha(X, random_state=seed)
        shesha_scores.append(score)
        results_t1.append([alpha, score])

    save_to_csv('results_test1_sensitivity.csv', ['Signal_Weight_Alpha', 'Shesha_Score'], results_t1)

    r, p = spearmanr(signal_weights, shesha_scores)
    print(f"   Correlation with Ground Truth (signal weight alpha): rho = {r:.4f}")
    if r > 0.95:
        print("   >> PASS: Shesha accurately tracks signal weight levels.")

    # --- TEST 2: DISSOCIATION & BIAS CHECK ---
    print("\n[TEST 2] Dissociation & Bias Check (Case D)")
    print("-" * 50)

    X_d = rng.standard_normal((200, 256))
    Y_d = rng.standard_normal((200, 256))

    shesha_d = split_half_shesha(X_d, random_state=RANDOM_SEED + 100)
    cka_debias = debiased_linear_cka(X_d, Y_d)

    print(f"   Case D (Random vs Random):")
    print(f"   Shesha (Stability):     {shesha_d:.3f} (Expected ~0.0)")
    print(f"   Debiased CKA (Sim):     {cka_debias:.3f} (Expected ~0.0)")

    save_to_csv('results_test2_bias.csv', ['Metric', 'Score'],
                [['Shesha', shesha_d], ['Debiased_CKA', cka_debias]])

    # --- TEST 3: SPECTRAL SENSITIVITY (The "Mic Drop") ---
    print("\n[TEST 3] Spectral Sensitivity (PC Deletion)")
    print("-" * 50)

    n, d = 200, 256
    rng_spec = np.random.default_rng(RANDOM_SEED + 200)
    U, _ = np.linalg.qr(rng_spec.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng_spec.standard_normal((d, d)))
    S = np.zeros((n, d))
    np.fill_diagonal(S, [100.0/(i+1) for i in range(min(n, d))])
    X_spec = U @ S @ V.T

    removal_levels = [i for i in range(50)]
    res_shesha = []
    res_cka = []
    results_t3 = []

    pca = PCA(n_components=min(n, d), random_state=RANDOM_SEED)
    pca.fit(X_spec)

    print(f"   {'PCs Removed':<12} {'Shesha':<10} {'Debiased CKA':<15}")
    
    # Compute transform once outside the loop for efficiency
    X_pca_full = pca.transform(X_spec)
    
    for k in removal_levels:
        X_pca = X_pca_full.copy()
        X_pca[:, :k] = 0.0
        X_mod = pca.inverse_transform(X_pca)

        # Use seed tied to this specific modified data
        s = split_half_shesha(X_mod, random_state=RANDOM_SEED + 200 + k)
        c = debiased_linear_cka(X_spec, X_mod)
        res_shesha.append(s)
        res_cka.append(c)
        results_t3.append([k, s, c])
        if k % 10 == 0 or k < 5:
            print(f"   {k:<12} {s:<10.3f} {c:<15.3f}")

    save_to_csv('results_test3_spectral.csv', ['PCs_Removed', 'Shesha', 'Debiased_CKA'], results_t3)
    
    # NOTE: Debiased CKA can go negative - this is expected for an unbiased estimator
    # when the true similarity is near zero. Negative values indicate no meaningful
    # alignment rather than "anti-alignment".

    # --- TEST 4: ORTHOGONALITY (Full Quadrant Sampling) ---
    print("\n[TEST 4] Cross-Metric Correlation (Orthogonality)")
    print("-" * 50)
    print("   Sampling balanced quadrants...")

    sample_shesha = []
    sample_cka = []
    quadrant_labels = []

    # Q1: High Stab / High Sim (Same Signal)
    for i in range(15):
        seed = RANDOM_SEED + 1000 + i
        base = generate_controlled_stability(signal_weight=0.9, random_state=seed)
        # Use same seed for Shesha as for data generation
        sample_shesha.append(split_half_shesha(base, random_state=seed))
        # Add small noise using seeded rng
        rng_q1 = np.random.default_rng(seed)
        sample_cka.append(debiased_linear_cka(base, base + rng_q1.standard_normal(base.shape) * 0.1))
        quadrant_labels.append('High/High')

    # Q2: High Stab / Low Sim (Different Signals)
    for i in range(15):
        seed_x = RANDOM_SEED + 2000 + i
        seed_y = RANDOM_SEED + 3000 + i
        X = generate_controlled_stability(signal_weight=0.9, random_state=seed_x)
        Y = generate_controlled_stability(signal_weight=0.9, random_state=seed_y)
        sample_shesha.append((split_half_shesha(X, random_state=seed_x) + 
                              split_half_shesha(Y, random_state=seed_y)) / 2)
        sample_cka.append(debiased_linear_cka(X, Y))
        quadrant_labels.append('High/Low')

    # Q3: Low Stab / Low Sim (Independent Noise)
    for i in range(15):
        seed_x = RANDOM_SEED + 4000 + i
        seed_y = RANDOM_SEED + 5000 + i
        X = generate_controlled_stability(signal_weight=0.1, random_state=seed_x)
        Y = generate_controlled_stability(signal_weight=0.1, random_state=seed_y)
        sample_shesha.append((split_half_shesha(X, random_state=seed_x) + 
                              split_half_shesha(Y, random_state=seed_y)) / 2)
        sample_cka.append(debiased_linear_cka(X, Y))
        quadrant_labels.append('Low/Low')

    # Q4: Low Stab / High Sim - ADVERSARIAL QUADRANT VIA REJECTION SAMPLING
    # Strategy: Create representations where samples are similar (high CKA) but the 
    # geometric structure doesn't replicate across feature splits (low Shesha).
    # 
    # NOTE: With 256 i.i.d. features, geometry can still be fairly consistent across
    # halves. We use rejection sampling to ensure we land in the target quadrant.
    # This is an engineered adversarial case, not a natural generative regime.
    # The acceptance rate should be reported if relying on this quadrant in papers.
    q4_misses = 0
    q4_target = 15
    q4_collected = 0
    max_total_attempts = q4_target * 100  # Cap total attempts to avoid infinite loop
    attempt_count = 0
    
    while q4_collected < q4_target and attempt_count < max_total_attempts:
        seed = RANDOM_SEED + 6000 + attempt_count
        rng_q4 = np.random.default_rng(seed)
        
        n_samples, n_features = 200, 256
        
        # Approach: Create features that are independent (each feature is random noise)
        # but X and Y share the same feature values (high CKA between X and Y)
        # Since features are independent, splitting features gives inconsistent RDMs (low Shesha)
        
        # Each feature is independent random noise
        X = rng_q4.standard_normal((n_samples, n_features))
        # Y is a noisy copy of X (preserves sample alignment, high CKA)
        Y = X + rng_q4.standard_normal((n_samples, n_features)) * 0.15
        
        shesha_x = split_half_shesha(X, random_state=seed)
        shesha_y = split_half_shesha(Y, random_state=seed)
        avg_shesha = (shesha_x + shesha_y) / 2
        cka_val = debiased_linear_cka(X, Y)
        
        attempt_count += 1
        
        # Check if we landed in the right quadrant
        if avg_shesha < QUADRANT_THRESHOLD and cka_val > QUADRANT_THRESHOLD:
            sample_shesha.append(avg_shesha)
            sample_cka.append(cka_val)
            quadrant_labels.append('Low/High')
            q4_collected += 1
        else:
            q4_misses += 1
    
    if q4_collected < q4_target:
        print(f"   WARNING: Only collected {q4_collected}/{q4_target} Q4 samples after {attempt_count} attempts")
    
    # Guard against division by zero
    rate = (q4_collected / attempt_count) if attempt_count > 0 else 0.0
    print(f"   Q4 (rejection-sampled): {q4_collected} collected, {q4_misses} rejected "
          f"(acceptance: {q4_collected}/{attempt_count} = {rate:.1%})")

    results_t4 = list(zip(sample_shesha, sample_cka, quadrant_labels))
    save_to_csv('results_test4_orthogonality.csv', ['Shesha', 'Debiased_CKA', 'Quadrant'], results_t4)

    print_quadrant_stats(sample_shesha, sample_cka, quadrant_labels)

    rho, p = spearmanr(sample_shesha, sample_cka)
    print(f"\n   Overall Shesha vs CKA Correlation: rho = {rho:.3f}")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))

    # Plot Spectral Sensitivity
    plt.subplot(1, 2, 1)
    plt.plot(removal_levels, res_shesha, 'b-o', label='Shesha (Stability)', markersize=3)
    plt.plot(removal_levels, res_cka, 'r-s', label='Debiased CKA (Similarity)', markersize=3)
    plt.xlabel('Top PCs Removed')
    plt.ylabel('Score')
    plt.title('Spectral Sensitivity (Tail Awareness)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Orthogonality
    plt.subplot(1, 2, 2)
    colors = {'High/High': 'green', 'High/Low': 'blue', 'Low/Low': 'red', 'Low/High': 'orange'}
    for lbl in ['High/High', 'High/Low', 'Low/Low', 'Low/High']:
        idx = [i for i, x in enumerate(quadrant_labels) if x == lbl]
        plt.scatter([sample_cka[i] for i in idx], [sample_shesha[i] for i in idx],
                    alpha=0.6, label=lbl, c=colors[lbl], s=50)

    plt.axhline(QUADRANT_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(QUADRANT_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Debiased CKA')
    plt.ylabel('Shesha')
    plt.title(f'Metric Dissociation (rho={rho:.2f})')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('validation_final.png', dpi=150)
    print(f"\n[Figure Saved] validation_final.png")
    plt.show()


if __name__ == "__main__":
    run_comprehensive_suite()