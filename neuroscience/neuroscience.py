"""
Shesha Neuroscience Analysis
- Behavioral ground truth analysis
- Regional analysis (shows hierarchy)
- Drift detection (temporal stability)
"""


import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, ttest_rel
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, orthogonal_procrustes
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# --- Configuration ---
N_BOOTSTRAP = 10000
SEED = 320
np.random.seed(SEED)

OUTPUT_PATH = Path("./shesha-neuroscience")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("NEUROSCIENCE ANALYSIS - (Behavioral Ground Truth Focus)")
print("=" * 80)
print(f"Output directory: {OUTPUT_PATH}")


# =============================================================================
# 1. METRIC DEFINITIONS
# =============================================================================

def compute_shesha_split_half(X_odd, X_even, distance_metric='cosine'):
    """
    TRUE SHESHA: Split-half RDM correlation.

    Measures STABILITY - how reliably the pairwise distance structure
    is preserved across independent subsets of data.

    X_odd, X_even: (n_conditions, n_neurons) - condition means from odd/even trials

    Returns: Spearman correlation of RDMs (scalar in [-1, 1])
    """
    if X_odd.shape[0] < 3 or X_even.shape[0] < 3:
        return np.nan

    n_conds = min(X_odd.shape[0], X_even.shape[0])
    X_odd = X_odd[:n_conds]
    X_even = X_even[:n_conds]

    rdm_odd = pdist(X_odd, metric=distance_metric)
    rdm_even = pdist(X_even, metric=distance_metric)

    if len(rdm_odd) == 0 or len(rdm_even) == 0:
        return np.nan
    if np.std(rdm_odd) < 1e-10 or np.std(rdm_even) < 1e-10:
        return np.nan

    rho, _ = spearmanr(rdm_odd, rdm_even)

    if not np.isfinite(rho):
        return np.nan

    return float(rho)


def compute_shesha_temporal(X_early, X_late, conditions_early, conditions_late,
                            distance_metric='cosine'):
    """
    SHESHA TEMPORAL: RDM correlation between early and late trials.
    Measures whether the pairwise distance structure drifts over time.
    """
    unique_conds = list(set(conditions_early) & set(conditions_late))

    if len(unique_conds) < 3:
        return np.nan

    means_early = []
    means_late = []

    for cond in unique_conds:
        mask_early = np.array([c == cond for c in conditions_early])
        mask_late = np.array([c == cond for c in conditions_late])

        if mask_early.sum() < 1 or mask_late.sum() < 1:
            continue

        means_early.append(X_early[mask_early].mean(axis=0))
        means_late.append(X_late[mask_late].mean(axis=0))

    if len(means_early) < 3:
        return np.nan

    means_early = np.array(means_early)
    means_late = np.array(means_late)

    rdm_early = pdist(means_early, metric=distance_metric)
    rdm_late = pdist(means_late, metric=distance_metric)

    if np.std(rdm_early) < 1e-10 or np.std(rdm_late) < 1e-10:
        return np.nan

    rho, _ = spearmanr(rdm_early, rdm_late)

    if not np.isfinite(rho):
        return np.nan

    return float(rho)


def compute_centroid_drift(X_early, X_late):
    """
    CENTROID DRIFT: Cosine similarity of population centroids.
    Simple temporal consistency measure.
    """
    norm_early = np.linalg.norm(X_early, axis=1, keepdims=True)
    norm_late = np.linalg.norm(X_late, axis=1, keepdims=True)

    X_early_norm = X_early / (norm_early + 1e-10)
    X_late_norm = X_late / (norm_late + 1e-10)

    c_early = X_early_norm.mean(axis=0)
    c_late = X_late_norm.mean(axis=0)

    denom = np.linalg.norm(c_early) * np.linalg.norm(c_late)
    if denom < 1e-10:
        return np.nan

    sim = np.dot(c_early, c_late) / denom
    return float(sim)


def compute_wuc(X_odd, X_even, shrinkage=0.1):
    """
    WUC: Whitened Unbiased Cosine (Diedrichsen et al, 2021).
    Addresses noise covariance bias in RSA.
    """
    if X_odd.shape[0] < 3 or X_even.shape[0] < 3:
        return np.nan

    n_conds = min(X_odd.shape[0], X_even.shape[0])
    X_odd = X_odd[:n_conds]
    X_even = X_even[:n_conds]

    def whitening_matrix(data, shrink):
        data_centered = data - data.mean(0, keepdims=True)
        n = data_centered.shape[0]
        if n < 2:
            return np.eye(data.shape[1])

        cov = np.cov(data_centered.T)
        if cov.ndim < 2:
            cov = np.atleast_2d(cov)

        diag = np.diag(np.diag(cov))
        cov_shrunk = (1 - shrink) * cov + shrink * diag
        cov_shrunk += 1e-6 * np.eye(cov_shrunk.shape[0])

        try:
            eigvals, eigvecs = np.linalg.eigh(cov_shrunk)
            eigvals = np.maximum(eigvals, 1e-10)
            W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return W
        except:
            return np.eye(data.shape[1])

    X_odd_c = X_odd - X_odd.mean(0, keepdims=True)
    X_even_c = X_even - X_even.mean(0, keepdims=True)

    W_odd = whitening_matrix(X_odd_c, shrinkage)
    W_even = whitening_matrix(X_even_c, shrinkage)

    X_odd_w = X_odd_c @ W_odd
    X_even_w = X_even_c @ W_even

    rdm_odd = pdist(X_odd_w, metric='cosine')
    rdm_even = pdist(X_even_w, metric='cosine')

    if np.std(rdm_odd) < 1e-10 or np.std(rdm_even) < 1e-10:
        return np.nan

    rho, _ = spearmanr(rdm_odd, rdm_even)

    if not np.isfinite(rho):
        return np.nan

    return float(rho)


def compute_crossvalidated_rsa(X, conditions, n_folds=5, distance_metric='cosine'):
    """
    Crossvalidated RSA (Walther et al. 2016).
    """
    from sklearn.model_selection import KFold

    if len(conditions) > 0 and isinstance(conditions[0], tuple):
        conditions_str = [str(c) for c in conditions]
    else:
        conditions_str = list(conditions)

    unique_conds = list(set(conditions_str))
    if len(unique_conds) < 3:
        return np.nan

    n_trials = len(conditions_str)

    if n_trials < n_folds * 2:
        return np.nan

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    correlations = []

    for train_idx, test_idx in kf.split(X):
        means_train = []
        means_test = []

        for cond in unique_conds:
            cond_mask = np.array([c == cond for c in conditions_str])

            train_cond = np.isin(np.arange(n_trials), train_idx) & cond_mask
            test_cond = np.isin(np.arange(n_trials), test_idx) & cond_mask

            if train_cond.sum() >= 1 and test_cond.sum() >= 1:
                means_train.append(X[train_cond].mean(axis=0))
                means_test.append(X[test_cond].mean(axis=0))

        if len(means_train) < 3:
            continue

        means_train = np.array(means_train)
        means_test = np.array(means_test)

        rdm_train = pdist(means_train, metric=distance_metric)
        rdm_test = pdist(means_test, metric=distance_metric)

        if np.std(rdm_train) < 1e-10 or np.std(rdm_test) < 1e-10:
            continue

        rho, _ = spearmanr(rdm_train, rdm_test)
        if np.isfinite(rho):
            correlations.append(rho)

    if not correlations:
        return np.nan

    return float(np.mean(correlations))


# =============================================================================
# 2. BOOTSTRAP HELPERS
# =============================================================================

def bootstrap_ci(data, statistic='mean', n_bootstrap=N_BOOTSTRAP, ci=95):
    """Bootstrap confidence interval."""
    data = np.array(data)
    data = data[~np.isnan(data)]

    if len(data) < 3:
        return np.nan, np.nan, np.nan

    stat_func = np.mean if statistic == 'mean' else np.median
    point_est = stat_func(data)

    boot_stats = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(boot_sample))

    boot_stats = np.array(boot_stats)
    alpha = (100 - ci) / 2
    ci_low = np.percentile(boot_stats, alpha)
    ci_high = np.percentile(boot_stats, 100 - alpha)

    return point_est, ci_low, ci_high


def bootstrap_correlation_ci(x, y, method='spearman', n_bootstrap=N_BOOTSTRAP, ci=95):
    """Bootstrap CI for correlation."""
    x, y = np.array(x), np.array(y)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan

    if method == 'spearman':
        point_est, p_val = spearmanr(x, y)
    else:
        point_est, p_val = pearsonr(x, y)

    boot_rhos = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        rho, _ = spearmanr(x[idx], y[idx]) if method == 'spearman' else pearsonr(x[idx], y[idx])
        if np.isfinite(rho):
            boot_rhos.append(rho)

    if len(boot_rhos) < 100:
        return point_est, np.nan, np.nan, p_val

    boot_rhos = np.array(boot_rhos)
    alpha = (100 - ci) / 2
    ci_low = np.percentile(boot_rhos, alpha)
    ci_high = np.percentile(boot_rhos, 100 - alpha)

    return point_est, ci_low, ci_high, p_val


# =============================================================================
# 3. DATA LOADING
# =============================================================================

def load_steinmetz_data():
    """Download and load Steinmetz dataset."""
    import requests

    urls = {
        'steinmetz_part1.npz': 'https://osf.io/agvxh/download',
        'steinmetz_part2.npz': 'https://osf.io/uv3mw/download',
    }

    all_sessions = []

    for filename, url in urls.items():
        print(f"Downloading {filename}...")
        try:
            r = requests.get(url)
            # Save strictly to the current OUTPUT_PATH
            save_path = os.path.join(OUTPUT_PATH, filename)
            
            with open(save_path, 'wb') as f:
                f.write(r.content)
            
            dat = np.load(save_path, allow_pickle=True)['dat']
            all_sessions.extend(dat)
            print(f"  -> {len(dat)} sessions downloaded and loaded")
            
        except Exception as e:
            print(f"  -> Failed to download {filename}: {e}")

    print(f"\nTotal sessions loaded: {len(all_sessions)}")
    return all_sessions


# =============================================================================
# 4. REGION MAPPING
# =============================================================================

REGION_MAPPING = {
    'VISp': 'Visual', 'VISl': 'Visual', 'VISrl': 'Visual',
    'VISam': 'Visual', 'VISpm': 'Visual', 'VISa': 'Visual', 'VISal': 'Visual',
    'LP': 'Thalamus', 'LD': 'Thalamus', 'LGd': 'Thalamus',
    'VPM': 'Thalamus', 'PO': 'Thalamus', 'MD': 'Thalamus',
    'MOp': 'Motor', 'MOs': 'Motor',
    'ACA': 'Frontal', 'PL': 'Frontal', 'ILA': 'Frontal', 'ORB': 'Frontal',
    'CA1': 'Hippocampus', 'CA3': 'Hippocampus', 'DG': 'Hippocampus', 'SUB': 'Hippocampus',
    'CP': 'Striatum', 'ACB': 'Striatum', 'LS': 'Striatum',
    'SNr': 'Midbrain', 'SCm': 'Midbrain', 'MRN': 'Midbrain', 'ZI': 'Midbrain',
}

def get_coarse_region(area):
    return REGION_MAPPING.get(area, 'Other')


# =============================================================================
# 5. MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_steinmetz():
    """Run analysis with TRUE SHESHA definition."""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    dat_list = load_steinmetz_data()

    results = []

    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)

    for session_idx, dat in enumerate(dat_list):
        n_trials = dat['spks'].shape[1]

        if n_trials < 60:
            continue

        brain_areas = dat['brain_area']
        unique_areas = np.unique(brain_areas)

        correct = dat['feedback_type']
        contrast_left = dat['contrast_left']
        contrast_right = dat['contrast_right']

        conditions = [(cl, cr) for cl, cr in zip(contrast_left, contrast_right)]
        unique_conditions = list(set(conditions))

        mid_trial = n_trials // 2
        early_trials = np.arange(mid_trial)
        late_trials = np.arange(mid_trial, n_trials)

        odd_trials = np.arange(0, n_trials, 2)
        even_trials = np.arange(1, n_trials, 2)

        acc_early = np.mean(correct[:mid_trial] == 1)
        acc_late = np.mean(correct[mid_trial:] == 1)
        behavior_delta = acc_late - acc_early

        for area in unique_areas:
            area_mask = brain_areas == area
            n_neurons = np.sum(area_mask)

            if n_neurons < 10:
                continue

            spikes = dat['spks'][area_mask, :, :]
            decision_epoch = spikes[:, :, 50:100].mean(axis=2)
            X = decision_epoch.T

            if np.isnan(X).any():
                X = np.nan_to_num(X)

            X_early = X[early_trials]
            X_late = X[late_trials]
            X_odd = X[odd_trials]
            X_even = X[even_trials]

            conditions_early = [conditions[i] for i in early_trials]
            conditions_late = [conditions[i] for i in late_trials]

            # Compute condition means
            cond_means_odd = []
            cond_means_even = []
            valid_conds = []

            for cond in unique_conditions:
                cond_mask = np.array([c == cond for c in conditions])
                cond_trials = X[cond_mask]
                if len(cond_trials) < 2:
                    continue

                cond_odd_mask = cond_mask & np.isin(np.arange(n_trials), odd_trials)
                cond_even_mask = cond_mask & np.isin(np.arange(n_trials), even_trials)

                if cond_odd_mask.sum() >= 1 and cond_even_mask.sum() >= 1:
                    cond_means_odd.append(X[cond_odd_mask].mean(axis=0))
                    cond_means_even.append(X[cond_even_mask].mean(axis=0))
                    valid_conds.append(cond)

            has_conditions = len(cond_means_odd) >= 3

            if has_conditions:
                cond_means_odd = np.array(cond_means_odd)
                cond_means_even = np.array(cond_means_even)

            # =================================================================
            # COMPUTE METRICS
            # =================================================================

            # Shesha: Split-half RDM correlation
            if has_conditions:
                shesha = compute_shesha_split_half(cond_means_odd, cond_means_even)
            else:
                shesha = np.nan

            # WUC
            if has_conditions:
                wuc = compute_wuc(cond_means_odd, cond_means_even)
            else:
                wuc = np.nan

            # Crossvalidated RSA
            rsa_cv = compute_crossvalidated_rsa(X, conditions)

            # Temporal stability
            shesha_temporal = compute_shesha_temporal(
                X_early, X_late, conditions_early, conditions_late
            )

            centroid_drift = compute_centroid_drift(X_early, X_late)

            # Null model for drift
            null_drifts = []
            n_perms = 500
            X_all = np.vstack([X_early, X_late])
            n_early = len(X_early)

            for _ in range(n_perms):
                perm_idx = np.random.permutation(len(X_all))
                X_perm_early = X_all[perm_idx[:n_early]]
                X_perm_late = X_all[perm_idx[n_early:]]
                null_sim = compute_centroid_drift(X_perm_early, X_perm_late)
                null_drifts.append(null_sim)

            null_mean = np.mean(null_drifts)
            null_std = np.std(null_drifts)
            z_score_drift = (centroid_drift - null_mean) / (null_std + 1e-10)

            # Tertile analysis
            tertile_1 = n_trials // 3
            tertile_2 = 2 * n_trials // 3

            X_early_tert = X[:tertile_1]
            X_mid_tert = X[tertile_1:tertile_2]
            X_late_tert = X[tertile_2:]

            stability_early_mid = compute_centroid_drift(X_early_tert, X_mid_tert)
            stability_mid_late = compute_centroid_drift(X_mid_tert, X_late_tert)

            results.append({
                'session': session_idx,
                'area': area,
                'region_coarse': get_coarse_region(area),
                'n_neurons': n_neurons,
                'n_trials': n_trials,
                'n_conditions': len(valid_conds),
                'behavior_delta': behavior_delta,

                'shesha': shesha,
                'wuc': wuc,
                'rsa_cv': rsa_cv,
                'shesha_temporal': shesha_temporal,
                'centroid_drift': centroid_drift,

                'null_mean_drift': null_mean,
                'null_std_drift': null_std,
                'z_score_drift': z_score_drift,

                'stability_early_mid': stability_early_mid,
                'stability_mid_late': stability_mid_late,
            })

        if (session_idx + 1) % 5 == 0:
            print(f"   Processed session {session_idx + 1}/{len(dat_list)}")

    df = pd.DataFrame(results)
    print(f"\nTotal observations: {len(df)}")
    print(f"Sessions: {df['session'].nunique()}")
    print(f"Brain areas: {df['area'].nunique()}")

    return df, dat_list


# =============================================================================
# 6. REGIONAL ANALYSIS
# =============================================================================

def compute_regional_analysis(df):
    """Compute regional statistics."""
    print("\n" + "=" * 80)
    print("REGIONAL ANALYSIS")
    print("=" * 80)

    regions = ['Visual', 'Thalamus', 'Hippocampus', 'Frontal', 'Motor',
               'Striatum', 'Midbrain', 'Other']
    metrics = ['shesha', 'wuc', 'rsa_cv', 'centroid_drift', 'shesha_temporal']

    regional_results = []

    for region in regions:
        subset = df[df['region_coarse'] == region]
        if len(subset) < 3:
            continue

        print(f"\n{region} (n = {len(subset)}):")

        for metric in metrics:
            if metric not in df.columns:
                continue

            values = subset[metric].dropna()
            if len(values) < 3:
                continue

            mean_val, ci_low, ci_high = bootstrap_ci(values, 'mean')

            regional_results.append({
                'region': region,
                'metric': metric,
                'n': len(values),
                'mean': mean_val,
                'ci_low': ci_low,
                'ci_high': ci_high
            })

            print(f"  {metric:20}: {mean_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    return pd.DataFrame(regional_results)


# =============================================================================
# 7. BEHAVIORAL GROUND TRUTH ANALYSIS
# =============================================================================

def compute_neural_tangling(X, dt=1):
    """
    Compute neural tangling (Russo et al., 2018).
    High tangling = unstable dynamics.
    """
    n = X.shape[0]
    if n < 3:
        return np.nan

    velocities = np.diff(X, axis=0)
    positions = X[:-1]

    if len(velocities) < 2:
        return np.nan

    tangling_values = []

    for i in range(len(positions)):
        max_tangling = 0
        for j in range(len(positions)):
            if i == j:
                continue

            pos_diff = np.linalg.norm(positions[i] - positions[j])
            if pos_diff < 1e-10:
                continue

            vel_diff = np.linalg.norm(velocities[i] - velocities[j])
            tangling = vel_diff / pos_diff
            max_tangling = max(max_tangling, tangling)

        tangling_values.append(max_tangling)

    if not tangling_values:
        return np.nan

    return float(np.mean(tangling_values))


def compute_behavioral_analysis(df, dat_list):
    """
    Comprehensive behavioral ground truth analysis.
    """
    print("\n" + "=" * 80)
    print("BEHAVIORAL GROUND TRUTH ANALYSIS")
    print("=" * 80)

    all_results = []
    area_session_behavior = []

    for session_idx, dat in enumerate(dat_list):
        n_trials = dat['spks'].shape[1]

        if n_trials < 60:
            continue

        brain_areas = dat['brain_area']
        unique_areas = np.unique(brain_areas)

        feedback = dat['feedback_type']
        trial_accuracy = (feedback == 1).astype(float)

        mid_trial = n_trials // 2
        acc_early = np.mean(trial_accuracy[:mid_trial])
        acc_late = np.mean(trial_accuracy[mid_trial:])

        contrast_left = dat['contrast_left']
        contrast_right = dat['contrast_right']
        conditions = [(cl, cr) for cl, cr in zip(contrast_left, contrast_right)]
        unique_conditions = list(set(conditions))

        for area in unique_areas:
            area_mask = brain_areas == area
            n_neurons = np.sum(area_mask)

            if n_neurons < 10:
                continue

            spikes = dat['spks'][area_mask, :, :]
            decision_epoch = spikes[:, :, 50:100].mean(axis=2)
            X = decision_epoch.T

            if np.isnan(X).any():
                X = np.nan_to_num(X)

            # Neural tangling
            cond_means = []
            for cond in unique_conditions:
                cond_mask = np.array([c == cond for c in conditions])
                if cond_mask.sum() >= 2:
                    cond_means.append(X[cond_mask].mean(axis=0))

            if len(cond_means) >= 3:
                cond_means_arr = np.array(cond_means)
                tangling = compute_neural_tangling(cond_means_arr)
            else:
                tangling = np.nan

            # Trial-by-trial neural-behavioral correlation
            neural_magnitude = np.linalg.norm(X, axis=1)

            if len(neural_magnitude) == len(trial_accuracy):
                rho_trial, _ = spearmanr(neural_magnitude, trial_accuracy)
                trial_neural_behavior_corr = rho_trial if np.isfinite(rho_trial) else np.nan
            else:
                trial_neural_behavior_corr = np.nan

            # Get metrics from df
            df_match = df[(df['session'] == session_idx) & (df['area'] == area)]

            if len(df_match) == 0:
                continue

            row = df_match.iloc[0]

            area_session_behavior.append({
                'session': session_idx,
                'area': area,
                'region_coarse': get_coarse_region(area),
                'n_neurons': n_neurons,
                'n_trials': n_trials,

                'shesha': row.get('shesha', np.nan),
                'wuc': row.get('wuc', np.nan),
                'centroid_drift': row.get('centroid_drift', np.nan),

                'acc_early': acc_early,
                'acc_late': acc_late,
                'acc_change': acc_late - acc_early,
                'mean_accuracy': np.mean(trial_accuracy),

                'tangling': tangling,
                'trial_neural_behavior_corr': trial_neural_behavior_corr,
            })

    df_behavior = pd.DataFrame(area_session_behavior)

    if len(df_behavior) == 0:
        print("No behavioral data computed.")
        return pd.DataFrame(all_results), df_behavior

    print(f"\nComputed behavioral measures for {len(df_behavior)} area-sessions")

    # =========================================================================
    # TEST EACH GROUND TRUTH
    # =========================================================================

    ground_truths = [
        ('mean_accuracy', 'Mean Accuracy'),
        ('acc_change', 'Accuracy Change'),
        ('tangling', 'Neural Tangling'),
        ('trial_neural_behavior_corr', 'Trial-by-Trial Neural-Behavior'),
    ]

    stability_metrics = ['shesha', 'wuc', 'centroid_drift']

    for gt_col, gt_name in ground_truths:
        print(f"\n--- {gt_name} ---")

        for metric in stability_metrics:
            valid = df_behavior[metric].notna() & df_behavior[gt_col].notna()
            if valid.sum() < 10:
                continue

            rho, ci_low, ci_high, p_val = bootstrap_correlation_ci(
                df_behavior.loc[valid, metric].values,
                df_behavior.loc[valid, gt_col].values,
                method='spearman'
            )

            all_results.append({
                'ground_truth': gt_col,
                'ground_truth_name': gt_name,
                'metric': metric,
                'rho': rho,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_value': p_val,
                'n': int(valid.sum()),
            })

            sig = "*" if p_val < 0.05 else ""
            sig = "**" if p_val < 0.01 else sig
            sig = "***" if p_val < 0.001 else sig

            print(f"  {metric:20} vs {gt_col:25}: rho = {rho:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}], p = {p_val:.4f} {sig}")

    return pd.DataFrame(all_results), df_behavior


# =============================================================================
# 8. DRIFT ANALYSIS
# =============================================================================

def compute_drift_analysis(df):
    """Analyze temporal drift patterns."""
    print("\n" + "=" * 80)
    print("DRIFT ANALYSIS")
    print("=" * 80)

    # Null model summary
    df_valid = df.dropna(subset=['z_score_drift'])

    z_mean, z_ci_low, z_ci_high = bootstrap_ci(df_valid['z_score_drift'], 'mean')
    obs_mean, obs_ci_low, obs_ci_high = bootstrap_ci(df_valid['centroid_drift'], 'mean')
    null_mean, null_ci_low, null_ci_high = bootstrap_ci(df_valid['null_mean_drift'], 'mean')

    print(f"\nNull Model Validation:")
    print(f"  Mean z-score: {z_mean:.1f} [{z_ci_low:.1f}, {z_ci_high:.1f}]")
    print(f"  Observed drift: {obs_mean:.3f} [{obs_ci_low:.3f}, {obs_ci_high:.3f}]")
    print(f"  Null drift: {null_mean:.3f} [{null_ci_low:.3f}, {null_ci_high:.3f}]")

    # Tertile dynamics
    valid = df['stability_early_mid'].notna() & df['stability_mid_late'].notna()
    df_valid = df[valid]

    if len(df_valid) >= 5:
        em_mean, em_ci_low, em_ci_high = bootstrap_ci(df_valid['stability_early_mid'], 'mean')
        ml_mean, ml_ci_low, ml_ci_high = bootstrap_ci(df_valid['stability_mid_late'], 'mean')

        t_stat, p_val = ttest_rel(df_valid['stability_early_mid'], df_valid['stability_mid_late'])

        print(f"\nTertile Dynamics:")
        print(f"  Early-Middle: {em_mean:.3f} [{em_ci_low:.3f}, {em_ci_high:.3f}]")
        print(f"  Middle-Late:  {ml_mean:.3f} [{ml_ci_low:.3f}, {ml_ci_high:.3f}]")
        print(f"  Paired t-test: t = {t_stat:.2f}, p = {p_val:.3f}")

    return {
        'z_mean': z_mean,
        'obs_mean': obs_mean,
        'null_mean': null_mean,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Run analysis
    df, dat_list = analyze_steinmetz()

    # Save raw results
    save_path = os.path.join(OUTPUT_PATH, "shesha_neuro_v3.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved: {save_path}")

    # Regional analysis
    regional_df = compute_regional_analysis(df)
    save_path = os.path.join(OUTPUT_PATH, "shesha_regional_v3.csv")
    regional_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    # Behavioral analysis (KEY VALIDATION)
    behavioral_df, df_behavior_detail = compute_behavioral_analysis(df, dat_list)
    save_path = os.path.join(OUTPUT_PATH, "shesha_behavioral_v3.csv")
    behavioral_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    save_path = os.path.join(OUTPUT_PATH, "behavioral_detailed_v3.csv")
    df_behavior_detail.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    # Drift analysis
    drift_results = compute_drift_analysis(df)
