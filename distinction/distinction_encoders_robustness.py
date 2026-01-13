"""
Shesha Distinction - Metric Robustness
Tests distinction between Shesha (stability) and PWCKA/Procrustes (similarity) 
specifically for the Language domain.

Compares results with standard CKA to see if PWCKA/Procrustes changes the 
relationship.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = Path("./shesha-distinction")
OUTDIR.mkdir(parents=True, exist_ok=True)



SEEDS = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]

CONFIG = {
    'language': {'n_samples': 500, 'max_len': 64},
}

# =============================================================================
# METRICS
# =============================================================================

def compute_shesha_features(X, n_splits=30, random_state=None):
    """Feature-split Shesha."""
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    
    if n_samples < 10 or n_features < 4:
        return np.nan
    
    rng = np.random.default_rng(random_state)
    corrs = []
    tri_idx = np.triu_indices(n_samples, k=1)
    
    for _ in range(n_splits):
        perm = rng.permutation(n_features)
        half = n_features // 2
        if half < 2:
            half = 2
        
        idx1 = perm[:half]
        idx2 = perm[half:2*half]
        if len(idx2) < 2:
            idx2 = idx1
        
        X1 = X[:, idx1]
        X2 = X[:, idx2]
        
        X1_norm = X1 / (np.linalg.norm(X1, axis=1, keepdims=True) + 1e-12)
        X2_norm = X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-12)
        
        rdm1 = 1.0 - (X1_norm @ X1_norm.T)
        rdm2 = 1.0 - (X2_norm @ X2_norm.T)
        
        v1 = rdm1[tri_idx]
        v2 = rdm2[tri_idx]
        
        if np.std(v1) < 1e-9 or np.std(v2) < 1e-9:
            continue
        
        rho, _ = spearmanr(v1, v2)
        if np.isfinite(rho):
            corrs.append(rho)
    
    return float(np.mean(corrs)) if len(corrs) >= 5 else np.nan


def compute_cka(X, Y, eps=1e-12):
    """
    Standard Linear CKA (linear-kernel HSIC with centered Gram matrices).
    X, Y: (n_samples, n_features)
    """
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


def procrustes_similarity(X, Y, center=True, scale=True):
    """
    Procrustes similarity.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    # Match sample size
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n], Y[:n]
    
    # Center (remove translation)
    if center:
        X = X - np.mean(X, axis=0, keepdims=True)
        Y = Y - np.mean(Y, axis=0, keepdims=True)
    
    # Handle dimension mismatch (pad with zeros)
    if X.shape[1] != Y.shape[1]:
        if X.shape[1] < Y.shape[1]:
            X = np.pad(X, ((0, 0), (0, Y.shape[1] - X.shape[1])), mode='constant')
        else:
            Y = np.pad(Y, ((0, 0), (0, X.shape[1] - Y.shape[1])), mode='constant')
    
    # Scale to unit norm (remove scaling)
    if scale:
        norm_X = np.linalg.norm(X, 'fro')
        norm_Y = np.linalg.norm(Y, 'fro')
        if norm_X > 1e-12:
            X = X / norm_X
        if norm_Y > 1e-12:
            Y = Y / norm_Y
    
    # SciPy's orthogonal_procrustes: finds R that minimizes ||X - Y@R||_F
    # Note: Takes Y then X (Y is transformed to match X)
    R, _ = orthogonal_procrustes(Y, X)
    
    # Apply rotation to align Y to X
    Y_aligned = Y @ R
    
    # Compute similarity
    distance_sq = np.sum((X - Y_aligned) ** 2)
    norm_X = np.linalg.norm(X, 'fro')
    norm_Y = np.linalg.norm(Y_aligned, 'fro')
    
    if norm_X < 1e-12 and norm_Y < 1e-12:
        return 1.0
    
    similarity = 1.0 - (distance_sq / (norm_X**2 + norm_Y**2 + 1e-12))
    return np.clip(similarity, 0.0, 1.0)


def compute_pwcka(X, Y, threshold=0.99, eps=1e-12):
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

    # PCA scores in sample space: U[:, :k] * S[:k]
    X_proj = Ux[:, :k] * Sx[:k]
    Y_proj = Uy[:, :k] * Sy[:k]

    # Correct CKA on projected reps
    return compute_cka(X_proj, Y_proj, eps=eps)


# =============================================================================
# ENCODER TRANSFORMATIONS
# =============================================================================

try:
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("sklearn required")


def build_encoder_transformations(X_base, seed):
    """Full encoder transformation suite."""
    X_base_native = np.asarray(X_base)
    rng = np.random.default_rng(seed)
    encoders = {}
    n_samples, n_features = X_base_native.shape

    # 1. PCA at various ranks
    for k in [5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 256, 300]:
        k_actual = min(k, n_samples - 1, n_features)
        if k_actual >= 5:
            try:
                pca = PCA(n_components=k_actual, random_state=seed)
                encoders[f"pca_{k:03d}"] = pca.fit_transform(X_base_native)
            except:
                pass

    # 2. Random Projections
    for k in [16, 32, 64, 128, 256]:
        k_actual = min(k, n_features)
        if k_actual >= 5:
            try:
                grp = GaussianRandomProjection(n_components=k_actual, random_state=seed)
                encoders[f"randproj_{k:03d}"] = grp.fit_transform(X_base_native)
            except:
                pass

    # 3. Top Variance Features
    try:
        vars = np.var(X_base_native, axis=0)
        for k in [50, 100, 200, 400, 800]:
            if k < n_features:
                idx = np.argsort(vars)[-k:]
                encoders[f"topvar_{k:03d}"] = X_base_native[:, idx]
    except:
        pass

    # 4. Random Feature Subsets
    for k in [50, 100, 200]:
        if k < n_features:
            idx = rng.choice(n_features, k, replace=False)
            encoders[f"randfeat_{k:03d}"] = X_base_native[:, idx]

    # 5. Noise Injection
    for noise_level in [0.05, 0.1, 0.25, 0.5, 1.0]:
        noise = rng.normal(0, noise_level * np.std(X_base_native), X_base_native.shape)
        encoders[f"noise_{int(noise_level*100):03d}"] = X_base_native + noise

    # 6. Controls
    encoders["original"] = X_base_native.copy()
    
    try:
        scaler = StandardScaler()
        encoders["zscore"] = scaler.fit_transform(X_base_native)
    except:
        pass

    try:
        norms = np.linalg.norm(X_base_native, axis=1, keepdims=True) + 1e-12
        encoders["l2norm"] = X_base_native / norms
    except:
        pass

    return encoders


def get_encoder_type(encoder_name):
    """Categorize encoder by type."""
    if encoder_name.startswith('pca_'):
        return 'PCA'
    elif encoder_name.startswith('randproj_'):
        return 'Random Projection'
    elif encoder_name.startswith('topvar_'):
        return 'Top Variance'
    elif encoder_name.startswith('randfeat_'):
        return 'Random Features'
    elif encoder_name.startswith('noise_'):
        return 'Noise Injection'
    elif encoder_name in ['original', 'log1p_full']:
        return 'Original'
    elif encoder_name in ['zscore', 'l2norm']:
        return 'Normalization'
    else:
        return 'Other'


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_language_models_and_data():
    """Load language data and compute base embeddings."""
    print("Loading SST-2 data...")
    ds = load_dataset("glue", "sst2", split="validation")
    texts = ds['sentence'][:CONFIG['language']['n_samples']]
    print(f"  Loaded {len(texts)} sentences")
    
    base_embeddings = {}
    
    # Model 1: MiniLM
    try:
        print("  Loading MiniLM...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        base_embeddings['minilm'] = emb
        print(f"    minilm: {emb.shape}")
        del model
    except Exception as e:
        print(f"    [ERROR] MiniLM: {e}")
    
    # Model 2: MPNet
    try:
        print("  Loading MPNet...")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        base_embeddings['mpnet'] = emb
        print(f"    mpnet: {emb.shape}")
        del model
    except Exception as e:
        print(f"    [ERROR] MPNet: {e}")
    
    # Model 3: DistilBERT
    try:
        print("  Loading DistilBERT...")
        model = SentenceTransformer("sentence-transformers/distilbert-base-nli-stsb-mean-tokens", device=DEVICE)
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        base_embeddings['distilbert'] = emb
        print(f"    distilbert: {emb.shape}")
        del model
    except Exception as e:
        print(f"    [ERROR] DistilBERT: {e}")
    
    # Model 4: RoBERTa
    try:
        print("  Loading RoBERTa...")
        model = SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1", device=DEVICE)
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        base_embeddings['roberta'] = emb
        print(f"    roberta: {emb.shape}")
        del model
    except Exception as e:
        print(f"    [ERROR] RoBERTa: {e}")
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return base_embeddings


def run_language_analysis():
    """Run language analysis comparing SHESHA vs CKA and PWCKA."""
    print("=" * 70)
    print("LANGUAGE ORTHOGONALITY TEST - SHESHA vs CKA vs PWCKA/Procrustes")
    print("=" * 70)
    
    # Load base embeddings
    base_embeddings = load_language_models_and_data()
    
    if not base_embeddings:
        print("[ERROR] No base embeddings loaded!")
        return None
    
    all_results = []
    
    for seed in tqdm(SEEDS, desc="Seeds"):
        for base_name, X_base in base_embeddings.items():
            encoders = build_encoder_transformations(X_base, seed)
            
            # Reference representations for similarity metrics
            refs = {
                "ref_original": encoders.get("original"),
                "ref_pca_100": encoders.get("pca_100", encoders.get("pca_075", encoders.get("original"))),
                "ref_zscore": encoders.get("zscore"),
            }
            
            for enc_name, X_enc in encoders.items():
                if X_enc is None:
                    continue
                
                X = np.nan_to_num(X_enc, nan=0.0)
                if X.shape[0] < 10 or X.shape[1] < 2:
                    continue
                if np.std(X) < 1e-9:
                    continue
                
                # Compute SHESHA
                shesha = compute_shesha_features(X, n_splits=30, random_state=seed)
                
                # Compute CKA against references
                cka_values = []
                pwcka_values = []
                procrustes_values = []
                
                for ref_name, ref_X in refs.items():
                    if ref_X is not None and ref_X.shape[0] == X.shape[0]:
                        ref_X = np.nan_to_num(ref_X, nan=0.0)
                        
                        # Standard CKA
                        cka = compute_cka(X, ref_X)
                        if np.isfinite(cka):
                            cka_values.append(cka)
                        
                        # PWCKA 
                        PWCKA (alpha=1.0)
                        pwcka = compute_pwcka(X, ref_X, alpha=1.0)
                        if np.isfinite(pwcka):
                            pwcka_values.append(pwcka)
                
                        # Procrustes
                        procrustes = procrustes_similarity(X, ref_X)
                        if np.isfinite(procrustes):
                            procrustes_values.append(procrustes)
                
                cka_avg = np.mean(cka_values) if cka_values else np.nan
                pwcka_avg = np.mean(pwcka_values) if pwcka_values else np.nan
                procrustes_avg = np.mean(procrustes_values) if procrustes_values else np.nan
                
                all_results.append({
                    'domain': 'Language',
                    'seed': seed,
                    'base_model': base_name,
                    'encoder': enc_name,
                    'encoder_type': get_encoder_type(enc_name),
                    'SHESHA': shesha,
                    'CKA': cka_avg,
                    'PWCKA': pwcka_avg,
                    'Procrustes': procrustes_avg,
                    'n_features': X.shape[1],
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv(OUTDIR / "language_robustness.csv", index=False)
    print(f"\nSaved {len(df)} raw results")
    
    # Aggregate by (base_model, encoder)
    df_agg = df.groupby(['domain', 'base_model', 'encoder', 'encoder_type']).agg({
        'SHESHA': 'mean',
        'CKA': 'mean',
        'PWCKA': 'mean',
        'Procrustes': 'mean',
        'n_features': 'first'
    }).reset_index()
    df_agg.to_csv(OUTDIR / "language_robustness_aggregated.csv", index=False)
    
    # =================================================================
    # ANALYSIS
    # =================================================================
    print("\n" + "=" * 70)
    print("RESULTS: SHESHA vs CKA vs PWCKA")
    print("=" * 70)
    
    valid = df_agg.dropna(subset=['SHESHA', 'CKA', 'PWCKA', 'Procrustes'])

    # Overall correlations
    rho_cka, p_cka = spearmanr(valid['SHESHA'], valid['CKA'])
    rho_pwcka, p_pwcka = spearmanr(valid['SHESHA'], valid['PWCKA'])
    rho_procrustes, p_procrostes = spearmanr(valid['SHESHA'], valid['Procrustes'])
    
    print(f"\nAggregate Correlations (N={len(valid)}):")
    print(f"  SHESHA vs CKA:      rho = {rho_cka:+.3f} (p={p_cka:.4f})")
    print(f"  SHESHA vs PWCKA:    rho = {rho_pwcka:+.3f} (p={p_pwcka:.4f})")
    print(f"  SHESHA vs Procrustes: rho = {rho_procrustes:+.3f} (p={p_procrostes:.4f})")
    
    rho_cka_pwcka, _ = spearmanr(valid['CKA'], valid['PWCKA'])
    rho_cka_procrustes, _ = spearmanr(valid['CKA'], valid['Procrustes'])

    print(f"  CKA vs PWCKA:    rho = {rho_cka_pwcka:+.3f}")
    print(f"  CKA vs Procrustes:    rho = {rho_cka_procrustes:+.3f}")

    
    # Per encoder type
    print(f"\n" + "-" * 70)
    print("Per Encoder Type:")
    print("-" * 70)
    
    for enc_type in sorted(valid['encoder_type'].unique()):
        sub = valid[valid['encoder_type'] == enc_type]
        if len(sub) < 5:
            continue
        
        rho_c, _ = spearmanr(sub['SHESHA'], sub['CKA'])
        rho_p, _ = spearmanr(sub['SHESHA'], sub['PWCKA'])
        rho_proc, _ = spearmanr(sub['SHESHA'], sub['Procrustes'])
        
        print(f"\n  {enc_type} (N={len(sub)}):")
        print(f"    vs CKA:      {rho_c:+.3f}")
        print(f"    vs PWCKA:    {rho_p:+.3f}")
        print(f"    vs Procustes: {rho_proc:+.3f}")
    
    # Descriptive stats
    print(f"\n" + "-" * 70)
    print("Descriptive Statistics:")
    print("-" * 70)
    print(f"\n  SHESHA:   M={valid['SHESHA'].mean():.3f}, SD={valid['SHESHA'].std():.3f}")
    print(f"  CKA:      M={valid['CKA'].mean():.3f}, SD={valid['CKA'].std():.3f}")
    print(f"  PWCKA:    M={valid['PWCKA'].mean():.3f}, SD={valid['PWCKA'].std():.3f}")
    print(f"  Procrustes: M={valid['Procrustes'].mean():.3f}, SD={valid['Procrustes'].std():.3f}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return df_agg


if __name__ == "__main__":
    run_language_analysis()