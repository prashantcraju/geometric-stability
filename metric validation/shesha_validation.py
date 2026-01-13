"""
Shesha Metric Validation
"""


import os
import sys

# =============================================================================
# 0) DETERMINISM GUARDS
# =============================================================================
SEED = 320
# Internal env vars for Colab/local compatibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)

import json
import random
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.manifold import trustworthiness
from sklearn.decomposition import PCA
import torch

# =============================================================================
# 1) CONFIGURATION
# =============================================================================

BATCH_SIZE = 128
RESULTS_DIR = Path("./shesha-validation")
EMBED_DIR = RESULTS_DIR / "embeds"
REPORT_DIR = RESULTS_DIR / "shesha_reports_csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# Torch Setup (Robust)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except: pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except: pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
MAX_N_FEATURE_SPLIT = 1600 
N_SPLITS = 30  

# Global Seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# 2) CORE METRICS
# =============================================================================

def compute_shesha_variance(X, y):
    classes = np.unique(y)
    if len(classes) < 2: return 0.0
    global_mean = np.mean(X, axis=0)
    X_centered = X - global_mean
    ss_total = np.sum(X_centered**2) + EPS
    ss_between = 0.0
    for c in classes:
        mask = (y == c)
        n_c = np.sum(mask)
        if n_c == 0: continue
        mean_c = np.mean(X[mask], axis=0)
        ss_between += n_c * np.sum((mean_c - global_mean)**2)
    return ss_between / ss_total

def compute_shesha_zscore(X, y):
    classes = np.unique(y)
    if len(classes) < 2: return 0.0
    scores = []
    centroids = {c: np.mean(X[y==c], axis=0) for c in classes}
    for c in classes:
        mask = (y == c)
        if np.sum(mask) < 2: continue
        X_c = X[mask]
        center_c = centroids[c]
        intra = np.mean(np.linalg.norm(X_c - center_c, axis=1))
        other_mask = ~mask
        if np.sum(other_mask) == 0: continue
        other_mean = np.mean(X[other_mask], axis=0)
        inter = np.mean(np.linalg.norm(X_c - other_mean, axis=1))
        scores.append((inter - intra) / (inter + intra + EPS))
    return np.mean(scores) if scores else 0.0

def compute_shesha_feature_split(X, n_splits=N_SPLITS, seed=SEED):
    n_samples, n_features = X.shape
    if n_features < 2 or n_samples < 4: return 0.0
    
    if n_samples > MAX_N_FEATURE_SPLIT:
        rng_sub = np.random.default_rng(seed)
        idx = rng_sub.choice(n_samples, MAX_N_FEATURE_SPLIT, replace=False)
        X = X[idx]

    correlations = []
    for i in range(n_splits):
        local_rng = np.random.default_rng(seed + i)
        feats = np.arange(n_features)
        local_rng.shuffle(feats)
        mid = n_features // 2
        f1, f2 = feats[:mid], feats[mid:]
        
        X1, X2 = X[:, f1], X[:, f2]
        # Explicit L2 Norm per half
        X1 = X1 / (np.linalg.norm(X1, axis=1, keepdims=True) + EPS)
        X2 = X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + EPS)
        
        # Cosine distance on unit vectors (equivalent to 1 - dot)
        # Handle Potential NaNs from zero-vectors
        rdm1 = pdist(X1, metric='cosine')
        rdm2 = pdist(X2, metric='cosine')
        rdm1 = np.nan_to_num(rdm1, nan=2.0) # 2.0 = max cosine dist (opposite)
        rdm2 = np.nan_to_num(rdm2, nan=2.0)
        
        rho, _ = spearmanr(rdm1, rdm2)
        # Handle NaN correlation (e.g. if RDM is constant)
        if np.isnan(rho): rho = 0.0
            
        correlations.append(rho)
        
    return np.mean(correlations)

def wrapper_shesha(X, y, variant, seed_override=None):
    if variant == "variance": 
        if y is None: raise ValueError("Variance requires labels")
        return compute_shesha_variance(X, y)
    if variant == "zscore": 
        if y is None: raise ValueError("Zscore requires labels")
        return compute_shesha_zscore(X, y)
    if variant == "feature_split": 
        s = seed_override if seed_override is not None else SEED
        return compute_shesha_feature_split(X, n_splits=N_SPLITS, seed=s)
    raise ValueError(f"Unknown variant: {variant}")

# =============================================================================
# 3) VALIDATION SUITE (NOW ITERATING ALL MODELS)
# =============================================================================

class SheshaValidator:
    def __init__(self, valid_models_list, labels_c10, labels_c100):
        self.models = valid_models_list 
        self.y_c10 = labels_c10
        self.y_c100 = labels_c100
        self.variants = ["variance", "zscore", "feature_split"]

        # Per-test RNGs
        self.rngs = {f"T{i}": np.random.default_rng(SEED + 1000 + i) for i in range(1, 12)}

    def _load(self, m, ds):
        path = EMBED_DIR / f"{ds}_{m}.npy"
        if not path.exists():
            raise FileNotFoundError(f"[FATAL] Missing: {path}")
        return np.load(path)

    def _get_labels(self, ds):
        if ds == "cifar10": return self.y_c10
        if ds == "cifar100": return self.y_c100
        raise ValueError(f"Unknown dataset: {ds}")

    # --- T1: Convergence ---
    def test_1_convergence(self):
        print("Running T1: Convergence (All Models)...")
        rng = self.rngs["T1"]
        res = []
        sample_sizes = [400, 1600]
        
        for m, ds in self.models:
            X = self._load(m, ds)
            y_full = self._get_labels(ds)
            
            for n in sample_sizes:
                idx = rng.choice(len(y_full), n, replace=False)
                X_sub = X[idx]
                y_sub = y_full[idx]
                
                for v in self.variants:
                    y_arg = None if v == "feature_split" else y_sub
                    if v == "feature_split" and n > MAX_N_FEATURE_SPLIT: continue
                    val = wrapper_shesha(X_sub, y_arg, v)
                    res.append({"Model": m, "Dataset": ds, "N": n, "Variant": v, "Score": val})
        return pd.DataFrame(res)

    # --- T2: Baselines ---
    def test_2_baselines(self):
        print("Running T2: Baselines (All Models)...")
        res = []
        for m, ds in self.models:
            y_full = self._get_labels(ds)
            X = self._load(m, ds)
            
            subset_size = min(2000, len(y_full))
            X_sub = X[:subset_size]
            y_sub = y_full[:subset_size]
            
            for v in self.variants:
                y_arg = None if v == "feature_split" else y_sub
                if v == "feature_split" and subset_size > MAX_N_FEATURE_SPLIT:
                    val = wrapper_shesha(X_sub[:MAX_N_FEATURE_SPLIT], None, v)
                else:
                    val = wrapper_shesha(X_sub, y_arg, v)
                res.append({"Model": m, "Dataset": ds, "Variant": v, "Score": val})
        return pd.DataFrame(res)

    # --- T3: Determinism ---
    def test_3_repeatability(self):
        print("Running T3: Determinism (All Models)...")
        res = []
        for m, ds in self.models:
            y = self._get_labels(ds)[:400]
            X = self._load(m, ds)[:400]
            
            for v in self.variants:
                y_arg = None if v == "feature_split" else y
                v1 = wrapper_shesha(X, y_arg, v)
                v2 = wrapper_shesha(X, y_arg, v)
                diff = abs(v1 - v2)
                res.append({
                    "Model": m, "Dataset": ds, "Variant": v, 
                    "Diff": diff, "Pass": diff < EPS
                })
        return pd.DataFrame(res)

    # --- T4: Validity ---
    def test_4_validity(self):
        print("Running T4: Validity (All Models)...")
        rng = self.rngs["T4"]
        res = []
        
        for m, ds in self.models:
            y_full = self._get_labels(ds)
            X = self._load(m, ds)
            idx = rng.choice(len(y_full), 1000, replace=False)
            X_sub = X[idx]
            y_sub = y_full[idx]
            
            for v in self.variants:
                y_arg = None if v == "feature_split" else y_sub
                val = wrapper_shesha(X_sub, y_arg, v)
                valid = not (np.isnan(val) or np.isinf(val))
                res.append({
                    "Model": m, "Dataset": ds, "Variant": v, 
                    "Valid": valid, "Score": val
                })
        return pd.DataFrame(res)

    # --- T5: Dimensionality ---
    def test_5_dimensionality(self):
        print("Running T5: Dimensionality (All Models)...")
        res = []
        
        for m, ds in self.models:
            y = self._get_labels(ds)[:400]
            X = self._load(m, ds)[:400]
            
            d = 64
            pca = PCA(n_components=d, random_state=SEED)
            X_low = pca.fit_transform(X)
            
            for v in self.variants:
                y_arg = None if v == "feature_split" else y
                val = wrapper_shesha(X_low, y_arg, v)
                res.append({
                    "Model": m, "Dataset": ds, "Dim": d, "Variant": v, "Score": val
                })
        return pd.DataFrame(res)

    # --- T6: Label Noise ---
    def test_6_label_noise(self):
        print("Running T6: Label Noise (All Models)...")
        rng = self.rngs["T6"]
        res = []
        
        for m, ds in self.models:
            y_true = self._get_labels(ds)[:500].copy()
            X = self._load(m, ds)[:500]
            
            for noise_p in [0.0, 1.0]:
                y_curr = y_true.copy()
                if noise_p > 0: rng.shuffle(y_curr)
                
                for v in ["variance", "zscore"]:
                    val = wrapper_shesha(X, y_curr, v)
                    res.append({
                        "Model": m, "Dataset": ds, 
                        "Noise": noise_p, "Variant": v, "Score": val
                    })
        return pd.DataFrame(res)

    # --- T7: Stratified ---
    def test_7_stratified(self):
        print("Running T7: Stratified (All Models)...")
        rng = self.rngs["T7"]
        res = []
        
        for m, ds in self.models:
            y = self._get_labels(ds)
            X = self._load(m, ds)
            
            idx = []
            uniq = np.unique(y)
            for c in uniq:
                cnt = 100 if c == uniq[0] else 5
                c_idx = np.where(y==c)[0]
                if len(c_idx) >= cnt:
                    idx.extend(rng.choice(c_idx, cnt, replace=False))
            
            X_imb = X[idx]
            y_imb = y[idx]
            
            for v in ["variance", "zscore"]:
                val = wrapper_shesha(X_imb, y_imb, v)
                res.append({
                    "Model": m, "Dataset": ds, 
                    "Condition": "Imbalanced", "Variant": v, "Score": val
                })
        return pd.DataFrame(res)

    # --- T8: Perturbations ---
    def test_8_perturbations(self):
        print("Running T8: Perturbations (All Models)...")
        rng = self.rngs["T8"]
        res = []
        
        for m, ds in self.models:
            y = self._get_labels(ds)[:400]
            X = self._load(m, ds)[:400]
            X_noisy = X + rng.normal(0, 0.1, X.shape)
            
            for v in self.variants:
                y_arg = None if v == "feature_split" else y
                v1 = wrapper_shesha(X, y_arg, v)
                v2 = wrapper_shesha(X_noisy, y_arg, v)
                res.append({
                    "Model": m, "Dataset": ds, "Variant": v, 
                    "Clean": v1, "Drift": abs(v1-v2)
                })
        return pd.DataFrame(res)

    # --- T9: Stability ---
    def test_9_stability(self):
        print("Running T9: FS Stability (All Models)...")
        res = []
        for m, ds in self.models:
            X = self._load(m, ds)[:MAX_N_FEATURE_SPLIT]
            val_a = wrapper_shesha(X, None, "feature_split", seed_override=100)
            val_b = wrapper_shesha(X, None, "feature_split", seed_override=200)
            diff = abs(val_a - val_b)
            res.append({
                "Model": m, "Dataset": ds, 
                "Score_SeedA": val_a, "Diff": diff, "Stable": diff < 0.05
            })
        return pd.DataFrame(res)

    # --- T10: Sanity ---
    def test_10_sanity(self):
        print("Running T10: Sanity (Random Data)...")
        rng = self.rngs["T10"]
        res = []
        X = rng.standard_normal((500, 128))
        y = rng.integers(0, 10, 500)
        for v in self.variants:
            y_arg = None if v == "feature_split" else y
            val = wrapper_shesha(X, y_arg, v)
            res.append({"Variant": v, "Score": val, "Expected": "Low"})
        return pd.DataFrame(res)


# =============================================================================
# 4) MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print(f"Starting FULL Shesha Validation (Seed={SEED})...")

    # Load Labels
    p_c10 = RESULTS_DIR / "cifar10_labels.npy"
    p_c100 = RESULTS_DIR / "cifar100_labels.npy"
    if not p_c10.exists(): print("Missing C10 labels"); sys.exit(1)
    
    y_c10 = np.load(p_c10)
    y_c100 = np.load(p_c100) if p_c100.exists() else np.array([])
    
    # Identify Models
    found_models = []
    for f in EMBED_DIR.glob("cifar10_*.npy"):
        found_models.append((f.stem.replace("cifar10_", ""), "cifar10"))
    for f in EMBED_DIR.glob("cifar100_*.npy"):
        if len(y_c100) > 0:
            found_models.append((f.stem.replace("cifar100_", ""), "cifar100"))
            
    found_models.sort(key=lambda x: (x[1], x[0]))
    
    if not found_models:
        print("[FATAL] No embedding files found.")
        sys.exit(1)
        
    print(f"Found {len(found_models)} models. Running ALL tests on ALL models.")

    validator = SheshaValidator(found_models, y_c10, y_c100)
    
    test_funcs = [
        ("T1_Convergence", validator.test_1_convergence),
        ("T2_Baselines", validator.test_2_baselines),
        ("T3_Determinism", validator.test_3_repeatability),
        ("T4_Validity", validator.test_4_validity),
        ("T5_Dimensionality", validator.test_5_dimensionality),
        ("T6_LabelNoise", validator.test_6_label_noise),
        ("T7_Stratified", validator.test_7_stratified),
        ("T8_Perturbations", validator.test_8_perturbations),
        ("T9_Stability", validator.test_9_stability),
        ("T10_Sanity", validator.test_10_sanity),
    ]

    print(f"Saving reports to: {REPORT_DIR}")
    for name, func in test_funcs:
        print(f"--- {name} ---")
        try:
            df = func()
            if not df.empty:
                df.to_csv(REPORT_DIR / f"{name}.csv", index=False)
                print(f"  -> Saved {len(df)} rows")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            
    print("\nDONE.")