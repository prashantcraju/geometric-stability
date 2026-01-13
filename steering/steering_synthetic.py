"""
Shesha Steering Analysis - Synthetic Data
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import warnings
import random
import gc
from transformers import AutoConfig

warnings.filterwarnings("ignore")

# =============================================================================
# GPU OPTIMIZATIONS
# =============================================================================

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cap = torch.cuda.get_device_capability()
    DTYPE = torch.bfloat16 if cap[0] >= 8 else torch.float16
    DEVICE = "cuda"
    BATCH_SIZE = 128
else:
    DTYPE = torch.float32
    DEVICE = "cpu"
    BATCH_SIZE = 32

print(f"[Config] Device: {DEVICE}, Dtype: {DTYPE}, Batch Size: {BATCH_SIZE}")

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]
N_STEERING_SPLITS = 5  # Multiple steering splits for stability
N_RANDOM_DIRS = 20  # Random directions per split for control baseline

OUTPUT_DIR = Path("./shesha-steering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("./shesha-steering/cache_synthetic")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL REGISTRY (same as before)
# =============================================================================
MODEL_REGISTRY = []

# Small Models
for m in ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "paraphrase-MiniLM-L3-v2",
          "paraphrase-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2",
          "multi-qa-MiniLM-L6-cos-v1", "msmarco-MiniLM-L6-cos-v5",
          "msmarco-distilbert-base-v4", "msmarco-distilbert-base-v3",
          "distilbert-base-nli-stsb-mean-tokens",
          "paraphrase-TinyBERT-L6-v2", "paraphrase-albert-small-v2",
          "nli-distilroberta-base-v2", "stsb-distilbert-base",
          "all-distilroberta-v1"]:
    MODEL_REGISTRY.append({"id": f"sentence-transformers/{m}", "group": "Small"})

for m in ["intfloat/e5-small-v2", "intfloat/e5-small",
          "BAAI/bge-small-en-v1.5", "thenlper/gte-small"]:
    MODEL_REGISTRY.append({"id": m, "group": "Small"})

# Base Models
for m in ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "paraphrase-mpnet-base-v2",
          "bert-base-nli-mean-tokens", "bert-base-nli-max-tokens", "bert-base-nli-cls-token",
          "bert-base-nli-stsb-mean-tokens", "roberta-base-nli-mean-tokens",
          "stsb-roberta-base", "stsb-bert-base", "nli-bert-base", "nli-roberta-base",
          "nli-mpnet-base-v2", "paraphrase-distilroberta-base-v1",
          "quora-distilbert-base", "gtr-t5-base", "sentence-t5-base"]:
    MODEL_REGISTRY.append({"id": f"sentence-transformers/{m}", "group": "Base"})

for m in ["BAAI/bge-base-en-v1.5", "thenlper/gte-base", "intfloat/e5-base-v2",
          "intfloat/e5-base", "jinaai/jina-embeddings-v2-base-en"]:
    MODEL_REGISTRY.append({"id": m, "group": "Base"})

# Large Models
for m in ["bert-large-nli-mean-tokens", "bert-large-nli-max-tokens",
          "bert-large-nli-cls-token", "bert-large-nli-stsb-mean-tokens",
          "roberta-large-nli-mean-tokens", "roberta-large-nli-stsb-mean-tokens",
          "stsb-roberta-large", "stsb-bert-large", "nli-roberta-large",
          "gtr-t5-large", "sentence-t5-large"]:
    MODEL_REGISTRY.append({"id": f"sentence-transformers/{m}", "group": "Large"})

for m in ["BAAI/bge-large-en-v1.5", "thenlper/gte-large", "intfloat/e5-large-v2",
          "intfloat/e5-large", "WhereIsAI/UAE-Large-V1",
          "mixedbread-ai/mxbai-embed-large-v1"]:
    MODEL_REGISTRY.append({"id": m, "group": "Large"})

# Unsupervised/Pretrained
for m in ["princeton-nlp/unsup-simcse-roberta-large", "princeton-nlp/unsup-simcse-roberta-base",
          "princeton-nlp/unsup-simcse-bert-large-uncased", "princeton-nlp/unsup-simcse-bert-base-uncased",
          "facebook/contriever", "roberta-large", "roberta-base",
          "bert-large-uncased", "bert-base-uncased", "microsoft/deberta-v3-large",
          "microsoft/deberta-v3-base"]:
    MODEL_REGISTRY.append({"id": m, "group": "Unsupervised"})

print(f"Total models: {len(MODEL_REGISTRY)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clear_gpu_memory():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_cache_key(model_name, seed):
    model_key = model_name.replace("/", "_").replace("-", "_")
    return f"steer_{model_key}_seed_{seed}.npz"

def get_cached_embeddings(model_name, seed):
    cache_key = get_cache_key(model_name, seed)
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            return data['E_train'], data['E_test'], True
        except:
            return None, None, False
    return None, None, False

def save_embeddings_to_cache(model_name, seed, E_train, E_test):
    cache_key = get_cache_key(model_name, seed)
    cache_path = CACHE_DIR / cache_key
    np.savez_compressed(cache_path, E_train=E_train, E_test=E_test)

def encode_with_cache(model_name, seed, X_train, X_test):
    E_train, E_test, hit = get_cached_embeddings(model_name, seed)

    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # NOTE: This is a rough proxy for model size, not actual parameter count.
        # Use only as approximate size signal, not exact param count.
        params = cfg.num_hidden_layers * (cfg.hidden_size ** 2) * 12
    except:
        params = np.nan

    if hit:
        return E_train, E_test, True, params

    print(f"   Computing embeddings for {model_name}...")
    try:
        model = SentenceTransformer(model_name, device=DEVICE)

        if DEVICE == "cuda":
            model = model.to(dtype=DTYPE)

        with torch.inference_mode():
            # FIXED: Only use autocast on CUDA, not CPU
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=DTYPE):
                    E_train = model.encode(X_train.tolist(), normalize_embeddings=True,
                                           show_progress_bar=False, batch_size=BATCH_SIZE)
                    E_test = model.encode(X_test.tolist(), normalize_embeddings=True,
                                          show_progress_bar=False, batch_size=BATCH_SIZE)
            else:
                E_train = model.encode(X_train.tolist(), normalize_embeddings=True,
                                       show_progress_bar=False, batch_size=BATCH_SIZE)
                E_test = model.encode(X_test.tolist(), normalize_embeddings=True,
                                      show_progress_bar=False, batch_size=BATCH_SIZE)

        del model
        clear_gpu_memory()

        save_embeddings_to_cache(model_name, seed, E_train, E_test)
        return E_train, E_test, False, params

    except Exception as e:
        print(f"   [Error embedding {model_name}]: {e}")
        clear_gpu_memory()
        return None, None, False, params

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_diverse_data(n_samples=1000, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adj_pos = ["adequate", "fine", "good", "decent", "solid", "excellent", "superb", "exceptional"]
    adj_neg = ["poor", "bad", "mediocre", "lacking", "subpar", "terrible", "awful", "dreadful"]
    nouns = ["aspect", "element", "part", "feature", "component", "unit", "item", "factor"]
    contexts = ["in my opinion", "overall", "considering everything", "to be honest"]

    X, y = [], []
    for _ in range(n_samples // 2):
        s = f"{random.choice(contexts)}, the {random.choice(nouns)} was {random.choice(adj_pos)}"
        X.append(s)
        y.append(1)
        s = f"{random.choice(contexts)}, the {random.choice(nouns)} was {random.choice(adj_neg)}"
        X.append(s)
        y.append(0)
    return np.array(X), np.array(y)

# =============================================================================
# METRICS IMPLEMENTATION
# =============================================================================

def compute_fisher_criterion(X, y):
    """Fisher criterion - returns np.nan on failure."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    unique_labels = np.unique(y)

    if len(unique_labels) < 2:
        return np.nan

    global_mean = X.mean(axis=0)
    S_b = np.zeros((X.shape[1], X.shape[1]))
    S_w = np.zeros((X.shape[1], X.shape[1]))

    for label in unique_labels:
        mask = y == label
        n_k = mask.sum()
        mean_k = X[mask].mean(axis=0)
        diff = (mean_k - global_mean).reshape(-1, 1)
        S_b += n_k * (diff @ diff.T)
        centered = X[mask] - mean_k
        S_w += centered.T @ centered

    S_w += 1e-6 * np.eye(S_w.shape[0])
    result = float(np.trace(S_b) / (np.trace(S_w) + 1e-10))
    return result if np.isfinite(result) else np.nan


def compute_silhouette(X, y, sample_size=1000, seed=320):
    """
    Silhouette score with local RandomState for reproducibility.
    """
    if len(np.unique(y)) < 2:
        return np.nan

    rng = np.random.RandomState(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
        X, y = X[idx], y[idx]

    try:
        result = float(silhouette_score(X, y, metric='cosine'))
        return result if np.isfinite(result) else np.nan
    except:
        return np.nan


def compute_anisotropy_pca(X):
    """
    PCA-based anisotropy measure.
    Returns fraction of variance explained by first PC.
    Returns np.nan on failure.
    """
    X = np.asarray(X, dtype=np.float64)
    centered = X - X.mean(axis=0)

    if np.allclose(centered, 0):
        return np.nan

    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            return np.nan
        result = float((S**2)[0] / total_var)
        return result if np.isfinite(result) else np.nan
    except Exception as e:
        print(f"   [Anisotropy SVD failed]: {e}")
        return np.nan


def compute_wuc_corrected(X, y, shrinkage=0.3, min_samples_per_class=5, seed=320):
    """
    Whitened Unbiased Cosine (WUC).
    1. LedoitWolf with assume_centered=False (safer default)
    2. Eigendecomposition for inverse square root (stable, deterministic)
    3. Local RandomState for reproducibility
    4. Returns np.nan on failure
    5. Scale-aware minimum residual count
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    rng = np.random.RandomState(seed)

    if n_classes < 2:
        return np.nan

    # Check minimum samples per class
    for label in unique_labels:
        if np.sum(y == label) < min_samples_per_class:
            return np.nan

    n = len(y)
    idx = rng.permutation(n)
    half = n // 2
    idx1, idx2 = idx[:half], idx[half:2*half]

    def get_means(emb, lab, indices):
        means = []
        valid = True
        for label in unique_labels:
            mask = lab[indices] == label
            if mask.sum() < 2:
                valid = False
                break
            means.append(emb[indices][mask].mean(axis=0))
        return np.array(means) if valid else None

    means1 = get_means(X, y, idx1)
    means2 = get_means(X, y, idx2)

    if means1 is None or means2 is None:
        return np.nan

    # Collect residuals
    residuals = []
    for i, label in enumerate(unique_labels):
        mask1 = y[idx1] == label
        mask2 = y[idx2] == label
        if mask1.sum() > 0:
            residuals.append(X[idx1][mask1] - means1[i])
        if mask2.sum() > 0:
            residuals.append(X[idx2][mask2] - means2[i])

    if not residuals:
        return np.nan

    residuals = np.vstack(residuals)

    # FIXED: Scale-aware minimum residual count
    # LedoitWolf handles n < d with shrinkage, but need enough samples
    min_residuals = max(50, 5 * n_classes)
    if len(residuals) < min_residuals:
        return np.nan

    try:
        # FIXED: assume_centered=False is the safer default
        lw = LedoitWolf(assume_centered=False)
        lw.fit(residuals)
        cov = lw.covariance_

        # Additional shrinkage toward identity
        trace_cov = np.trace(cov)
        if trace_cov < 1e-10:
            return np.nan
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) * trace_cov / cov.shape[0]

        # FIXED: Eigendecomposition for stable inverse square root
        cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, 1e-10)
        cov_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    except Exception as e:
        print(f"   [WUC covariance failed]: {e}")
        return np.nan

    # Whiten condition means
    m1_w = means1 @ cov_inv_sqrt
    m2_w = means2 @ cov_inv_sqrt

    # Compute RDMs and correlate
    rdm1 = pdist(m1_w, metric='cosine')
    rdm2 = pdist(m2_w, metric='cosine')

    if len(rdm1) < 2:
        return np.nan

    rho, _ = stats.spearmanr(rdm1, rdm2)
    return float(rho) if np.isfinite(rho) else np.nan


def compute_shesha_supervised(X, y, max_samples=300, seed=320):
    """Supervised Shesha - returns np.nan on failure."""
    rng = np.random.RandomState(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    X_centered = X - np.mean(X, axis=0)
    model_dists = pdist(X_centered, metric='correlation')
    ideal_dists = pdist(y.reshape(-1, 1), metric='hamming')

    rho, _ = stats.spearmanr(model_dists, ideal_dists)
    return float(rho) if np.isfinite(rho) else np.nan


def compute_shesha_unsupervised_dim(X, n_splits=50, seed=320):
    """
    Unsupervised Shesha (dimension split).
    This is valid because it keeps the same items and splits features.
    Returns np.nan on failure.
    """
    d = X.shape[1]
    if d < 4:
        return np.nan

    corrs = []
    for i in range(n_splits):
        rng = np.random.RandomState(seed + i)
        perm = rng.permutation(d)
        half = d // 2
        dims1, dims2 = perm[:half], perm[half:2*half]

        rdm1 = pdist(X[:, dims1], 'correlation')
        rdm2 = pdist(X[:, dims2], 'correlation')

        rho, _ = stats.spearmanr(rdm1, rdm2)
        if np.isfinite(rho):
            corrs.append(rho)

    if not corrs:
        return np.nan
    return float(np.mean(corrs))


def compute_procrustes(X, y):
    """
    Procrustes alignment to ideal behavior matrix.
    
    Measures how well embeddings align with concept directions.
    Higher scores indicate better conditions for steering vectors.
    
    Parameters:
    -----------
    X : array, shape (n_samples, embedding_dim)
        Model embeddings or activations
    y : array, shape (n_samples,)
        Concept labels
        
    Returns:
    --------
    alignment : float in [0, 1]
        Alignment score where:
        - 0.9-1.0: Excellent for steering
        - 0.8-0.9: Very good for steering
        - 0.7-0.8: Good for steering
        - 0.6-0.7: Moderate for steering
        - 0.5-0.6: Fair for steering
        - <0.5: Poor for steering
    """
    import numpy as np
    
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    # Remap labels to 0..C-1
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[label] for label in y])

    n_classes = len(unique_labels)
    n_samples = len(y)

    # Create ideal representation (one-hot)
    ideal = np.zeros((n_samples, n_classes))
    for i, l in enumerate(y_remapped):
        ideal[i, l] = 1.0

    # Match dimensions
    if X.shape[1] > n_classes:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        X = U[:, :n_classes] * S[:n_classes]
    elif X.shape[1] < n_classes:
        X = np.hstack([X, np.zeros((n_samples, n_classes - X.shape[1]))])

    # Frobenius normalize both
    norm_X = np.linalg.norm(X, 'fro')
    norm_ideal = np.linalg.norm(ideal, 'fro')

    if norm_X < 1e-10 or norm_ideal < 1e-10:
        return np.nan

    X_norm = X / norm_X
    ideal_norm = ideal / norm_ideal

    try:
        # Compute SVD of cross-covariance matrix
        M = X_norm.T @ ideal_norm
        U, S, Vt = np.linalg.svd(M)

        # Standard bounded Procrustes similarity
        similarity = float(np.clip(np.sum(S), 0.0, 1.0))

        return similarity if np.isfinite(similarity) else np.nan

    except Exception:
        return np.nan


# =============================================================================
# STEERING EVALUATION WITH CONTROLS
# =============================================================================

def evaluate_steering_corrected(E_train, y_train, E_test, y_test,
                                n_fewshot=50, n_splits=5, n_random_per_split=20, seed=320):
    """
    Steering evaluation with multiple splits and random direction control.

    Returns:
        dict with mean/std across splits for:
        - acc_0: baseline accuracy
        - max_drop: max accuracy drop with true steering direction
        - max_drop_random: max accuracy drop with random direction (control)
        - min_acc: minimum accuracy achieved
        - class_flip_rate: fraction of predictions that changed
        - asymmetry: difference between +alpha and -alpha accuracy
    """
    alphas = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    n_classes = len(np.unique(y_train))

    all_acc_0 = []
    all_max_drop = []
    all_max_drop_random = []
    all_min_acc = []
    all_acc_neg_alpha = []
    all_acc_pos_alpha = []
    all_flip_rate_pos = []
    all_flip_rate_neg = []
    all_pred_pos_rate_delta = []

    for split_idx in range(n_splits):
        rng = np.random.RandomState(seed + split_idx * 1000)

        # Few-shot probe training (stratified)
        if len(E_train) > n_fewshot:
            fs_idx = []
            for label in np.unique(y_train):
                label_idx = np.where(y_train == label)[0]
                # FIXED: proper stratification for any number of classes
                n_per = max(2, n_fewshot // n_classes)
                sampled = rng.choice(label_idx, min(n_per, len(label_idx)), replace=False)
                fs_idx.extend(sampled)
            fs_idx = np.array(fs_idx)
        else:
            fs_idx = np.arange(len(E_train))

        E_fs, y_fs = E_train[fs_idx], y_train[fs_idx]

        # Train probe (use appropriate solver for n_classes)
        solver = 'liblinear' if n_classes == 2 else 'lbfgs'
        clf = LogisticRegression(C=1.0, solver=solver, random_state=seed, max_iter=1000)
        clf.fit(E_fs, y_fs)

        # Get TRUE steering direction (from classifier weights)
        if n_classes == 2:
            w = clf.coef_[0]
        else:
            # FUTURE-PROOFED: For multiclass, use top singular vector of coef_
            # This is the principal direction in weight space, avoids
            # mean(coef_) which can cancel out and point nowhere
            U, S, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
            w = Vt[0]  # Top right singular vector (in embedding space)
        w_hat = w / (np.linalg.norm(w) + 1e-12)

        # Baseline accuracy on TEST set
        preds_baseline = clf.predict(E_test)
        acc_0 = accuracy_score(y_test, preds_baseline)

        # Baseline predicted positive rate (for binary, class 1)
        pred_pos_rate_baseline = np.mean(preds_baseline == 1) if n_classes == 2 else np.nan

        # Steering sweep with TRUE direction
        accuracies_true = []
        for alpha in alphas:
            E_steered = E_test + alpha * w_hat
            norms = np.linalg.norm(E_steered, axis=1, keepdims=True)
            E_steered = E_steered / np.maximum(norms, 1e-12)
            acc = accuracy_score(y_test, clf.predict(E_steered))
            accuracies_true.append(acc)

        # FIXED: Compute flip rate at BOTH extreme alphas
        # Positive direction (+2.0)
        E_steered_pos = E_test + 2.0 * w_hat
        norms = np.linalg.norm(E_steered_pos, axis=1, keepdims=True)
        E_steered_pos = E_steered_pos / np.maximum(norms, 1e-12)
        preds_pos = clf.predict(E_steered_pos)
        flip_rate_pos = np.mean(preds_pos != preds_baseline)

        # Negative direction (-2.0)
        E_steered_neg = E_test - 2.0 * w_hat
        norms = np.linalg.norm(E_steered_neg, axis=1, keepdims=True)
        E_steered_neg = E_steered_neg / np.maximum(norms, 1e-12)
        preds_neg = clf.predict(E_steered_neg)
        flip_rate_neg = np.mean(preds_neg != preds_baseline)

        # Delta in predicted positive rate between +2 and -2 (directional control)
        if n_classes == 2:
            pred_pos_rate_pos = np.mean(preds_pos == 1)
            pred_pos_rate_neg = np.mean(preds_neg == 1)
            pred_pos_rate_delta = pred_pos_rate_pos - pred_pos_rate_neg
        else:
            pred_pos_rate_delta = np.nan

        # FIXED: Average over multiple random directions for robust control
        random_drops = []
        for _ in range(n_random_per_split):
            w_random = rng.randn(len(w))
            w_random_hat = w_random / (np.linalg.norm(w_random) + 1e-12)

            accuracies_rand = []
            for alpha in alphas:
                E_steered = E_test + alpha * w_random_hat
                norms = np.linalg.norm(E_steered, axis=1, keepdims=True)
                E_steered = E_steered / np.maximum(norms, 1e-12)
                acc = accuracy_score(y_test, clf.predict(E_steered))
                accuracies_rand.append(acc)
            random_drops.append(acc_0 - min(accuracies_rand))

        max_drop = acc_0 - min(accuracies_true)
        max_drop_random = np.mean(random_drops)
        min_acc = min(accuracies_true)

        all_acc_0.append(acc_0)
        all_max_drop.append(max_drop)
        all_max_drop_random.append(max_drop_random)
        all_min_acc.append(min_acc)
        all_acc_neg_alpha.append(accuracies_true[0])   # alpha = -2.0
        all_acc_pos_alpha.append(accuracies_true[-1])  # alpha = +2.0
        all_flip_rate_pos.append(flip_rate_pos)
        all_flip_rate_neg.append(flip_rate_neg)
        all_pred_pos_rate_delta.append(pred_pos_rate_delta)

    return {
        'acc_0_mean': np.mean(all_acc_0),
        'acc_0_std': np.std(all_acc_0),
        'max_drop_mean': np.mean(all_max_drop),
        'max_drop_std': np.std(all_max_drop),
        'max_drop_random_mean': np.mean(all_max_drop_random),
        'max_drop_random_std': np.std(all_max_drop_random),
        'min_acc_mean': np.mean(all_min_acc),
        'min_acc_std': np.std(all_min_acc),
        'acc_neg_alpha_mean': np.mean(all_acc_neg_alpha),
        'acc_pos_alpha_mean': np.mean(all_acc_pos_alpha),
        'asymmetry': np.mean(all_acc_pos_alpha) - np.mean(all_acc_neg_alpha),
        'flip_rate_pos_mean': np.mean(all_flip_rate_pos),
        'flip_rate_neg_mean': np.mean(all_flip_rate_neg),
        'flip_rate_pos_std': np.std(all_flip_rate_pos),
        'flip_rate_neg_std': np.std(all_flip_rate_neg),
        'pred_pos_rate_delta_mean': np.nanmean(all_pred_pos_rate_delta),
        'pred_pos_rate_delta_std': np.nanstd(all_pred_pos_rate_delta),
    }


def partial_correlation(x, y, covariates):
    """Partial correlation controlling for covariates."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    valid = ~(np.isnan(x) | np.isnan(y) | np.any(np.isnan(covariates), axis=1))
    x, y, cov = x[valid], y[valid], covariates[valid]

    if len(x) < 5:
        return np.nan, np.nan

    cov_int = np.hstack([np.ones((len(x), 1)), cov])

    try:
        beta_x = np.linalg.lstsq(cov_int, x, rcond=None)[0]
        x_resid = x - cov_int @ beta_x
        beta_y = np.linalg.lstsq(cov_int, y, rcond=None)[0]
        y_resid = y - cov_int @ beta_y
        rho, p = stats.spearmanr(x_resid, y_resid)
        return rho, p
    except:
        return np.nan, np.nan


# =============================================================================
# EXPERIMENT RUNNER WITH NEGATIVE CONTROLS
# =============================================================================

def run_experiment_for_seed(seed):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate Synthetic Data
    X_all, y_all = generate_diverse_data(1000, seed=seed)

    # Split: 700 for metrics, 300 for steering
    X_metrics, y_metrics = X_all[:700], y_all[:700]
    X_steer, y_steer = X_all[700:], y_all[700:]

    # Create shuffled labels for negative control
    rng = np.random.RandomState(seed)
    y_metrics_shuffled = rng.permutation(y_metrics)

    results = []

    for i, m in enumerate(MODEL_REGISTRY):
        try:
            # Get embeddings
            E_full_train, E_full_test, hit, params = encode_with_cache(
                m['id'], seed, X_all[:700], X_all[700:]
            )

            if E_full_train is None:
                continue

            E_metrics = E_full_train
            E_steer_all = E_full_test

            # Further split steering data into train/test
            steer_train_idx, steer_test_idx = train_test_split(
                np.arange(len(y_steer)),
                test_size=0.5,
                random_state=seed,
                stratify=y_steer
            )

            E_steer_train = E_steer_all[steer_train_idx]
            y_steer_train = y_steer[steer_train_idx]
            E_steer_test = E_steer_all[steer_test_idx]
            y_steer_test = y_steer[steer_test_idx]

            # --- METRICS (on metrics set with TRUE labels) ---
            shesha_sup = compute_shesha_supervised(E_metrics, y_metrics, seed=seed)
            shesha_unsup_dim = compute_shesha_unsupervised_dim(E_metrics, seed=seed)
            fisher = compute_fisher_criterion(E_metrics, y_metrics)
            sil = compute_silhouette(E_metrics, y_metrics, seed=seed)
            pca_aniso = compute_anisotropy_pca(E_metrics)
            wuc = compute_wuc_corrected(E_metrics, y_metrics, seed=seed)
            proc_behav = compute_procrustes(E_metrics, y_metrics)

            # --- NEGATIVE CONTROL: Metrics with SHUFFLED labels ---
            shesha_sup_shuffled = compute_shesha_supervised(E_metrics, y_metrics_shuffled, seed=seed)
            fisher_shuffled = compute_fisher_criterion(E_metrics, y_metrics_shuffled)
            proc_behav_shuffled = compute_procrustes(E_metrics, y_metrics_shuffled)

            # --- STEERING (with multiple splits and random direction control) ---
            steering = evaluate_steering_corrected(
                E_steer_train, y_steer_train,
                E_steer_test, y_steer_test,
                n_fewshot=50,
                n_splits=N_STEERING_SPLITS,
                n_random_per_split=N_RANDOM_DIRS,
                seed=seed
            )

            wuc_str = f"{wuc:.3f}" if not np.isnan(wuc) else "NaN"
            print(f"[{i+1}/{len(MODEL_REGISTRY)}] {m['id'].split('/')[-1]:<30} | "
                  f"Sh={shesha_sup:.3f} | WUC={wuc_str} | "
                  f"Drop={steering['max_drop_mean']:.3f}+/-{steering['max_drop_std']:.3f} | "
                  f"RandDrop={steering['max_drop_random_mean']:.3f}")

            results.append({
                "seed": seed,
                "model": m['id'],
                "group": m['group'],
                "params": params,
                # True label metrics
                "shesha_supervised": shesha_sup,
                "shesha_unsup_dim": shesha_unsup_dim,
                "fisher_criterion": fisher,
                "silhouette": sil,
                "anisotropy_pca": pca_aniso,
                "wuc": wuc,
                "procrustes_to_behavior": proc_behav,
                # Shuffled label controls
                "shesha_supervised_shuffled": shesha_sup_shuffled,
                "fisher_criterion_shuffled": fisher_shuffled,
                "procrustes_shuffled": proc_behav_shuffled,
                # Steering metrics (mean across splits)
                "steering_acc_0": steering['acc_0_mean'],
                "steering_acc_0_std": steering['acc_0_std'],
                "max_drop": steering['max_drop_mean'],
                "max_drop_std": steering['max_drop_std'],
                "max_drop_random": steering['max_drop_random_mean'],
                "max_drop_random_std": steering['max_drop_random_std'],
                "min_acc": steering['min_acc_mean'],
                "min_acc_std": steering['min_acc_std'],
                # Controllability metrics
                "acc_neg_alpha": steering['acc_neg_alpha_mean'],
                "acc_pos_alpha": steering['acc_pos_alpha_mean'],
                "asymmetry": steering['asymmetry'],
                "flip_rate_pos": steering['flip_rate_pos_mean'],
                "flip_rate_neg": steering['flip_rate_neg_mean'],
                "flip_rate_pos_std": steering['flip_rate_pos_std'],
                "flip_rate_neg_std": steering['flip_rate_neg_std'],
                "pred_pos_rate_delta": steering['pred_pos_rate_delta_mean'],
                "pred_pos_rate_delta_std": steering['pred_pos_rate_delta_std'],
            })

        except Exception as e:
            print(f"Error processing {m['id']}: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    all_res = []

    for seed in SEEDS:
        clear_gpu_memory()
        df = run_experiment_for_seed(seed)
        all_res.append(df)

        # Save intermediate
        df.to_csv(f"{OUTPUT_DIR}/synthetic_results_seed_{seed}.csv", index=False)

    df_final = pd.concat(all_res, ignore_index=True)
    df_final.to_csv(f"{OUTPUT_DIR}/synthetic_all_results.csv", index=False)

    # ==========================================================================
    # ANALYSIS SUMMARY
    # ==========================================================================
    print("\n" + "="*70 + "\nSUMMARY ANALYSIS\n" + "="*70)

    # Aggregate by model (mean across seeds)
    agg_cols = ['shesha_supervised', 'shesha_unsup_dim', 'fisher_criterion',
                'silhouette', 'anisotropy_pca', 'wuc', 'procrustes_to_behavior',
                'shesha_supervised_shuffled', 'fisher_criterion_shuffled', 'procrustes_shuffled',
                'max_drop', 'max_drop_random', 'min_acc',
                'acc_neg_alpha', 'acc_pos_alpha', 'asymmetry',
                'flip_rate_pos', 'flip_rate_neg', 'pred_pos_rate_delta']

    df_agg = df_final.groupby('model').agg({col: 'mean' for col in agg_cols}).reset_index()

    # --------------------------------------------------------------------------
    # NEGATIVE CONTROL CHECK: Shuffled labels should break supervised metrics
    # --------------------------------------------------------------------------
    print("\n--- NEGATIVE CONTROL: True vs Shuffled Labels ---")
    print("(Supervised metrics should drop significantly with shuffled labels)\n")

    for metric_pair in [('shesha_supervised', 'shesha_supervised_shuffled'),
                        ('fisher_criterion', 'fisher_criterion_shuffled'),
                        ('procrustes_to_behavior', 'procrustes_shuffled')]:
        true_col, shuf_col = metric_pair
        valid = df_agg[true_col].notna() & df_agg[shuf_col].notna()
        if valid.sum() >= 5:
            true_mean = df_agg.loc[valid, true_col].mean()
            shuf_mean = df_agg.loc[valid, shuf_col].mean()
            t_stat, p_val = stats.ttest_rel(df_agg.loc[valid, true_col],
                                            df_agg.loc[valid, shuf_col])
            print(f"{true_col:<25}: True={true_mean:.3f}, Shuffled={shuf_mean:.3f}, "
                  f"t={t_stat:.2f}, p={p_val:.4f}")

    # --------------------------------------------------------------------------
    # NEGATIVE CONTROL CHECK: Random direction should have less steering effect
    # --------------------------------------------------------------------------
    print("\n--- NEGATIVE CONTROL: True vs Random Steering Direction ---")
    print("(Random direction should produce much smaller max_drop)\n")

    valid = df_agg['max_drop'].notna() & df_agg['max_drop_random'].notna()
    if valid.sum() >= 5:
        true_mean = df_agg.loc[valid, 'max_drop'].mean()
        rand_mean = df_agg.loc[valid, 'max_drop_random'].mean()
        t_stat, p_val = stats.ttest_rel(df_agg.loc[valid, 'max_drop'],
                                        df_agg.loc[valid, 'max_drop_random'])
        print(f"max_drop: True={true_mean:.3f}, Random={rand_mean:.3f}, "
              f"t={t_stat:.2f}, p={p_val:.4f}")

    # --------------------------------------------------------------------------
    # Raw Correlations with Max Drop
    # --------------------------------------------------------------------------
    print("\n--- Raw Correlations with Max Drop ---")
    for metric in ['shesha_supervised', 'shesha_unsup_dim', 'fisher_criterion',
                   'silhouette', 'anisotropy_pca', 'wuc', 'procrustes_to_behavior']:
        valid = df_agg[metric].notna() & df_agg['max_drop'].notna()
        if valid.sum() >= 5:
            rho, p = stats.spearmanr(df_agg.loc[valid, metric],
                                     df_agg.loc[valid, 'max_drop'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{metric:<25}: rho={rho:+.3f}, p={p:.4f} {sig}")
        else:
            print(f"{metric:<25}: insufficient data (n={valid.sum()})")

    # --------------------------------------------------------------------------
    # Partial Correlations (controlling for separability)
    # --------------------------------------------------------------------------
    print("\n--- Partial Correlations (Controlling for Fisher + Silhouette) ---")

    valid_all = (df_agg['fisher_criterion'].notna() &
                 df_agg['silhouette'].notna() &
                 df_agg['max_drop'].notna())

    if valid_all.sum() >= 5:
        covs = df_agg.loc[valid_all, ['fisher_criterion', 'silhouette']].values
        for metric in ['shesha_supervised', 'shesha_unsup_dim', 'wuc', 'procrustes_to_behavior']:
            metric_valid = df_agg[metric].notna()
            combined_valid = valid_all & metric_valid
            if combined_valid.sum() >= 5:
                rho, p = partial_correlation(
                    df_agg.loc[combined_valid, metric].values,
                    df_agg.loc[combined_valid, 'max_drop'].values,
                    df_agg.loc[combined_valid, ['fisher_criterion', 'silhouette']].values
                )
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"{metric:<25}: rho={rho:+.3f}, p={p:.4f} {sig}")

    # --------------------------------------------------------------------------
    # Value ranges
    # --------------------------------------------------------------------------
    print("\n--- Value Ranges (aggregated across models) ---")
    for col in ['shesha_supervised', 'wuc', 'procrustes_to_behavior',
                'max_drop', 'max_drop_random', 'min_acc', 'anisotropy_pca',
                'asymmetry', 'flip_rate_pos', 'flip_rate_neg', 'pred_pos_rate_delta']:
        valid = df_agg[col].notna()
        if valid.sum() > 0:
            print(f"{col:<25}: {df_agg.loc[valid, col].min():.3f} - {df_agg.loc[valid, col].max():.3f}")

    # --------------------------------------------------------------------------
    # Controllability Analysis
    # --------------------------------------------------------------------------
    print("\n--- CONTROLLABILITY ANALYSIS ---")
    print("(Bidirectional flip rates and pred_pos_rate_delta show directional control)")

    valid = df_agg['asymmetry'].notna()
    if valid.sum() > 0:
        asym_mean = df_agg.loc[valid, 'asymmetry'].mean()
        asym_std = df_agg.loc[valid, 'asymmetry'].std()
        print(f"Asymmetry (acc@+2 - acc@-2):    {asym_mean:.3f} +/- {asym_std:.3f}")

    valid = df_agg['flip_rate_pos'].notna()
    if valid.sum() > 0:
        flip_pos_mean = df_agg.loc[valid, 'flip_rate_pos'].mean()
        flip_neg_mean = df_agg.loc[valid, 'flip_rate_neg'].mean()
        print(f"Flip rate at alpha=+2.0:       {flip_pos_mean:.3f}")
        print(f"Flip rate at alpha=-2.0:       {flip_neg_mean:.3f}")

    valid = df_agg['pred_pos_rate_delta'].notna()
    if valid.sum() > 0:
        delta_mean = df_agg.loc[valid, 'pred_pos_rate_delta'].mean()
        delta_std = df_agg.loc[valid, 'pred_pos_rate_delta'].std()
        print(f"Pred pos rate delta (+2 vs -2): {delta_mean:.3f} +/- {delta_std:.3f}")
        print("  (Should be large and consistent for true directional control)")

    print(f"\nSaved results to {OUTPUT_DIR}/synthetic_all_results.csv")