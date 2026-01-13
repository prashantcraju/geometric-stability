"""
Shesha Steering Analysis - MNLI and SST2
"""

import os
import gc
import warnings
import hashlib
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, ttest_rel
from scipy.spatial.distance import pdist
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.covariance import LedoitWolf
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from huggingface_hub import login

# Authenticate with Hugging Face
token = os.environ.get("HF_TOKEN")
if token:
    login(token)
else:
    print("Set HF_TOKEN environment variable")

warnings.filterwarnings("ignore")

# ================================================================
# 1. CONFIGURATION
# ================================================================

class CONFIG:

    OUTPUT_DIR = Path("./shesha-steering")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = Path("./shesha-steering/cache_real")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]

    n_total = 800
    holdout_fraction = 0.5

    reference_model = "bert-base-uncased"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    batch_size = 256  # Reduced from 512 to avoid OOM on large models
    n_steering_splits = 5
    n_random_per_split = 20

    use_cache = True  # Enable embedding caching for reliability


os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)

print(f"Device: {CONFIG.device}")
print(f"Precision: {CONFIG.dtype}")
print(f"Output Dir: {CONFIG.OUTPUT_DIR}")

# ================================================================
# 2. MODEL LIST
# ================================================================

MODELS_TO_TEST = [
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/paraphrase-albert-small-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-MiniLM-L12-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/paraphrase-distilroberta-base-v1",
    "sentence-transformers/distilbert-base-nli-mean-tokens",
    "sentence-transformers/msmarco-distilbert-base-v4",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "sentence-transformers/bert-base-nli-mean-tokens",
    "sentence-transformers/stsb-roberta-base",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en",
    "BAAI/bge-base-en",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-small",
    "intfloat/e5-base",
    "intfloat/e5-base-unsupervised",
    "thenlper/gte-small",
    "thenlper/gte-base",
    "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/unsup-simcse-bert-base-uncased",
    "princeton-nlp/sup-simcse-roberta-base",
    "princeton-nlp/unsup-simcse-roberta-base",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
    "thenlper/gte-large",
    "sentence-transformers/all-roberta-large-v1",
]

# ================================================================
# 3. DATA LOADING
# ================================================================

def load_sst2_data():
    print("Loading SST-2 dataset...")
    ds = load_dataset("glue", "sst2", split="train")
    texts = list(ds["sentence"])
    labels = np.array(ds["label"])
    print(f"  Total: {len(texts)} samples")
    return texts, None, labels  # None for text_pairs


def load_mnli_data():
    print("Loading MNLI dataset...")
    ds = load_dataset("glue", "mnli", split="train")
    premises = list(ds["premise"])
    hypotheses = list(ds["hypothesis"])
    labels = np.array(ds["label"])
    print(f"  Total: {len(premises)} samples")
    return premises, hypotheses, labels


def sample_data_balanced(texts, labels, seed, n_total, texts_pair=None):
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    n_per_class = n_total // n_classes

    selected_idx = []
    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        n_sample = min(n_per_class, len(label_idx))
        sampled = rng.choice(label_idx, n_sample, replace=False)
        selected_idx.extend(sampled)

    selected_idx = np.array(selected_idx)
    rng.shuffle(selected_idx)

    sampled_texts = [texts[i] for i in selected_idx]
    sampled_labels = labels[selected_idx]

    if texts_pair is not None:
        sampled_pairs = [texts_pair[i] for i in selected_idx]
        return sampled_texts, sampled_pairs, sampled_labels

    return sampled_texts, None, sampled_labels


# ================================================================
# 4. EMBEDDING CACHING
# ================================================================

CACHE_VERSION = "v1"  # Bump this if caching logic changes

def get_cache_key(model_name, dataset_name, seed, n_total, max_len):
    """Generate a unique cache key for embeddings."""
    # Include version, max_len to invalidate cache if logic changes
    key_str = f"{CACHE_VERSION}_{model_name}_{dataset_name}_{seed}_{n_total}_{max_len}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_embeddings(model_name, dataset_name, seed, max_len, expected_n=None):
    """Try to load cached embeddings with integrity check."""
    if not CONFIG.use_cache:
        return None

    cache_key = get_cache_key(model_name, dataset_name, seed, CONFIG.n_total, max_len)
    cache_path = Path(CONFIG.CACHE_DIR) / f"{cache_key}.pkl"

    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            embeddings = data['embeddings']

            # Version check
            cached_version = data.get('version', None)
            if cached_version != CACHE_VERSION:
                print(f"   [Cache STALE - version mismatch: {cached_version} vs {CACHE_VERSION}] {model_name}")
                return None

            # Shape integrity checks
            cached_n = data.get('n', embeddings.shape[0])
            cached_d = data.get('d', embeddings.shape[1])

            if embeddings.shape != (cached_n, cached_d):
                print(f"   [Cache CORRUPT - shape mismatch] {model_name}")
                return None

            if expected_n is not None and embeddings.shape[0] != expected_n:
                print(f"   [Cache STALE - n mismatch: {embeddings.shape[0]} vs {expected_n}] {model_name}")
                return None

            print(f"   [Cache HIT] {model_name}")
            return embeddings
        except Exception as e:
            print(f"   [Cache CORRUPT] {model_name}: {e}")
            return None
    return None


def save_embeddings_to_cache(model_name, dataset_name, seed, max_len, embeddings):
    """Save embeddings to cache with metadata."""
    if not CONFIG.use_cache:
        return

    cache_key = get_cache_key(model_name, dataset_name, seed, CONFIG.n_total, max_len)
    cache_path = Path(CONFIG.CACHE_DIR) / f"{cache_key}.pkl"

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'n': embeddings.shape[0],
                'd': embeddings.shape[1],
                'version': CACHE_VERSION,
            }, f)
    except Exception as e:
        print(f"   [Cache SAVE FAILED] {model_name}: {e}")


# ================================================================
# 5. EMBEDDING FUNCTION
# ================================================================

def embed_texts(model_name, texts, texts_pair=None, max_len=128, batch_size=None,
                dataset_name=None, seed=None):
    """
    Embedding function with caching and proper pair handling for NLI tasks.
    Returns L2-normalized embeddings.
    """
    # Try cache first (with integrity check)
    if dataset_name is not None and seed is not None:
        cached = get_cached_embeddings(
            model_name, dataset_name, seed, max_len=max_len, expected_n=len(texts)
        )
        if cached is not None:
            return cached

    if batch_size is None:
        batch_size = CONFIG.batch_size

    print(f"   Loading {model_name}...")
    try:
        # Removed trust_remote_code=True (not needed)
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if tok.pad_token is None:
            if tok.sep_token is not None:
                tok.pad_token = tok.sep_token
            elif tok.unk_token is not None:
                tok.pad_token = tok.unk_token
            elif tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({'pad_token': '[PAD]'})

        # Removed output_hidden_states=True (unused)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=CONFIG.dtype,
            low_cpu_mem_usage=True,
        ).to(CONFIG.device)

        if tok.pad_token == '[PAD]':
            model.resize_token_embeddings(len(tok))

        model.eval()

    except Exception as e:
        print(f"   [Error loading {model_name}]: {e}")
        return None

    feats = []
    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i:i+batch_size]

        # Proper pair encoding for NLI tasks
        if texts_pair is not None:
            chunk_pairs = texts_pair[i:i+batch_size]
            inputs = tok(
                chunk_texts, chunk_pairs,
                return_tensors="pt", padding=True,
                truncation=True, max_length=max_len
            ).to(CONFIG.device)
        else:
            inputs = tok(
                chunk_texts,
                return_tensors="pt", padding=True,
                truncation=True, max_length=max_len
            ).to(CONFIG.device)

        with torch.no_grad():
            if CONFIG.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=CONFIG.dtype):
                    out = model(**inputs)
            else:
                out = model(**inputs)

            H = out.last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            vec = (H * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            feats.append(vec.cpu().numpy())
        del inputs, out, H, vec

    del model, tok
    if CONFIG.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    embeddings = np.vstack(feats)

    # L2 normalize upfront to avoid normalization mismatch
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    # Save to cache
    if dataset_name is not None and seed is not None:
        save_embeddings_to_cache(model_name, dataset_name, seed, max_len, embeddings)

    return embeddings


# ================================================================
# 6. METRICS
# ================================================================

def compute_fisher_criterion(embeddings, labels):
    embeddings = np.asarray(embeddings, dtype=np.float64)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    global_mean = embeddings.mean(axis=0)
    S_b = np.zeros((embeddings.shape[1], embeddings.shape[1]))
    S_w = np.zeros((embeddings.shape[1], embeddings.shape[1]))

    for label in unique_labels:
        mask = labels == label
        n_k = mask.sum()
        class_embs = embeddings[mask]
        mean_k = class_embs.mean(axis=0)
        diff = (mean_k - global_mean).reshape(-1, 1)
        S_b += n_k * (diff @ diff.T)
        centered = class_embs - mean_k
        S_w += centered.T @ centered

    S_w += 1e-6 * np.eye(S_w.shape[0])
    result = float(np.trace(S_b) / (np.trace(S_w) + 1e-10))
    return result if np.isfinite(result) else np.nan


def compute_silhouette(embeddings, labels, sample_size=1000, seed=320):
    rng = np.random.RandomState(seed)
    if len(embeddings) > sample_size:
        idx = rng.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
    try:
        result = float(silhouette_score(embeddings, labels, metric='cosine'))
        return result if np.isfinite(result) else np.nan
    except:
        return np.nan


def compute_anisotropy_pca(embeddings):
    centered = embeddings - embeddings.mean(axis=0)
    if np.allclose(centered, 0):
        return np.nan
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            return np.nan
        result = float((S**2)[0] / total_var)
        return result if np.isfinite(result) else np.nan
    except:
        return np.nan


def compute_wuc_corrected(embeddings, labels, shrinkage=0.3, seed=320):
    """
    Compute Word-Unit Consistency (WUC) using split-half reliability.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) - categorical labels (e.g., word identities)
        shrinkage: Regularization parameter for covariance matrix
        seed: Random seed for reproducibility
    
    Returns:
        WUC score (Spearman correlation between split-half RDMs)
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    
    # Input validation
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D array, got shape {embeddings.shape}")
    if len(labels) != len(embeddings):
        raise ValueError(f"labels length ({len(labels)}) != embeddings length ({len(embeddings)})")
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    rng = np.random.RandomState(seed)

    # Basic checks
    if n_classes < 2:
        return np.nan
        # Consider raising: raise ValueError("Need at least 2 unique labels")

    # Check minimum samples per class
    min_samples_per_class = np.min([np.sum(labels == label) for label in unique_labels])
    if min_samples_per_class < 10:
        return np.nan
        # Consider: raise ValueError(f"Need at least 10 samples per class, min={min_samples_per_class}")

    n = len(labels)
    
    # Ensure we have enough samples for split-half
    if n < 20:  # Arbitrary minimum
        return np.nan
    
    # Create balanced splits while preserving class distribution
    idx = rng.permutation(n)
    half = n // 2
    
    # Alternative: Create more balanced splits
    def create_balanced_splits(labels, seed):
        """Create balanced splits preserving class distribution"""
        rng = np.random.RandomState(seed)
        split1_indices = []
        split2_indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            rng.shuffle(label_indices)
            n_label = len(label_indices)
            half_label = n_label // 2
            
            split1_indices.extend(label_indices[:half_label])
            split2_indices.extend(label_indices[half_label:2*half_label])
        
        return np.array(split1_indices), np.array(split2_indices)
    
    # Use balanced splits instead of simple permutation
    idx1, idx2 = create_balanced_splits(labels, seed)
    
    def get_class_means(emb, lab, indices):
        """Compute class means for given indices"""
        means = []
        for label in unique_labels:
            mask = lab[indices] == label
            if mask.sum() < 3:
                # Too few samples for reliable mean estimation
                return None
            class_emb = emb[indices][mask]
            # Consider robust mean estimation
            mean_emb = np.median(class_emb, axis=0)  # More robust than mean
            means.append(mean_emb)
        return np.array(means)

    means1 = get_class_means(embeddings, labels, idx1)
    means2 = get_class_means(embeddings, labels, idx2)

    if means1 is None or means2 is None:
        return np.nan

    # Compute residuals for covariance estimation
    residuals = []
    for i, label in enumerate(unique_labels):
        # Split 1 residuals
        mask1 = labels[idx1] == label
        if mask1.sum() > 0:
            residuals1 = embeddings[idx1][mask1] - means1[i]
            residuals.append(residuals1)
        
        # Split 2 residuals
        mask2 = labels[idx2] == label
        if mask2.sum() > 0:
            residuals2 = embeddings[idx2][mask2] - means2[i]
            residuals.append(residuals2)

    if not residuals:
        return np.nan

    residuals = np.vstack(residuals)
    
    # More sophisticated sample size check
    if len(residuals) < max(50, 5 * embeddings.shape[1]):
        # Need more samples for stable covariance estimation
        return np.nan

    try:
        # Covariance estimation with regularization
        if len(residuals) > embeddings.shape[1]:
            # Only use LedoitWolf if we have enough samples
            lw = LedoitWolf(assume_centered=False)
            lw.fit(residuals)
            cov = lw.covariance_
        else:
            # Use regularized empirical covariance
            cov = np.cov(residuals, rowvar=False)
            if np.linalg.matrix_rank(cov) < cov.shape[0]:
                # Add diagonal regularization
                cov = cov + 1e-3 * np.eye(cov.shape[0])
        
        trace_cov = np.trace(cov)
        if trace_cov < 1e-12:  # Stricter tolerance
            return np.nan
        
        # Apply shrinkage regularization
        d = cov.shape[0]
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(d) * (trace_cov / d)
        
        # Ensure positive definiteness
        cov_reg = cov + 1e-8 * np.eye(d)
        
        # Eigendecomposition for whitening
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        
        # Filter out near-zero eigenvalues
        eigvals = np.maximum(eigvals, 1e-10 * np.max(eigvals))
        
        # Compute whitening transform
        cov_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    except np.linalg.LinAlgError:
        # Handle numerical issues
        return np.nan
    except Exception as e:
        # Handle unexpected errors
        return np.nan

    # Apply whitening transform
    means1_white = means1 @ cov_inv_sqrt
    means2_white = means2 @ cov_inv_sqrt

    # Compute RDMs (Representational Dissimilarity Matrices)
    rdm1 = pdist(means1_white, metric='cosine')
    rdm2 = pdist(means2_white, metric='cosine')
    
    # Alternative distance metrics to consider:
    # rdm1 = pdist(means1_white, metric='euclidean')
    # rdm1 = pdist(means1_white, metric='correlation')

    if len(rdm1) < 2:
        return np.nan

    # Compute correlation between RDMs
    rho, p_value = spearmanr(rdm1, rdm2)
    
    # Additional quality checks
    if not np.isfinite(rho):
        return np.nan
    
    # Optional: Compute confidence intervals via bootstrapping
    # if compute_ci:
    #     ci = bootstrap_confidence_interval(embeddings, labels, seed=seed)
    #     return float(rho), ci
    
    return float(rho)


def compute_shesha_supervised(embeddings, labels, max_samples=300, seed=320):
    rng = np.random.RandomState(seed)
    if len(embeddings) > max_samples:
        idx = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    embeddings = embeddings - embeddings.mean(axis=0)
    model_dists = pdist(embeddings, metric='correlation')
    ideal_dists = pdist(labels.reshape(-1, 1), metric='hamming')

    rho, _ = spearmanr(model_dists, ideal_dists)
    return float(rho) if np.isfinite(rho) else np.nan


def compute_rdm_noise_sensitivity(embeddings, n_iterations=50, noise_scale=0.01, seed=320):
    """
    RDM noise sensitivity - measures numerical smoothness under small perturbations.
    Will be near 1.0 for most models. NOT bootstrap stability.
    """
    n = embeddings.shape[0]
    if n < 10:
        return np.nan

    rng = np.random.RandomState(seed)
    actual_noise_scale = noise_scale * np.std(embeddings)
    corrs = []

    for _ in range(n_iterations):
        emb_view1 = embeddings + rng.randn(*embeddings.shape) * actual_noise_scale
        emb_view2 = embeddings + rng.randn(*embeddings.shape) * actual_noise_scale

        rdm1 = pdist(emb_view1, 'cosine')
        rdm2 = pdist(emb_view2, 'cosine')

        rho, _ = spearmanr(rdm1, rdm2)
        if np.isfinite(rho):
            corrs.append(rho)

    return float(np.mean(corrs)) if corrs else np.nan


def compute_feature_partition_stability(embeddings, n_splits=50, seed=320):
    """
    Feature-partition stability - measures internal redundancy across feature partitions.
    NOT stability under data resampling.
    """
    d = embeddings.shape[1]
    if d < 4:
        return np.nan

    corrs = []
    for i in range(n_splits):
        rng = np.random.RandomState(seed + i)
        perm = rng.permutation(d)
        half = d // 2
        dims1, dims2 = perm[:half], perm[half:2*half]

        rdm1 = pdist(embeddings[:, dims1], 'correlation')
        rdm2 = pdist(embeddings[:, dims2], 'correlation')

        rho, _ = spearmanr(rdm1, rdm2)
        if np.isfinite(rho):
            corrs.append(rho)

    return float(np.mean(corrs)) if corrs else np.nan


def compute_procrustes(embeddings, labels):
    """
    Original function with mathematical corrections.
    
    Measures subspace alignment between embeddings and concept directions.
    Similar to compute_steering_alignment but without baseline adjustment.
    
    Parameters:
    -----------
    embeddings : array (n_samples, embedding_dim)
        Model embeddings/activations
    labels : array (n_samples,)
        Class labels
        
    Returns:
    --------
    alignment : float in [0, 1]
        Subspace alignment score (higher = better alignment)
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)

    # Get unique labels and remap to 0, 1, 2, ...
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[l] for l in labels])

    n_classes = len(unique_labels)
    n_samples = len(labels)

    # Create one-hot encoded ideal representation
    ideal = np.zeros((n_samples, n_classes))
    for i, l in enumerate(y_remapped):
        ideal[i, l] = 1.0

    # Center both matrices
    embeddings = embeddings - embeddings.mean(axis=0)
    ideal = ideal - ideal.mean(axis=0)

    # Handle dimension mismatches
    if embeddings.shape[1] > n_classes:
        # Reduce dimensions to n_classes (preserve most variance)
        U, S, _ = np.linalg.svd(embeddings, full_matrices=False)
        embeddings = U[:, :n_classes] * S[:n_classes]
    elif embeddings.shape[1] < n_classes:
        # Pad with zeros
        embeddings = np.hstack([
            embeddings, 
            np.zeros((n_samples, n_classes - embeddings.shape[1]))
        ])

    # Normalize to unit Frobenius norm
    emb_norm = np.linalg.norm(embeddings, 'fro')
    ideal_norm = np.linalg.norm(ideal, 'fro')

    if emb_norm < 1e-10 or ideal_norm < 1e-10:
        return np.nan

    embeddings = embeddings / emb_norm
    ideal = ideal / ideal_norm

    # Compute subspace alignment
    try:
        # Correlation matrix
        M = embeddings.T @ ideal
        
        # Singular values of correlation matrix
        S = np.linalg.svd(M, compute_uv=False)
        
        # FIXED NORMALIZATION:
        # The maximum possible sum(S) is min(dim_X, dim_Y) = n_classes
        # So we normalize by n_classes to get [0, 1]
        alignment = np.sum(S) / n_classes
        
        return float(np.clip(alignment, 0.0, 1.0))
    except:
        return np.nan



def compute_cka(X, Y):
    """
    Simple linear CKA (not debiased) for sanity checks.
    More numerically stable for self-similarity tests.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    num = np.linalg.norm(X.T @ Y, 'fro') ** 2
    den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    return float(num / (den + 1e-12))

def compute_steering_alignment(
    embeddings,
    labels,
    use_baseline=True,
    n_permutations=100,
    random_seed=None
):
    """
    Measure alignment between embeddings and concept directions for LLM steering tests.
    
    Higher scores indicate better conditions for steering vectors.
    
    Parameters:
    -----------
    embeddings : array (n_samples, embedding_dim)
        Model activations or embeddings
    labels : array (n_samples,)
        Concept labels for each sample
    use_baseline : bool, default=True
        Subtract random baseline via label permutations
    n_permutations : int, default=100
        Number of permutations for baseline
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    score : float in [0, 1]
        Steering alignment score where:
        - 0.9-1.0: Excellent - Clear concept directions
        - 0.7-0.9: Good - Steering should work well
        - 0.6-0.7: Moderate - Steering may work
        - 0.5-0.6: Weak - Steering unlikely to work well
        - 0.0-0.5: Poor - No concept structure
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    
    # Validate inputs
    if len(embeddings) != len(labels):
        raise ValueError(f"Length mismatch: embeddings ({len(embeddings)}) != labels ({len(labels)})")
    
    if len(embeddings) < 2:
        return np.nan
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    n_samples = len(labels)
    
    if n_classes < 2:
        return np.nan
    
    # Map labels to indices
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[label] for label in labels])
    
    # Create one-hot encoded concept matrix
    ideal = np.zeros((n_samples, n_classes), dtype=np.float64)
    ideal[np.arange(n_samples), label_indices] = 1.0
    
    # Center matrices
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    Y = ideal - ideal.mean(axis=0, keepdims=True)
    
    # Normalize to unit Frobenius norm
    X_norm = np.linalg.norm(X, 'fro')
    Y_norm = np.linalg.norm(Y, 'fro')
    
    if X_norm < 1e-12 or Y_norm < 1e-12:
        return np.nan
    
    X = X / X_norm
    Y = Y / Y_norm
    
    # Compute raw alignment score via singular values
    try:
        M = X.T @ Y  # Correlation matrix
        singular_values = np.linalg.svd(M, compute_uv=False)
        raw_score = np.mean(singular_values)  # Average correlation strength
    except np.linalg.LinAlgError:
        return np.nan
    
    # Return raw score if baseline not requested
    if not use_baseline:
        return float(np.clip(raw_score, 0.0, 1.0))
    
    # Compute baseline via label permutations
    baseline_scores = []
    labels_perm = label_indices.copy()
    
    for _ in range(n_permutations):
        np.random.shuffle(labels_perm)
        
        ideal_perm = np.zeros((n_samples, n_classes), dtype=np.float64)
        ideal_perm[np.arange(n_samples), labels_perm] = 1.0
        
        Y_perm = ideal_perm - ideal_perm.mean(axis=0, keepdims=True)
        Y_perm_norm = np.linalg.norm(Y_perm, 'fro')
        
        if Y_perm_norm > 1e-12:
            Y_perm = Y_perm / Y_perm_norm
            M_perm = X.T @ Y_perm
            try:
                sv_perm = np.linalg.svd(M_perm, compute_uv=False)
                baseline_scores.append(np.mean(sv_perm))
            except np.linalg.LinAlgError:
                continue
    
    # If baseline computation failed, return raw score
    if not baseline_scores:
        return float(np.clip(raw_score, 0.0, 1.0))
    
    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores)
    
    # Adjust score relative to baseline
    if baseline_std > 0:
        z_score = (raw_score - baseline_mean) / baseline_std
        adjusted_score = 1.0 / (1.0 + np.exp(-z_score))
    else:
        adjusted_score = 1.0 if raw_score > baseline_mean else 0.5
    
    return float(np.clip(adjusted_score, 0.0, 1.0))



# ================================================================
# 7. STEERING EVALUATION
# ================================================================

def evaluate_steering_corrected(emb_train, lab_train, emb_test, lab_test,
                                n_fewshot=25, n_splits=5, n_random_per_split=20, seed=320):
    alphas = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    n_classes = len(np.unique(lab_train))

    all_results = {
        'acc_0': [], 'max_drop': [], 'max_drop_random': [], 'min_acc': [],
        'acc_neg_alpha': [], 'acc_pos_alpha': [],
        'flip_rate_pos': [], 'flip_rate_neg': [],
    }

    for split_idx in range(n_splits):
        rng = np.random.RandomState(seed + split_idx * 1000)

        # Few-shot stratified sampling
        if len(emb_train) > n_fewshot:
            fs_idx = []
            for label in np.unique(lab_train):
                label_idx = np.where(lab_train == label)[0]
                n_per = max(2, n_fewshot // n_classes)
                sampled = rng.choice(label_idx, min(n_per, len(label_idx)), replace=False)
                fs_idx.extend(sampled)
            fs_idx = np.array(fs_idx)
        else:
            fs_idx = np.arange(len(emb_train))

        emb_fs, lab_fs = emb_train[fs_idx], lab_train[fs_idx]

        # Train probe
        solver = 'liblinear' if n_classes == 2 else 'lbfgs'
        # Explicit multinomial for multiclass (more stable across sklearn versions)
        multi_class = 'ovr' if n_classes == 2 else 'multinomial'
        clf = LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed, solver=solver,
            multi_class=multi_class
        )
        clf.fit(emb_fs, lab_fs)

        # Get steering direction
        if n_classes == 2:
            w = clf.coef_.flatten()
        else:
            # Top singular vector for multiclass
            U, S, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
            w = Vt[0]
        w_hat = w / (np.linalg.norm(w) + 1e-12)

        # Baseline
        preds_baseline = clf.predict(emb_test)
        acc_0 = accuracy_score(lab_test, preds_baseline)

        # Steering sweep
        accuracies_true = []
        for alpha in alphas:
            emb_steered = emb_test + alpha * w_hat
            norms = np.linalg.norm(emb_steered, axis=1, keepdims=True)
            emb_steered = emb_steered / np.maximum(norms, 1e-12)
            acc = accuracy_score(lab_test, clf.predict(emb_steered))
            accuracies_true.append(acc)

        # Bidirectional flip rates
        emb_pos = emb_test + 2.0 * w_hat
        emb_pos = emb_pos / np.maximum(np.linalg.norm(emb_pos, axis=1, keepdims=True), 1e-12)
        flip_rate_pos = np.mean(clf.predict(emb_pos) != preds_baseline)

        emb_neg = emb_test - 2.0 * w_hat
        emb_neg = emb_neg / np.maximum(np.linalg.norm(emb_neg, axis=1, keepdims=True), 1e-12)
        flip_rate_neg = np.mean(clf.predict(emb_neg) != preds_baseline)

        # Random direction control
        random_drops = []
        for _ in range(n_random_per_split):
            w_rand = rng.randn(len(w))
            w_rand_hat = w_rand / (np.linalg.norm(w_rand) + 1e-12)

            accuracies_rand = []
            for alpha in alphas:
                emb_steered = emb_test + alpha * w_rand_hat
                norms = np.linalg.norm(emb_steered, axis=1, keepdims=True)
                emb_steered = emb_steered / np.maximum(norms, 1e-12)
                acc = accuracy_score(lab_test, clf.predict(emb_steered))
                accuracies_rand.append(acc)
            random_drops.append(acc_0 - min(accuracies_rand))

        all_results['acc_0'].append(acc_0)
        all_results['max_drop'].append(acc_0 - min(accuracies_true))
        all_results['max_drop_random'].append(np.mean(random_drops))
        all_results['min_acc'].append(min(accuracies_true))
        all_results['acc_neg_alpha'].append(accuracies_true[0])
        all_results['acc_pos_alpha'].append(accuracies_true[-1])
        all_results['flip_rate_pos'].append(flip_rate_pos)
        all_results['flip_rate_neg'].append(flip_rate_neg)

    return {
        'acc_0_mean': np.mean(all_results['acc_0']),
        'max_drop_mean': np.mean(all_results['max_drop']),
        'max_drop_std': np.std(all_results['max_drop']),
        'max_drop_random_mean': np.mean(all_results['max_drop_random']),
        'min_acc_mean': np.mean(all_results['min_acc']),
        'asymmetry': np.mean(all_results['acc_pos_alpha']) - np.mean(all_results['acc_neg_alpha']),
        'flip_rate_pos_mean': np.mean(all_results['flip_rate_pos']),
        'flip_rate_neg_mean': np.mean(all_results['flip_rate_neg']),
    }


# ================================================================
# 8. MAIN EXPERIMENT
# ================================================================

def run_experiment():
    print("=" * 70)
    print("STEERING MNLI AND SST2")
    print(f"Cache: {'ENABLED' if CONFIG.use_cache else 'DISABLED'} ({CONFIG.CACHE_DIR})")
    print("=" * 70)

    # Load datasets
    sst2_texts, _, sst2_labels = load_sst2_data()
    mnli_premises, mnli_hypotheses, mnli_labels = load_mnli_data()

    all_results = []

    # Process each dataset
    for dataset_name, (texts, texts_pair, all_labels) in [
        ('sst2', (sst2_texts, None, sst2_labels)),
        ('mnli', (mnli_premises, mnli_hypotheses, mnli_labels))
    ]:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")

        for seed in CONFIG.seeds:
            print(f"\n--- Seed: {seed} ---")

            # Sample data
            sampled_texts, sampled_pairs, labels = sample_data_balanced(
                texts, all_labels, seed, CONFIG.n_total, texts_pair=texts_pair
            )

            # Shuffled labels for negative control
            rng = np.random.RandomState(seed)
            labels_shuffled = rng.permutation(labels)

            # Reference embeddings (with explicit error reporting)
            ref_embeddings = None
            try:
                ref_embeddings = embed_texts(
                    CONFIG.reference_model, sampled_texts, texts_pair=sampled_pairs,
                    dataset_name=dataset_name, seed=seed
                )
            except Exception as e:
                print(f"   [Reference model error - CKA will be NaN]: {e}")

            for model_name in MODELS_TO_TEST:
                print(f"Processing {model_name}...")

                try:
                    embeddings = embed_texts(
                        model_name, sampled_texts, texts_pair=sampled_pairs,
                        dataset_name=dataset_name, seed=seed
                    )
                    if embeddings is None:
                        continue

                    # Split: A for metrics, B for steering
                    idx_A, idx_B = train_test_split(
                        np.arange(len(labels)),
                        test_size=CONFIG.holdout_fraction,
                        random_state=seed,
                        stratify=labels
                    )

                    emb_A, lab_A = embeddings[idx_A], labels[idx_A]
                    lab_A_shuffled = labels_shuffled[idx_A]
                    emb_B, lab_B = embeddings[idx_B], labels[idx_B]

                    # Further split B for steering
                    idx_B_train, idx_B_test = train_test_split(
                        np.arange(len(lab_B)),
                        test_size=0.5,
                        random_state=seed,
                        stratify=lab_B
                    )

                    emb_B_train, lab_B_train = emb_B[idx_B_train], lab_B[idx_B_train]
                    emb_B_test, lab_B_test = emb_B[idx_B_test], lab_B[idx_B_test]

                    # Compute metrics
                    shesha_sup = compute_shesha_supervised(emb_A, lab_A, seed=seed)
                    rdm_noise = compute_rdm_noise_sensitivity(emb_A, seed=seed)
                    feat_part_stab = compute_feature_partition_stability(emb_A, seed=seed)
                    wuc = compute_wuc_corrected(emb_A, lab_A, seed=seed)
                    fisher = compute_fisher_criterion(emb_A, lab_A)
                    sil = compute_silhouette(emb_A, lab_A, seed=seed)
                    aniso = compute_anisotropy_pca(emb_A)
                    proc = compute_procrustes(emb_A, lab_A)
                    steer_align = compute_steering_alignment(emb_A, lab_A)

                    cka_ref = np.nan
                    if ref_embeddings is not None:
                        cka_ref = compute_cka(emb_A, ref_embeddings[idx_A])

                    # Shuffled label controls
                    shesha_shuffled = compute_shesha_supervised(emb_A, lab_A_shuffled, seed=seed)
                    fisher_shuffled = compute_fisher_criterion(emb_A, lab_A_shuffled)
                    proc_shuffled = compute_procrustes(emb_A, lab_A_shuffled)
                    steer_align_shuffled = compute_steering_alignment(emb_A, lab_A_shuffled)

                    # Steering
                    steering = evaluate_steering_corrected(
                        emb_B_train, lab_B_train,
                        emb_B_test, lab_B_test,
                        n_fewshot=25,
                        n_splits=CONFIG.n_steering_splits,
                        n_random_per_split=CONFIG.n_random_per_split,
                        seed=seed
                    )

                    res = {
                        'model': model_name,
                        'seed': seed,
                        'dataset': dataset_name,
                        'shesha_supervised': shesha_sup,
                        'rdm_noise_sensitivity': rdm_noise,
                        'feature_partition_stability': feat_part_stab,
                        'wuc': wuc,
                        'fisher': fisher,
                        'silhouette': sil,
                        'anisotropy': aniso,
                        'procrustes_behavior': proc,
                        'steering aligned': steer_align,
                        'cka_reference': cka_ref,
                        'shesha_supervised_shuffled': shesha_shuffled,
                        'fisher_shuffled': fisher_shuffled,
                        'procrustes_shuffled': proc_shuffled,
                        'steering_aligned_shuffled': steer_align_shuffled,
                        'steering_acc_0': steering['acc_0_mean'],
                        'steering_max_drop': steering['max_drop_mean'],
                        'steering_max_drop_std': steering['max_drop_std'],
                        'steering_max_drop_random': steering['max_drop_random_mean'],
                        'steering_min_acc': steering['min_acc_mean'],
                        'steering_asymmetry': steering['asymmetry'],
                        'steering_flip_rate_pos': steering['flip_rate_pos_mean'],
                        'steering_flip_rate_neg': steering['flip_rate_neg_mean'],
                    }
                    all_results.append(res)

                    wuc_str = f"{wuc:.3f}" if not np.isnan(wuc) else "NaN"
                    print(f"  > Sh={shesha_sup:.3f}, WUC={wuc_str}, "
                          f"Drop={steering['max_drop_mean']:.3f}, "
                          f"RandDrop={steering['max_drop_random_mean']:.3f}")

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()

            # Save intermediate
            df_interim = pd.DataFrame(all_results)
            df_interim.to_csv(f"{CONFIG.OUTPUT_DIR}/real_results_interim_{dataset_name}_{seed}.csv", index=False)

    # Save final
    df = pd.DataFrame(all_results)
    df.to_csv(f"{CONFIG.OUTPUT_DIR}/real_steering_results.csv", index=False)
    print(f"\nSaved to {CONFIG.OUTPUT_DIR}/real_steering_results.csv")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    for dataset in df['dataset'].unique():
        print(f"\n{'='*50}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*50}")

        sub = df[df['dataset'] == dataset]
        agg = sub.groupby('model').agg({
            'shesha_supervised': 'mean',
            'feature_partition_stability': 'mean',
            'wuc': 'mean',
            'fisher': 'mean',
            'procrustes_behavior': 'mean',
            'shesha_supervised_shuffled': 'mean',
            'fisher_shuffled': 'mean',
            'procrustes_shuffled': 'mean',
            'steering_max_drop': 'mean',
            'steering_max_drop_random': 'mean',
            'steering_asymmetry': 'mean',
            'steering_flip_rate_pos': 'mean',
            'steering_flip_rate_neg': 'mean',
        }).reset_index()

        # Negative control: shuffled labels
        print("\n--- NEGATIVE CONTROL: True vs Shuffled Labels ---")
        for true_col, shuf_col in [
            ('shesha_supervised', 'shesha_supervised_shuffled'),
            ('fisher', 'fisher_shuffled'),
            ('procrustes_behavior', 'procrustes_shuffled')
        ]:
            valid = agg[true_col].notna() & agg[shuf_col].notna()
            if valid.sum() >= 5:
                true_mean = agg.loc[valid, true_col].mean()
                shuf_mean = agg.loc[valid, shuf_col].mean()
                t, p = ttest_rel(agg.loc[valid, true_col], agg.loc[valid, shuf_col])
                print(f"  {true_col:<25}: True={true_mean:.3f}, Shuf={shuf_mean:.3f}, p={p:.4f}")

        # Negative control: random direction
        print("\n--- NEGATIVE CONTROL: True vs Random Direction ---")
        valid = agg['steering_max_drop'].notna() & agg['steering_max_drop_random'].notna()
        if valid.sum() >= 5:
            true_mean = agg.loc[valid, 'steering_max_drop'].mean()
            rand_mean = agg.loc[valid, 'steering_max_drop_random'].mean()
            t, p = ttest_rel(agg.loc[valid, 'steering_max_drop'],
                            agg.loc[valid, 'steering_max_drop_random'])
            print(f"  max_drop: True={true_mean:.3f}, Random={rand_mean:.3f}, p={p:.4f}")

        # Correlations
        print(f"\n--- Correlations with steering_max_drop ---")
        for col in ['shesha_supervised', 'feature_partition_stability', 'wuc', 'fisher', 'procrustes_behavior']:
            valid = agg[col].notna() & agg['steering_max_drop'].notna()
            if valid.sum() >= 5:
                rho, p = spearmanr(agg.loc[valid, col], agg.loc[valid, 'steering_max_drop'])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {col:<30}: rho={rho:+.3f}, p={p:.4f} {sig}")

        # Controllability
        print(f"\n--- CONTROLLABILITY ---")
        valid = agg['steering_flip_rate_pos'].notna()
        if valid.sum() > 0:
            print(f"  Flip rate at +2.0: {agg.loc[valid, 'steering_flip_rate_pos'].mean():.3f}")
            print(f"  Flip rate at -2.0: {agg.loc[valid, 'steering_flip_rate_neg'].mean():.3f}")
            print(f"  Asymmetry:         {agg.loc[valid, 'steering_asymmetry'].mean():.3f}")


if __name__ == "__main__":
    run_experiment()