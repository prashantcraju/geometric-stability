"""
Shesha Transfer Experiments
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import random
import numpy as np
import pandas as pd
import torch
import hashlib
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr, rankdata
from scipy.spatial.distance import pdist, cdist
from scipy.linalg import svd
import warnings
import time
from LogME import LogME

# Authenticate with Hugging Face to access Llama/Gemma models
from huggingface_hub import login

# Authenticate with Hugging Face
token = os.environ.get("HF_TOKEN")
if token:
    login(token)
else:
    print("Set HF_TOKEN environment variable")


warnings.filterwarnings("ignore")

# =============================================================================
# A100 OPTIMIZATIONS
# =============================================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print(f"[A100 Optimizations]")
print(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
print(f"  Default dtype: {DTYPE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Compute capability: {torch.cuda.get_device_capability()}")

# ==========================================
# CONFIGURATION
# ==========================================

RUN_CONFIG = {
    (320, "exp1"): True,
    (320, "exp2"): True,
    (1991, "exp1"): True,
    (1991, "exp2"): True,
    (9, "exp1"): True,
    (9, "exp2"): True,
    (724, "exp1"): True,
    (724, "exp2"): True,
}

SEEDS = [320, 1991, 9, 724]
IMDB_SAMPLE_SEED = 7

N_IMDB_SHESHA_EXP1 = 2000
N_IMDB_SHESHA_EXP2 = 2000
N_BOOTSTRAP = 20

MAX_TEXT_LENGTH = 256
# Increased from 32 to 256: A100 can easily handle larger batches
# for MiniLM/MPNet models, speeds up embedding by ~8x
BATCH_SIZE = 256
SAMPLE_SIZES_TOTAL = [16, 32, 64, 128, 256, 512]

OUTPUT_DIR = Path("./shesha-transfer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("./shesha-transfer/cache")
CACHE_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ==========================================
# MODELS
# ==========================================
MODELS = [
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/all-MiniLM-L12-v2", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/paraphrase-MiniLM-L6-v2", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/paraphrase-MiniLM-L12-v2", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/paraphrase-MiniLM-L3-v2", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/paraphrase-albert-small-v2", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "nreimers/MiniLM-L6-H384-uncased", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "intfloat/e5-small-v2", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "intfloat/e5-small", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "BAAI/bge-small-en-v1.5", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "thenlper/gte-small", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/all-distilroberta-v1", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/msmarco-MiniLM-L6-cos-v5", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "capacity": "Small", "train_type": "unsupervised"},
    {"name": "sentence-transformers/all-mpnet-base-v2", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "sentence-transformers/paraphrase-mpnet-base-v2", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "intfloat/e5-base-v2", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "intfloat/e5-base", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "BAAI/bge-base-en-v1.5", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "thenlper/gte-base", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "sentence-transformers/gtr-t5-base", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "sentence-transformers/sentence-t5-base", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "jinaai/jina-embeddings-v2-base-en", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "nomic-ai/nomic-embed-text-v1", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "avsolatorio/GIST-small-Embedding-v0", "capacity": "Base", "train_type": "unsupervised"},
    {"name": "intfloat/e5-large-v2", "capacity": "Large", "train_type": "unsupervised"},
    {"name": "intfloat/e5-large", "capacity": "Large", "train_type": "unsupervised"},
    {"name": "BAAI/bge-large-en-v1.5", "capacity": "Large", "train_type": "unsupervised"},
    {"name": "thenlper/gte-large", "capacity": "Large", "train_type": "unsupervised"},
    {"name": "sentence-transformers/gtr-t5-large", "capacity": "Large", "train_type": "unsupervised"},
    {"name": "sentence-transformers/sentence-t5-large", "capacity": "Large", "train_type": "unsupervised"},
    {"name": "Alibaba-NLP/gte-large-en-v1.5", "capacity": "Large", "train_type": "unsupervised"},
    {"name": "WhereIsAI/UAE-Large-V1", "capacity": "Large", "train_type": "unsupervised"},
]

SKIP_MODELS = {
    "intfloat/e5-mistral-7b-instruct",
    "Salesforce/SFR-Embedding-Mistral",
    "dunzhang/stella_en_1.5B_v5",
    "nvidia/NV-Embed-v1",
    "WhereIsAI/UAE-Large-V1",
}

print(f"Total models: {len(MODELS)}")
print(f"Seeds: {SEEDS}")


# ==========================================
# TRANSFERABILITY BASELINES
# ==========================================

def compute_centroid_softmax_score(embeddings, labels):
    """
    Centroid-based soft assignment score.

    NOTE: This is NOT LEEP. True LEEP (Nguyen et al., ICML 2020) requires
    source-model class probabilities, which are not available for embedding-only
    models without training a classifier head.

    This is a proxy that uses centroid distances with softmax to create
    pseudo-probabilities. Included as an additional heuristic baseline.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)

    n = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    centroids = np.array([X[y == c].mean(axis=0) for c in classes])

    dists = np.zeros((n, n_classes))
    for i, c in enumerate(classes):
        dists[:, i] = np.linalg.norm(X - centroids[i], axis=1)

    neg_dists = -dists
    exp_dists = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
    source_probs = exp_dists / (exp_dists.sum(axis=1, keepdims=True) + 1e-10)

    joint = np.zeros((n_classes, n_classes))
    for i in range(n):
        target_class_idx = np.where(classes == y[i])[0][0]
        joint[:, target_class_idx] += source_probs[i]
    joint /= n

    marginal_z = joint.sum(axis=1, keepdims=True)
    conditional = joint / (marginal_z + 1e-10)

    score = 0.0
    for i in range(n):
        target_class_idx = np.where(classes == y[i])[0][0]
        prob = (source_probs[i] @ conditional[:, target_class_idx])
        score += np.log(prob + 1e-10)

    return score / n


def compute_hscore(embeddings, labels):
    """Minimal H-score: Inter-class variance / intra-class variance ratio"""
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)
    
    # Center data
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    # Get classes
    classes = np.unique(y)
    
    # Compute class means and probabilities
    class_means = []
    class_probs = []
    
    for c in classes:
        mask = (y == c)
        n_c = mask.sum()
        class_means.append(X_centered[mask].mean(axis=0))
        class_probs.append(n_c / len(X))
    
    class_means = np.array(class_means)
    class_probs = np.array(class_probs)
    
    # Between-class covariance
    between_cov = np.zeros((X.shape[1], X.shape[1]))
    for i, prob in enumerate(class_probs):
        mu = class_means[i]
        between_cov += prob * np.outer(mu, mu)
    
    # Total covariance
    total_cov = np.cov(X_centered.T)
    
    # Add small regularization
    total_cov_reg = total_cov + 1e-10 * np.eye(total_cov.shape[0])
    
    # H-score = trace(Σ_b Σ_t^(-1))
    inv_total_cov = np.linalg.pinv(total_cov_reg)
    h_score = np.trace(inv_total_cov @ between_cov)
    
    return float(h_score)


def compute_fisher_discriminate(embeddings, labels):
    """
    H-Score: Inter-class variance / intra-class variance ratio
    """
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)
    
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples, n_features = X.shape
    
    # Center data
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    # Compute within-class scatter matrix
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))
    
    overall_mean = X_centered.mean(axis=0)
    
    for c in classes:
        class_mask = (y == c)
        n_c = class_mask.sum()
        
        # Class mean
        class_mean = X_centered[class_mask].mean(axis=0)
        
        # Within-class scatter
        X_class = X_centered[class_mask] - class_mean
        S_W += X_class.T @ X_class
        
        # Between-class scatter (weighted by class size)
        mean_diff = class_mean - overall_mean
        S_B += n_c * np.outer(mean_diff, mean_diff)
    
    # Fisher criterion (generalized Rayleigh quotient)
    try:
        # Solve generalized eigenvalue problem: S_B * w = λ * S_W * w
        eigvals = np.linalg.eigvals(np.linalg.inv(S_W + 1e-8*np.eye(n_features)) @ S_B)
        # Sum of eigenvalues (trace)
        score = np.sum(np.real(eigvals))
    except:
        # Fallback to scalar version
        between_var = np.trace(S_B)
        within_var = np.trace(S_W)
        score = between_var / (within_var + 1e-10)
    
    score = score / (n_classes - 1)
    
    return score


def compute_nce(embeddings, labels, temperature='auto', metric='cosine'):
    """
    NCE: Negative Conditional Entropy for transfer learning evaluation.
    
    Parameters:
    - embeddings: feature vectors from pretrained model
    - labels: downstream task labels
    - temperature: 'auto' or float. Auto uses median distance heuristic
    - metric: 'cosine' or 'euclidean' distance
    
    Returns:
    - nce: Negative Conditional Entropy (higher is better)
    - accuracy: Classification accuracy from 1-NN centroid classifier
    """
    X = np.asarray(embeddings)
    y = np.asarray(labels)
    
    # Normalize for cosine distance
    if metric == 'cosine':
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    
    classes = np.unique(y)
    centroids = np.array([X[y == c].mean(axis=0) for c in classes])
    
    # Compute distances
    if metric == 'cosine':
        # Cosine similarity (higher = closer)
        similarities = X @ centroids.T  # shape (n_samples, n_classes)
        dists = 1 - similarities  # Convert to distance
    else:  # euclidean
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Auto temperature: median of distances to correct class centroids
    if temperature == 'auto':
        correct_class_indices = [np.where(classes == label)[0][0] for label in y]
        correct_dists = dists[np.arange(len(y)), correct_class_indices]
        temperature = np.median(correct_dists) + 1e-10
    
    # Convert distances to probabilities (softmax over -distance/temperature)
    logits = -dists / temperature
    logits = logits - logits.max(axis=1, keepdims=True)  # Numerical stability
    probs = np.exp(logits)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
    
    # Get probabilities for true labels
    true_probs = probs[np.arange(len(y)), 
                      [np.where(classes == label)[0][0] for label in y]]
    
    # Conditional entropy
    cond_entropy = -np.mean(np.log(true_probs + 1e-10))
    nce = -cond_entropy
    
    # Also compute accuracy from 1-NN centroids
    predicted = np.argmin(dists, axis=1)
    predicted_labels = classes[predicted]
    accuracy = np.mean(predicted_labels == y)
    
    return {
        'nce': nce,
        'accuracy': accuracy,
        'temperature': temperature,
        'cond_entropy': cond_entropy,
        'mean_prob_true': np.mean(true_probs)
    }


def compute_bhattacharyya_distance(embeddings, labels):
    """
    Bhattacharyya Distance between class-conditional Gaussians (diagonal covariance)
    """
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)

    classes = np.unique(y)
    if len(classes) != 2:
        return np.nan

    class0 = X[y == classes[0]]
    class1 = X[y == classes[1]]

    mu0, mu1 = class0.mean(axis=0), class1.mean(axis=0)
    var0 = class0.var(axis=0) + 1e-10
    var1 = class1.var(axis=0) + 1e-10

    avg_var = (var0 + var1) / 2
    term1 = 0.125 * np.sum((mu0 - mu1) ** 2 / avg_var)
    term2 = 0.5 * np.sum(np.log(avg_var / np.sqrt(var0 * var1)))

    return term1 + term2


def compute_margin_score(embeddings, labels):
    """
    Margin-based transferability score (heuristic)

    Computes inter-class distance / intra-class spread ratio with a
    complexity penalty. NOT a PAC-Bayes bound - just a simple heuristic.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)

    n, d = X.shape
    classes = np.unique(y)

    centroids = np.array([X[y == c].mean(axis=0) for c in classes])

    inter_dist = np.mean(pdist(centroids, metric='euclidean'))

    intra_spread = 0.0
    for c in classes:
        class_data = X[y == c]
        if len(class_data) > 1:
            intra_spread += np.mean(pdist(class_data, metric='euclidean'))
    intra_spread /= len(classes)

    score = inter_dist / (intra_spread + 1e-10) - np.log(n) / n

    return score


# ==========================================
# SHESHA METRICS
# ==========================================

def shesha_rdm_bootstrap(embeddings, n_boot=50, subsample=0.25, metric="cosine",
                         corr_type="spearman", max_samples=1500, rng=None):
    """
    Canonical Shesha stability metric: RDM self-consistency across data resamples.

    Measures how stable the pairwise distance structure is when computed
    on different random subsets of the data.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    n = X.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X = X[idx]
        n = X.shape[0]

    m = int(n * subsample)

    rhos = []
    for _ in range(n_boot):
        idx1 = rng.choice(n, m, replace=False)
        idx2 = rng.choice(n, m, replace=False)

        rdm1 = pdist(X[idx1], metric=metric)
        rdm2 = pdist(X[idx2], metric=metric)

        if corr_type == "spearman":
            rho, _ = spearmanr(rdm1, rdm2)
        else:
            rho = np.corrcoef(rdm1, rdm2)[0, 1]

        if not np.isnan(rho):
            rhos.append(rho)

    return float(np.mean(rhos)) if rhos else np.nan


def shesha_split_half_dims(embeddings, n_splits=50, max_samples=1000, rng=None):
    """
    Shesha split-half (DIMENSION variant): RDM consistency across feature splits.

    Splits feature dimensions in half and checks if the two halves produce
    consistent distance structures.
    """
    n = len(embeddings)

    if rng is None:
        rng = np.random.default_rng()

    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X = embeddings[idx]
    else:
        X = embeddings

    d = X.shape[1]
    correlations = []

    for _ in range(n_splits):
        perm = rng.permutation(d)
        half = d // 2
        dims1, dims2 = perm[:half], perm[half:2*half]

        rdm1 = pdist(X[:, dims1], metric='correlation')
        rdm2 = pdist(X[:, dims2], metric='correlation')

        rho, _ = spearmanr(rdm1, rdm2)
        if not np.isnan(rho):
            correlations.append(rho)

    return np.mean(correlations) if correlations else np.nan


def shesha_anchor_distance_stability(embeddings, n_splits=50, n_anchors=100, n_per_split=200,
                                      max_samples=1500, metric="cosine", rank_normalize=True,
                                      rng=None):
    """
    Anchored distance profile stability metric.

    Measures consistency of distance profiles from fixed anchor points to
    random sample splits. This is NOT a same-items RDM split-half; rather,
    it tests whether the geometric structure (distances from anchors) is
    stable across different random subsets of the data.

    Method:
    1. Select a fixed set of anchor points R
    2. Randomly split remaining data into disjoint sets A and B
    3. Compute distance matrices D_A = dist(R, A) and D_B = dist(R, B)
    4. Optionally rank-normalize within each anchor row (more robust to outliers)
    5. Correlate flattened distance matrices

    Args:
        rank_normalize: If True, convert each anchor's distances to ranks before
                        correlating. This makes the metric more robust to anchor
                        outliers and more "Spearman-like" in spirit.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    n = X.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X = X[idx]
        n = X.shape[0]

    # Ensure we have enough samples
    min_required = n_anchors + 2 * n_per_split
    if n < min_required:
        scale = n / min_required * 0.9
        n_anchors = max(10, int(n_anchors * scale))
        n_per_split = max(20, int(n_per_split * scale))

    if n < n_anchors + 2 * n_per_split:
        return np.nan

    correlations = []

    for _ in range(n_splits):
        perm = rng.permutation(n)

        anchor_idx = perm[:n_anchors]
        a_idx = perm[n_anchors:n_anchors + n_per_split]
        b_idx = perm[n_anchors + n_per_split:n_anchors + 2 * n_per_split]

        # Distance matrices from anchors to each split
        D_A = cdist(X[anchor_idx], X[a_idx], metric=metric)  # n_anchors x n_per_split
        D_B = cdist(X[anchor_idx], X[b_idx], metric=metric)  # n_anchors x n_per_split

        if rank_normalize:
            # Rank-transform within each anchor row
            # This makes each anchor contribute equally and reduces outlier sensitivity
            D_A_ranked = np.zeros_like(D_A)
            D_B_ranked = np.zeros_like(D_B)
            for i in range(n_anchors):
                D_A_ranked[i] = rankdata(D_A[i])
                D_B_ranked[i] = rankdata(D_B[i])
            D_A = D_A_ranked
            D_B = D_B_ranked
            # Use Pearson after rank transform (cleaner than Spearman-on-ranks)
            rho, _ = pearsonr(D_A.flatten(), D_B.flatten())
        else:
            # Use Spearman on raw distances
            rho, _ = spearmanr(D_A.flatten(), D_B.flatten())

        if not np.isnan(rho):
            correlations.append(rho)

    return np.mean(correlations) if correlations else np.nan


def label_rdm_alignment(embeddings, labels, n_boot=50, subsample=0.5, rng=None):
    """
    Label-RDM alignment (NOT a stability metric)

    Measures alignment between model geometry and label structure.
    """
    n = len(embeddings)
    n_sub = int(n * subsample)

    if rng is None:
        rng = np.random.default_rng()

    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, n_sub, replace=False)
        X_sub = embeddings[idx]
        y_sub = labels[idx]

        d_model = pdist(X_sub, metric='cosine')
        d_label = pdist(y_sub.reshape(-1, 1), metric='hamming')

        rho, _ = spearmanr(d_model, d_label)
        if not np.isnan(rho):
            rhos.append(rho)

    return np.mean(rhos) if rhos else np.nan


def shesha_class_separation(embeddings, labels, n_boot=50, subsample=0.5, rng=None):
    """
    Class separation ratio (LABEL-INFORMED)
    """
    n = len(embeddings)
    n_sub = int(n * subsample)

    if rng is None:
        rng = np.random.default_rng()

    ratios = []
    for _ in range(n_boot):
        idx = rng.choice(n, n_sub, replace=False)
        X_sub = embeddings[idx]
        y_sub = labels[idx]

        class0 = X_sub[y_sub == 0]
        class1 = X_sub[y_sub == 1]

        if len(class0) < 2 or len(class1) < 2:
            continue

        within0 = pdist(class0, metric='cosine')
        within1 = pdist(class1, metric='cosine')
        within_mean = (np.mean(within0) + np.mean(within1)) / 2

        dists_matrix = cdist(class0, class1, metric='euclidean')
        between_mean = np.mean(dists_matrix)

        if within_mean > 0:
            ratios.append(between_mean / within_mean)

    return np.mean(ratios) if ratios else np.nan


def shesha_subspace_lda(embeddings, labels, n_boot=50, subsample=0.5, rng=None):
    """
    Subspace LDA stability (LABEL-INFORMED)
    """
    n = len(embeddings)
    n_sub = int(n * subsample)

    if rng is None:
        rng = np.random.default_rng()

    try:
        lda_ref = LinearDiscriminantAnalysis(n_components=1)
        lda_ref.fit(embeddings, labels)
        w_ref = lda_ref.coef_.flatten()
        w_ref = w_ref / np.linalg.norm(w_ref)
    except:
        return np.nan

    similarities = []
    for _ in range(n_boot):
        idx = rng.choice(n, n_sub, replace=False)
        X_sub = embeddings[idx]
        y_sub = labels[idx]

        if len(np.unique(y_sub)) < 2:
            continue

        try:
            lda_boot = LinearDiscriminantAnalysis(n_components=1)
            lda_boot.fit(X_sub, y_sub)
            w_boot = lda_boot.coef_.flatten()
            w_boot = w_boot / np.linalg.norm(w_boot)

            cos_sim = np.abs(np.dot(w_ref, w_boot))
            similarities.append(cos_sim)
        except:
            continue

    return np.mean(similarities) if similarities else np.nan


def compute_all_transferability_metrics(embeddings, labels, n_boot=50, subsample=0.5, seed=42):
    """
    Compute all transferability metrics for comparison.
    """
    results = {}

    # Shesha stability metrics (UNSUPERVISED)
    print("    Computing shesha_rdm_bootstrap (canonical)...")
    results['shesha_rdm_bootstrap'] = shesha_rdm_bootstrap(
        embeddings, n_boot, subsample=0.25, max_samples=1500,
        rng=np.random.default_rng(seed)
    )

    print("    Computing shesha_split_half_dims...")
    results['shesha_split_half_dims'] = shesha_split_half_dims(
        embeddings, n_boot, rng=np.random.default_rng(seed + 1)
    )

    print("    Computing shesha_anchor_stability...")
    results['shesha_anchor_stability'] = shesha_anchor_distance_stability(
        embeddings, n_boot, rank_normalize=True, rng=np.random.default_rng(seed + 2)
    )

    # Label-informed metrics (NOT stability)
    print("    Computing label_rdm_alignment...")
    results['label_rdm_alignment'] = label_rdm_alignment(
        embeddings, labels, n_boot, subsample, rng=np.random.default_rng(seed + 3)
    )

    print("    Computing shesha_class_sep...")
    results['shesha_class_sep'] = shesha_class_separation(
        embeddings, labels, n_boot, subsample, rng=np.random.default_rng(seed + 4)
    )

    print("    Computing shesha_subspace_lda...")
    results['shesha_subspace_lda'] = shesha_subspace_lda(
        embeddings, labels, n_boot, subsample, rng=np.random.default_rng(seed + 5)
    )

    # Baseline transferability metrics
    print("    Computing logme...")
    results['logme'] = LogME(regression=False).fit(embeddings, labels)


    print("    Computing centroid_softmax...")
    results['centroid_softmax'] = compute_centroid_softmax_score(embeddings, labels)

    print("    Computing hscore...")
    results['hscore'] = compute_hscore(embeddings, labels)

    print("    Computing fisher_discriminate...")
    results['fisher_discriminate'] = compute_fisher_discriminate(embeddings, labels)

    print("    Computing nce...")
    nce_output = compute_nce(embeddings, labels, temperature=1.0)
    results['nce'] = nce_output['nce']  # Extract just the score

    print("    Computing bhattacharyya_dist...")
    results['bhattacharyya_dist'] = compute_bhattacharyya_distance(embeddings, labels)

    print("    Computing margin_score...")
    results['margin_score'] = compute_margin_score(embeddings, labels)

    return results


def compute_unsupervised_metrics_on_pool(embeddings, n_boot=50, seed=42):
    """
    Compute unsupervised metrics on target pool (unlabeled data available at selection time).

    Note: These are computed on the training pool only, representing the unlabeled
    data available when selecting a model. This is the realistic setting for
    training-free model selection.
    """
    results = {}

    print("    Computing shesha_rdm_bootstrap (pool)...")
    results['shesha_rdm_bootstrap_pool'] = shesha_rdm_bootstrap(
        embeddings, n_boot, subsample=0.25, max_samples=1500,
        rng=np.random.default_rng(seed)
    )

    print("    Computing shesha_split_half_dims (pool)...")
    results['shesha_split_half_dims_pool'] = shesha_split_half_dims(
        embeddings, n_boot, rng=np.random.default_rng(seed + 1)
    )

    print("    Computing shesha_anchor_stability (pool)...")
    results['shesha_anchor_stability_pool'] = shesha_anchor_distance_stability(
        embeddings, n_boot, rank_normalize=True, rng=np.random.default_rng(seed + 2)
    )

    return results


# ==========================================
# Helper Functions
# ==========================================

def truncate_texts(texts, max_tokens=256):
    if max_tokens is None:
        return texts
    truncated = []
    for text in texts:
        words = text.split()
        if len(words) > max_tokens:
            truncated.append(" ".join(words[:max_tokens]))
        else:
            truncated.append(text)
    return truncated


def hash_texts(texts, max_tokens=None, dataset_id=""):
    hasher = hashlib.md5()
    hasher.update(str(len(texts)).encode())
    hasher.update(str(max_tokens).encode())
    hasher.update(dataset_id.encode())
    sample_indices = list(range(min(50, len(texts))))
    sample_indices += list(range(max(0, len(texts) - 50), len(texts)))
    sample_indices += list(range(0, len(texts), 100))
    sample_indices = sorted(set(sample_indices))
    for i in sample_indices:
        hasher.update(texts[i][:200].encode())
    return hasher.hexdigest()[:16]


def get_cache_key(model_name, text_hash, prefix="", max_tokens=None):
    model_key = model_name.replace("/", "_").replace("-", "_")
    tokens_key = f"_t{max_tokens}" if max_tokens else ""
    if prefix:
        return f"{prefix}_{model_key}{tokens_key}_{text_hash}.npy"
    return f"{model_key}{tokens_key}_{text_hash}.npy"


def get_embeddings_cached(model_name, texts, cache_prefix="", batch_size=64,
                          max_tokens=MAX_TEXT_LENGTH, dataset_id=""):
    texts_truncated = truncate_texts(texts, max_tokens)

    text_hash = hash_texts(texts_truncated, max_tokens, dataset_id)
    cache_key = get_cache_key(model_name, text_hash, cache_prefix, max_tokens)
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        print(f"    [CACHE HIT] {cache_key}")
        return np.load(cache_path)

    print(f"    [CACHE MISS] Computing embeddings...")
    model = None
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = SentenceTransformer(model_name, device=device)

        with torch.inference_mode():
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    embeddings = model.encode(
                        texts_truncated,
                        batch_size=batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
            else:
                embeddings = model.encode(
                    texts_truncated,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )

        embeddings = embeddings.astype(np.float32)
        np.save(cache_path, embeddings)
        print(f"    [CACHED] {cache_key}")

        return embeddings

    except Exception as e:
        print(f"    [ERROR] {e}")
        return None

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ==========================================
# Probe Functions
# ==========================================

def get_probes():
    """
    Returns dictionary of probe classifiers to evaluate.

    Note: Some probes (especially CalibratedClassifierCV with cv=3) may fail
    at very small k_total (e.g., 16) if cross-validation folds don't contain
    both classes. These failures are caught by try/except in evaluate_fewshot
    and the probe is simply skipped.
    """
    return {
        "logreg": [
            LogisticRegression(C=c, max_iter=1000, solver='lbfgs', random_state=42)
            for c in [0.01, 0.1, 1.0, 10.0]
        ],
        "ridge": [
            RidgeClassifier(alpha=a, random_state=42)
            for a in [0.1, 1.0, 10.0, 100.0]
        ],
        "sgd": [
            SGDClassifier(alpha=a, max_iter=1000, random_state=42)
            for a in [0.0001, 0.001, 0.01]
        ],
        "lda": [LinearDiscriminantAnalysis()],
        "nearest_centroid": [NearestCentroid()],
    }


def get_fixed_probe():
    return LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)


def evaluate_fewshot(X_train, y_train, X_val, y_val, X_test, y_test):
    probes = get_probes()

    best_val_acc = -1
    best_probe = None
    best_probe_name = None

    for name, probe_list in probes.items():
        for probe in probe_list:
            try:
                probe.fit(X_train, y_train)
                val_acc = accuracy_score(y_val, probe.predict(X_val))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_probe = probe
                    best_probe_name = name
            except:
                continue

    best_test_acc = None
    if best_probe is not None:
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        best_probe.fit(X_trainval, y_trainval)
        best_test_acc = accuracy_score(y_test, best_probe.predict(X_test))

    fixed_probe = get_fixed_probe()
    try:
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        fixed_probe.fit(X_trainval, y_trainval)
        fixed_test_acc = accuracy_score(y_test, fixed_probe.predict(X_test))
    except:
        fixed_test_acc = None

    return best_test_acc, best_probe_name, fixed_test_acc


# ==========================================
# Checkpoint Functions
# ==========================================

def get_checkpoint_file(seed, exp):
    return f"{OUTPUT_DIR}/checkpoint_{exp}_seed_{seed}.pkl"


def save_checkpoint(seed, exp, results, completed_models):
    with open(get_checkpoint_file(seed, exp), "wb") as f:
        pickle.dump({
            "results": results,
            "completed_models": completed_models
        }, f)


def load_checkpoint(seed, exp):
    checkpoint_file = get_checkpoint_file(seed, exp)
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            print(f"[CHECKPOINT] Resuming {exp} seed {seed} ({len(checkpoint['completed_models'])} models done)")
            return checkpoint
    except Exception as e:
        print(f"[CHECKPOINT] Could not load: {e}")
    return {"results": [], "completed_models": set()}


# ==========================================
# Load Data
# ==========================================

print("\n" + "="*70)
print("LOADING DATASETS")
print("="*70)

print("\nLoading IMDB...")
imdb = load_dataset("imdb")
imdb_texts_all = np.array(list(imdb["train"]["text"]) + list(imdb["test"]["text"]))
imdb_labels_all = np.array(list(imdb["train"]["label"]) + list(imdb["test"]["label"]))

np.random.seed(IMDB_SAMPLE_SEED)

imdb_shesha_idx_exp1 = np.random.choice(len(imdb_texts_all), N_IMDB_SHESHA_EXP1, replace=False)
imdb_shesha_texts_exp1 = [imdb_texts_all[i] for i in imdb_shesha_idx_exp1]
imdb_shesha_labels_exp1 = imdb_labels_all[imdb_shesha_idx_exp1]
print(f"  Exp1 IMDB: {len(imdb_shesha_texts_exp1)} samples")

imdb_pos_idx = np.where(imdb_labels_all == 1)[0]
imdb_neg_idx = np.where(imdb_labels_all == 0)[0]
n_per_class = N_IMDB_SHESHA_EXP2 // 2

imdb_pos_sample = np.random.choice(imdb_pos_idx, n_per_class, replace=False)
imdb_neg_sample = np.random.choice(imdb_neg_idx, n_per_class, replace=False)
imdb_shesha_idx_exp2 = np.concatenate([imdb_pos_sample, imdb_neg_sample])
np.random.shuffle(imdb_shesha_idx_exp2)
imdb_shesha_texts_exp2 = [imdb_texts_all[i] for i in imdb_shesha_idx_exp2]
imdb_shesha_labels_exp2 = imdb_labels_all[imdb_shesha_idx_exp2]
print(f"  Exp2 IMDB: {len(imdb_shesha_texts_exp2)} samples (balanced)")

del imdb, imdb_texts_all, imdb_labels_all, imdb_pos_idx, imdb_neg_idx
gc.collect()

print("\nLoading SST-2...")
sst2 = load_dataset("glue", "sst2")
sst2_train_texts = np.array(sst2["train"]["sentence"])
sst2_train_labels = np.array(sst2["train"]["label"])
sst2_test_texts = np.array(sst2["validation"]["sentence"])
sst2_test_labels = np.array(sst2["validation"]["label"])
print(f"  Train: {len(sst2_train_texts)}, Test: {len(sst2_test_texts)}")
del sst2
gc.collect()

print("\nLoading Yelp (sampling 6k)...")
yelp_full = load_dataset("yelp_polarity", split="train")
yelp_labels_full = np.array(yelp_full["label"])
yelp_pos_idx_all = np.where(yelp_labels_full == 1)[0]
yelp_neg_idx_all = np.where(yelp_labels_full == 0)[0]

np.random.seed(IMDB_SAMPLE_SEED)
yelp_sample_pos = np.random.choice(yelp_pos_idx_all, 3000, replace=False)
yelp_sample_neg = np.random.choice(yelp_neg_idx_all, 3000, replace=False)
yelp_sample_idx = np.concatenate([yelp_sample_pos, yelp_sample_neg])

yelp_texts_sampled = np.array([yelp_full["text"][int(i)] for i in yelp_sample_idx])
yelp_labels_sampled = yelp_labels_full[yelp_sample_idx]
print(f"  Sampled: {len(yelp_texts_sampled)}")

del yelp_full, yelp_labels_full, yelp_pos_idx_all, yelp_neg_idx_all
gc.collect()


# ==========================================
# Experiment 1: Few-shot SST-2
# ==========================================

def run_exp1(seed):
    print("\n" + "="*70)
    print(f"EXPERIMENT 1 - SEED {seed}")
    print("="*70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint = load_checkpoint(seed, "exp1")
    results = checkpoint["results"]
    completed_models = checkpoint["completed_models"]

    pool_texts, val_texts, pool_labels, val_labels = train_test_split(
        sst2_train_texts, sst2_train_labels,
        test_size=0.2, stratify=sst2_train_labels, random_state=seed
    )
    pool_labels_arr = np.array(pool_labels)

    fewshot_indices = {}
    for k_total in SAMPLE_SIZES_TOTAL:
        k_per_class = k_total // 2
        pos_idx = np.where(pool_labels_arr == 1)[0]
        neg_idx = np.where(pool_labels_arr == 0)[0]
        pos_sample = np.random.choice(pos_idx, k_per_class, replace=False)
        neg_sample = np.random.choice(neg_idx, k_per_class, replace=False)
        train_idx = np.concatenate([pos_sample, neg_sample])
        np.random.shuffle(train_idx)
        fewshot_indices[k_total] = train_idx

    for i, model_info in enumerate(MODELS):
        m_name = model_info["name"]
        capacity = model_info["capacity"]
        train_type = model_info["train_type"]

        if m_name in SKIP_MODELS:
            print(f"\n[{i+1}/{len(MODELS)}] {m_name} [SKIP - too large]")
            continue

        if m_name in completed_models:
            continue

        print(f"\n[{i+1}/{len(MODELS)}] {m_name}")
        model_start = time.time()

        print(f"  Encoding IMDB ({len(imdb_shesha_texts_exp1)} samples)...")
        emb_imdb = get_embeddings_cached(
            m_name, imdb_shesha_texts_exp1,
            cache_prefix="exp1_imdb_2k",
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TEXT_LENGTH,
            dataset_id="imdb_exp1"
        )

        if emb_imdb is None:
            print(f"  [SKIP] Failed - marking complete")
            completed_models.add(m_name)
            save_checkpoint(seed, "exp1", results, completed_models)
            continue

        emb_dim = emb_imdb.shape[1]

        print(f"  Computing transferability metrics on IMDB...")
        transfer_metrics = compute_all_transferability_metrics(
            emb_imdb, imdb_shesha_labels_exp1, N_BOOTSTRAP, subsample=0.5, seed=seed
        )

        for method, val in transfer_metrics.items():
            if val is not None and not np.isnan(val):
                print(f"    {method}: {val:.4f}")

        print("  Encoding SST-2...")
        emb_pool = get_embeddings_cached(
            m_name, list(pool_texts),
            cache_prefix="sst2_pool",
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TEXT_LENGTH,
            dataset_id="sst2_pool"
        )
        emb_val = get_embeddings_cached(
            m_name, list(val_texts),
            cache_prefix="sst2_val",
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TEXT_LENGTH,
            dataset_id="sst2_val"
        )
        emb_test = get_embeddings_cached(
            m_name, list(sst2_test_texts),
            cache_prefix="sst2_test",
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TEXT_LENGTH,
            dataset_id="sst2_test"
        )

        if emb_pool is None or emb_val is None or emb_test is None:
            print(f"  [SKIP] SST-2 failed - marking complete")
            completed_models.add(m_name)
            save_checkpoint(seed, "exp1", results, completed_models)
            continue

        print("  Computing unsupervised metrics on SST-2 pool...")
        pool_unsup_metrics = compute_unsupervised_metrics_on_pool(emb_pool, N_BOOTSTRAP, seed)

        print("  Few-shot:", end=" ")
        for k_total in SAMPLE_SIZES_TOTAL:
            train_idx = fewshot_indices[k_total]
            X_train = emb_pool[train_idx]
            y_train = pool_labels_arr[train_idx]

            best_acc, probe_name, fixed_acc = evaluate_fewshot(
                X_train, y_train, emb_val, val_labels, emb_test, sst2_test_labels
            )

            if best_acc is not None:
                print(f"k={k_total}:{best_acc:.3f}", end=" ")

                result = {
                    "seed": seed,
                    "model": m_name,
                    "capacity": capacity,
                    "train_type": train_type,
                    "emb_dim": emb_dim,
                    "k_total": k_total,
                    "k_per_class": k_total // 2,
                    "experiment": "few_shot_sst2",
                    "metric": "accuracy",
                    "value": best_acc,
                    "probe": probe_name,
                    "fixed_probe_acc": fixed_acc,
                }
                for method, val in transfer_metrics.items():
                    result[f"source_{method}"] = val
                for method, val in pool_unsup_metrics.items():
                    result[method] = val

                results.append(result)
        print()

        completed_models.add(m_name)
        if len(completed_models) % 5 == 0 or (i + 1) == len(MODELS):
            save_checkpoint(seed, "exp1", results, completed_models)
        print(f"  Time: {format_time(time.time() - model_start)}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_checkpoint(seed, "exp1", results, completed_models)

    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/exp1_seed_{seed}.csv", index=False)
    print(f"\nExp1 seed {seed} complete! Saved to exp1_seed_{seed}.csv")
    return df


# ==========================================
# Experiment 2: Yelp Transfer
# ==========================================

def run_exp2(seed):
    print("\n" + "="*70)
    print(f"EXPERIMENT 2 - SEED {seed}")
    print("="*70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint = load_checkpoint(seed, "exp2")
    results = checkpoint["results"]
    completed_models = checkpoint["completed_models"]

    yelp_pos_idx = np.where(yelp_labels_sampled == 1)[0]
    yelp_neg_idx = np.where(yelp_labels_sampled == 0)[0]

    yelp_pos_train = np.random.choice(yelp_pos_idx, 1200, replace=False)
    yelp_neg_train = np.random.choice(yelp_neg_idx, 1200, replace=False)
    yelp_train_idx = np.concatenate([yelp_pos_train, yelp_neg_train])

    remaining_pos = np.setdiff1d(yelp_pos_idx, yelp_pos_train)
    remaining_neg = np.setdiff1d(yelp_neg_idx, yelp_neg_train)

    yelp_pos_val = np.random.choice(remaining_pos, 300, replace=False)
    yelp_neg_val = np.random.choice(remaining_neg, 300, replace=False)
    yelp_val_idx = np.concatenate([yelp_pos_val, yelp_neg_val])

    remaining_pos = np.setdiff1d(remaining_pos, yelp_pos_val)
    remaining_neg = np.setdiff1d(remaining_neg, yelp_neg_val)

    yelp_pos_test = np.random.choice(remaining_pos, 500, replace=False)
    yelp_neg_test = np.random.choice(remaining_neg, 500, replace=False)
    yelp_test_idx = np.concatenate([yelp_pos_test, yelp_neg_test])

    yelp_train_texts = yelp_texts_sampled[yelp_train_idx]
    yelp_train_labels = yelp_labels_sampled[yelp_train_idx]
    yelp_val_texts = yelp_texts_sampled[yelp_val_idx]
    yelp_val_labels = yelp_labels_sampled[yelp_val_idx]
    yelp_test_texts = yelp_texts_sampled[yelp_test_idx]
    yelp_test_labels = yelp_labels_sampled[yelp_test_idx]

    for i, model_info in enumerate(MODELS):
        m_name = model_info["name"]
        capacity = model_info["capacity"]
        train_type = model_info["train_type"]

        if m_name in SKIP_MODELS:
            print(f"\n[{i+1}/{len(MODELS)}] {m_name} [SKIP - too large]")
            continue

        if m_name in completed_models:
            continue

        print(f"\n[{i+1}/{len(MODELS)}] {m_name}")
        model_start = time.time()

        print(f"  Encoding IMDB ({len(imdb_shesha_texts_exp2)} samples)...")
        emb_imdb = get_embeddings_cached(
            m_name, imdb_shesha_texts_exp2,
            cache_prefix="exp2_imdb_2k",
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TEXT_LENGTH,
            dataset_id="imdb_exp2"
        )

        if emb_imdb is None:
            print(f"  [SKIP] Failed - marking complete")
            completed_models.add(m_name)
            save_checkpoint(seed, "exp2", results, completed_models)
            continue

        emb_dim = emb_imdb.shape[1]

        print(f"  Computing transferability metrics on IMDB...")
        transfer_metrics = compute_all_transferability_metrics(
            emb_imdb, imdb_shesha_labels_exp2, N_BOOTSTRAP, subsample=0.5, seed=seed
        )

        for method, val in transfer_metrics.items():
            if val is not None and not np.isnan(val):
                print(f"    {method}: {val:.4f}")

        print("  Encoding Yelp...")
        emb_yelp_train = get_embeddings_cached(
            m_name, list(yelp_train_texts),
            cache_prefix="yelp_train",
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TEXT_LENGTH,
            dataset_id="yelp_train"
        )
        emb_yelp_val = get_embeddings_cached(
            m_name, list(yelp_val_texts),
            cache_prefix="yelp_val",
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TEXT_LENGTH,
            dataset_id="yelp_val"
        )
        emb_yelp_test = get_embeddings_cached(
            m_name, list(yelp_test_texts),
            cache_prefix="yelp_test",
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TEXT_LENGTH,
            dataset_id="yelp_test"
        )

        if emb_yelp_train is None or emb_yelp_val is None or emb_yelp_test is None:
            print(f"  [SKIP] Yelp failed - marking complete")
            completed_models.add(m_name)
            save_checkpoint(seed, "exp2", results, completed_models)
            continue

        print("  Computing unsupervised metrics on Yelp pool...")
        pool_unsup_metrics = compute_unsupervised_metrics_on_pool(emb_yelp_train, N_BOOTSTRAP, seed)

        best_acc, probe_name, fixed_acc = evaluate_fewshot(
            emb_yelp_train, yelp_train_labels,
            emb_yelp_val, yelp_val_labels,
            emb_yelp_test, yelp_test_labels
        )

        if best_acc is not None:
            print(f"  Acc: {best_acc:.3f} ({probe_name}), Fixed: {fixed_acc:.3f}")

            result = {
                "seed": seed,
                "model": m_name,
                "capacity": capacity,
                "train_type": train_type,
                "emb_dim": emb_dim,
                "experiment": "yelp_transfer",
                "metric": "accuracy",
                "value": best_acc,
                "probe": probe_name,
                "fixed_probe_acc": fixed_acc,
            }
            for method, val in transfer_metrics.items():
                result[f"source_{method}"] = val
            for method, val in pool_unsup_metrics.items():
                result[method] = val

            results.append(result)

        completed_models.add(m_name)
        if len(completed_models) % 5 == 0 or (i + 1) == len(MODELS):
            save_checkpoint(seed, "exp2", results, completed_models)
        print(f"  Time: {format_time(time.time() - model_start)}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_checkpoint(seed, "exp2", results, completed_models)

    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/exp2_seed_{seed}.csv", index=False)
    print(f"\nExp2 seed {seed} complete! Saved to exp2_seed_{seed}.csv")
    return df


# ==========================================
# Analysis Functions
# ==========================================

def bootstrap_correlation_by_unit(df, metric_col, target_col, unit_cols=['model', 'seed'],
                                   n_boot=1000, seed=42):
    """Bootstrap over (model, seed) units, not individual rows."""
    rng = np.random.default_rng(seed)

    units = df[unit_cols].drop_duplicates()

    unit_data = []
    for _, unit in units.iterrows():
        mask = (df[unit_cols] == unit).all(axis=1)
        subset = df[mask]
        if len(subset) > 0 and not subset[metric_col].isna().all():
            unit_data.append({
                'metric': subset[metric_col].mean(),
                'target': subset[target_col].mean()
            })

    if len(unit_data) < 5:
        return np.nan, np.nan, np.nan

    unit_df = pd.DataFrame(unit_data)
    x = unit_df['metric'].values
    y = unit_df['target'].values

    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        rho, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(rho)

    if len(rhos) < 10:
        return np.nan, np.nan, np.nan

    rhos = np.array(rhos)
    return np.mean(rhos), np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)


def get_metric_type(metric_name):
    """Correct ordering - check label-informed first."""
    if 'pool' in metric_name:
        return "Pool-unsup"
    elif ('label' in metric_name) or ('class' in metric_name) or ('lda' in metric_name):
        return "Label-inform"
    elif ('shesha' in metric_name) or ('rdm' in metric_name) or ('split' in metric_name) or ('anchor' in metric_name):
        return "Shesha-unsup"
    else:
        return "Baseline"


def analyze_transferability_metrics(df, target_col='value'):
    """Compare all transferability metrics with proper bootstrap CIs."""
    source_metrics = [
        'source_shesha_rdm_bootstrap', 'source_shesha_split_half_dims',
        'source_shesha_anchor_stability', 'source_label_rdm_alignment',
        'source_shesha_class_sep', 'source_shesha_subspace_lda',
        'source_logme', 'source_centroid_softmax', 'source_hscore',
        'source_nce', 'source_bhattacharyya_dist', 'source_margin_score'
    ]
    pool_metrics = [
        'shesha_rdm_bootstrap_pool', 'shesha_split_half_dims_pool',
        'shesha_anchor_stability_pool'
    ]

    all_metrics = source_metrics + pool_metrics

    print("\n" + "="*70)
    print("TRANSFERABILITY METRIC COMPARISON")
    print("="*70)

    print(f"\n{'Metric':<40} | {'rho':<8} | {'95% CI':<20} | {'Type'}")
    print("-" * 85)

    results = []
    for metric in all_metrics:
        if metric in df.columns:
            valid = df.dropna(subset=[metric, target_col])
            if len(valid) > 5:
                rho, p = spearmanr(valid[metric], valid[target_col])

                rho_mean, ci_lo, ci_hi = bootstrap_correlation_by_unit(
                    valid, metric, target_col
                )

                metric_type = get_metric_type(metric)

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "[NA]"
                print(f"{metric:<40} | {rho:<8.3f} | {ci_str:<20} | {metric_type} {sig}")

                results.append({
                    'metric': metric,
                    'rho': rho,
                    'rho_boot': rho_mean,
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'p': p,
                    'type': metric_type
                })

    return pd.DataFrame(results)


def partial_correlation_rank_based(df, target_col='value'):
    """Rank-based partial correlation (consistent with Spearman)."""
    print("\n" + "="*70)
    print("PARTIAL CORRELATION ANALYSIS (rank-based, controlling for emb_dim)")
    print("="*70)

    if 'emb_dim' not in df.columns:
        print("No emb_dim column found, skipping partial correlations")
        return

    from sklearn.linear_model import LinearRegression

    metrics = ['source_shesha_rdm_bootstrap', 'source_logme', 'source_hscore',
               'source_margin_score', 'source_nce']

    for metric in metrics:
        if metric not in df.columns:
            continue

        valid = df.dropna(subset=[metric, target_col, 'emb_dim'])
        if len(valid) < 10:
            continue

        metric_ranks = rankdata(valid[metric].values)
        target_ranks = rankdata(valid[target_col].values)
        dim_ranks = rankdata(np.log(valid['emb_dim'].values))

        rho_raw, p_raw = spearmanr(valid[metric], valid[target_col])

        reg_metric = LinearRegression().fit(dim_ranks.reshape(-1, 1), metric_ranks)
        resid_metric = metric_ranks - reg_metric.predict(dim_ranks.reshape(-1, 1))

        reg_target = LinearRegression().fit(dim_ranks.reshape(-1, 1), target_ranks)
        resid_target = target_ranks - reg_target.predict(dim_ranks.reshape(-1, 1))

        rho_partial, p_partial = pearsonr(resid_metric, resid_target)

        print(f"{metric}:")
        print(f"  Raw Spearman:     rho={rho_raw:.3f}, p={p_raw:.4f}")
        print(f"  Partial (rank):   rho={rho_partial:.3f}, p={p_partial:.4f}")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    total_start = time.time()

    for seed in SEEDS:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if RUN_CONFIG.get((seed, "exp1"), False):
            run_exp1(seed)
        else:
            print(f"\n[SKIP] Exp1 seed {seed}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if RUN_CONFIG.get((seed, "exp2"), False):
            run_exp2(seed)
        else:
            print(f"\n[SKIP] Exp2 seed {seed}")

    # ==========================================
    # Combine and Analyze Results
    # ==========================================
    print("\n" + "="*70)
    print("COMBINING AND ANALYZING RESULTS")
    print("="*70)

    exp1_files = [f"{OUTPUT_DIR}/exp1_seed_{s}.csv" for s in SEEDS if os.path.exists(f"{OUTPUT_DIR}/exp1_seed_{s}.csv")]
    if exp1_files:
        combined_exp1 = pd.concat([pd.read_csv(f) for f in exp1_files], ignore_index=True)
        combined_exp1.to_csv(f"{OUTPUT_DIR}/exp1_all_seeds.csv", index=False)
        print(f"\nCombined Exp1: {len(combined_exp1)} rows")

        for k in SAMPLE_SIZES_TOTAL:
            df_k = combined_exp1[combined_exp1['k_total'] == k]
            if len(df_k) > 5:
                print(f"\n--- Few-shot k_total={k} (k_per_class={k//2}) ---")
                analyze_transferability_metrics(df_k)

        df_64 = combined_exp1[combined_exp1['k_total'] == 64]
        if len(df_64) > 10:
            partial_correlation_rank_based(df_64)

    exp2_files = [f"{OUTPUT_DIR}/exp2_seed_{s}.csv" for s in SEEDS if os.path.exists(f"{OUTPUT_DIR}/exp2_seed_{s}.csv")]
    if exp2_files:
        combined_exp2 = pd.concat([pd.read_csv(f) for f in exp2_files], ignore_index=True)
        combined_exp2.to_csv(f"{OUTPUT_DIR}/exp2_all_seeds.csv", index=False)
        print(f"\nCombined Exp2: {len(combined_exp2)} rows")

        print("\n--- Yelp Transfer ---")
        analyze_transferability_metrics(combined_exp2)
        partial_correlation_rank_based(combined_exp2)

    print(f"\nTotal time: {format_time(time.time() - total_start)}")
    print("Done!")