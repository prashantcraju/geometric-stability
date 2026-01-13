"""
Shesha Drift - Canary Validation Extended: Quantization & LoRA Perturbations

Based on Canary Validation but replaces Gaussian noise with structured perturbations.

Key features:
- Quantization via bitsandbytes (INT8, INT4)
- LoRA via peft library (ranks 1-64)
- Uses CausalLM models (not sentence transformers) for compatibility
- L2 normalized embeddings for consistent cosine RDM
- Resume-safe reproducibility via stable per-(model, perturbation) seeding
"""

import os
import gc
import shutil
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pathlib import Path
import warnings

# Authenticate with Hugging Face
from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if token:
    login(token)
else:
    print("Set HF_TOKEN environment variable")

warnings.filterwarnings("ignore")

# Check for bitsandbytes (quantization)
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
    print(">>> bitsandbytes available")
except ImportError:
    HAS_BITSANDBYTES = False
    print(">>> bitsandbytes NOT available - quantization experiments will be skipped")

# Check for peft (LoRA)
try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_PEFT = True
    print(">>> peft available")
except ImportError:
    HAS_PEFT = False
    print(">>> peft NOT available - LoRA experiments will be skipped")

# --- CONFIGURATION ---
RANDOM_STATE = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set all seeds for reproducibility
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    DTYPE = torch.bfloat16
    BATCH_SIZE = 256
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {total_mem:.1f}GB")
else:
    DTYPE = torch.float32
    BATCH_SIZE = 8

print(f"Device: {DEVICE}, Dtype: {DTYPE}, Batch: {BATCH_SIZE}")

OUTPUT_DIR = Path("./shesha-drift")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Experiment Settings
N_SAMPLES = 800
MAX_SEQ_LEN = 128
LAYER = -1  # Layer to extract embeddings from

# Quantization levels to test
QUANT_CONFIGS = [
    {"name": "fp16", "bits": None},      # Baseline (no quantization)
    {"name": "int8", "bits": 8},          # INT8 quantization
    {"name": "int4_nf4", "bits": 4},      # INT4 NF4 quantization
]

# LoRA ranks to test (simulating different adaptation magnitudes)
# We'll also vary the initialization scale to simulate "amount of fine-tuning"
LORA_CONFIGS = [
    {"rank": 0, "alpha": 0, "init_scale": 0.0},      # Baseline (no LoRA)
    {"rank": 1, "alpha": 2, "init_scale": 0.01},     # Minimal rank
    {"rank": 2, "alpha": 4, "init_scale": 0.01},
    {"rank": 4, "alpha": 8, "init_scale": 0.01},
    {"rank": 8, "alpha": 16, "init_scale": 0.01},    # Standard
    {"rank": 16, "alpha": 32, "init_scale": 0.01},
    {"rank": 32, "alpha": 64, "init_scale": 0.01},
    {"rank": 64, "alpha": 128, "init_scale": 0.01},  # High rank
    # Also test varying init scales at fixed rank
    {"rank": 8, "alpha": 16, "init_scale": 0.001},   # Small init
    {"rank": 8, "alpha": 16, "init_scale": 0.05},    # Medium init
    {"rank": 8, "alpha": 16, "init_scale": 0.1},     # Large init
]

# Gaussian noise levels (as fraction of parameter std)
GAUSSIAN_NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# Known causal LM model types for padding side detection
CAUSAL_TYPES = {
    "llama", "mistral", "falcon", "gpt2", "gpt_neo", "gpt_neox", "bloom", "opt",
    "qwen", "qwen2", "gemma", "gemma2", "phi", "stablelm", "mpt", "pythia",
    "tinyllama", "smollm", "starcoder", "codegen", "cohere", "dbrx"
}


# --- CLEANUP FUNCTIONS ---
def force_delete_model(model_id):
    """Delete model from HF cache."""
    try:
        cache_root = "/root/.cache/huggingface/hub"
        if not os.path.exists(cache_root):
            return

        safe_name = model_id.replace("/", "--")
        patterns = [
            f"models--{safe_name}",
            f"sentence-transformers--{safe_name}",
        ]

        for item in os.listdir(cache_root):
            for pattern in patterns:
                if item == pattern or item.startswith(pattern + "."):
                    path = os.path.join(cache_root, item)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"   [Cleanup] Removed: {item}")
    except Exception as e:
        print(f"   [Cleanup Error] {model_id}: {e}")


def deep_cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def mem_info():
    if DEVICE == "cuda":
        return f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
    return "CPU"


# --- METRICS ---
def rdm_spearman_from_clean(rdm_clean, Y):
    """Compute RDM Spearman using precomputed clean RDM."""
    ry = pdist(Y, metric='cosine')
    rho = spearmanr(rdm_clean, ry).correlation
    return float(rho) if np.isfinite(rho) else 0.0


def rdm_pearson_from_clean(rdm_clean, Y):
    """Compute RDM Pearson using precomputed clean RDM."""
    ry = pdist(Y, metric='cosine')
    r, _ = pearsonr(rdm_clean, ry)
    return float(r) if np.isfinite(r) else 0.0

def procrustes_similarity(X, Y):
    """Procrustes similarity using SciPy's procrustes with robust numerical checks."""
    try:
        # Convert to float64 for better numerical stability
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        # Check for NaN/Inf values
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            print("      Warning: NaN values in procrustes input")
            return float('nan')
        if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
            print("      Warning: Inf values in procrustes input")
            return float('nan')

        # Check for all-zero or constant columns
        X_std = X.std(axis=0)
        Y_std = Y.std(axis=0)
        if np.any(X_std < 1e-12) or np.any(Y_std < 1e-12):
            print("      Warning: Low variance columns in procrustes input")
            # Add small noise to break degeneracy
            rng = np.random.default_rng(42)
            X = X + rng.normal(0, 1e-8, X.shape)
            Y = Y + rng.normal(0, 1e-8, Y.shape)

        # Center the data
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        # Check if matrices are degenerate (all points identical after centering)
        X_norm = np.linalg.norm(X_centered, 'fro')
        Y_norm = np.linalg.norm(Y_centered, 'fro')

        if X_norm < 1e-12 or Y_norm < 1e-12:
            print("      Warning: Degenerate matrices in procrustes (norm too small)")
            return float('nan')

        # Scale to unit Frobenius norm
        X_scaled = X_centered / X_norm
        Y_scaled = Y_centered / Y_norm

        # Use try-except for SVD convergence issues
        try:
            # Use orthogonal_procrustes instead of scipy.procrustes for better stability
            R, scale = orthogonal_procrustes(X_scaled, Y_scaled)
        except np.linalg.LinAlgError as e:
            if "SVD did not converge" in str(e):
                print(f"      Warning: SVD did not converge, using fallback")
                # Fallback: compute similarity via correlation of principal angles
                Ux, _, Vx = np.linalg.svd(X_scaled, full_matrices=False)
                Uy, _, Vy = np.linalg.svd(Y_scaled, full_matrices=False)
                cos_angles = np.abs(np.diag(Ux.T @ Uy))
                return float(np.mean(cos_angles))
            else:
                raise

        # Compute distance and convert to similarity
        dist = np.linalg.norm(X_scaled @ R - Y_scaled, 'fro')

        # Theoretical maximum distance for unit-norm matrices is sqrt(2)
        similarity = 1 - dist / np.sqrt(2)

        # Ensure the result is valid
        similarity = np.clip(similarity, 0, 1)

        # Check if result is reasonable
        if not np.isfinite(similarity):
            print(f"      Warning: Non-finite procrustes similarity: {similarity}")
            return float('nan')

        return float(similarity)

    except ValueError as e:
        if "must contain >1 unique points" in str(e):
            print("      Warning: Degenerate case in procrustes (insufficient unique points)")
            return float('nan')
        print(f"      Warning: ValueError in procrustes: {e}")
        return float('nan')
    except np.linalg.LinAlgError as e:
        print(f"      Warning: LinAlgError in procrustes: {e}")
        return float('nan')
    except Exception as e:
        print(f"      Warning: Unexpected error in procrustes: {e}")
        return float('nan')

def cka_debiased(X, Y):
    """Cleaner implementation of debiased CKA."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    # Center the data
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    n = X.shape[0]
    if n < 4:
        return 0.0
    
    # Center kernel helper
    def center_gram_matrix(G):
        """Center a Gram matrix: H @ G @ H"""
        row_means = G.mean(axis=1, keepdims=True)
        col_means = G.mean(axis=0, keepdims=True)
        grand_mean = G.mean()
        return G - row_means - col_means + grand_mean
    
    # Compute and center Gram matrices
    K = center_gram_matrix(X @ X.T)
    L = center_gram_matrix(Y @ Y.T)
    
    # Zero diagonals for debiasing terms
    K_no_diag = K.copy()
    L_no_diag = L.copy()
    np.fill_diagonal(K_no_diag, 0)
    np.fill_diagonal(L_no_diag, 0)
    
    # Debiased HSIC estimator (Kornblith et al., 2019)
    hsic = (np.sum(K * L) 
            + (np.sum(K_no_diag) * np.sum(L_no_diag)) / ((n-1)*(n-2))
            - 2 * np.sum(np.sum(K_no_diag, axis=1) * np.sum(L_no_diag, axis=1)) / (n-2)
           ) / (n * (n-3))
    
    # Self-HSIC for normalization
    hsic_xx = (np.sum(K * K) 
               + np.sum(K_no_diag)**2 / ((n-1)*(n-2))
               - 2 * np.sum(np.sum(K_no_diag, axis=1)**2) / (n-2)
              ) / (n * (n-3))
    
    hsic_yy = (np.sum(L * L) 
               + np.sum(L_no_diag)**2 / ((n-1)*(n-2))
               - 2 * np.sum(np.sum(L_no_diag, axis=1)**2) / (n-2)
              ) / (n * (n-3))
    
    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0
    
    return hsic / np.sqrt(hsic_xx * hsic_yy)


def sliced_wasserstein(X, Y, n_proj=50, seed=320):
    """Sliced Wasserstein distance."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((X.shape[1], n_proj))
    dirs /= np.linalg.norm(dirs, axis=0)
    Xp = np.sort(X @ dirs, axis=0)
    Yp = np.sort(Y @ dirs, axis=0)
    return float(np.mean(np.abs(Xp - Yp)))


def subspace_overlap(X, Y, k=10):
    """
    Compute subspace overlap between top-k principal components.
    Returns the mean squared cosine similarity between subspaces.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
        Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
        
        k = min(k, Ux.shape[1], Uy.shape[1])
        Ux_k = Ux[:, :k]
        Uy_k = Uy[:, :k]
        
        overlap_matrix = Ux_k.T @ Uy_k
        overlap = np.sum(overlap_matrix ** 2) / k
        
        return float(np.clip(overlap, 0, 1))
    except Exception as e:
        return float('nan')


def eigenspectrum_similarity(X, Y):
    """
    Compare eigenspectrum (singular value) distributions.
    Returns cosine similarity between normalized singular value vectors.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        Sx = np.linalg.svd(X, compute_uv=False)
        Sy = np.linalg.svd(Y, compute_uv=False)
        
        Sx_norm = Sx / (np.linalg.norm(Sx) + 1e-12)
        Sy_norm = Sy / (np.linalg.norm(Sy) + 1e-12)
        
        max_len = max(len(Sx_norm), len(Sy_norm))
        Sx_padded = np.zeros(max_len)
        Sy_padded = np.zeros(max_len)
        Sx_padded[:len(Sx_norm)] = Sx_norm
        Sy_padded[:len(Sy_norm)] = Sy_norm
        
        similarity = np.dot(Sx_padded, Sy_padded)
        
        return float(np.clip(similarity, 0, 1))
    except Exception as e:
        return float('nan')


def effective_rank(X, use_log2=False):
    """
    Compute effective rank using spectral ENTROPY of ENERGY (normalized squared singular values).
    Robust measure of effective dimensionality.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - X.mean(axis=0, keepdims=True)
        
        # SVD
        s = np.linalg.svd(X_centered, compute_uv=False)
        
        # --- CRITICAL: Use Energy (Eigenvalues), not Magnitude (Singular Values) ---
        # This aligns with PCA variance explained and Participation Ratio
        eigenvalues = s ** 2
        
        # Filter numerical noise
        ev = eigenvalues[eigenvalues > 1e-12]
        
        if len(ev) == 0:
            return 0.0
            
        # Normalize to probability distribution
        p = ev / ev.sum()
        
        # Entropy
        if use_log2:
            h = -np.sum(p * np.log2(p))
            erank = 2 ** h
        else:
            h = -np.sum(p * np.log(p))
            erank = np.exp(h)
            
        return float(erank)
        
    except Exception as e:
        print(f"Warning: effective_rank error: {e}")
        return float('nan')

def effective_rank_ratio(X, Y, use_log2=False):
    """
    Compute the ratio of effective ranks.
    Returns: float in [0, 1]
    1.0 = Identical effective dimensionality
    <1.0 = Rank collapse or expansion
    """
    try:
        r_x = effective_rank(X, use_log2)
        r_y = effective_rank(Y, use_log2)
        
        if r_x == 0 or r_y == 0:
            return 0.0
            
        # Ratio is always <= 1.0
        return min(r_x, r_y) / max(r_x, r_y)
        
    except Exception as e:
        return float('nan')



def eigenspectrum_js_divergence(X, Y, epsilon=1e-10):
    """
    Jensen-Shannon divergence between eigenspectrum distributions.
    Lower values indicate more similar spectra.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        Sx = np.linalg.svd(X, compute_uv=False) ** 2
        Sy = np.linalg.svd(Y, compute_uv=False) ** 2
        
        Px = Sx / (Sx.sum() + epsilon)
        Py = Sy / (Sy.sum() + epsilon)
        
        max_len = max(len(Px), len(Py))
        Px_padded = np.full(max_len, epsilon)
        Py_padded = np.full(max_len, epsilon)
        Px_padded[:len(Px)] = Px + epsilon
        Py_padded[:len(Py)] = Py + epsilon
        
        Px_padded = Px_padded / Px_padded.sum()
        Py_padded = Py_padded / Py_padded.sum()
        
        M = 0.5 * (Px_padded + Py_padded)
        kl_pm = np.sum(Px_padded * np.log(Px_padded / M))
        kl_qm = np.sum(Py_padded * np.log(Py_padded / M))
        js_divergence = 0.5 * (kl_pm + kl_qm)
        
        return float(js_divergence)
    except Exception as e:
        return float('nan')


def participation_ratio(X):
    """
    Compute participation ratio (effective dimensionality) of a representation.
    PR = (sum of eigenvalues)^2 / sum of eigenvalues^2
    """
    try:
        X = np.asarray(X, dtype=np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        
        S = np.linalg.svd(X, compute_uv=False) ** 2
        pr = (S.sum() ** 2) / (np.sum(S ** 2) + 1e-12)
        
        return float(pr)
    except Exception as e:
        return float('nan')


def participation_ratio_similarity(X, Y):
    """
    Compare participation ratios of two representations.
    """
    try:
        pr_x = participation_ratio(X)
        pr_y = participation_ratio(Y)
        
        if not np.isfinite(pr_x) or not np.isfinite(pr_y):
            return float('nan')
        
        max_pr = max(pr_x, pr_y)
        if max_pr < 1e-12:
            return 1.0
        
        similarity = 1 - abs(pr_x - pr_y) / max_pr
        return float(np.clip(similarity, 0, 1))
    except Exception as e:
        return float('nan')


def compute_drift_metrics(rdm_clean, emb_clean, emb_noisy, seed=320):
    """Compute all drift metrics using precomputed clean RDM."""
    X = np.asarray(emb_clean, dtype=np.float64)
    Y = np.asarray(emb_noisy, dtype=np.float64)
    
    metrics = {
        'shesha': 1.0 - rdm_spearman_from_clean(rdm_clean, Y),
        'rdm_pearson': 1.0 - rdm_pearson_from_clean(rdm_clean, Y),
        'cka_debiased': 1.0 - cka_debiased(X, Y),
        'procrustes': 1.0 - procrustes_similarity(X, Y),
        'wasserstein': sliced_wasserstein(X, Y, seed=seed),
        # New metrics
        'subspace_overlap_k5': subspace_overlap(X, Y, k=5),
        'subspace_overlap_k10': subspace_overlap(X, Y, k=10),
        'subspace_overlap_k20': subspace_overlap(X, Y, k=20),
        'effective_rank': effective_rank(X),
        'effective_rank_ratio': effective_rank_ratio(X, Y),
        'eigenspectrum_sim': eigenspectrum_similarity(X, Y),
        'eigenspectrum_js': eigenspectrum_js_divergence(X, Y),
        'pr_x': participation_ratio(X),
        'pr_y': participation_ratio(Y),
        'participation_ratio_sim': participation_ratio_similarity(X, Y),
    }
    
    # Convert overlaps to drift (1 - similarity)
    metrics['subspace_drift_k5'] = 1.0 - metrics['subspace_overlap_k5'] if np.isfinite(metrics['subspace_overlap_k5']) else np.nan
    metrics['subspace_drift_k10'] = 1.0 - metrics['subspace_overlap_k10'] if np.isfinite(metrics['subspace_overlap_k10']) else np.nan
    metrics['subspace_drift_k20'] = 1.0 - metrics['subspace_overlap_k20'] if np.isfinite(metrics['subspace_overlap_k20']) else np.nan
    metrics['eigenspectrum_drift'] = 1.0 - metrics['eigenspectrum_sim'] if np.isfinite(metrics['eigenspectrum_sim']) else np.nan
    metrics['pr_drift'] = 1.0 - metrics['participation_ratio_sim'] if np.isfinite(metrics['participation_ratio_sim']) else np.nan
    
    return metrics


def evaluate_accuracy(X, y):
    """Evaluate classification accuracy with cross-validation."""
    clf = LogisticRegression(solver='liblinear', max_iter=200, random_state=RANDOM_STATE)
    return float(np.mean(cross_val_score(clf, X, y, cv=5, scoring='accuracy')))


# --- L2 NORMALIZATION ---
def l2_normalize(emb):
    """L2 normalize embeddings to unit norm."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return emb / norms


# --- MODEL LOADING ---
def load_model_fp16(model_name):
    """Load model in FP16/BF16 (baseline)."""
    print(f"   Loading {model_name} [FP16] [{mem_info()}]")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    
    # Set padding side
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type in CAUSAL_TYPES:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"
    
    return model, tokenizer


def load_model_quantized(model_name, bits=8):
    """Load model with INT8 or INT4 quantization."""
    if not HAS_BITSANDBYTES:
        return None, None
    
    print(f"   Loading {model_name} [INT{bits}] [{mem_info()}]")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_type="nf4"
        )
    else:
        raise ValueError(f"Unsupported bits: {bits}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()
    
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type in CAUSAL_TYPES:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"
    
    return model, tokenizer


def load_model_with_lora(model_name, lora_rank, lora_alpha, init_scale, seed=320):
    """Load model and apply LoRA adapter with random initialization."""
    if not HAS_PEFT:
        return None, None
    
    if lora_rank == 0:
        # No LoRA, just return base model
        return load_model_fp16(model_name)
    
    print(f"   Loading {model_name} [LoRA r={lora_rank}, scale={init_scale}] [{mem_info()}]")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    
    # Determine target modules based on model type
    model_type = getattr(model.config, "model_type", "").lower()
    
    # Common projection names across architectures
    if model_type in ["llama", "mistral", "qwen", "qwen2", "gemma", "gemma2"]:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif model_type in ["gpt2", "gpt_neo", "gpt_neox"]:
        target_modules = ["c_attn", "c_proj"]
    elif model_type in ["bloom"]:
        target_modules = ["query_key_value", "dense"]
    elif model_type in ["falcon"]:
        target_modules = ["query_key_value", "dense"]
    elif model_type in ["opt"]:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    elif model_type in ["phi"]:
        target_modules = ["q_proj", "v_proj", "k_proj", "dense"]
    else:
        # Fallback to common names
        target_modules = ["q_proj", "v_proj"]
    
    try:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        
        # Initialize LoRA weights with controlled random values
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_" in name:
                    if "lora_A" in name:
                        # Kaiming-like init scaled down
                        fan_in = param.shape[1] if len(param.shape) > 1 else param.shape[0]
                        std = np.sqrt(2.0 / fan_in)
                        param.data = torch.randn_like(param) * std
                    elif "lora_B" in name:
                        # Initialize B with small random (not zero, to create perturbation)
                        param.data = torch.randn_like(param) * init_scale
        
        model.eval()
        
    except Exception as e:
        print(f"      LoRA application failed: {e}")
        # Fallback: return base model without LoRA
        return load_model_fp16(model_name)
    
    if model_type in CAUSAL_TYPES:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"
    
    return model, tokenizer


def inject_gaussian_noise(model, alpha, seed=320):
    """
    Inject Gaussian noise into model parameters.
    alpha: noise level as fraction of each parameter's std
    """
    if alpha == 0:
        return
    
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.numel() > 1:
                std = float(p.std().item())
                if std > 0:
                    noise = torch.randn_like(p) * (std * alpha)
                    p.add_(noise)


def get_model_state_dict(model):
    """Get a copy of model state dict on CPU."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def load_model_state_dict(model, state_dict):
    """Load state dict back into model."""
    device_state = {k: v.to(DEVICE) for k, v in state_dict.items()}
    model.load_state_dict(device_state)


# --- EMBEDDING ---
def get_embeddings(model, tokenizer, texts, layer=LAYER):
    """Get embeddings using mean pooling of hidden states, L2 normalized."""
    model.eval()
    all_vecs = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN
        ).to(DEVICE)
        
        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast(enabled=True, dtype=DTYPE):
                    out = model(**inputs, output_hidden_states=True, return_dict=True)
            else:
                out = model(**inputs, output_hidden_states=True, return_dict=True)
            
            # Get specified layer
            h = out.hidden_states[layer]
            
            # Mean pooling over non-padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1)
            vecs = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_vecs.append(vecs.float().cpu().numpy())
        
        del inputs, out, h, vecs
    
    emb = np.vstack(all_vecs)
    return l2_normalize(emb)


# --- MODELS ---
# Using decoder-only models for quantization/LoRA compatibility
MODELS = [
    # Small models (fast iteration)
    ("HuggingFaceTB/SmolLM-135M", 0.135),
    ("HuggingFaceTB/SmolLM2-135M", 0.135),
    ("HuggingFaceTB/SmolLM-360M", 0.36),
    ("HuggingFaceTB/SmolLM2-360M", 0.36),
    ("Qwen/Qwen2-0.5B", 0.5),
    # 1B models
    ("meta-llama/Llama-3.2-1B", 1.0),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", 1.1),
    ("Qwen/Qwen2-1.5B", 1.5),
    ("stabilityai/stablelm-2-1_6b", 1.6),
    ("HuggingFaceTB/SmolLM-1.7B", 1.7),
    # 2-3B models
    ("google/gemma-2b", 2.0),
    ("google/gemma-2-2b", 2.0),
    ("meta-llama/Llama-3.2-3B", 3.0),
    # 7B models (run last, memory intensive)
    ("Qwen/Qwen2-7B", 7.0),
    ("mistralai/Mistral-7B-v0.1", 7.0),
]


# --- QUANTIZATION EXPERIMENT ---
def run_quantization_experiment():
    """Compare FP16 vs INT8 vs INT4 representations."""
    if not HAS_BITSANDBYTES:
        print("\n[SKIP] Quantization experiment - bitsandbytes not available")
        return
    
    print(f"\n{'='*60}")
    print("QUANTIZATION EXPERIMENT")
    print(f"{'='*60}")
    
    # Load dataset
    print("Loading SST-2...")
    dataset = load_dataset("glue", "sst2", split="validation")
    df = pd.DataFrame({'text': dataset['sentence'], 'label': dataset['label']})
    g = df.groupby('label')
    df = g.apply(lambda x: x.sample(N_SAMPLES // 2, random_state=RANDOM_STATE)).reset_index(drop=True)
    texts = df['text'].tolist()
    labels = df['label'].values
    print(f"Loaded {len(texts)} samples")
    
    results_file = f"{OUTPUT_DIR}/quantization_canary_results.csv"
    
    # Resume logic
    completed = set()
    all_results = []
    if os.path.exists(results_file):
        try:
            existing = pd.read_csv(results_file)
            completed = set(existing['model'].unique())
            all_results = existing.to_dict('records')
            print(f"Resuming... {len(completed)} models completed")
        except:
            pass
    
    for model_name, size in MODELS:
        if model_name in completed:
            print(f"\n[Skip] {model_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"{model_name} ({size}B)")
        print(f"{'='*60}")
        
        # Skip large models on small GPUs
        if DEVICE == "cuda" and size >= 7:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem < 35:
                print("   Skipping (GPU too small)")
                continue
        
        model_hash = int(hashlib.md5(model_name.encode("utf-8")).hexdigest()[:8], 16)
        
        # Store embeddings for each quantization level
        embeddings = {}
        accuracies = {}
        
        try:
            for qconfig in QUANT_CONFIGS:
                qname = qconfig["name"]
                bits = qconfig["bits"]
                
                # Load model
                if bits is None:
                    model, tokenizer = load_model_fp16(model_name)
                else:
                    model, tokenizer = load_model_quantized(model_name, bits)
                
                if model is None:
                    print(f"   Failed to load {qname}")
                    continue
                
                # Get embeddings
                emb = get_embeddings(model, tokenizer, texts)
                embeddings[qname] = emb
                
                # Evaluate accuracy
                acc = evaluate_accuracy(emb, labels)
                accuracies[qname] = acc
                print(f"   {qname}: dim={emb.shape[1]}, acc={acc:.3f}")
                
                # Cleanup
                del model, tokenizer
                deep_cleanup()
            
            # Compute drift metrics (FP16 as baseline)
            if "fp16" not in embeddings:
                print("   No FP16 baseline, skipping metrics")
                continue
            
            emb_clean = embeddings["fp16"]
            rdm_clean = pdist(emb_clean.astype(np.float64), metric='cosine')
            acc_clean = accuracies["fp16"]
            
            for qname, emb in embeddings.items():
                seed = RANDOM_STATE + model_hash + hash(qname) % 10000
                
                if qname == "fp16":
                    # Sanity check: self-similarity
                    m = compute_drift_metrics(rdm_clean, emb_clean, emb_clean, seed=seed)
                    print(f"   Sanity (fp16 vs fp16): shesha={m['shesha']:.6f}")
                else:
                    m = compute_drift_metrics(rdm_clean, emb_clean, emb, seed=seed)
                
                row = {
                    'model': model_name,
                    'size': size,
                    'quantization': qname,
                    'acc_clean': acc_clean,
                    'acc_quant': accuracies[qname],
                    'acc_drop': acc_clean - accuracies[qname],
                    **m
                }
                all_results.append(row)
                
                if qname != "fp16":
                    print(f"   {qname} drift: shesha={m['shesha']:.4f}, cka={m['cka_debiased']:.4f}")
            
            # Save after each model
            pd.DataFrame(all_results).to_csv(results_file, index=False)
            completed.add(model_name)
            print(f"   Saved ({len(all_results)} rows)")
            
        except Exception as e:
            print(f"   [Error] {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            del embeddings, accuracies
            deep_cleanup()
            force_delete_model(model_name)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Quantization experiment complete: {results_file}")
    
    if all_results:
        df_res = pd.DataFrame(all_results)
        print(f"\nTotal rows: {len(df_res)}")
        
        # Average drift by quantization level
        print("\n--- MEAN DRIFT BY QUANTIZATION ---")
        for qname in ["int8", "int4_nf4"]:
            subset = df_res[df_res['quantization'] == qname]
            if len(subset) > 0:
                print(f"  {qname}:")
                print(f"    shesha: {subset['shesha'].mean():.4f}")
                print(f"    cka: {subset['cka_debiased'].mean():.4f}")
                print(f"    acc_drop: {subset['acc_drop'].mean():.4f}")


# --- LORA EXPERIMENT ---
def run_lora_experiment():
    """Compare base model vs models with LoRA adapters of varying ranks."""
    if not HAS_PEFT:
        print("\n[SKIP] LoRA experiment - peft not available")
        return
    
    print(f"\n{'='*60}")
    print("LORA EXPERIMENT")
    print(f"{'='*60}")
    
    # Load dataset
    print("Loading SST-2...")
    dataset = load_dataset("glue", "sst2", split="validation")
    df = pd.DataFrame({'text': dataset['sentence'], 'label': dataset['label']})
    g = df.groupby('label')
    df = g.apply(lambda x: x.sample(N_SAMPLES // 2, random_state=RANDOM_STATE)).reset_index(drop=True)
    texts = df['text'].tolist()
    labels = df['label'].values
    print(f"Loaded {len(texts)} samples")
    
    results_file = f"{OUTPUT_DIR}/lora_canary_results.csv"
    
    # Resume logic
    completed = set()
    all_results = []
    if os.path.exists(results_file):
        try:
            existing = pd.read_csv(results_file)
            # Create compound key for resume
            for _, row in existing.iterrows():
                key = f"{row['model']}_r{row['lora_rank']}_s{row['init_scale']}"
                completed.add(key)
            all_results = existing.to_dict('records')
            print(f"Resuming... {len(completed)} configs completed")
        except:
            pass
    
    for model_name, size in MODELS:
        print(f"\n{'='*60}")
        print(f"{model_name} ({size}B)")
        print(f"{'='*60}")
        
        # Skip large models on small GPUs
        if DEVICE == "cuda" and size >= 7:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem < 35:
                print("   Skipping (GPU too small)")
                continue
        
        model_hash = int(hashlib.md5(model_name.encode("utf-8")).hexdigest()[:8], 16)
        
        # Get baseline embeddings first
        baseline_key = f"{model_name}_r0_s0.0"
        
        emb_clean = None
        rdm_clean = None
        acc_clean = None
        
        try:
            # Helper to ensure baseline is loaded and recorded
            def ensure_baseline():
                nonlocal emb_clean, rdm_clean, acc_clean, all_results, completed
                
                if emb_clean is not None:
                    return  # Already loaded
                
                print("   Loading baseline...")
                base_model, base_tokenizer = load_model_fp16(model_name)
                emb_clean = get_embeddings(base_model, base_tokenizer, texts)
                rdm_clean = pdist(emb_clean.astype(np.float64), metric='cosine')
                acc_clean = evaluate_accuracy(emb_clean, labels)
                print(f"   Baseline: dim={emb_clean.shape[1]}, acc={acc_clean:.3f}")
                del base_model, base_tokenizer
                deep_cleanup()
                
                # Always save baseline record if not already in results
                baseline_key = f"{model_name}_r0_s0.0"
                if baseline_key not in completed:
                    seed = RANDOM_STATE + model_hash
                    m = compute_drift_metrics(rdm_clean, emb_clean, emb_clean, seed=seed)
                    print(f"   Sanity: shesha={m['shesha']:.6f}")
                    
                    row = {
                        'model': model_name,
                        'size': size,
                        'lora_rank': 0,
                        'lora_alpha': 0,
                        'init_scale': 0.0,
                        'acc_clean': acc_clean,
                        'acc_lora': acc_clean,
                        'acc_drop': 0.0,
                        **m
                    }
                    all_results.append(row)
                    completed.add(baseline_key)
            
            # Check if we have any configs to run for this model
            configs_to_run = [
                lconfig for lconfig in LORA_CONFIGS
                if f"{model_name}_r{lconfig['rank']}_s{lconfig['init_scale']}" not in completed
            ]
            
            if not configs_to_run:
                print(f"   All configs completed, skipping")
                continue
            
            for lconfig in LORA_CONFIGS:
                lora_rank = lconfig["rank"]
                lora_alpha = lconfig["alpha"]
                init_scale = lconfig["init_scale"]
                
                config_key = f"{model_name}_r{lora_rank}_s{init_scale}"
                
                if config_key in completed:
                    print(f"   [Skip] r={lora_rank}, scale={init_scale}")
                    continue
                
                seed = RANDOM_STATE + model_hash + lora_rank * 100 + int(init_scale * 1000)
                
                # Load model with LoRA
                model, tokenizer = load_model_with_lora(
                    model_name, lora_rank, lora_alpha, init_scale, seed=seed
                )
                
                if model is None:
                    print(f"   Failed to load r={lora_rank}")
                    continue
                
                # Get embeddings
                emb = get_embeddings(model, tokenizer, texts)
                
                # Evaluate accuracy
                acc = evaluate_accuracy(emb, labels)
                
                # For baseline (rank 0), it's handled by ensure_baseline
                if lora_rank == 0:
                    # Just use this to set emb_clean if not already set
                    if emb_clean is None:
                        emb_clean = emb.copy()
                        rdm_clean = pdist(emb_clean.astype(np.float64), metric='cosine')
                        acc_clean = acc
                        print(f"   Baseline: dim={emb.shape[1]}, acc={acc:.3f}")
                        
                        # Sanity check
                        m = compute_drift_metrics(rdm_clean, emb_clean, emb_clean, seed=seed)
                        print(f"   Sanity: shesha={m['shesha']:.6f}")
                        
                        row = {
                            'model': model_name,
                            'size': size,
                            'lora_rank': 0,
                            'lora_alpha': 0,
                            'init_scale': 0.0,
                            'acc_clean': acc,
                            'acc_lora': acc,
                            'acc_drop': 0.0,
                            **m
                        }
                        all_results.append(row)
                        completed.add(config_key)
                    
                    del model, tokenizer, emb
                    deep_cleanup()
                    continue
                else:
                    # Ensure baseline is loaded for comparison
                    ensure_baseline()
                    
                    m = compute_drift_metrics(rdm_clean, emb_clean, emb, seed=seed)
                    
                    row = {
                        'model': model_name,
                        'size': size,
                        'lora_rank': lora_rank,
                        'lora_alpha': lora_alpha,
                        'init_scale': init_scale,
                        'acc_clean': acc_clean,
                        'acc_lora': acc,
                        'acc_drop': acc_clean - acc,
                        **m
                    }
                    
                    print(f"   r={lora_rank}, scale={init_scale}: shesha={m['shesha']:.4f}, cka={m['cka_debiased']:.4f}, acc={acc:.3f}")
                    
                    all_results.append(row)
                    completed.add(config_key)
                
                # Cleanup
                del model, tokenizer, emb
                deep_cleanup()
            
            # Save after each model
            pd.DataFrame(all_results).to_csv(results_file, index=False)
            print(f"   Saved ({len(all_results)} rows)")
            
        except Exception as e:
            print(f"   [Error] {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            del emb_clean, rdm_clean
            deep_cleanup()
            force_delete_model(model_name)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"LoRA experiment complete: {results_file}")
    
    if all_results:
        df_res = pd.DataFrame(all_results)
        print(f"\nTotal rows: {len(df_res)}")
        
        # Average drift by rank
        print("\n--- MEAN DRIFT BY LORA RANK ---")
        for rank in [1, 2, 4, 8, 16, 32, 64]:
            subset = df_res[(df_res['lora_rank'] == rank) & (df_res['init_scale'] == 0.01)]
            if len(subset) > 0:
                print(f"  r={rank}:")
                print(f"    shesha: {subset['shesha'].mean():.4f}")
                print(f"    cka: {subset['cka_debiased'].mean():.4f}")
        
        # Average drift by init scale (at rank 8)
        print("\n--- MEAN DRIFT BY INIT SCALE (r=8) ---")
        for scale in [0.001, 0.01, 0.05, 0.1]:
            subset = df_res[(df_res['lora_rank'] == 8) & (df_res['init_scale'] == scale)]
            if len(subset) > 0:
                print(f"  scale={scale}:")
                print(f"    shesha: {subset['shesha'].mean():.4f}")
                print(f"    cka: {subset['cka_debiased'].mean():.4f}")


# --- GAUSSIAN NOISE EXPERIMENT ---
def run_gaussian_noise_experiment():
    """Compare clean model vs models with Gaussian noise injection."""
    
    print(f"\n{'='*60}")
    print("GAUSSIAN NOISE EXPERIMENT")
    print(f"{'='*60}")
    
    # Load dataset
    print("Loading SST-2...")
    dataset = load_dataset("glue", "sst2", split="validation")
    df = pd.DataFrame({'text': dataset['sentence'], 'label': dataset['label']})
    g = df.groupby('label')
    df = g.apply(lambda x: x.sample(N_SAMPLES // 2, random_state=RANDOM_STATE)).reset_index(drop=True)
    texts = df['text'].tolist()
    labels = df['label'].values
    print(f"Loaded {len(texts)} samples")
    
    results_file = f"{OUTPUT_DIR}/gaussian_noise_canary_results.csv"
    
    # Resume logic
    completed = set()
    all_results = []
    if os.path.exists(results_file):
        try:
            existing = pd.read_csv(results_file)
            for _, row in existing.iterrows():
                key = f"{row['model']}_noise{row['noise_level']}"
                completed.add(key)
            all_results = existing.to_dict('records')
            print(f"Resuming... {len(completed)} configs completed")
        except:
            pass
    
    for model_name, size in MODELS:
        print(f"\n{'='*60}")
        print(f"{model_name} ({size}B)")
        print(f"{'='*60}")
        
        if DEVICE == "cuda" and size >= 7:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem < 35:
                print("   Skipping (GPU too small)")
                continue
        
        # Check if we have any noise levels to run
        configs_to_run = [
            alpha for alpha in GAUSSIAN_NOISE_LEVELS
            if f"{model_name}_noise{alpha}" not in completed
        ]
        
        if not configs_to_run:
            print(f"   All noise levels completed, skipping")
            continue
        
        model_hash = int(hashlib.md5(model_name.encode("utf-8")).hexdigest()[:8], 16)
        
        emb_clean = None
        rdm_clean = None
        acc_clean = None
        clean_state_dict = None
        
        try:
            # Load model once
            model, tokenizer = load_model_fp16(model_name)
            
            # Save clean state dict for restoration
            clean_state_dict = get_model_state_dict(model)
            
            # Get clean embeddings
            emb_clean = get_embeddings(model, tokenizer, texts)
            rdm_clean = pdist(emb_clean.astype(np.float64), metric='cosine')
            acc_clean = evaluate_accuracy(emb_clean, labels)
            print(f"   Baseline: dim={emb_clean.shape[1]}, acc={acc_clean:.3f}")
            
            # Save baseline record if needed
            baseline_key = f"{model_name}_noise0.0"
            if baseline_key not in completed:
                seed = RANDOM_STATE + model_hash
                m = compute_drift_metrics(rdm_clean, emb_clean, emb_clean, seed=seed)
                print(f"   Sanity: shesha={m['shesha']:.6f}")
                
                row = {
                    'model': model_name,
                    'size': size,
                    'noise_level': 0.0,
                    'acc_clean': acc_clean,
                    'acc_noisy': acc_clean,
                    'acc_drop': 0.0,
                    **m
                }
                all_results.append(row)
                completed.add(baseline_key)
            
            # Test each noise level
            for alpha in GAUSSIAN_NOISE_LEVELS:
                if alpha == 0.0:
                    continue  # Already handled
                
                config_key = f"{model_name}_noise{alpha}"
                if config_key in completed:
                    continue
                
                # Restore clean weights
                load_model_state_dict(model, clean_state_dict)
                
                # Inject noise
                seed = RANDOM_STATE + model_hash + int(alpha * 1000)
                inject_gaussian_noise(model, alpha, seed=seed)
                
                # Get noisy embeddings
                emb_noisy = get_embeddings(model, tokenizer, texts)
                acc_noisy = evaluate_accuracy(emb_noisy, labels)
                
                m = compute_drift_metrics(rdm_clean, emb_clean, emb_noisy, seed=seed)
                
                row = {
                    'model': model_name,
                    'size': size,
                    'noise_level': alpha,
                    'acc_clean': acc_clean,
                    'acc_noisy': acc_noisy,
                    'acc_drop': acc_clean - acc_noisy,
                    **m
                }
                
                all_results.append(row)
                completed.add(config_key)
                
                print(f"   noise={alpha}: shesha={m['shesha']:.4f}, cka={m['cka_debiased']:.4f}, acc_drop={row['acc_drop']:.3f}")
                
                del emb_noisy
                deep_cleanup()
            
            del model, tokenizer
            pd.DataFrame(all_results).to_csv(results_file, index=False)
            print(f"   Saved ({len(all_results)} rows)")
            
        except Exception as e:
            print(f"   [Error] {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            del emb_clean, rdm_clean, clean_state_dict
            deep_cleanup()
            force_delete_model(model_name)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Gaussian noise experiment complete: {results_file}")
    
    if all_results:
        df_res = pd.DataFrame(all_results)
        print(f"\nTotal rows: {len(df_res)}")
        
        print("\n--- MEAN DRIFT BY NOISE LEVEL ---")
        for alpha in GAUSSIAN_NOISE_LEVELS[1:]:  # Skip 0
            subset = df_res[df_res['noise_level'] == alpha]
            if len(subset) > 0:
                print(f"  noise={alpha}:")
                print(f"    shesha: {subset['shesha'].mean():.4f}")
                print(f"    cka: {subset['cka_debiased'].mean():.4f}")
                print(f"    acc_drop: {subset['acc_drop'].mean():.3f}")


# --- MAIN ---
def run_all():
    """Run all three experiments."""
    print("\n" + "="*70)
    print("STRUCTURED PERTURBATION EXPERIMENTS")
    print("="*70)
    
    # # 1. Gaussian noise experiment
    run_gaussian_noise_experiment()
    
    # # 2. Quantization experiment
    run_quantization_experiment()
    
    # 3. LoRA experiment
    run_lora_experiment()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_all()