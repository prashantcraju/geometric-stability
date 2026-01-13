"""
Shesha Drift - Canary Validation

Validated experiment for measuring representational drift under
parameter noise injection across sentence embedding models.

- SentenceTransformer with full-model noise (try/fallback to HF)
- Device-safe state dict save/restore with strict=False
- L2 normalized embeddings for consistent cosine RDM
- E5 prefix handling
- Resume-safe reproducibility via stable per-(model, alpha) seeding
- Alpha=0 sanity check for numerical precision
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
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import warnings

from huggingface_hub import login

# Authenticate with Hugging Face
token = os.environ.get("HF_TOKEN")
if token:
    login(token)
else:
    print("Set HF_TOKEN environment variable")

warnings.filterwarnings("ignore")

# Try importing SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("WARNING: sentence-transformers not installed. Using AutoModel for all.")

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
else:
    DTYPE = torch.float32
    BATCH_SIZE = 16

print(f"Device: {DEVICE}, Dtype: {DTYPE}, Batch: {BATCH_SIZE}")

OUTPUT_DIR = Path("./shesha-drift")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Experiment Settings
NOISE_LEVELS = [i * 0.01 for i in range(51)]  # 0% to 50%
N_SAMPLES = 800
MAX_SEQ_LEN = 128

# E5 models that need prefix
E5_MODELS = {"intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2"}


# --- CLEANUP FUNCTIONS ---
def force_delete_model(model_id):
    """
    Delete model from HF cache. Handles both regular and ST cache patterns.
    """
    try:
        cache_root = "/root/.cache/huggingface/hub"
        if not os.path.exists(cache_root):
            return

        # Convert model_id to possible folder patterns
        safe_name = model_id.replace("/", "--")
        patterns = [
            f"models--{safe_name}",
            f"sentence-transformers--{safe_name}",
        ]

        for item in os.listdir(cache_root):
            for pattern in patterns:
                if pattern in item:
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


def sliced_wasserstein(X, Y, n_proj=50):
    """Sliced Wasserstein distance."""
    rng = np.random.default_rng(42)
    dirs = rng.standard_normal((X.shape[1], n_proj))
    dirs /= np.linalg.norm(dirs, axis=0)
    Xp = np.sort(X @ dirs, axis=0)
    Yp = np.sort(Y @ dirs, axis=0)
    return float(np.mean(np.abs(Xp - Yp)))


def compute_drift_metrics(rdm_clean, emb_clean, emb_noisy):
    """Compute all drift metrics using precomputed clean RDM."""
    X = np.asarray(emb_clean, dtype=np.float64)
    Y = np.asarray(emb_noisy, dtype=np.float64)
    return {
        'shesha': 1.0 - rdm_spearman_from_clean(rdm_clean, Y),
        'rdm_pearson': 1.0 - rdm_pearson_from_clean(rdm_clean, Y),
        'cka_debiased': 1.0 - cka_debiased(X, Y),
        'procrustes': 1.0 - procrustes_similarity(X, Y),
        'wasserstein': sliced_wasserstein(X, Y),
    }


# --- MODEL LOADING ---
def try_load_sentence_transformer(model_name):
    """
    Try to load as SentenceTransformer. Returns (model, True) on success,
    (None, False) on failure.
    """
    if not ST_AVAILABLE:
        return None, False

    try:
        model = SentenceTransformer(model_name, device=DEVICE)
        # Ensure all modules are on DEVICE
        model.to(DEVICE)
        # Quick sanity check
        _ = model.encode(["test"], show_progress_bar=False)
        return model, True
    except Exception as e:
        print(f"   ST load failed ({e}), falling back to AutoModel")
        return None, False


def load_model(model_name):
    """
    Load model, trying SentenceTransformer first, then AutoModel.
    Returns (model, tokenizer_or_none, is_st)
    """
    st_model, success = try_load_sentence_transformer(model_name)
    if success:
        return st_model, None, True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    return model, tokenizer, False


# --- EMBEDDING FUNCTIONS ---
def l2_normalize(emb):
    """L2 normalize embeddings to unit norm."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)  # Avoid division by zero
    return emb / norms


def get_embeddings_st(model, texts):
    """Get embeddings using SentenceTransformer (already normalized by most ST models)."""
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    # ST models typically normalize, but ensure consistency
    return l2_normalize(emb)


def get_embeddings_hf(model, tokenizer, texts):
    """Get embeddings using HuggingFace AutoModel with mean pooling + L2 normalization."""
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
                    out = model(**inputs)
            else:
                out = model(**inputs)

            mask = inputs['attention_mask'].unsqueeze(-1)
            vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_vecs.append(vecs.float().cpu().numpy())

    emb = np.vstack(all_vecs)
    # L2 normalize to match ST behavior
    return l2_normalize(emb)


# --- STATE DICT MANAGEMENT (Device-safe) ---
def get_full_state_dict_st(st_model):
    """
    Get state dict for ALL modules in a SentenceTransformer.
    Stores on CPU for device-safety, clones to avoid aliasing.
    """
    full_state = {}
    for idx, module in enumerate(st_model):
        module_state = module.state_dict()
        for k, v in module_state.items():
            # Store on CPU for safety
            full_state[f"module_{idx}.{k}"] = v.cpu().clone()
    return full_state


def load_full_state_dict_st(st_model, state_dict):
    """
    Load state dict back into all ST modules.
    All tensors go to DEVICE (model is already on DEVICE from loading).
    Uses strict=False to handle version-specific buffers gracefully.
    """
    for idx, module in enumerate(st_model):
        prefix = f"module_{idx}."
        module_state = {
            k[len(prefix):]: v.to(DEVICE)
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if module_state:
            module.load_state_dict(module_state, strict=False)


def inject_noise_st(st_model, clean_state_dict, alpha):
    """
    Inject noise into ALL SentenceTransformer modules.
    Restores from clean state, then adds noise to all parameters.
    """
    # Restore clean weights
    load_full_state_dict_st(st_model, clean_state_dict)

    if alpha == 0:
        return

    # Add noise to all modules
    with torch.no_grad():
        for module in st_model:
            for p in module.parameters():
                if p.requires_grad and p.numel() > 1:
                    std = float(p.std().item())
                    if std > 0:
                        noise = torch.randn_like(p) * (std * alpha)
                        p.add_(noise)


def get_state_dict_hf(model):
    """
    Get HF model state dict. Tries GPU clone for speed, falls back to CPU if OOM.
    """
    try:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("   [Warning] GPU OOM on state dict clone, using CPU fallback")
            torch.cuda.empty_cache()
            return {k: v.cpu().clone() for k, v in model.state_dict().items()}
        raise


def load_state_dict_hf(model, state_dict):
    """Load HF model state dict, handling CPU or GPU source."""
    # Check if state dict is on CPU (fallback case)
    first_val = next(iter(state_dict.values()))
    if first_val.device.type == "cpu":
        device_state = {k: v.to(DEVICE) for k, v in state_dict.items()}
        model.load_state_dict(device_state)
    else:
        model.load_state_dict(state_dict)


def inject_noise_hf(model, clean_state_dict, alpha):
    """
    Inject noise into HuggingFace model.
    Restores from clean state, then adds noise.
    """
    load_state_dict_hf(model, clean_state_dict)

    if alpha == 0:
        return

    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.numel() > 1:
                std = float(p.std().item())
                if std > 0:
                    noise = torch.randn_like(p) * (std * alpha)
                    p.add_(noise)


def evaluate_accuracy(X, y):
    """Evaluate classification accuracy with cross-validation."""
    clf = LogisticRegression(solver='liblinear', max_iter=200, random_state=RANDOM_STATE)
    return float(np.mean(cross_val_score(clf, X, y, cv=5, scoring='accuracy')))


# --- MODELS ---
MODELS = [
    # Small models
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/paraphrase-albert-small-v2",
    "thenlper/gte-small",
    "intfloat/e5-small-v2",
    "BAAI/bge-small-en-v1.5",
    # Distil models
    "sentence-transformers/distilbert-base-nli-mean-tokens",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/paraphrase-distilroberta-base-v1",
    # Base models
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "sentence-transformers/bert-base-nli-mean-tokens",
    "sentence-transformers/stsb-roberta-base",
    "sentence-transformers/nli-roberta-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
    "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/unsup-simcse-bert-base-uncased",
    # Large models
    "thenlper/gte-large",
    "intfloat/e5-large-v2",
    "BAAI/bge-large-en-v1.5",
]


# --- MAIN LOOP ---
def run_experiment():
    print("Loading SST-2...")
    dataset = load_dataset("glue", "sst2", split="validation")
    df = pd.DataFrame({'text': dataset['sentence'], 'label': dataset['label']})
    g = df.groupby('label')
    df = g.apply(lambda x: x.sample(N_SAMPLES // 2, random_state=RANDOM_STATE)).reset_index(drop=True)
    texts_raw = df['text'].tolist()
    labels = df['label'].values
    print(f"Loaded {len(texts_raw)} samples")

    results_file = f"{OUTPUT_DIR}/canary_results_v14.csv"

    # Resume logic
    completed = set()
    all_results = []
    if os.path.exists(results_file):
        try:
            existing = pd.read_csv(results_file)
            completed = set(existing['model'].unique())
            all_results = existing.to_dict('records')
            print(f"Resuming... {len(completed)} models completed, {len(all_results)} rows loaded.")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    for model_name in MODELS:
        # Initialize variables for clean cleanup
        model = None
        tokenizer = None
        emb_clean = None
        rdm_clean = None
        clean_state_dict = None
        is_st = False

        if model_name in completed:
            print(f"\n[Skip] {model_name}")
            continue

        print(f"\n{'='*60}\n{model_name}\n{'='*60}")

        # Prepare texts (with E5 prefix if needed)
        if model_name in E5_MODELS:
            texts = [f"passage: {t}" for t in texts_raw]
            print("   Applied E5 'passage:' prefix")
        else:
            texts = texts_raw

        try:
            # Load model
            model, tokenizer, is_st = load_model(model_name)
            print(f"   Loaded as: {'SentenceTransformer' if is_st else 'AutoModel'}")

            # Save clean state dict (on CPU for device safety)
            if is_st:
                clean_state_dict = get_full_state_dict_st(model)
            else:
                clean_state_dict = get_state_dict_hf(model)

            # Get clean embeddings
            if is_st:
                emb_clean = get_embeddings_st(model, texts)
            else:
                emb_clean = get_embeddings_hf(model, tokenizer, texts)

            # Log embedding stats
            mean_norm = float(np.linalg.norm(emb_clean, axis=1).mean())
            print(f"   Embedding dim: {emb_clean.shape[1]}, mean L2 norm: {mean_norm:.4f}")

            # Evaluate clean accuracy
            acc_clean = evaluate_accuracy(emb_clean, labels)
            print(f"   Clean accuracy: {acc_clean:.3f}")

            # Precompute clean RDM once
            rdm_clean = pdist(emb_clean.astype(np.float64), metric='cosine')
            print(f"   Precomputed clean RDM: {len(rdm_clean)} distances")

            # CRITICAL: Restore clean weights before noise loop
            # This ensures model is in known clean state
            if is_st:
                load_full_state_dict_st(model, clean_state_dict)
            else:
                load_state_dict_hf(model, clean_state_dict)

            # Precompute stable hash for this model (for resume-safe seeding)
            model_hash = int(hashlib.md5(model_name.encode("utf-8")).hexdigest()[:8], 16)

            model_results = []

            # Noise loop
            for alpha in NOISE_LEVELS:
                # Per-(model, alpha) seed for resume-safe reproducibility
                seed = RANDOM_STATE + int(alpha * 1000) + model_hash
                torch.manual_seed(seed)
                if DEVICE == "cuda":
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)

                # Inject noise (or restore clean for alpha=0)
                if is_st:
                    inject_noise_st(model, clean_state_dict, alpha)
                    emb_noisy = get_embeddings_st(model, texts)
                else:
                    inject_noise_hf(model, clean_state_dict, alpha)
                    emb_noisy = get_embeddings_hf(model, tokenizer, texts)

                # Compute drift metrics (even for alpha=0 to verify numerical precision)
                m = compute_drift_metrics(rdm_clean, emb_clean, emb_noisy)

                # Skip CV for alpha=0 (embeddings are identical, so acc_noisy = acc_clean)
                if alpha == 0:
                    acc_noisy = acc_clean
                else:
                    acc_noisy = evaluate_accuracy(emb_noisy, labels)

                row = {
                    'model': model_name,
                    'noise': alpha,
                    'acc_clean': acc_clean,
                    'acc_noisy': acc_noisy,
                    'acc_drop': acc_clean - acc_noisy,
                    **m
                }

                del emb_noisy
                deep_cleanup()

                model_results.append(row)

                # Log alpha=0 drift as sanity check (should be ~0)
                if alpha == 0:
                    print(f"   alpha=0 sanity: shesha={row['shesha']:.6f}, cka={row['cka_debiased']:.6f}")

                # Progress logging every 10%
                if alpha > 0 and round(alpha * 100) % 10 == 0:
                    print(f"   noise={alpha:.2f}: acc_drop={row['acc_drop']:.3f}, shesha={row['shesha']:.4f}")

            # Restore clean weights before cleanup
            if is_st:
                load_full_state_dict_st(model, clean_state_dict)
            else:
                load_state_dict_hf(model, clean_state_dict)

            # Save results
            all_results.extend(model_results)
            pd.DataFrame(all_results).to_csv(results_file, index=False)
            completed.add(model_name)
            print(f"   Saved ({len(all_results)} total rows)")

        except Exception as e:
            print(f"   [Error] {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Explicit cleanup
            del model, tokenizer, emb_clean, rdm_clean, clean_state_dict
            deep_cleanup()
            force_delete_model(model_name)

    # Final summary
    print(f"\n{'='*60}")
    print(f"DONE - Saved to {results_file}")
    print(f"{'='*60}")

    if all_results:
        df_res = pd.DataFrame(all_results)
        print(f"\nTotal rows: {len(df_res)}")
        print(f"Models completed: {df_res['model'].nunique()}")

        # Correlation summary
        valid = df_res[df_res['noise'] > 0]
        print("\n--- CORRELATION WITH ACCURACY DROP ---")
        for metric in ['shesha', 'cka_debiased', 'rdm_pearson', 'wasserstein']:
            if metric in valid.columns:
                corr = valid['acc_drop'].corr(valid[metric])
                print(f"  {metric}: r = {corr:.3f}")


if __name__ == "__main__":
    run_experiment()