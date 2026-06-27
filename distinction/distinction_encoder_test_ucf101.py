"""
Shesha Distinction - Encoder Test (7 Domains) — UCF-101 Video
Domains:
1. Language (SST-2) - 4 models
2. Vision (CIFAR-100) - 4 models
3. Audio (LibriSpeech) - 2 models
4. Video (UCF-101) - 4 models
5. Neuroscience (Steinmetz) - All sessions
6. Protein (Swiss-Prot) - Multiple encoders
7. Molecular (PBMC3k) - Multiple encoders

Metric: FEATURE-SPLIT SHESHA (Internal Geometric Consistency)
Scale: 15 SEEDS

Video data loading is tiered:
  Tier 1 — UCF-101 .avi files already on disk at UCF101_ROOT
  Tier 2 — torchvision.datasets.UCF101 (downloads annotation splits only)
  Tier 3 — synthetic colour-coded clips (always works, smoke-test only)
"""


import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True

import os
import warnings
import tarfile
import requests
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from scipy.sparse import issparse
import scanpy as sc

# Transformers & Data
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoImageProcessor,
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor, HubertModel,
    VideoMAEImageProcessor, VideoMAEModel, CLIPModel, CLIPProcessor
)
from sentence_transformers import SentenceTransformer

# GPU Acceleration
try:
    from cuml.decomposition import PCA
    from cuml.random_projection import GaussianRandomProjection
    print("[INFO] Using GPU-accelerated PCA (cuML)")
    IS_GPU_PCA = True
except ImportError:
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection
    print("[INFO] Falling back to CPU PCA (sklearn)")
    IS_GPU_PCA = False

try:
    if torch.cuda.is_available():
        from cuml.preprocessing import StandardScaler
    else:
        from sklearn.preprocessing import StandardScaler
except:
    from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = Path(__file__).resolve().parent / "shesha-distinction"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# UCF-101 paths — edit if your data lives elsewhere
# ---------------------------------------------------------------------------
UCF101_ROOT     = "./data/UCF101/UCF-101"          # folder of class sub-dirs
ANNOTATION_PATH = "./data/UCF101/ucfTrainTestlist"  # annotation split folder

# ---------------------------------------------------------------------------
# UCF-101 auto-download
# ---------------------------------------------------------------------------
def _auto_download_ucf101():
    """
    Attempt to download UCF-101 automatically.

    Strategy (in order):
      1. kaggle-hub  — pulls from Kaggle dataset 'matthewjansen/ucf101-action-recognition'
      2. opendatalab — huggingface mirror (rar, needs unrar/p7zip)
      3. Print clear manual instructions and return False.

    Returns True if UCF-101 is now available at UCF101_ROOT, else False.
    """
    import shutil, subprocess as _sp

    dst = Path(UCF101_ROOT)
    if dst.exists() and any(dst.rglob("*.avi")):
        return True  # already present

    dst.parent.mkdir(parents=True, exist_ok=True)

    # --- Strategy 1: kaggle-hub ---
    try:
        import kagglehub                                        # pip install kagglehub
        path = kagglehub.dataset_download(
            "matthewjansen/ucf101-action-recognition"
        )
        # kagglehub returns the cache path; locate the UCF-101 root inside it
        found = list(Path(path).rglob("UCF-101"))
        if not found:
            found = [p.parent for p in Path(path).rglob("*.avi")]
        if found:
            src = found[0]
            if not dst.exists():
                shutil.copytree(str(src), str(dst))
            print(f"    [UCF-101] Downloaded via kagglehub → {dst}")
            return True
    except Exception as e:
        print(f"    [UCF-101] kagglehub failed ({e})")

    # --- Strategy 2: direct RAR from CRCV (needs curl + unrar/p7zip) ---
    rar_path = dst.parent / "UCF101.rar"
    try:
        if not rar_path.exists():
            result = _sp.run(
                ["curl", "--insecure", "-L", "--progress-bar",
                 "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
                 "-o", str(rar_path)],
                check=True,
            )
        # Try unrar, then 7z
        for cmd in [
            ["unrar", "x", "-y", str(rar_path), str(dst.parent)],
            ["7z",    "x", f"-o{dst.parent}", str(rar_path), "-y"],
            ["unar",  "-o", str(dst.parent), str(rar_path)],
        ]:
            if shutil.which(cmd[0]):
                _sp.run(cmd, check=True)
                if dst.exists() and any(dst.rglob("*.avi")):
                    print(f"    [UCF-101] Extracted via {cmd[0]} → {dst}")
                    return True
    except Exception as e:
        print(f"    [UCF-101] CRCV RAR strategy failed ({e})")

    # --- Nothing worked: print instructions ---
    print(
        "\n" + "=" * 60 +
        "\n  UCF-101 AUTO-DOWNLOAD FAILED\n"
        "  To use real UCF-101 videos, do ONE of the following:\n\n"
        "  Option A — Kaggle:\n"
        "    pip install kagglehub\n"
        "    import kagglehub\n"
        "    kagglehub.dataset_download('matthewjansen/ucf101-action-recognition')\n\n"
        "  Option B — Manual download:\n"
        "    Place UCF101.rar under ./data/UCF101/, then extract with unrar or 7z.\n\n"
        "  After extraction set UCF101_ROOT to the folder that contains\n"
        "  sub-directories like 'ApplyEyeMakeup', 'Archery', etc.\n"
        "  (default: ./data/UCF101/UCF-101)\n" +
        "=" * 60 + "\n"
        "  Continuing with SYNTHETIC clips for now.\n" +
        "=" * 60
    )
    return False


# FULL 15 SEEDS
SEEDS = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]

CONFIG = {
    'language': {'n_samples': 500, 'max_len': 64},
    'vision': {'n_images': 400, 'image_size': 224},
    'audio': {'n_audio': 200, 'sample_rate': 16000},
    'video': {'n_videos': 100, 'frames_per_video': 16, 'video_size': 224},
    'neuroscience': {'min_neurons': 20, 'min_trials': 50},
    'protein': {'n_proteins': 200},
    'molecular': {'n_cells': 1000},
}

# Number of bootstrap resamples for confidence intervals
N_BOOTSTRAP = 10_000

# =============================================================================
# METRICS
# =============================================================================

def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05, statistic=np.mean, rng_seed=0):
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    values   : array-like of floats
    n_boot   : int   — number of bootstrap resamples (default 10 000)
    alpha    : float — two-tailed alpha level (default 0.05 → 95 % CI)
    statistic: callable applied to each resample (default np.mean)
    rng_seed : int

    Returns
    -------
    (estimate, ci_low, ci_high)
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(rng_seed)
    boot_stats = np.array([
        statistic(rng.choice(vals, size=len(vals), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(statistic(vals)), float(lo), float(hi))


def bootstrap_ci_rho(x, y, n_boot=N_BOOTSTRAP, alpha=0.05, rng_seed=0):
    """Bootstrap CI for Spearman rho between two paired arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(rng_seed)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        rho, _ = spearmanr(x[idx], y[idx])
        if np.isfinite(rho):
            boot_rhos.append(rho)
    if not boot_rhos:
        return (np.nan, np.nan, np.nan)
    boot_rhos = np.array(boot_rhos)
    lo = np.percentile(boot_rhos, 100 * alpha / 2)
    hi = np.percentile(boot_rhos, 100 * (1 - alpha / 2))
    point_rho, point_p = spearmanr(x, y)
    return (float(point_rho), float(lo), float(hi))


def compute_shesha_features(X, n_splits=30, random_state=None):
    """Feature-split Shesha on GPU."""
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    elif X.device.type != 'cuda':
        X = X.to(DEVICE).float()
    else:
        X = X.float()

    n_samples, n_features = X.shape
    if n_samples < 10 or n_features < 2:
        return np.nan

    tri_idx = torch.triu_indices(n_samples, n_samples, offset=1, device=DEVICE)
    corrs = []
    rng = np.random.default_rng(random_state)

    for _ in range(n_splits):
        perm = torch.randperm(n_features, device=DEVICE)
        half = n_features // 2
        if half < 1:
            half = 1

        idx1 = perm[:half]
        idx2 = perm[half:2*half]
        if len(idx2) == 0:
            idx2 = idx1

        X1 = X[:, idx1]
        X2 = X[:, idx2]

        X1_n = torch.nn.functional.normalize(X1, p=2, dim=1)
        X2_n = torch.nn.functional.normalize(X2, p=2, dim=1)

        rdm1 = 1.0 - torch.mm(X1_n, X1_n.t())
        rdm2 = 1.0 - torch.mm(X2_n, X2_n.t())

        v1 = rdm1[tri_idx[0], tri_idx[1]].cpu().numpy()
        v2 = rdm2[tri_idx[0], tri_idx[1]].cpu().numpy()

        if np.std(v1) < 1e-9 or np.std(v2) < 1e-9:
            continue

        rho, _ = spearmanr(v1, v2)
        if np.isfinite(rho):
            corrs.append(rho)

    return float(np.mean(corrs)) if len(corrs) >= 5 else np.nan


def compute_cka(X, Y):
    """GPU Linear CKA"""
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float64, device='cuda')
    else:
        X = X.double().cuda()
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float64, device='cuda')
    else:
        Y = Y.double().cuda()
    
    n = X.shape[0]
    if n != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of samples")
    
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    
    K = torch.matmul(X, X.T)
    L = torch.matmul(Y, Y.T)
    
    H = torch.eye(n, dtype=torch.float64, device='cuda') - torch.ones((n, n), dtype=torch.float64, device='cuda') / n
    K = H @ K @ H
    L = H @ L @ H
    
    num = (K * L).sum()
    den = torch.sqrt((K * K).sum() * (L * L).sum()) + 1e-12
    return float(torch.clamp(num / den, 0.0, 1.0).item())


# =============================================================================
# ENCODER TRANSFORMATIONS
# =============================================================================

def build_encoder_transformations(X_base, seed):
    """Full encoder transformation suite (~30 variants)."""
    if hasattr(X_base, 'cpu'):
        X_base_native = X_base.cpu().numpy()
    elif hasattr(X_base, 'get'):
        X_base_native = X_base.get()
    else:
        X_base_native = np.asarray(X_base)

    rng = np.random.default_rng(seed)
    encoders = {}
    n_samples, n_features = X_base_native.shape

    # 1. PCA at various ranks
    for k in [5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 256, 300]:
        k_actual = min(k, n_samples - 1, n_features)
        if k_actual >= 5:
            try:
                if IS_GPU_PCA:
                    pca = PCA(n_components=k_actual)
                else:
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


def run_encoder_analysis(base_embeddings, seed, domain_name):
    """Run analysis on all encoder transformations."""
    all_rows = []
    
    for base_name, X_base in base_embeddings.items():
        encoders = build_encoder_transformations(X_base, seed)
        
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

            shesha = compute_shesha_features(X, n_splits=30, random_state=seed)

            cka_values = []
            for ref_name, ref_X in refs.items():
                if ref_X is not None and ref_X.shape[0] == X.shape[0]:
                    ref_X = np.nan_to_num(ref_X, nan=0.0)
                    cka = compute_cka(X, ref_X)
                    if np.isfinite(cka):
                        cka_values.append(cka)
            
            cka_avg = np.mean(cka_values) if cka_values else np.nan

            all_rows.append({
                'domain': domain_name,
                'seed': seed,
                'base_model': base_name,
                'encoder': enc_name,
                'SHESHA': shesha,
                'CKA': cka_avg,
                'n_features': X.shape[1],
            })
    
    return all_rows


# =============================================================================
# DOMAIN 1: LANGUAGE (4 Models)
# =============================================================================

def _load_sst2_texts(n_samples):
    """Load SST-2 sentences, trying multiple dataset paths for compatibility."""
    # Try new canonical HF path first
    for dataset_id, config_name, split in [
        ("stanfordnlp/sst2",  None,   "validation"),
        ("glue",              "sst2", "validation"),
        ("sst2",              None,   "validation"),
    ]:
        try:
            if config_name:
                ds = load_dataset(dataset_id, config_name, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
            col = "sentence" if "sentence" in ds.column_names else ds.column_names[0]
            return list(ds[col])[:n_samples]
        except Exception:
            continue
    # Last resort: a small set of hard-coded sentences so the domain never fails
    print("    [WARN] SST-2 download failed — using built-in fallback sentences.")
    return [
        "a great film", "terrible acting", "loved every minute",
        "boring and dull", "masterpiece of cinema", "awful waste of time",
        "surprisingly enjoyable", "deeply moving story", "predictable plot",
        "outstanding performances", "mediocre at best", "a must watch",
        "completely forgettable", "brilliantly directed", "poorly written",
        "charming and funny", "slow and tedious", "emotionally powerful",
        "disappointing sequel", "genuinely hilarious",
    ] * (n_samples // 20 + 1)


def run_language_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 1: LANGUAGE (4 Models)")
    print("=" * 60)

    try:
        texts = _load_sst2_texts(CONFIG['language']['n_samples'])
        print(f"  Loaded {len(texts)} sentences")
        
        base_embeddings = {}
        
        # Model 1: MiniLM
        try:
            print("    Loading MiniLM...")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            base_embeddings['minilm'] = emb
            print(f"    minilm: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] MiniLM: {e}")
        
        # Model 2: MPNet
        try:
            print("    Loading MPNet...")
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            base_embeddings['mpnet'] = emb
            print(f"    mpnet: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] MPNet: {e}")
        
        # Model 3: DistilBERT
        try:
            print("    Loading DistilBERT...")
            model = SentenceTransformer("sentence-transformers/distilbert-base-nli-stsb-mean-tokens", device=DEVICE)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            base_embeddings['distilbert'] = emb
            print(f"    distilbert: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] DistilBERT: {e}")
        
        # Model 4: RoBERTa
        try:
            print("    Loading RoBERTa...")
            model = SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1", device=DEVICE)
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            base_embeddings['roberta'] = emb
            print(f"    roberta: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] RoBERTa: {e}")
        
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        if not base_embeddings:
            return []
        
        all_results = []
        for seed in tqdm(SEEDS, desc="  Seeds"):
            all_results.extend(run_encoder_analysis(base_embeddings, seed, "Language"))
        
        return all_results
    
    except Exception as e:
        print(f"Language Failed: {e}")
        return []


# =============================================================================
# DOMAIN 2: VISION (4 Models)
# =============================================================================

def _load_cifar100_pil(n_images):
    """
    Load CIFAR-100 images as PIL, no torchvision required.
    Uses HuggingFace datasets (cifar100) with a numpy/pickle fallback.
    Returns list of PIL Images (224×224 RGB).
    """
    pil_images = []
    try:
        ds = load_dataset("uoft-cs/cifar100", split="test", trust_remote_code=True)
        indices = np.linspace(0, len(ds) - 1, n_images, dtype=int)
        for i in indices:
            img = ds[int(i)]["img"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img, dtype=np.uint8))
            pil_images.append(img.resize((224, 224)).convert("RGB"))
        return pil_images
    except Exception as e:
        print(f"    [WARN] HF cifar100 load failed ({e}) — trying direct download...")

    # Fallback: download CIFAR-100 binary directly
    try:
        import pickle, urllib.request, gzip
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        tar_path = data_dir / "cifar100.tar.gz"
        if not tar_path.exists():
            urllib.request.urlretrieve(url, tar_path)
        import tarfile
        with tarfile.open(tar_path) as tf:
            member = tf.getmember("cifar-100-python/test")
            f = tf.extractfile(member)
            data = pickle.load(f, encoding="bytes")
        imgs = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        indices = np.linspace(0, len(imgs) - 1, n_images, dtype=int)
        for i in indices:
            pil_images.append(
                Image.fromarray(imgs[i]).resize((224, 224)).convert("RGB")
            )
        return pil_images
    except Exception as e2:
        print(f"    [WARN] Direct CIFAR-100 download also failed ({e2}) — using noise images.")

    # Last resort: random noise images
    rng = np.random.default_rng(320)
    for _ in range(n_images):
        arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        pil_images.append(Image.fromarray(arr))
    return pil_images


def _pil_batch_to_tensor(pil_images):
    """Stack PIL images → (N, 3, 224, 224) float32 tensor, no torchvision."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensors = []
    for img in pil_images:
        arr = np.array(img.resize((224, 224)).convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))
    return torch.stack(tensors)


def run_vision_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 2: VISION (4 Models)")
    print("=" * 60)

    try:
        n_images = CONFIG['vision']['n_images']
        pil_images = _load_cifar100_pil(n_images)
        batch = _pil_batch_to_tensor(pil_images).to(DEVICE)
        print(f"  Loaded {len(pil_images)} images  batch={tuple(batch.shape)}")

        base_embeddings = {}

        # Model 1: ViT
        try:
            print("    Loading ViT...")
            model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()
            with torch.no_grad():
                out = model(pixel_values=batch)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            base_embeddings['vit'] = emb
            print(f"    vit: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] ViT: {e}")

        # Model 2: CLIP
        try:
            print("    Loading CLIP...")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
            feats = []
            for img_pil in pil_images:
                inputs = processor(images=img_pil, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != 'input_ids'}
                with torch.no_grad():
                    out = model.get_image_features(**inputs)
                emb = out if isinstance(out, torch.Tensor) else \
                      (out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output)
                feats.append(emb.cpu().numpy().squeeze())
            base_embeddings['clip'] = np.vstack(feats)
            print(f"    clip: {base_embeddings['clip'].shape}")
            del model, processor
        except Exception as e:
            print(f"    [ERROR] CLIP: {e}")

        # Model 3: DeiT
        try:
            print("    Loading DeiT...")
            model = AutoModel.from_pretrained("facebook/deit-base-patch16-224").to(DEVICE).eval()
            with torch.no_grad():
                out = model(pixel_values=batch)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            base_embeddings['deit'] = emb
            print(f"    deit: {emb.shape}")
            del model
        except Exception as e:
            print(f"    [ERROR] DeiT: {e}")

        # Model 4: ResNet-50 via HuggingFace (no torchvision)
        try:
            print("    Loading ResNet-50 (HF)...")
            from transformers import AutoFeatureExtractor, ResNetModel
            fe    = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
            model = ResNetModel.from_pretrained("microsoft/resnet-50").to(DEVICE).eval()
            feats = []
            for img_pil in pil_images:
                inp = fe(images=img_pil, return_tensors="pt")
                inp = {k: v.to(DEVICE) for k, v in inp.items()}
                with torch.no_grad():
                    out = model(**inp)
                feats.append(out.pooler_output.squeeze().cpu().numpy())
            base_embeddings['resnet50'] = np.vstack(feats)
            print(f"    resnet50: {base_embeddings['resnet50'].shape}")
            del model, fe
        except Exception as e:
            print(f"    [ERROR] ResNet50: {e}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if not base_embeddings:
            return []

        all_results = []
        for seed in tqdm(SEEDS, desc="  Seeds"):
            all_results.extend(run_encoder_analysis(base_embeddings, seed, "Vision"))

        return all_results

    except Exception as e:
        print(f"Vision Failed: {e}")
        return []


# =============================================================================
# DOMAIN 3: AUDIO (2 Models)
# =============================================================================

def run_audio_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 3: AUDIO (2 Models)")
    print("=" * 60)
    
    tar_path = "librispeech.tar.gz"
    extract_dir = "libri_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    
    if not os.path.exists(tar_path):
        print("    Downloading LibriSpeech...")
        url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
        r = requests.get(url, stream=True)
        with open(tar_path, 'wb') as f:
            f.write(r.content)
    
    audios = []
    try:
        with tarfile.open(tar_path, "r") as tar:
            for m in tar:
                if len(audios) >= CONFIG['audio']['n_audio']:
                    break
                if m.name.endswith('.flac'):
                    tar.extract(m, path=extract_dir)
                    d, sr = sf.read(os.path.join(extract_dir, m.name))
                    d = librosa.resample(d, orig_sr=sr, target_sr=16000)
                    if len(d) > 16000:
                        d = d[:16000]
                    else:
                        d = np.pad(d, (0, 16000-len(d)))
                    audios.append(d)
    except Exception as e:
        print(f"    [ERROR] Extracting: {e}")
    
    if not audios:
        print("  No audio loaded")
        return []
    
    print(f"  Loaded {len(audios)} audio samples")
    base_embeddings = {}
    
    # Model 1: Wav2Vec2
    try:
        print("    Loading Wav2Vec2...")
        proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()
        
        feats = []
        for a in audios:
            inp = proc(a, sampling_rate=16000, return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state.mean(1).cpu().numpy())
        
        base_embeddings['wav2vec2'] = np.vstack(feats)
        print(f"    wav2vec2: {base_embeddings['wav2vec2'].shape}")
        del model, proc
    except Exception as e:
        print(f"    [ERROR] Wav2Vec2: {e}")
    
    # Model 2: HuBERT
    try:
        print("    Loading HuBERT...")
        proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE).eval()
        
        feats = []
        for a in audios:
            inp = proc(a, sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state.mean(1).cpu().numpy())
        
        base_embeddings['hubert'] = np.vstack(feats)
        print(f"    hubert: {base_embeddings['hubert'].shape}")
        del model, proc
    except Exception as e:
        print(f"    [ERROR] HuBERT: {e}")
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    if not base_embeddings:
        return []
    
    all_results = []
    for seed in tqdm(SEEDS, desc="  Seeds"):
        all_results.extend(run_encoder_analysis(base_embeddings, seed, "Audio"))
    
    return all_results


# =============================================================================
# DOMAIN 4: VIDEO UTILITIES (UCF-101 tiered loader + fixed extractors)
# =============================================================================

def _find_ucf_videos(root: str):
    root = Path(root)
    videos = []
    for ext in ("*.avi", "*.mp4", "*.AVI", "*.MP4"):
        videos.extend(root.rglob(ext))
    return sorted(videos)


def _pil_to_tensor_manual(img, size: int = 224):
    """PIL → normalised (3,H,W) float32 tensor, no torchvision required."""
    img = img.resize((size, size)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _read_clip(video_path: str, n_frames: int, size: int):
    """Decode n_frames uniformly from a video. Uses decord if available."""
    try:
        import decord as _decord
        vr = _decord.VideoReader(video_path, width=size, height=size)
        total = len(vr)
        if total < n_frames:
            return None
        idx = np.linspace(0, total - 1, n_frames, dtype=int)
        batch = vr.get_batch(idx)
        return [Image.fromarray(batch[i].numpy()) for i in range(n_frames)]
    except Exception:
        pass
    try:
        from torchvision.io import read_video
        vframes, _, _ = read_video(video_path, pts_unit="sec")
        if len(vframes) < n_frames:
            return None
        idx = np.linspace(0, len(vframes) - 1, n_frames, dtype=int)
        return [Image.fromarray(vframes[i].numpy()).resize((size, size)) for i in idx]
    except Exception:
        return None


def _load_ucf101_from_disk(n_videos, frames_per_clip, size, seed):
    video_files = _find_ucf_videos(UCF101_ROOT)
    if not video_files:
        return None
    rng = np.random.default_rng(seed)
    class_dirs   = sorted({p.parent.name for p in video_files})
    class_to_idx = {c: i for i, c in enumerate(class_dirs)}
    chosen       = [video_files[i] for i in
                    rng.choice(len(video_files), min(n_videos, len(video_files)), replace=False)]
    clips, labels = [], []
    for vpath in chosen:
        try:
            frames = _read_clip(str(vpath), frames_per_clip, size)
            if frames and len(frames) == frames_per_clip:
                clips.append(frames)
                labels.append(class_to_idx[vpath.parent.name])
        except Exception:
            pass
    return (clips, labels) if clips else None


def _load_ucf101_torchvision(n_videos, frames_per_clip, size, seed):
    try:
        import torchvision.datasets as tvd
        ann = Path(ANNOTATION_PATH)
        if not ann.exists():
            ann.mkdir(parents=True, exist_ok=True)
            import urllib.request, zipfile
            url = ("https://www.crcv.ucf.edu/data/UCF101/"
                   "UCF101TrainTestSplits-RecognitionTask.zip")
            zp = ann / "splits.zip"
            urllib.request.urlretrieve(url, zp)
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(ann.parent)
            zp.unlink()
        ds = tvd.UCF101(
            root=str(Path(UCF101_ROOT).parent),
            annotation_path=str(ann),
            frames_per_clip=frames_per_clip,
            step_between_clips=1,
            fold=1, train=False, transform=None,
        )
        rng  = np.random.default_rng(seed)
        idxs = rng.choice(len(ds), min(n_videos, len(ds)), replace=False)
        clips, labels = [], []
        for i in idxs:
            try:
                video, _, label = ds[int(i)]
                frames = [
                    Image.fromarray(video[j].numpy()).resize((size, size))
                    for j in np.linspace(0, len(video) - 1, frames_per_clip, dtype=int)
                ]
                clips.append(frames)
                labels.append(int(label))
            except Exception:
                pass
        return (clips, labels) if clips else None
    except Exception:
        return None


def _load_synthetic_clips(n_videos, frames_per_clip, size, seed, n_classes=10):
    print("    [Video] No UCF-101 found — using synthetic colour clips (smoke-test mode).")
    rng     = np.random.default_rng(seed)
    colours = rng.integers(0, 256, (n_classes, 3), dtype=np.uint8)
    clips, labels = [], []
    for i in range(n_videos):
        label  = i % n_classes
        colour = colours[label]
        frames = []
        for _ in range(frames_per_clip):
            noise = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            arr   = np.clip(colour.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(arr))
        clips.append(frames)
        labels.append(label)
    return clips, labels


def _load_video_clips(n_videos, frames_per_clip, size, seed):
    """Tiered loader: disk → auto-download → torchvision → synthetic."""
    # Tier 1: already on disk
    result = _load_ucf101_from_disk(n_videos, frames_per_clip, size, seed)
    if result:
        clips, labels = result
        print(f"    [Video] Loaded {len(clips)} UCF-101 clips from disk.")
        return clips, labels

    # Tier 2: try to download automatically, then retry from disk
    print("    [Video] UCF-101 not found on disk — attempting auto-download...")
    if _auto_download_ucf101():
        result = _load_ucf101_from_disk(n_videos, frames_per_clip, size, seed)
        if result:
            clips, labels = result
            print(f"    [Video] Loaded {len(clips)} UCF-101 clips after download.")
            return clips, labels

    # Tier 3: torchvision UCF101 (if installed)
    result = _load_ucf101_torchvision(n_videos, frames_per_clip, size, seed)
    if result:
        clips, labels = result
        print(f"    [Video] Loaded {len(clips)} UCF-101 clips via torchvision.")
        return clips, labels

    return _load_synthetic_clips(n_videos, frames_per_clip, size, seed)


# =============================================================================
# DOMAIN 4: VIDEO (4 Models)
# =============================================================================

def run_video_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 4: VIDEO (UCF-101, 4 Models)")
    print("=" * 60)

    n_videos       = CONFIG['video']['n_videos']
    frames_per_clip = CONFIG['video']['frames_per_video']
    video_size     = CONFIG['video']['video_size']

    # ------------------------------------------------------------------
    # Load clips: disk → torchvision UCF101 → synthetic fallback
    # ------------------------------------------------------------------
    try:
        videos, _labels = _load_video_clips(
            n_videos, frames_per_clip, video_size, seed=SEEDS[0]
        )
    except Exception as e:
        print(f"  [ERROR] Video loading failed: {e}")
        return []

    print(f"  Loaded {len(videos)} clips")
    base_embeddings = {}

    # ------------------------------------------------------------------
    # Model 1: TimeSformer — uses VideoMAEImageProcessor (no torchvision)
    # ------------------------------------------------------------------
    try:
        print("    Loading TimeSformer...")
        proc  = VideoMAEImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        model = AutoModel.from_pretrained("facebook/timesformer-base-finetuned-k400").to(DEVICE).eval()
        feats = []
        for v in videos:
            inp = proc(images=v[:8], return_tensors="pt")
            inp = {k: val.to(DEVICE) for k, val in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state.mean(1).cpu().numpy())
        base_embeddings['timesformer'] = np.vstack(feats)
        print(f"    timesformer: {base_embeddings['timesformer'].shape}")
        del model, proc
    except Exception as e:
        print(f"    [ERROR] TimeSformer: {e}")

    # ------------------------------------------------------------------
    # Model 2: VideoMAE
    # ------------------------------------------------------------------
    try:
        print("    Loading VideoMAE...")
        proc  = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(DEVICE).eval()
        feats = []
        for v in videos:
            inp = proc(images=v[:16], return_tensors="pt")
            inp = {k: val.to(DEVICE) for k, val in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state.mean(1).cpu().numpy())
        base_embeddings['videomae'] = np.vstack(feats)
        print(f"    videomae: {base_embeddings['videomae'].shape}")
        del model, proc
    except Exception as e:
        print(f"    [ERROR] VideoMAE: {e}")

    # ------------------------------------------------------------------
    # Model 3: ViT-B/16 on temporal mean frame (no torchvision)
    # ------------------------------------------------------------------
    try:
        print("    Loading ViT (mean frame)...")
        model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()
        try:
            from transformers import AutoFeatureExtractor
            vit_proc = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            use_proc = True
        except Exception:
            use_proc = False

        feats = []
        for v in videos:
            arr       = np.stack([np.array(f) for f in v]).mean(axis=0).astype(np.uint8)
            mean_pil  = Image.fromarray(arr)
            if use_proc:
                inp = vit_proc(images=mean_pil, return_tensors="pt")
                pixel_values = inp["pixel_values"].to(DEVICE)
            else:
                pixel_values = _pil_to_tensor_manual(mean_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(pixel_values=pixel_values)
            feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        base_embeddings['vit_meanframe'] = np.vstack(feats)
        print(f"    vit_meanframe: {base_embeddings['vit_meanframe'].shape}")
        del model
    except Exception as e:
        print(f"    [ERROR] ViT mean frame: {e}")

    # ------------------------------------------------------------------
    # Model 4: CLIP ViT-B/32 multi-frame
    # ------------------------------------------------------------------
    try:
        print("    Loading CLIP (multi-frame)...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        feats     = []
        for v in videos:
            fi = [0, 4, 8, 12] if len(v) >= 13 else list(range(min(4, len(v))))
            frame_embs = []
            for idx in fi:
                inputs = processor(images=v[idx], return_tensors="pt")
                inputs = {k: val.to(DEVICE) for k, val in inputs.items() if k != 'input_ids'}
                with torch.no_grad():
                    out = model.get_image_features(**inputs)
                # handle both tensor and BaseModelOutputWithPooling
                if isinstance(out, torch.Tensor):
                    emb = out
                else:
                    emb = out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output
                frame_embs.append(emb.cpu().numpy())
            feats.append(np.mean(frame_embs, axis=0))
        base_embeddings['clip_multiframe'] = np.vstack(feats)
        print(f"    clip_multiframe: {base_embeddings['clip_multiframe'].shape}")
        del model, processor
    except Exception as e:
        print(f"    [ERROR] CLIP multi-frame: {e}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if not base_embeddings:
        return []

    all_results = []
    for seed in tqdm(SEEDS, desc="  Seeds"):
        all_results.extend(run_encoder_analysis(base_embeddings, seed, "Video"))

    return all_results


# =============================================================================
# DOMAIN 5: NEUROSCIENCE (Steinmetz)
# =============================================================================

def run_neuroscience_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 5: NEUROSCIENCE (Full Dataset)")
    print("=" * 60)
    
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    urls = ["https://osf.io/agvxh/download", "https://osf.io/uv3mw/download"]
    fnames = [data_dir / "steinmetz_part1.npz", data_dir / "steinmetz_part2.npz"]

    for url, fname in zip(urls, fnames):
        if not fname.exists():
            print(f"    Downloading {fname.name}...")
            r = requests.get(url, timeout=300)
            with open(fname, "wb") as f:
                f.write(r.content)

    try:
        alldat = []
        for fname in fnames:
            if fname.exists():
                alldat.extend(np.load(fname, allow_pickle=True)['dat'])
        
        base_embeddings = {}
        for i, d in enumerate(alldat):
            spikes = d['spks']
            X = spikes.mean(axis=1).T  # (Trials, Neurons)
            
            if X.shape[1] >= CONFIG['neuroscience']['min_neurons'] and X.shape[0] >= CONFIG['neuroscience']['min_trials']:
                base_embeddings[f'session_{i:02d}'] = X
        
        print(f"  Loaded {len(base_embeddings)} valid sessions")
        
        if not base_embeddings:
            return []
        
        all_results = []
        for seed in tqdm(SEEDS, desc="  Seeds"):
            all_results.extend(run_encoder_analysis(base_embeddings, seed, "Neuroscience"))
        
        return all_results
    
    except Exception as e:
        print(f"Neuroscience Failed: {e}")
        return []


# =============================================================================
# DOMAIN 6: PROTEIN (Swiss-Prot + Multiple Encoders)
# =============================================================================

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
HYDRO = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
         'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
         'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
CHARGE = {'D':-1,'E':-1,'K':1,'R':1,'H':0.1,'A':0,'C':0,'F':0,'G':0,'I':0,
          'L':0,'M':0,'N':0,'P':0,'Q':0,'S':0,'T':0,'V':0,'W':0,'Y':0}


def load_swissprot(n_proteins, seed):
    """Load Swiss-Prot sequences."""
    filename = "uniprot_sprot.fasta"
    url = "https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28reviewed%3Atrue%29+AND+%28model_organism%3A9606%29&size=500"
    
    if not os.path.exists(filename):
        print("    Downloading Swiss-Prot...")
        try:
            r = requests.get(url, timeout=120)
            with open(filename, 'w') as f:
                f.write(r.text)
        except:
            return None
    
    seqs = []
    curr = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(">"):
                if curr:
                    seqs.append("".join(curr))
                curr = []
            else:
                curr.append(line.strip())
    if curr:
        seqs.append("".join(curr))
    
    AA_SET = set(AA_ALPHABET)
    seqs = [s for s in seqs if 50 <= len(s) <= 2000 and all(c in AA_SET for c in s)]
    
    rng = np.random.default_rng(seed)
    if len(seqs) > n_proteins:
        seqs = list(rng.choice(seqs, size=n_proteins, replace=False))
    
    return seqs


def build_protein_encoders(seqs, seed):
    """Build diverse protein encoders."""
    encoders = {}
    n_seqs = len(seqs)
    rng = np.random.default_rng(seed)
    
    # 1. AA Composition (20-dim)
    X_comp = np.zeros((n_seqs, 20))
    for i, s in enumerate(seqs):
        for c in s:
            if c in AA_INDEX:
                X_comp[i, AA_INDEX[c]] += 1
        X_comp[i] /= max(len(s), 1)
    encoders['aa_composition'] = X_comp
    
    # 2. Dipeptide (400-dim)
    dipeps = [a+b for a in AA_ALPHABET for b in AA_ALPHABET]
    dp_map = {dp: i for i, dp in enumerate(dipeps)}
    X_dp = np.zeros((n_seqs, 400))
    for i, s in enumerate(seqs):
        for j in range(len(s)-1):
            dp = s[j:j+2]
            if dp in dp_map:
                X_dp[i, dp_map[dp]] += 1
        X_dp[i] /= max(len(s)-1, 1)
    encoders['dipeptide'] = X_dp
    
    # 3. Hydrophobicity profiles at different resolutions
    for bins in [25, 50, 100]:
        X_hydro = []
        for s in seqs:
            vals = [HYDRO.get(c, 0) for c in s]
            if len(vals) < 2:
                vals = [0, 0]
            resampled = np.interp(np.linspace(0, len(vals)-1, bins), np.arange(len(vals)), vals)
            X_hydro.append(resampled)
        encoders[f'hydro_{bins}'] = np.vstack(X_hydro)
    
    # 4. Charge profiles
    for bins in [25, 50]:
        X_charge = []
        for s in seqs:
            vals = [CHARGE.get(c, 0) for c in s]
            if len(vals) < 2:
                vals = [0, 0]
            resampled = np.interp(np.linspace(0, len(vals)-1, bins), np.arange(len(vals)), vals)
            X_charge.append(resampled)
        encoders[f'charge_{bins}'] = np.vstack(X_charge)
    
    # 5. K-mer spectrum (k=3, hashed)
    X_kmer = np.zeros((n_seqs, 500))
    for i, s in enumerate(seqs):
        for j in range(len(s)-2):
            km = s[j:j+3]
            idx = hash(km) % 500
            X_kmer[i, idx] += 1
        X_kmer[i] /= max(len(s)-2, 1)
    encoders['kmer_3'] = X_kmer
    
    # 6. Noise Injection (Stress Test - crucial for consistency)
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
        noise = rng.normal(0, noise_level * np.std(X_dp), X_dp.shape)
        encoders[f'noise_{int(noise_level*100):03d}'] = X_dp + noise
    
    # 7. Combined
    encoders['combined'] = np.hstack([X_comp, X_dp, encoders['hydro_50'], encoders['charge_50']])
    
    return encoders


def run_protein_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 6: PROTEIN (Swiss-Prot)")
    print("=" * 60)
    
    all_results = []
    
    for seed in tqdm(SEEDS, desc="  Seeds"):
        seqs = load_swissprot(CONFIG['protein']['n_proteins'], seed)
        if not seqs:
            continue
        
        print(f"  Loaded {len(seqs)} sequences")
        encoders = build_protein_encoders(seqs, seed)
        
        # Run through transformation pipeline
        all_results.extend(run_encoder_analysis(encoders, seed, "Protein"))
    
    return all_results


# =============================================================================
# DOMAIN 7: MOLECULAR (PBMC3k + Multiple Encoders)
# =============================================================================

def build_molecular_encoders(X_raw, seed):
    """Build diverse molecular encoders."""
    X_log = np.log1p(X_raw)
    rng = np.random.default_rng(seed)
    encoders = {}
    n_cells, n_genes = X_log.shape
    
    # 1. PCA at various dimensions
    for k in [10, 25, 50, 75, 100, 150, 200]:
        k_actual = min(k, n_cells - 1, n_genes)
        if k_actual >= 5:
            try:
                if IS_GPU_PCA:
                    pca = PCA(n_components=k_actual)
                else:
                    pca = PCA(n_components=k_actual, random_state=seed)
                encoders[f"pca_{k:03d}"] = pca.fit_transform(X_log)
            except:
                pass
    
    # 2. Top variance genes
    gene_vars = X_log.var(axis=0)
    for k in [100, 500, 1000, 2000, 5000]:
        if k <= n_genes:
            idx = np.argsort(gene_vars)[-k:]
            encoders[f"topvar_{k:04d}"] = X_log[:, idx]
    
    # 3. Random gene subsets
    for k in [100, 500, 1000]:
        if k <= n_genes:
            idx = rng.choice(n_genes, k, replace=False)
            encoders[f"randgenes_{k:04d}"] = X_log[:, idx]
    
    # 4. Noise Injection (The "Stress Test")
    # Crucial for consistency with other domains
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
        noise = rng.normal(0, noise_level * np.std(X_log), X_log.shape)
        encoders[f"noise_{int(noise_level*100):03d}"] = X_log + noise
    
    # 5. Binarized (Biological "Presence/Absence")
    # Highly relevant for sparse scRNA-seq data
    encoders["binary"] = (X_raw > 0).astype(np.float32)
    
    # 6. Normalization variants
    scaler = StandardScaler()
    encoders["zscore"] = scaler.fit_transform(X_log)
    
    norms = np.linalg.norm(X_log, axis=1, keepdims=True) + 1e-12
    encoders["l2norm"] = X_log / norms
    
    # CPM-like normalization
    total_counts = X_raw.sum(axis=1, keepdims=True) + 1e-12
    encoders["cpm"] = np.log1p(X_raw / total_counts * 1e4)
    
    # 7. Original
    encoders["log1p_full"] = X_log.copy()
    
    return encoders


def build_molecular_encoders_gpu(X_raw, seed):
    """
    GPU-accelerated version of diverse molecular encoders.
    Replaces NumPy/Sklearn with PyTorch/cuML.
    """
    # 0. Infrastructure: Move to GPU and clear cache
    torch.cuda.empty_cache()
    if not isinstance(X_raw, torch.Tensor):
        X_raw_gpu = torch.tensor(X_raw, device=DEVICE, dtype=torch.float32)
    else:
        X_raw_gpu = X_raw.to(DEVICE).float()

    X_log = torch.log1p(X_raw_gpu)
    encoders = {}
    n_cells, n_genes = X_log.shape
    
    # Use PyTorch's generator for seeding GPU operations
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)

    # 1. PCA at various dimensions
    X_log_np = X_log.cpu().numpy() 
    
    for k in [10, 25, 50, 75, 100, 150, 200]:
        k_actual = min(k, n_cells - 1, n_genes)
        if k_actual >= 5:
            try:
                if IS_GPU_PCA:
                    pca = PCA(n_components=k_actual) 
                else:
                    from sklearn.decomposition import PCA as skPCA
                    pca = skPCA(n_components=k_actual, random_state=seed)
                encoders[f"pca_{k:03d}"] = pca.fit_transform(X_log_np)
            except Exception:
                pass
    
    # 2. Top variance genes (Vectorized on GPU)
    gene_vars = torch.var(X_log, dim=0)
    for k in [100, 500, 1000, 2000, 5000]:
        if k <= n_genes:
            _, idx = torch.topk(gene_vars, k)
            encoders[f"topvar_{k:04d}"] = X_log[:, idx].cpu().numpy()
    
    # 3. Random gene subsets
    for k in [100, 500, 1000]:
        if k <= n_genes:
            idx = torch.randperm(n_genes, generator=g, device=DEVICE)[:k]
            encoders[f"randgenes_{k:04d}"] = X_log[:, idx].cpu().numpy()
    
    # 4. Noise Injection (Parallelized on GPU cores)
    std_val = torch.std(X_log)
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
        noise = torch.randn(X_log.shape, generator=g, device=DEVICE) * (noise_level * std_val)
        encoders[f"noise_{int(noise_level*100):03d}"] = (X_log + noise).cpu().numpy()
    
    # 5. Binarized
    encoders["binary"] = (X_raw_gpu > 0).float().cpu().numpy()
    
    # 6. Normalization variants (With Manual Fallback)
    try:
        # Attempt cuML StandardScaler
        scaler = StandardScaler() 
        encoders["zscore"] = scaler.fit_transform(X_log_np)
    except Exception:
        # Fallback: Manual Torch Z-scoring (much faster than Sklearn CPU)
        mean = X_log.mean(0)
        std = X_log.std(0) + 1e-12
        encoders["zscore"] = ((X_log - mean) / std).cpu().numpy()
    
    # L2 Norm
    norms = torch.norm(X_log, p=2, dim=1, keepdim=True) + 1e-12
    encoders["l2norm"] = (X_log / norms).cpu().numpy()
    
    # CPM-like normalization
    total_counts = X_raw_gpu.sum(dim=1, keepdim=True) + 1e-12
    encoders["cpm"] = torch.log1p(X_raw_gpu / total_counts * 1e4).cpu().numpy()
    
    # 7. Original
    encoders["log1p_full"] = X_log.cpu().numpy()
    
    return encoders


def run_molecular_domain():
    print("\n" + "=" * 60)
    print("DOMAIN 7: MOLECULAR (PBMC3k)")
    print("=" * 60)
    
    try:
        adata = sc.datasets.pbmc3k()
        sc.pp.filter_genes(adata, min_cells=3)
        
        if adata.n_obs > CONFIG['molecular']['n_cells']:
            sc.pp.subsample(adata, n_obs=CONFIG['molecular']['n_cells'])
        
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        print(f"  Loaded PBMC3k: {X.shape}")
        
        all_results = []
        
        for seed in tqdm(SEEDS, desc="  Seeds"):
            # encoders = build_molecular_encoders(X, seed)
            encoders = build_molecular_encoders_gpu(X, seed)
            all_results.extend(run_encoder_analysis(encoders, seed, "Molecular"))
        
        return all_results
    
    except Exception as e:
        print(f"Molecular Failed: {e}")
        return []


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("DISTINCTION TEST - 7 DOMAINS")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Output: {OUTDIR}")
    
    all_results = []
    
    all_results.extend(run_language_domain())
    all_results.extend(run_vision_domain())
    all_results.extend(run_audio_domain())
    all_results.extend(run_video_domain())
    all_results.extend(run_neuroscience_domain())
    all_results.extend(run_protein_domain())
    all_results.extend(run_molecular_domain())
    
    if not all_results:
        print("\n[FATAL] No results collected!")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(OUTDIR / "raw_results_all_seeds.csv", index=False)
    print(f"\nSaved {len(df)} raw results")
    
    # Aggregate by (domain, base_model, encoder)
    df_agg = df.groupby(['domain', 'base_model', 'encoder']).agg({
        'SHESHA': 'mean',
        'CKA': 'mean',
        'n_features': 'first'
    }).reset_index()
    df_agg.to_csv(OUTDIR / "aggregated_by_encoder.csv", index=False)
    
    # Summary with bootstrap CIs
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\nTotal encoder configurations: {len(df_agg)}")
    print(f"Bootstrap resamples: {N_BOOTSTRAP:,}  (95 % CI)")
    print(f"\nPer-domain counts:")
    print(df_agg.groupby('domain').size())

    print(f"\nPer-domain statistics (mean [95% CI]) and rho:")
    all_shesha = []
    all_cka    = []

    for domain in sorted(df_agg['domain'].unique()):
        d     = df_agg[df_agg['domain'] == domain]
        valid = d.dropna(subset=['SHESHA', 'CKA'])
        if valid.empty:
            print(f"\n{domain}: no valid results")
            continue

        s_est, s_lo, s_hi = bootstrap_ci(valid['SHESHA'].values)
        c_est, c_lo, c_hi = bootstrap_ci(valid['CKA'].values)
        r_est, r_lo, r_hi = bootstrap_ci_rho(valid['SHESHA'].values,
                                              valid['CKA'].values)
        _, pval = spearmanr(valid['SHESHA'], valid['CKA']) \
                  if len(valid) >= 5 else (np.nan, np.nan)

        print(f"\n{domain} (N={len(valid)}):")
        print(f"  SHESHA: {s_est:.4f}  95% CI [{s_lo:.4f}, {s_hi:.4f}]")
        print(f"  CKA:    {c_est:.4f}  95% CI [{c_lo:.4f}, {c_hi:.4f}]")
        if np.isfinite(r_est):
            print(f"  rho:    {r_est:+.4f}  95% CI [{r_lo:+.4f}, {r_hi:+.4f}]"
                  f"  (p={pval:.4f})")

        all_shesha.extend(valid['SHESHA'].tolist())
        all_cka.extend(valid['CKA'].tolist())

    # Save per-domain CI table
    ci_rows = []
    for domain in sorted(df_agg['domain'].unique()):
        d     = df_agg[df_agg['domain'] == domain]
        valid = d.dropna(subset=['SHESHA', 'CKA'])
        if valid.empty:
            continue
        s_est, s_lo, s_hi = bootstrap_ci(valid['SHESHA'].values)
        c_est, c_lo, c_hi = bootstrap_ci(valid['CKA'].values)
        r_est, r_lo, r_hi = bootstrap_ci_rho(valid['SHESHA'].values,
                                              valid['CKA'].values)
        _, pval = spearmanr(valid['SHESHA'], valid['CKA']) \
                  if len(valid) >= 5 else (np.nan, np.nan)
        ci_rows.append({
            'domain': domain, 'N': len(valid),
            'SHESHA_mean': s_est, 'SHESHA_ci_lo': s_lo, 'SHESHA_ci_hi': s_hi,
            'CKA_mean':    c_est, 'CKA_ci_lo':    c_lo, 'CKA_ci_hi':    c_hi,
            'rho':         r_est, 'rho_ci_lo':    r_lo, 'rho_ci_hi':    r_hi,
            'rho_pval':    pval,  'n_bootstrap':  N_BOOTSTRAP,
        })
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(OUTDIR / "bootstrap_ci_summary.csv", index=False)
        print(f"\nBootstrap CI table saved → {OUTDIR / 'bootstrap_ci_summary.csv'}")

    # Aggregate across all domains
    print("\n" + "-" * 60)
    print("AGGREGATE (all domains):")
    if len(all_shesha) >= 5:
        r_est, r_lo, r_hi = bootstrap_ci_rho(all_shesha, all_cka)
        _, pval_agg = spearmanr(all_shesha, all_cka)
        print(f"  N   = {len(all_shesha)}")
        print(f"  rho = {r_est:+.4f}  95% CI [{r_lo:+.4f}, {r_hi:+.4f}]"
              f"  (p={pval_agg:.4f})")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()