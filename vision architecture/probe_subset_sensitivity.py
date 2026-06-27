"""
Probe Subset-Sensitivity Experiment
====================================
Tests the interpretability-relevance claim directly: does low SheshaFS
predict that linear probes are sensitive to WHICH feature subset is used?

Hypothesis
----------
A representation with low SheshaFS concentrates its geometry into a small
number of coordinate dimensions.  A linear probe trained on a random subset
of features will therefore give highly variable accuracy depending on which
subset is sampled.  A representation with high SheshaFS distributes its
geometry across the feature basis, so probe accuracy is stable regardless
of subset choice.

Prediction: across representations, the standard deviation of probe accuracy
over random feature subsets correlates negatively with SheshaFS.

This operationalises basis-dependent redundancy in concrete terms a
practitioner cares about (linear probing).

Design
------
For each representation X (n_samples, d) with labels y:
  1. Compute SheshaFS(X).
  2. For B random feature subsets of size floor(d * subset_frac):
       train a logistic-regression probe on the subset (train split),
       evaluate accuracy on a held-out test split.
  3. Record mean and std of probe accuracy across the B subsets.
The headline metric per representation is probe_acc_std: how much probe
accuracy swings depending on which features you happen to probe.

Across representations, correlate SheshaFS with probe_acc_std (expect
negative) and with probe_acc_mean (secondary; not the main claim).

Reuses the same models, datasets, and subset indices as
vision_architecture.py so the SheshaFS values match the main benchmark.

Usage:
    python probe_subset_sensitivity.py

    python probe_subset_sensitivity.py --dataset cifar10

    python probe_subset_sensitivity.py --dataset cifar10 --model-subsample 5

    python probe_subset_sensitivity.py \\
        --results-dir ./shesha-vision_architecture
"""

import argparse
import csv
import gc
import io
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# =============================================================================
# 0) CONFIGURATION
# =============================================================================

SEED = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12

# ---------------------------------------------------------------------------
# Output directory — same folder as vision_architecture.py for shared artifacts.
# ---------------------------------------------------------------------------
GDRIVE_FOLDER = None
DEFAULT_OUTPUT_DIR = "./shesha-vision_architecture"

DATASET_CONFIG = {
    "cifar10":    {"n_samples": 5000, "n_classes": 10},
    "cifar100":   {"n_samples": 5000, "n_classes": 100},
    "flowers102": {"n_samples": 5000, "n_classes": 102},
    "dtd":        {"n_samples": 1600, "n_classes": 47},
    "pets":       {"n_samples": 1500, "n_classes": 37},
    "eurosat":    {"n_samples": 5000, "n_classes": 10},
}

# =============================================================================
# 1) OUTPUT DIRECTORY
# =============================================================================

def _output_dir_accessible(folder: str) -> bool:
    if not folder:
        return False
    try:
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception:
        return False


def _resolve_output_dir(gdrive_folder: str = GDRIVE_FOLDER) -> Path:
    """Return output Path: optional override if writable, else local default."""
    if gdrive_folder and _output_dir_accessible(gdrive_folder):
        p = Path(gdrive_folder)
        print(f"[Output] Saving to: {p.resolve()}")
        return p
    local_path = Path(DEFAULT_OUTPUT_DIR)
    local_path.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Saving locally: {local_path.resolve()}")
    return local_path


OUTPUT_DIR = _resolve_output_dir()

# =============================================================================
# 2) FULL MODEL LIST  (self-contained; no import from vision_architecture.py)
# =============================================================================

_RAW_MODEL_LIST = [
    # ── CLIP ──────────────────────────────────────────────────────────────
    'vit_base_patch32_clip_224.openai',
    'vit_base_patch16_clip_224.openai',
    'vit_large_patch14_clip_224.openai',
    'vit_base_patch32_clip_224.laion400m_e32',
    'vit_base_patch16_clip_224.laion400m_e32',
    'vit_large_patch14_clip_224.laion400m_e32',
    'vit_base_patch32_clip_224.laion2b',
    'vit_base_patch16_clip_224.laion2b',
    'vit_large_patch14_clip_224.laion2b',
    'vit_huge_patch14_clip_224.laion2b',
    'vit_giant_patch14_clip_224.laion2b',
    'vit_base_patch32_clip_224.datacompxl',
    'vit_base_patch16_clip_224.datacompxl',
    'vit_large_patch14_clip_224.datacompxl',
    'eva02_enormous_patch14_clip_224.laion2b',
    'eva02_enormous_patch14_clip_224.laion2b_plus',
    'convnext_base.clip_laion2b',
    'convnext_large_mlp.clip_laion2b_augreg',
    'convnext_xxlarge.clip_laion2b_rewind',
    # ── DINOv2 ────────────────────────────────────────────────────────────
    'vit_small_patch14_dinov2.lvd142m',
    'vit_small_patch14_reg4_dinov2.lvd142m',
    'vit_base_patch14_dinov2.lvd142m',
    'vit_base_patch14_reg4_dinov2.lvd142m',
    'vit_large_patch14_dinov2.lvd142m',
    'vit_large_patch14_reg4_dinov2.lvd142m',
    'vit_giant_patch14_dinov2.lvd142m',
    'vit_giant_patch14_reg4_dinov2.lvd142m',
    # ── DINOv3 ────────────────────────────────────────────────────────────
    'vit_small_patch16_dinov3.lvd1689m',
    'vit_base_patch16_dinov3.lvd1689m',
    'vit_large_patch16_dinov3.lvd1689m',
    'vit_small_patch16_dinov3_qkvb.lvd1689m',
    'vit_base_patch16_dinov3_qkvb.lvd1689m',
    'vit_large_patch16_dinov3_qkvb.lvd1689m',
    # ── SigLIP ────────────────────────────────────────────────────────────
    'vit_base_patch16_siglip_224.webli',
    'vit_large_patch16_siglip_256.webli',
    'vit_so400m_patch14_siglip_224.webli',
    'vit_base_patch16_siglip_224.v2_webli',
    'vit_large_patch16_siglip_256.v2_webli',
    'vit_so400m_patch14_siglip_224.v2_webli',
    # ── BEiT family ───────────────────────────────────────────────────────
    'beit_base_patch16_224.in22k_ft_in22k',
    'beit_large_patch16_224.in22k_ft_in22k',
    'beitv2_base_patch16_224.in1k_ft_in22k',
    'beitv2_large_patch16_224.in1k_ft_in22k',
    'beit3_base_patch16_224.in22k_ft_in1k',
    'beit3_large_patch16_224.in22k_ft_in1k',
    # ── MAE / Hiera ───────────────────────────────────────────────────────
    'vit_base_patch16_224.mae',
    'vit_large_patch16_224.mae',
    'vit_huge_patch14_224.mae',
    'hiera_tiny_224.mae',
    'hiera_small_224.mae',
    'hiera_base_224.mae',
    'hiera_large_224.mae',
    'hiera_huge_224.mae',
    # ── EVA / EVA-02 ──────────────────────────────────────────────────────
    'eva02_tiny_patch14_224.mim_in22k',
    'eva02_small_patch14_224.mim_in22k',
    'eva02_base_patch14_224.mim_in22k',
    'eva02_large_patch14_224.mim_in22k',
    'eva02_large_patch14_224.mim_m38m',
    'eva_large_patch14_196.in22k_ft_in1k',
    'eva_giant_patch14_336.m30m_ft_in22k_in1k',
    # ── SAM ViT ───────────────────────────────────────────────────────────
    'samvit_base_patch16.sa1b',
    'samvit_large_patch16.sa1b',
    'samvit_huge_patch16.sa1b',
    'vit_base_patch16_224.sam_in1k',
    # ── I-JEPA ────────────────────────────────────────────────────────────
    'vit_huge_patch14_gap_224.in1k_ijepa',
    'vit_huge_patch14_gap_224.in22k_ijepa',
    'vit_giant_patch16_gap_224.in22k_ijepa',
    # ── SwinV2 22k ────────────────────────────────────────────────────────
    'swinv2_base_window12_192.ms_in22k',
    'swinv2_large_window12_192.ms_in22k',
    'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
    'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
    # ── ViTamin ───────────────────────────────────────────────────────────
    'vitamin_small_224.datacomp1b_clip',
    'vitamin_base_224.datacomp1b_clip',
    'vitamin_large_224.datacomp1b_clip',
    'vitamin_large2_224.datacomp1b_clip',
    'vitamin_xlarge_256.datacomp1b_clip',
    # ── DINO v1 + ResMLP ──────────────────────────────────────────────────
    'vit_small_patch16_224.dino',
    'vit_base_patch16_224.dino',
    'vit_small_patch8_224.dino',
    'vit_base_patch8_224.dino',
    'resmlp_12_224.fb_dino',
    'resmlp_24_224.fb_dino',
    # ── RegNetY large (SWAG/SEER) ─────────────────────────────────────────
    'regnety_160.swag_ft_in1k',
    'regnety_320.swag_ft_in1k',
    'regnety_1280.swag_ft_in1k',
    'regnety_160.deit_in1k',
    'regnety_320.seer_ft_in1k',
    'regnety_1280.seer_ft_in1k',
    # ── Transformers ──────────────────────────────────────────────────────
    'swin_tiny_patch4_window7_224.ms_in1k',
    'swin_small_patch4_window7_224.ms_in1k',
    'swin_base_patch4_window7_224.ms_in1k',
    'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
    'swinv2_tiny_window8_256.ms_in1k',
    'swinv2_small_window8_256.ms_in1k',
    'pvt_v2_b0.in1k',
    'pvt_v2_b1.in1k',
    'pvt_v2_b2.in1k',
    'pvt_v2_b3.in1k',
    'pvt_v2_b5.in1k',
    'poolformer_s12.sail_in1k',
    'poolformer_s24.sail_in1k',
    'poolformer_m36.sail_in1k',
    'deit_tiny_patch16_224.fb_in1k',
    'deit_small_patch16_224.fb_in1k',
    'deit_base_patch16_224.fb_in1k',
    'deit3_small_patch16_224.fb_in1k',
    'deit3_base_patch16_224.fb_in1k',
    'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
    'vit_small_patch16_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'vit_large_patch16_224.augreg_in21k_ft_in1k',
    'maxvit_tiny_tf_224.in1k',
    'maxvit_small_tf_224.in1k',
    'coatnet_0_rw_224.sw_in1k',
    'coatnet_1_rw_224.sw_in1k',
    # ── CNNs ──────────────────────────────────────────────────────────────
    'convnext_atto.d2_in1k',
    'convnext_femto.d1_in1k',
    'convnext_pico.d1_in1k',
    'convnext_nano.d1h_in1k',
    'convnext_tiny.fb_in1k',
    'convnext_small.fb_in1k',
    'convnext_base.fb_in1k',
    'convnext_large.fb_in1k',
    'convnextv2_atto.fcmae_ft_in1k',
    'convnextv2_nano.fcmae_ft_in1k',
    'convnextv2_tiny.fcmae_ft_in1k',
    'convnextv2_base.fcmae_ft_in1k',
    'efficientnet_b0.ra_in1k',
    'efficientnet_b1.ft_in1k',
    'efficientnet_b2.ra_in1k',
    'efficientnet_b3.ra2_in1k',
    'efficientnetv2_rw_s.ra2_in1k',
    'efficientnetv2_rw_m.agc_in1k',
    'tf_efficientnetv2_s.in1k',
    'tf_efficientnetv2_m.in1k',
    'tf_efficientnetv2_b0.in1k',
    'tf_efficientnetv2_b3.in1k',
    'regnety_002.pycls_in1k',
    'regnety_004.pycls_in1k',
    'regnety_008.pycls_in1k',
    'regnety_016.pycls_in1k',
    'regnety_032.pycls_in1k',
    'regnety_064.pycls_in1k',
    'regnetx_002.pycls_in1k',
    'regnetx_004.pycls_in1k',
    'regnetx_008.pycls_in1k',
    'resnet18.a1_in1k',
    'resnet34.a1_in1k',
    'resnet50.a1_in1k',
    'resnet101.a1_in1k',
    'resnet152.a1_in1k',
    'resnext50_32x4d.a1_in1k',
    'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'densenet121.ra_in1k',
    'densenet169.tv_in1k',
    'densenet201.tv_in1k',
    'mobilenetv3_small_100.lamb_in1k',
    'mobilenetv3_large_100.ra_in1k',
    'inception_v3.tf_in1k',
    'inception_v4.tf_in1k',
    # ── Robust variants ───────────────────────────────────────────────────
    'resnet50_gn.a1h_in1k',
    'resnet50.a2_in1k',
    'resnet50.a3_in1k',
    'vit_base_patch16_224.augreg_in21k',
    'vit_base_patch16_224.augreg_in1k',
    'wide_resnet50_2.racm_in1k',
    'wide_resnet101_2.tv_in1k',
    'resnetv2_50.a1h_in1k',
    'resnetv2_101.a1h_in1k',
    'resnetrs50.tf_in1k',
    'resnetrs101.tf_in1k',
]

_EXCLUDED_MODELS = {'vitamin_base_224.datacomp1b_clip'}
_seen: set = set()
FULL_MODEL_LIST: list = []
for _m in _RAW_MODEL_LIST:
    if _m not in _seen:
        _seen.add(_m)
        FULL_MODEL_LIST.append(_m)
FULL_MODEL_LIST = [m for m in FULL_MODEL_LIST if m not in _EXCLUDED_MODELS]


# =============================================================================
# 3) UTILITIES
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# 4) SHESHA-FS  (inlined from vision_architecture.py — no external dependency)
# =============================================================================

def compute_shesha_feature_split(X: np.ndarray, n_splits: int = 30,
                                  seed: int = SEED) -> float:
    """Reliability via correlation of random feature-subspace RDMs."""
    n_samples, n_features = X.shape
    if n_features < 2 or n_samples < 4:
        return 0.0
    correlations = []
    rng = np.random.default_rng(seed)
    for _ in range(n_splits):
        feats = np.arange(n_features)
        rng.shuffle(feats)
        mid = n_features // 2
        X1, X2 = X[:, feats[:mid]], X[:, feats[mid:]]
        valid = (
            (np.linalg.norm(X1, axis=1) > EPS) &
            (np.linalg.norm(X2, axis=1) > EPS)
        )
        if valid.sum() < 4:
            continue
        d1 = pdist(X1[valid], 'cosine')
        d2 = pdist(X2[valid], 'cosine')
        rho, _ = spearmanr(d1, d2)
        correlations.append(rho if not np.isnan(rho) else 0.0)
    return float(np.mean(correlations)) if correlations else 0.0


# =============================================================================
# 5) PROBE SUBSET-SENSITIVITY CORE
# =============================================================================

def probe_subset_sensitivity(
    X: np.ndarray,
    y: np.ndarray,
    n_subsets: int = 20,
    subset_frac: float = 0.5,
    test_size: float = 0.4,
    max_iter: int = 200,
    seed: int = SEED,
) -> dict:
    """
    Train logistic-regression probes on random feature subsets and measure
    variability of test accuracy across subsets.

    Returns a dict with probe accuracy mean, std, and per-subset values.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = max(2, int(np.floor(d * subset_frac)))

    # Fixed train/test split over samples, shared across all subsets so that
    # variability reflects feature choice, not sample choice.
    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(n), y,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    accuracies = []
    for _ in range(n_subsets):
        cols = rng.choice(d, size=k, replace=False)
        Xs = X[:, cols]

        scaler = StandardScaler().fit(Xs[idx_train])
        Xtr = scaler.transform(Xs[idx_train])
        Xte = scaler.transform(Xs[idx_test])

        clf = LogisticRegression(max_iter=max_iter, multi_class="auto", n_jobs=-1)
        clf.fit(Xtr, y_train)
        accuracies.append(clf.score(Xte, y_test))

    accuracies = np.array(accuracies)
    return {
        "probe_acc_mean":  float(accuracies.mean()),
        "probe_acc_std":   float(accuracies.std()),
        "probe_acc_min":   float(accuracies.min()),
        "probe_acc_max":   float(accuracies.max()),
        "probe_acc_range": float(accuracies.max() - accuracies.min()),
        "subset_size":     int(k),
        "feature_dim":     int(d),
        "accuracies":      accuracies.tolist(),
    }


# =============================================================================
# 6) DATA LOADING  (mirrors vision_architecture.py)
# =============================================================================

def get_raw_dataset(name: str):
    if name == "cifar10":
        ds = datasets.CIFAR10(root="./data", train=False, download=True)
        labels = np.array(ds.targets)
    elif name == "cifar100":
        ds = datasets.CIFAR100(root="./data", train=False, download=True)
        labels = np.array(ds.targets)
    elif name == "flowers102":
        ds = datasets.Flowers102(root="./data", split="test", download=True)
        labels = np.array(ds._labels)
    elif name == "dtd":
        ds = datasets.DTD(root="./data", split="test", download=True)
        labels = np.array(ds._labels)
    elif name == "pets":
        ds = datasets.OxfordIIITPet(root="./data", split="test", download=True)
        labels = np.array(ds._labels)
    elif name == "eurosat":
        ds = datasets.EuroSAT(root="./data", download=True)
        labels = np.array(ds.targets)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds, labels


_FALLBACK_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def get_transform_from_model(model) -> transforms.Compose:
    """
    Extract the correct preprocessing transform from an already-loaded timm
    model.  Uses the live model object so no second model allocation is needed.
    """
    try:
        from timm.data import resolve_data_config, create_transform
        cfg = resolve_data_config({}, model=model)
        return create_transform(**cfg, is_training=False)
    except Exception:
        return _FALLBACK_TRANSFORM


def _pool(feats):
    if isinstance(feats, (tuple, list)):
        feats = feats[-1]
    if feats.dim() == 3:
        feats = feats.mean(dim=1)
    elif feats.dim() == 4:
        feats = feats.mean(dim=(2, 3))
    return feats


@torch.no_grad()
def extract(model, loader) -> np.ndarray:
    model.eval()
    chunks = []
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        chunks.append(_pool(model(imgs)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


class _ApplyTransform(torch.utils.data.Dataset):
    """
    Wraps a base dataset and applies `transform` in __getitem__.

    Avoids mutating dataset.transform / dataset.transforms, which is
    unreliable across torchvision versions: newer datasets (OxfordIIITPet,
    EuroSAT, DTD, Flowers102) use an internal self.transforms object that
    is frozen at __init__ time, so post-hoc assignment of .transform is
    silently ignored and the DataLoader receives raw PIL images.
    """
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), label


# =============================================================================
# 7) MODEL LIST HELPERS
# =============================================================================

def get_model_list() -> list:
    """Return the canonical model list, filtered to timm-available models."""
    all_timm = set(timm.list_models(pretrained=True))
    available = [m for m in FULL_MODEL_LIST if m in all_timm]
    if len(available) < len(FULL_MODEL_LIST):
        print(f"  [ModelList] {len(FULL_MODEL_LIST) - len(available)} models "
              "not found in timm (pretrained); skipping them.")
    return available


def models_from_results(results_dir: Path, dataset: str) -> list | None:
    """Fallback: read model names from an existing result CSV."""
    for pat in [
        f"{dataset.upper()}_AVERAGED_*.csv",
        f"{dataset.upper()}_ALL_SEEDS_*.csv",
        f"{dataset.upper()}_SEED{SEED}_*.csv",
    ]:
        for f in sorted(results_dir.glob(pat)):
            df = pd.read_csv(f)
            if "Model" in df.columns:
                return df["Model"].unique().tolist()
    return None


# =============================================================================
# 8) CHECKPOINT / RESUME HELPERS
# =============================================================================

# Canonical column order — every row (successful or skip) always has all of
# these, so the CSV schema never changes mid-file.
_CKPT_COLUMNS = [
    "Model", "Dataset", "Seed",
    "SHESHA_FS",
    "probe_acc_mean", "probe_acc_std", "probe_acc_range",
    "probe_acc_min",  "probe_acc_max",
    "subset_size", "feature_dim",
    "skip_reason",
]


def _checkpoint_path(output_dir: Path, dataset: str) -> Path:
    """Deterministic path for the per-dataset incremental checkpoint file."""
    return output_dir / f".ckpt_probe_sensitivity_{dataset}_seed{SEED}.csv"


def _load_checkpoint(ckpt_path: Path) -> tuple:
    """
    Load an existing checkpoint CSV robustly.

    Handles the common corruption case where earlier rows have 11 columns
    (old schema without skip_reason) and later rows have 12 columns (new
    schema).  pandas refuses to read this because the field count doesn't
    match the header.

    Strategy: parse every line with csv.reader, pad short rows / trim long
    rows to the canonical column list, and build a DataFrame manually.

    Returns (list_of_row_dicts, set_of_completed_model_names).
    """
    if not (ckpt_path.exists() and ckpt_path.stat().st_size > 0):
        return [], set()

    # ── Attempt 1: strict pandas read (fast path for clean files) ─────────
    try:
        df = pd.read_csv(ckpt_path)
        if "Model" in df.columns:
            done = set(df["Model"].dropna().tolist())
            print(f"  [Resume] Loaded {len(done)} completed models from {ckpt_path.name}")
            return df.to_dict("records"), done
    except Exception as exc:
        print(f"  [Resume] Strict read failed ({exc}); trying row-by-row recovery ...")

    # ── Attempt 2: row-by-row with csv.reader ────────────────────────────
    # This handles mixed column counts (11 vs 12), unquoted commas in
    # skip_reason, and other minor corruption.
    try:
        text = ckpt_path.read_text(encoding="utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        file_header = next(reader)
        n_file_cols = len(file_header)

        # Use the canonical header so all rows end up with the same schema
        target_cols = list(_CKPT_COLUMNS)
        n_target = len(target_cols)

        recovered_rows = []
        n_skipped = 0
        for line_no, fields in enumerate(reader, start=2):
            if not fields or not fields[0].strip():
                n_skipped += 1
                continue

            # If the file header is shorter than canonical (old schema), the
            # data rows may also be short — pad with empty strings.
            # If a row is wider than canonical (unquoted comma in last field),
            # rejoin the excess fields into the last canonical column.
            if len(fields) < n_target:
                fields = fields + [""] * (n_target - len(fields))
            elif len(fields) > n_target:
                # Everything past the last expected column is part of skip_reason
                excess = fields[n_target - 1:]
                fields = fields[:n_target - 1] + [",".join(excess)]

            # If the file's own header was shorter than canonical, the fields
            # are already padded above; map them positionally.
            row_dict = dict(zip(target_cols, fields))

            # Secondary header rows from mid-file appends (duplicate headers)
            if row_dict.get("Model") == "Model":
                continue

            recovered_rows.append(row_dict)

        if not recovered_rows:
            print(f"  [Resume] No valid rows recovered from {ckpt_path.name}")
            return [], set()

        df = pd.DataFrame(recovered_rows, columns=target_cols)

        # Convert numeric columns back from strings
        for col in ["Seed", "subset_size", "feature_dim"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["SHESHA_FS", "probe_acc_mean", "probe_acc_std",
                     "probe_acc_range", "probe_acc_min", "probe_acc_max"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        done = set(df["Model"].dropna().tolist())
        status = f"{len(done)} models recovered"
        if n_skipped:
            status += f" ({n_skipped} bad lines skipped)"
        print(f"  [Resume] {status} from {ckpt_path.name}")
        return df.to_dict("records"), done

    except Exception as exc2:
        print(f"  [Resume] Row-by-row recovery also failed ({exc2}). Starting fresh.")
        broken = ckpt_path.with_suffix(".broken.csv")
        try:
            ckpt_path.rename(broken)
            print(f"  [Resume] Broken checkpoint preserved as {broken.name}")
        except Exception:
            pass
        return [], set()


def _append_checkpoint(ckpt_path: Path, row: dict) -> None:
    """
    Append a single model-result row to the checkpoint CSV.

    Always writes ALL canonical columns (fills missing ones with empty string)
    and uses QUOTE_NONNUMERIC so that any commas inside skip_reason strings
    are safely wrapped in quotes and never break the column count on re-read.
    """
    # Ensure every column is present; fill gaps with empty string
    ordered = {col: row.get(col, "") for col in _CKPT_COLUMNS}
    df_row = pd.DataFrame([ordered])[_CKPT_COLUMNS]
    write_header = not ckpt_path.exists() or ckpt_path.stat().st_size == 0
    df_row.to_csv(
        ckpt_path, mode="a", header=write_header, index=False,
        quoting=csv.QUOTE_NONNUMERIC,  # quotes all string fields — commas are safe
    )


# =============================================================================
# 9) MAIN LOOP
# =============================================================================

def _print_correlations(df: pd.DataFrame, dataset: str) -> None:
    if len(df) >= 5:
        rho_std, p_std = spearmanr(df["SHESHA_FS"], df["probe_acc_std"])
        rho_rng, p_rng = spearmanr(df["SHESHA_FS"], df["probe_acc_range"])
        print(f"\n{'=' * 70}")
        print(f"Results: {dataset.upper()} (n={len(df)} models)")
        print(f"{'=' * 70}")
        print(f"  rho(SheshaFS, probe_acc_std)   = {rho_std:+.3f}  p={p_std:.2e}")
        print(f"  rho(SheshaFS, probe_acc_range) = {rho_rng:+.3f}  p={p_rng:.2e}")
        print(f"  (negative rho supports the hypothesis)")


def run(
    dataset: str,
    results_dir: Path,
    output_dir: Path,
    n_subsets: int = 20,
    subset_frac: float = 0.5,
    batch_size: int = 64,
    num_workers: int = 4,
    model_subsample: int | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    set_seed(SEED)
    cfg = DATASET_CONFIG[dataset]

    # Reuse the same sample subset as the original benchmark run
    idx_path = results_dir / f"{dataset}_seed{SEED}_subset_idx.npy"
    if idx_path.exists():
        subset_idx = np.load(idx_path)
        print(f"  Loaded subset indices from {idx_path.name} (n={len(subset_idx)})")
    else:
        print(f"  [WARN] No subset index at {idx_path} — generating fresh with seed {SEED}")
        raw_tmp, _ = get_raw_dataset(dataset)
        n = min(cfg["n_samples"], len(raw_tmp))
        np.random.seed(SEED)
        subset_idx = np.random.choice(len(raw_tmp), n, replace=False)

    model_names = get_model_list()
    if not model_names:
        model_names = models_from_results(results_dir, dataset) or []
    if not model_names:
        raise RuntimeError(
            f"Cannot determine model list for {dataset}. "
            "Ensure timm is installed or provide results CSVs in results_dir."
        )

    if model_subsample:
        model_names = model_names[:model_subsample]

    raw, labels_full = get_raw_dataset(dataset)
    y = labels_full[subset_idx]

    # ── Checkpoint / resume ───────────────────────────────────────────────
    ckpt_path = _checkpoint_path(output_dir, dataset)
    rows, done_models = [], set()
    if resume:
        rows, done_models = _load_checkpoint(ckpt_path)

        # Fallback: if no checkpoint exists (previous run completed and
        # deleted it), recover done-model names from any final output CSV
        # so we don't needlessly re-run every model.
        if not done_models:
            for pat in [
                f"{dataset.upper()}_PROBE_SENSITIVITY_SEED{SEED}_2*.csv",
                f"{dataset.upper()}_PROBE_SENSITIVITY_SEED{SEED}_PARTIAL_*.csv",
            ]:
                for csv_path in sorted(output_dir.glob(pat), reverse=True):
                    try:
                        prev = pd.read_csv(csv_path)
                        if "Model" in prev.columns:
                            prev_models = set(prev["Model"].dropna().tolist())
                            if prev_models:
                                rows = prev.to_dict("records")
                                done_models = prev_models
                                print(f"  [Resume] Recovered {len(done_models)} models "
                                      f"from completed CSV: {csv_path.name}")
                                break
                    except Exception:
                        continue
                if done_models:
                    break

    # Filter checkpoint/CSV rows to the current model list and deduplicate.
    # Previous runs may have used a different (larger) model list; carrying
    # those extra rows forward inflates the result count silently.
    valid_set = set(model_names)
    seen_in_rows: dict[str, int] = {}
    filtered_rows: list[dict] = []
    for r in rows:
        m = r.get("Model")
        if m not in valid_set:
            continue
        if m in seen_in_rows:
            filtered_rows[seen_in_rows[m]] = r
        else:
            seen_in_rows[m] = len(filtered_rows)
            filtered_rows.append(r)
    if len(filtered_rows) < len(rows):
        n_dropped = len(rows) - len(filtered_rows)
        print(f"  [Resume] Dropped {n_dropped} rows not in current model list "
              f"({len(rows)} -> {len(filtered_rows)})")
    rows = filtered_rows
    done_models = done_models & valid_set

    models_todo = [m for m in model_names if m not in done_models]
    n_resumed = len(done_models)

    print(f"\n{'=' * 70}")
    print(f"Probe Subset-Sensitivity: {dataset.upper()} | {len(model_names)} models")
    print(f"  n_subsets={n_subsets}  subset_frac={subset_frac}  resume={resume}")
    if n_resumed:
        print(f"  Resumed: {n_resumed} already done, {len(models_todo)} remaining")
    print(f"  results_dir : {results_dir}")
    print(f"  output_dir  : {output_dir}")
    print(f"{'=' * 70}")

    subset_idx_list = subset_idx.tolist()  # compute once outside the loop

    def _is_oom(exc):
        if not isinstance(exc, RuntimeError):
            return False
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg

    def _make_skip_row(model_name, reason):
        return {
            "Model":            model_name,
            "Dataset":          dataset,
            "Seed":             SEED,
            "SHESHA_FS":        float("nan"),
            "probe_acc_mean":   float("nan"),
            "probe_acc_std":    float("nan"),
            "probe_acc_range":  float("nan"),
            "probe_acc_min":    float("nan"),
            "probe_acc_max":    float("nan"),
            "subset_size":      0,
            "feature_dim":      0,
            "skip_reason":      reason,
        }

    interrupted = False
    for m_name in tqdm(models_todo, desc=dataset,
                       initial=n_resumed, total=len(model_names)):
        tqdm.write(f"  -> {m_name}")
        model  = None
        loader = None
        X      = None
        try:
            # ── Load model once; derive transform from it ────────────────
            # IMPORTANT: do NOT call timm.create_model a second time just
            # for the transform config — that leaks a full CPU model per
            # iteration and exhausts RAM after ~15 large models.
            model = timm.create_model(m_name, pretrained=True, num_classes=0).to(DEVICE)
            tf = get_transform_from_model(model)

            # ── Feature extraction with OOM batch-size backoff ──────────
            cur_bs = batch_size
            while True:
                loader = DataLoader(
                    _ApplyTransform(raw, subset_idx_list, tf),
                    batch_size=cur_bs, shuffle=False,
                    num_workers=num_workers,
                    pin_memory=(DEVICE.type == "cuda"),
                )
                try:
                    X = extract(model, loader)
                    break
                except RuntimeError as oom_exc:
                    del loader
                    loader = None
                    release()
                    if not _is_oom(oom_exc):
                        raise
                    next_bs = max(8, cur_bs // 2)
                    if next_bs >= cur_bs:
                        raise  # already at minimum
                    tqdm.write(f"  [OOM] batch_size {cur_bs} -> {next_bs}, retrying")
                    cur_bs = next_bs

            # ── Free GPU memory before CPU-bound probe step ──────────────
            del model
            model = None
            if loader is not None:
                del loader
                loader = None
            release()

            # ── Metrics ──────────────────────────────────────────────────
            shesha_fs = compute_shesha_feature_split(X, n_splits=30, seed=SEED)
            sens = probe_subset_sensitivity(
                X, y, n_subsets=n_subsets, subset_frac=subset_frac, seed=SEED,
            )
            del X
            X = None

            row = {
                "Model":            m_name,
                "Dataset":          dataset,
                "Seed":             SEED,
                "SHESHA_FS":        shesha_fs,
                "probe_acc_mean":   sens["probe_acc_mean"],
                "probe_acc_std":    sens["probe_acc_std"],
                "probe_acc_range":  sens["probe_acc_range"],
                "probe_acc_min":    sens["probe_acc_min"],
                "probe_acc_max":    sens["probe_acc_max"],
                "subset_size":      sens["subset_size"],
                "feature_dim":      sens["feature_dim"],
                "skip_reason":      "",
            }
            rows.append(row)
            _append_checkpoint(ckpt_path, row)
            release()

        except KeyboardInterrupt:
            print(f"\n  [Interrupted] Progress saved ({len(rows)}/{len(model_names)} models).")
            print(f"  Re-run with the same command to resume from here.")
            if model  is not None: del model
            if loader is not None: del loader
            if X      is not None: del X
            release()
            interrupted = True
            break

        except RuntimeError as exc:
            reason = "oom_skip" if _is_oom(exc) else f"runtime_error: {exc}"
            tqdm.write(f"  [Skip] {m_name}: {reason}")
            row = _make_skip_row(m_name, reason)
            rows.append(row)
            _append_checkpoint(ckpt_path, row)
            if model  is not None: del model
            if loader is not None: del loader
            if X      is not None: del X
            release()

        except Exception as exc:
            tqdm.write(f"  [Error] {m_name}: {exc}")
            if model  is not None: del model
            if loader is not None: del loader
            if X      is not None: del X
            release()

    df = pd.DataFrame(rows)

    if interrupted:
        # Keep checkpoint so the next run can resume; save a partial CSV too.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partial_path = output_dir / f"{dataset.upper()}_PROBE_SENSITIVITY_SEED{SEED}_PARTIAL_{timestamp}.csv"
        df.to_csv(partial_path, index=False)
        print(f"  Partial results saved: {partial_path.name}")
        print(f"  Checkpoint kept at   : {ckpt_path.name}")
        _print_correlations(df, dataset)
        return df

    # Full run complete — write final CSV and remove checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{dataset.upper()}_PROBE_SENSITIVITY_SEED{SEED}_{timestamp}.csv"
    df.to_csv(out_path, index=False)

    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  [Checkpoint] Removed {ckpt_path.name} (run complete)")

    _print_correlations(df, dataset)
    print(f"\n  Saved: {out_path.name}")
    return df


# =============================================================================
# 10) MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Probe subset-sensitivity experiment — checkpoints after every model so it can resume if interrupted.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python probe_subset_sensitivity.py --all-datasets --num-workers 0

  python probe_subset_sensitivity.py --dataset cifar10

  python probe_subset_sensitivity.py --dataset cifar10 --no-resume

  python probe_subset_sensitivity.py --dataset cifar10 --model-subsample 5
        """,
    )
    ap.add_argument(
        "--results-dir", default=None,
        help="Directory containing subset-index .npy files and benchmark CSVs. "
             f"Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Where to write result CSVs (default: same as --results-dir / OUTPUT_DIR).",
    )
    ap.add_argument(
        "--dataset", default="cifar10",
        choices=list(DATASET_CONFIG.keys()),
        help="Single dataset to run (default: cifar10). Ignored if --all-datasets is set.",
    )
    ap.add_argument(
        "--all-datasets", action="store_true",
        help="Run all 6 datasets sequentially, resuming each from its checkpoint.",
    )
    ap.add_argument("--n-subsets",      type=int,   default=20)
    ap.add_argument("--subset-frac",    type=float, default=0.5)
    ap.add_argument("--batch-size",     type=int,   default=64)
    ap.add_argument("--num-workers",    type=int,   default=4)
    ap.add_argument(
        "--model-subsample", type=int, default=None,
        help="Process only the first N models (quick smoke-test).",
    )
    ap.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing checkpoint and start fresh.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else OUTPUT_DIR
    output_dir  = Path(args.output_dir)  if args.output_dir  else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = (
        list(DATASET_CONFIG.keys()) if args.all_datasets else [args.dataset]
    )

    print(f"\n[ProbeExp] datasets    : {datasets_to_run}")
    print(f"[ProbeExp] results_dir : {results_dir}")
    print(f"[ProbeExp] output_dir  : {output_dir}")
    print(f"[ProbeExp] device      : {DEVICE}")
    print(f"[ProbeExp] resume      : {not args.no_resume}\n")

    for ds in datasets_to_run:
        run(
            dataset=ds,
            results_dir=results_dir,
            output_dir=output_dir,
            n_subsets=args.n_subsets,
            subset_frac=args.subset_frac,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_subsample=args.model_subsample,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
