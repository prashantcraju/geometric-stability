"""
Shesha Vision Architecture Experiment — Corrupted Benchmarks
Runs on CIFAR-10-C and CIFAR-100-C (Hendrycks & Dietterich corruption benchmarks).

Expected dataset layout on disk:
  ./data/CIFAR-10-C/
      labels.npy
      <corruption_type>.npy

  ./data/CIFAR-100-C/
      labels.npy
      <corruption_type>.npy

  Results CSVs: ./shesha-vision_architecture-corrupt/  (or --output-dir override)

Run download_cifar_c.py to fetch CIFAR-C archives into ./data/.
"""
import gc
import os
import hashlib
import torch
import timm
import numpy as np
import pandas as pd
import random
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from scipy.special import softmax
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from LogME import LogME


# =============================================================================
# 0) CONFIGURATION
# =============================================================================
SEEDS = [320, 1991, 9]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
# CIFAR-C images are 32x32; start high and let OOM backoff (down to 8) find the limit.
DEFAULT_BATCH_SIZE = 2048

# Models excluded from benchmarking (persistent OOM / unusable for this paper).
EXCLUDED_MODELS = {
    'vitamin_base_224.datacomp1b_clip',
}

VALID_LEEP_SOURCE_CLASSES = {1000, 21841, 21843, 11821, 11221, 10450, 12000}

# All 19 corruption types in CIFAR-C (same for both 10 and 100)
ALL_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate',
]

DATASET_CONFIG = {
    'cifar10c': {
        'n_samples': 5000,
        'n_classes': 10,
        'data_subdir': 'CIFAR-10-C',
        'local_data_dir': './data/CIFAR-10-C',
    },
    'cifar100c': {
        'n_samples': 5000,
        'n_classes': 100,
        'data_subdir': 'CIFAR-100-C',
        'local_data_dir': './data/CIFAR-100-C',
    },
}

# ---------------------------------------------------------------------------
# Output and data paths
# ---------------------------------------------------------------------------
GDRIVE_BASE = None
GDRIVE_FOLDER = None
DEFAULT_OUTPUT_DIR = "./shesha-vision_architecture-corrupt"


def _gdrive_accessible(base_path: str) -> bool:
    if not base_path:
        return False
    try:
        p = Path(base_path)
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception:
        return False


def _resolve_data_dir(dataset_name: str) -> Path:
    """Return local data directory for a CIFAR-C dataset."""
    cfg = DATASET_CONFIG[dataset_name.lower()]
    local_path = Path(cfg['local_data_dir'])

    if GDRIVE_BASE and _gdrive_accessible(GDRIVE_BASE):
        gdrive_data = Path(GDRIVE_BASE) / "data" / cfg['data_subdir']
        if gdrive_data.exists() and any(gdrive_data.glob("*.npy")):
            return gdrive_data
        if gdrive_data.exists():
            print(f"[Data] Override dir exists but no .npy yet: {gdrive_data}")
            return gdrive_data

    local_path.mkdir(parents=True, exist_ok=True)
    print(f"[Data] Using: {local_path.resolve()}")
    return local_path


def _resolve_output_dir(gdrive_folder=GDRIVE_FOLDER) -> Path:
    """Return output Path: optional override if writable, else local default."""
    if gdrive_folder and _gdrive_accessible(gdrive_folder):
        gdrive_path = Path(gdrive_folder)
        print(f"[Output] Saving to: {gdrive_path.resolve()}")
        return gdrive_path
    local_path = Path(DEFAULT_OUTPUT_DIR)
    local_path.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Saving locally: {local_path.resolve()}")
    return local_path


def get_data_dir(dataset_name: str) -> Path:
    """Resolved data directory for a CIFAR-C dataset key."""
    return _resolve_data_dir(dataset_name.lower())


# Populate resolved paths (used by loaders and --list-available)
for _ds, _cfg in DATASET_CONFIG.items():
    _cfg['data_dir'] = str(get_data_dir(_ds))


OUTPUT_DIR = _resolve_output_dir()


# =============================================================================
# 1) DETERMINISM UTILITIES
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn_factory(seed):
    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _worker_init_fn


# =============================================================================
# 2) CORE SHESHA & TRANSFERABILITY METRICS
# =============================================================================

def compute_shesha_variance(X, y):
    """Ratio of between-class variance to total variance."""
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    global_mean = np.mean(X, axis=0)
    ss_total = np.sum((X - global_mean)**2) + EPS
    ss_between = 0.0
    for c in classes:
        mask = (y == c)
        if np.sum(mask) == 0:
            continue
        mean_c = np.mean(X[mask], axis=0)
        ss_between += np.sum(mask) * np.sum((mean_c - global_mean)**2)
    return ss_between / ss_total


def compute_shesha_feature_split(X, n_splits=10, seed=320):
    """Reliability via correlation of random feature subspace RDMs."""
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
        valid = (np.linalg.norm(X1, axis=1) > EPS) & (np.linalg.norm(X2, axis=1) > EPS)
        if valid.sum() < 4:
            continue
        d1, d2 = pdist(X1[valid], 'cosine'), pdist(X2[valid], 'cosine')
        rho, _ = spearmanr(d1, d2)
        correlations.append(rho if not np.isnan(rho) else 0.0)
    return np.mean(correlations) if correlations else 0.0


def compute_leep(logits, y_target):
    """LEEP: Requires standard source labels and verified alignment."""
    if logits is None or logits.ndim != 2 or logits.shape[0] != len(y_target):
        return np.nan

    n_samples, n_source = logits.shape

    if n_source not in VALID_LEEP_SOURCE_CLASSES:
        return np.nan

    unique_y = np.unique(y_target)
    label_map = {val: i for i, val in enumerate(unique_y)}
    y_remapped = np.array([label_map[y] for y in y_target])

    prob_source = softmax(logits.astype(np.float64), axis=1)
    joint = np.zeros((n_source, len(unique_y)))
    for i in range(n_samples):
        joint[:, y_remapped[i]] += prob_source[i]
    joint /= n_samples

    marginal_z = joint.sum(axis=1, keepdims=True)
    conditional = joint / (marginal_z + EPS)

    score = sum(
        np.log(np.dot(prob_source[i], conditional[:, y_remapped[i]]) + EPS)
        for i in range(n_samples)
    )
    return score / n_samples


# =============================================================================
# 3) CIFAR-C DATASET
# =============================================================================

class CIFARCDataset(Dataset):
    """
    Wraps a pre-loaded numpy array of corrupted CIFAR images.

    images : uint8 numpy array of shape (N, 32, 32, 3)
    labels : int64 numpy array of shape (N,)
    transform: torchvision transform to apply
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert len(images) == len(labels), "images/labels length mismatch"
        self.images = images      # (N, 32, 32, 3) uint8
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def load_cifar_c(dataset_name: str, corruption: str, severity: int):
    """
    Load a single (corruption, severity) slice from CIFAR-C .npy files.

    severity in {1, 2, 3, 4, 5}  (1 = mildest, 5 = strongest)

    Returns
    -------
    images : np.ndarray  (10000, 32, 32, 3) uint8
    labels : np.ndarray  (10000,) int64
    """
    assert 1 <= severity <= 5, "severity must be 1–5"
    data_dir = Path(DATASET_CONFIG[dataset_name]['data_dir'])
    corruption_path = data_dir / f"{corruption}.npy"
    labels_path = data_dir / "labels.npy"

    if not corruption_path.exists():
        raise FileNotFoundError(
            f"Corruption file not found: {corruption_path}\n"
            f"Download CIFAR-10-C / CIFAR-100-C from:\n"
            f"  https://zenodo.org/record/2535967  (CIFAR-10-C)\n"
            f"  https://zenodo.org/record/3555552  (CIFAR-100-C)"
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    all_images = np.load(corruption_path)   # (50000, 32, 32, 3) uint8
    all_labels = np.load(labels_path)       # (10000,) or (50000,) depending on release

    # Severity index: severity 1 → indices [0, 10000), severity 5 → [40000, 50000)
    start = (severity - 1) * 10000
    end = start + 10000
    images_slice = all_images[start:end]

    # Some releases ship labels.npy as 10000 (one level, reused for all severities),
    # others ship it as 50000 (all severities stacked). Handle both.
    if len(all_labels) == 50000:
        labels_slice = all_labels[start:end]
    elif len(all_labels) == 10000:
        labels_slice = all_labels
    else:
        # Unexpected shape — try to align by taking modulo
        labels_slice = all_labels[:10000]

    assert len(images_slice) == len(labels_slice), (
        f"Could not align images ({len(images_slice)}) and labels ({len(labels_slice)}) "
        f"for {corruption_path}"
    )
    return images_slice, labels_slice.astype(np.int64)


def list_available_corruptions(dataset_name: str):
    """Return corruption types that are actually present on disk."""
    data_dir = Path(DATASET_CONFIG[dataset_name]['data_dir'])
    available = [c for c in ALL_CORRUPTIONS if (data_dir / f"{c}.npy").exists()]
    return available


CORRUPTION_GROUPS = {
    "noise": [
        "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    ],
    "blur": [
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    ],
    "weather": [
        "snow", "frost", "fog", "brightness", "spatter",
    ],
    "digital": [
        "contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate",
    ],
}


def resolve_corruptions(dataset_name: str, corruptions=None, corruption_group=None):
    """Resolve corruption list from explicit names, a preset group, or all on disk."""
    available = set(list_available_corruptions(dataset_name))
    if not available:
        raise FileNotFoundError(
            f"No corruption .npy files found in {DATASET_CONFIG[dataset_name]['data_dir']}"
        )

    if corruptions is not None:
        chosen = corruptions
    elif corruption_group is not None:
        group = corruption_group.lower()
        if group == "all":
            chosen = sorted(available)
        elif group not in CORRUPTION_GROUPS:
            valid = ", ".join(sorted(CORRUPTION_GROUPS.keys()) + ["all"])
            raise ValueError(f"Unknown corruption group '{corruption_group}'. Choose: {valid}")
        else:
            chosen = CORRUPTION_GROUPS[group]
    else:
        chosen = sorted(available)

    missing = [c for c in chosen if c not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing corruption files for {dataset_name}: {missing}\n"
            f"Available: {sorted(available)}"
        )
    # Preserve canonical ALL_CORRUPTIONS order
    order = {c: i for i, c in enumerate(ALL_CORRUPTIONS)}
    return sorted(chosen, key=lambda c: order.get(c, 999))


def build_corrupt_jobs(datasets, severities, corruptions_by_dataset):
    """Flat job list: (dataset_name, corruption, severity)."""
    jobs = []
    for dataset_name in datasets:
        for corruption in corruptions_by_dataset[dataset_name]:
            for severity in severities:
                jobs.append((dataset_name, corruption, severity))
    return jobs


def shard_jobs(jobs, shard_index, num_shards):
    """Deterministic split for parallel benchmark shards."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")
    if num_shards == 1:
        return jobs
    return [job for i, job in enumerate(jobs) if i % num_shards == shard_index]


def scan_completed_jobs(output_dir=None, datasets=None, severities=None):
    """
    Scan output_dir for CIFAR10C_*_sev{N}_AVERAGED_*.csv files.
    Returns set of (dataset_name, corruption, severity) tuples.
    """
    out = Path(output_dir or OUTPUT_DIR)
    if not out.exists():
        return set()

    completed = set()
    for path in out.glob("*_AVERAGED_*.csv"):
        name = path.name.upper()
        if not name.startswith(("CIFAR10C_", "CIFAR100C_")):
            continue
        dataset_name = "cifar10c" if name.startswith("CIFAR10C_") else "cifar100c"
        if datasets and dataset_name not in datasets:
            continue
        rest = path.name[len("CIFAR10C_") if dataset_name == "cifar10c" else len("CIFAR100C_") :]
        parts = rest.split("_sev")
        if len(parts) != 2:
            continue
        corruption = parts[0]
        try:
            severity = int(parts[1].split("_")[0])
        except ValueError:
            continue
        if severities and severity not in severities:
            continue
        completed.add((dataset_name, corruption, severity))
    return completed


def print_run_status(datasets, severities, corruptions_by_dataset, output_dir=None):
    """Compare expected jobs to AVERAGED CSVs on disk."""
    expected = build_corrupt_jobs(datasets, severities, corruptions_by_dataset)
    completed = scan_completed_jobs(output_dir, datasets=datasets, severities=severities)
    expected_set = set(expected)
    done = [j for j in expected if j in completed]
    todo = [j for j in expected if j not in completed]
    extra = sorted(completed - expected_set)

    out = Path(output_dir or OUTPUT_DIR)
    print("CORRUPT BENCHMARK STATUS")
    print("=" * 70)
    print(f"  Output directory: {out}")
    print(f"  Expected jobs:    {len(expected)}")
    print(f"  Completed:        {len(done)}")
    print(f"  Remaining:        {len(todo)}")
    if extra:
        print(f"  Extra on disk:    {len(extra)} (outside current filter)")

    if done:
        print(f"\n--- Done ({len(done)}) ---")
        for ds, corruption, severity in done:
            print(f"  [x] {ds.upper()} | {corruption} | sev{severity}")

    if todo:
        print(f"\n--- Still to run ({len(todo)}) ---")
        for ds, corruption, severity in todo:
            print(f"  [ ] {ds.upper()} | {corruption} | sev{severity}")

    if extra:
        print(f"\n--- On disk but not in current filter ({len(extra)}) ---")
        for ds, corruption, severity in extra[:20]:
            print(f"  [?] {ds.upper()} | {corruption} | sev{severity}")
        if len(extra) > 20:
            print(f"  ... and {len(extra) - 20} more")

    if todo:
        ds0, corr0, sev0 = todo[0]
        print("\nNext single job:")
        print(
            f"  python vision_architecture_corrupt.py "
            f"--datasets {ds0} --corruptions {corr0} --severities {sev0} --seeds 320"
        )
    return done, todo


def print_job_plan(jobs, shard_index=None, num_shards=None):
    """Print scheduled jobs without running."""
    header = "PLANNED CORRUPT BENCHMARK JOBS"
    if num_shards and num_shards > 1:
        header += f" (shard {shard_index}/{num_shards - 1})"
    print(header)
    print("=" * 70)
    for i, (ds, corruption, severity) in enumerate(jobs, start=1):
        print(f"  {i:3d}. {ds.upper()} | {corruption} | severity {severity}")
    print(f"\nTotal jobs in this run: {len(jobs)}")
    print(f"Output directory: {OUTPUT_DIR}")


def parse_csv_list(value):
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


# =============================================================================
# 4) MODEL RETRIEVAL & EXTRACTION LOGIC
# =============================================================================

def get_strategic_model_list():
    """Get curated list of pretrained models including expanded CLIP variants."""
    clip_models = [
        # OpenAI CLIP backbones
        'vit_base_patch32_clip_224.openai',
        'vit_base_patch16_clip_224.openai',
        'vit_large_patch14_clip_224.openai',
        # LAION-400M CLIP
        'vit_base_patch32_clip_224.laion400m_e32',
        'vit_base_patch16_clip_224.laion400m_e32',
        'vit_large_patch14_clip_224.laion400m_e32',
        # LAION-2B CLIP (pure pretrain checkpoints, no fine-tuning)
        'vit_base_patch32_clip_224.laion2b',
        'vit_base_patch16_clip_224.laion2b',
        'vit_large_patch14_clip_224.laion2b',
        'vit_huge_patch14_clip_224.laion2b',
        'vit_giant_patch14_clip_224.laion2b',
        # DataComp CLIP
        'vit_base_patch32_clip_224.datacompxl',
        'vit_base_patch16_clip_224.datacompxl',
        'vit_large_patch14_clip_224.datacompxl',
        # EVA-CLIP (LAION-2B)
        'eva02_enormous_patch14_clip_224.laion2b',
        'eva02_enormous_patch14_clip_224.laion2b_plus',
        # ConvNeXt CLIP (LAION-2B)
        'convnext_base.clip_laion2b',
        'convnext_large_mlp.clip_laion2b_augreg',
        'convnext_xxlarge.clip_laion2b_rewind',
    ]

    # DINOv2 — original + register variants (improved patch artifacts)
    dinov2 = [
        'vit_small_patch14_dinov2.lvd142m',
        'vit_small_patch14_reg4_dinov2.lvd142m',
        'vit_base_patch14_dinov2.lvd142m',
        'vit_base_patch14_reg4_dinov2.lvd142m',
        'vit_large_patch14_dinov2.lvd142m',
        'vit_large_patch14_reg4_dinov2.lvd142m',
        'vit_giant_patch14_dinov2.lvd142m',
        'vit_giant_patch14_reg4_dinov2.lvd142m',
    ]

    # DINOv3 — Meta's next-gen self-supervised (LVD-1689M)
    dinov3 = [
        'vit_small_patch16_dinov3.lvd1689m',
        'vit_base_patch16_dinov3.lvd1689m',
        'vit_large_patch16_dinov3.lvd1689m',
        'vit_small_patch16_dinov3_qkvb.lvd1689m',
        'vit_base_patch16_dinov3_qkvb.lvd1689m',
        'vit_large_patch16_dinov3_qkvb.lvd1689m',
    ]

    # SigLIP — Google sigmoid-loss CLIP, strong zero-shot (224px only)
    siglip = [
        'vit_base_patch16_siglip_224.webli',
        'vit_large_patch16_siglip_256.webli',
        'vit_so400m_patch14_siglip_224.webli',
        'vit_base_patch16_siglip_224.v2_webli',
        'vit_large_patch16_siglip_256.v2_webli',
        'vit_so400m_patch14_siglip_224.v2_webli',
    ]

    # BEiT / BEiTv2 / BEiT3 — masked image modelling family
    beit_family = [
        'beit_base_patch16_224.in22k_ft_in22k',
        'beit_large_patch16_224.in22k_ft_in22k',
        'beitv2_base_patch16_224.in1k_ft_in22k',
        'beitv2_large_patch16_224.in1k_ft_in22k',
        'beit3_base_patch16_224.in22k_ft_in1k',
        'beit3_large_patch16_224.in22k_ft_in1k',
    ]

    # MAE — ViT + Hiera variants (masked autoencoder)
    mae_family = [
        'vit_base_patch16_224.mae',
        'vit_large_patch16_224.mae',
        'vit_huge_patch14_224.mae',
        'hiera_tiny_224.mae',
        'hiera_small_224.mae',
        'hiera_base_224.mae',
        'hiera_large_224.mae',
        'hiera_huge_224.mae',
    ]

    # EVA / EVA-02 — Florence-style hybrid CLIP+MIM pretraining
    eva_family = [
        'eva02_tiny_patch14_224.mim_in22k',
        'eva02_small_patch14_224.mim_in22k',
        'eva02_base_patch14_224.mim_in22k',
        'eva02_large_patch14_224.mim_in22k',
        'eva02_large_patch14_224.mim_m38m',
        'eva_large_patch14_196.in22k_ft_in1k',
        'eva_giant_patch14_336.m30m_ft_in22k_in1k',
    ]

    # SAM ViT — segment-anything pretraining (SA-1B)
    sam_vit = [
        'samvit_base_patch16.sa1b',
        'samvit_large_patch16.sa1b',
        'samvit_huge_patch16.sa1b',
        'vit_base_patch16_224.sam_in1k',
    ]

    # I-JEPA — joint-embedding predictive architecture (IN22k/IN1k)
    ijepa = [
        'vit_huge_patch14_gap_224.in1k_ijepa',
        'vit_huge_patch14_gap_224.in22k_ijepa',
        'vit_giant_patch16_gap_224.in22k_ijepa',
    ]

    # SwinV2 22k pretrained
    swinv2_22k = [
        'swinv2_base_window12_192.ms_in22k',
        'swinv2_large_window12_192.ms_in22k',
        'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
        'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
    ]

    # ViTamin — efficient CLIP with MBConv mixer
    vitamin = [
        'vitamin_small_224.datacomp1b_clip',
        'vitamin_base_224.datacomp1b_clip',
        'vitamin_large_224.datacomp1b_clip',
        'vitamin_large2_224.datacomp1b_clip',
        'vitamin_xlarge_256.datacomp1b_clip',
    ]

    # DINO v1 + ResMLP-DINO
    dino_v1 = [
        'vit_small_patch16_224.dino',
        'vit_base_patch16_224.dino',
        'vit_small_patch8_224.dino',
        'vit_base_patch8_224.dino',
        'resmlp_12_224.fb_dino',
        'resmlp_24_224.fb_dino',
    ]

    # Large RegNetY (SEER / SWAG self-supervised)
    regnety_large = [
        'regnety_160.swag_ft_in1k',
        'regnety_320.swag_ft_in1k',
        'regnety_1280.swag_ft_in1k',
        'regnety_160.deit_in1k',
        'regnety_320.seer_ft_in1k',
        'regnety_1280.seer_ft_in1k',
    ]

    foundation = (
        dinov2 + dinov3 + siglip + beit_family + mae_family +
        eva_family + sam_vit + ijepa + swinv2_22k + vitamin + dino_v1 + regnety_large
    )

    transformers = [
        'swin_tiny_patch4_window7_224.ms_in1k', 'swin_small_patch4_window7_224.ms_in1k',
        'swin_base_patch4_window7_224.ms_in1k', 'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
        'swinv2_tiny_window8_256.ms_in1k', 'swinv2_small_window8_256.ms_in1k',
        'pvt_v2_b0.in1k', 'pvt_v2_b1.in1k', 'pvt_v2_b2.in1k', 'pvt_v2_b3.in1k', 'pvt_v2_b5.in1k',
        'poolformer_s12.sail_in1k', 'poolformer_s24.sail_in1k', 'poolformer_m36.sail_in1k',
        'deit_tiny_patch16_224.fb_in1k', 'deit_small_patch16_224.fb_in1k', 'deit_base_patch16_224.fb_in1k',
        'deit3_small_patch16_224.fb_in1k', 'deit3_base_patch16_224.fb_in1k',
        'vit_tiny_patch16_224.augreg_in21k_ft_in1k', 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        'vit_base_patch16_224.augreg_in21k_ft_in1k', 'vit_large_patch16_224.augreg_in21k_ft_in1k',
        'maxvit_tiny_tf_224.in1k', 'maxvit_small_tf_224.in1k',
        'coatnet_0_rw_224.sw_in1k', 'coatnet_1_rw_224.sw_in1k',
    ]

    cnns = [
        'convnext_atto.d2_in1k', 'convnext_femto.d1_in1k', 'convnext_pico.d1_in1k', 'convnext_nano.d1h_in1k',
        'convnext_tiny.fb_in1k', 'convnext_small.fb_in1k', 'convnext_base.fb_in1k', 'convnext_large.fb_in1k',
        'convnextv2_atto.fcmae_ft_in1k', 'convnextv2_nano.fcmae_ft_in1k',
        'convnextv2_tiny.fcmae_ft_in1k', 'convnextv2_base.fcmae_ft_in1k',
        'efficientnet_b0.ra_in1k', 'efficientnet_b1.ft_in1k', 'efficientnet_b2.ra_in1k', 'efficientnet_b3.ra2_in1k',
        'efficientnetv2_rw_s.ra2_in1k', 'efficientnetv2_rw_m.agc_in1k',
        'tf_efficientnetv2_s.in1k', 'tf_efficientnetv2_m.in1k', 'tf_efficientnetv2_b0.in1k', 'tf_efficientnetv2_b3.in1k',
        'regnety_002.pycls_in1k', 'regnety_004.pycls_in1k', 'regnety_008.pycls_in1k',
        'regnety_016.pycls_in1k', 'regnety_032.pycls_in1k', 'regnety_064.pycls_in1k',
        'regnetx_002.pycls_in1k', 'regnetx_004.pycls_in1k', 'regnetx_008.pycls_in1k',
        'resnet18.a1_in1k', 'resnet34.a1_in1k', 'resnet50.a1_in1k', 'resnet101.a1_in1k', 'resnet152.a1_in1k',
        'resnext50_32x4d.a1_in1k', 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
        'densenet121.ra_in1k', 'densenet169.tv_in1k', 'densenet201.tv_in1k',
        'mobilenetv3_small_100.lamb_in1k', 'mobilenetv3_large_100.ra_in1k',
        'inception_v3.tf_in1k', 'inception_v4.tf_in1k',
    ]

    robust = [
        'resnet50_gn.a1h_in1k', 'resnet50.a1_in1k', 'resnet50.a2_in1k', 'resnet50.a3_in1k',
        'vit_base_patch16_224.augreg_in21k', 'vit_base_patch16_224.augreg_in1k',
        'wide_resnet50_2.racm_in1k', 'wide_resnet101_2.tv_in1k',
        'resnetv2_50.a1h_in1k', 'resnetv2_101.a1h_in1k',
        'resnetrs50.tf_in1k', 'resnetrs101.tf_in1k',
    ]

    all_timm = timm.list_models(pretrained=True)
    combined = clip_models + foundation + transformers + cnns + robust

    # Deduplicate while preserving order (clip_models first)
    seen = set()
    combined = [m for m in combined if not (m in seen or seen.add(m))]

    if len(combined) < 90:
        combined += [m for m in all_timm if 'mobilenetv3' in m or 'densenet' in m][:30]
        seen2 = set()
        combined = [m for m in combined if not (m in seen2 or seen2.add(m))]

    all_timm_set = set(all_timm)
    available   = [m for m in combined if m in all_timm_set and m not in EXCLUDED_MODELS]
    unavailable = [m for m in combined if m not in all_timm_set]
    excluded    = [m for m in combined if m in EXCLUDED_MODELS]
    if unavailable:
        print(f"[ModelList] {len(unavailable)} models not found in this timm version and will be skipped:")
        for m in unavailable:
            print(f"  - {m}")
    if excluded:
        print(f"[ModelList] {len(excluded)} models excluded for this paper:")
        for m in excluded:
            print(f"  - {m}")
    print(f"[ModelList] Final count: {len(available)} models")
    return available


def get_pooled_features(feats):
    """Extract and pool features with strict rules."""
    if isinstance(feats, (tuple, list)):
        feats = feats[-1]

    if isinstance(feats, dict):
        for key in ['pre_logits', 'pooled', 'global_pool', 'features']:
            if key in feats and isinstance(feats[key], torch.Tensor):
                feats = feats[key]
                break
        else:
            tensor_values = [v for v in feats.values() if isinstance(v, torch.Tensor)]
            if len(tensor_values) == 1:
                feats = tensor_values[0]
            else:
                raise ValueError(f"Ambiguous dict output with {len(tensor_values)} tensors, keys: {list(feats.keys())}")

    if not isinstance(feats, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(feats)}")

    if feats.ndim == 4:
        return feats.mean(dim=(2, 3))
    if feats.ndim == 3:
        return feats.mean(dim=1)
    if feats.ndim == 2:
        return feats

    raise ValueError(f"Unexpected tensor shape: {feats.shape}")


def get_robust_logits(out):
    """Extract logits with strict rules."""
    logits = None

    if isinstance(out, torch.Tensor):
        logits = out
    elif isinstance(out, (tuple, list)) and len(out) > 0:
        if isinstance(out[0], torch.Tensor):
            logits = out[0]
        else:
            return None, 'unsupported_format'
    elif isinstance(out, dict):
        if 'logits' in out and isinstance(out['logits'], torch.Tensor):
            logits = out['logits']
        else:
            return None, 'missing_logits_key'
    else:
        return None, 'unsupported_format'

    if logits is None:
        return None, 'unsupported_format'

    if logits.ndim == 4 and logits.shape[2] == 1 and logits.shape[3] == 1:
        logits = logits[..., 0, 0]

    if logits.ndim != 2:
        return None, 'bad_shape'

    if logits.shape[1] not in VALID_LEEP_SOURCE_CLASSES:
        return None, 'bad_classcount'

    return logits, 'ok'


class CudaOomExhaustedError(RuntimeError):
    """Inference OOM persisted after batch-size backoff."""


def _is_cuda_oom(exc):
    if isinstance(exc, CudaOomExhaustedError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def _release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_corrupt_dataloader(
    raw_images,
    raw_labels,
    subset_idx,
    transform,
    batch_size,
    num_workers,
    seed,
):
    full_ds = CIFARCDataset(raw_images, raw_labels, transform=transform)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == 'cuda'),
    )
    if num_workers > 0:
        prefetch = 1 if batch_size >= 512 or num_workers >= 6 else 2
        loader_kwargs.update(dict(
            persistent_workers=(num_workers <= 4),
            prefetch_factor=prefetch,
            worker_init_fn=worker_init_fn_factory(seed),
        ))
    return DataLoader(
        torch.utils.data.Subset(full_ds, subset_idx),
        **loader_kwargs,
    )


def _run_combined_pass(model_with_head, model_backbone, loader):
    """Single data pass: extract both logits and pooled features."""
    all_logits = []
    all_feats = []
    batch_statuses = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            l_out, status = get_robust_logits(model_with_head(imgs))
            batch_statuses.append(status)
            if l_out is not None:
                all_logits.append(l_out.cpu().numpy())
            f_out = get_pooled_features(model_backbone(imgs))
            all_feats.append(f_out.cpu().numpy())
    return all_feats, all_logits, batch_statuses


def _run_with_batch_retry(infer_fn, initial_batch_size, min_batch_size=8):
    """Run inference; on CUDA OOM halve batch size and retry."""
    batch_size = initial_batch_size
    last_exc = None
    while batch_size >= min_batch_size:
        try:
            return infer_fn(batch_size), batch_size
        except RuntimeError as exc:
            last_exc = exc
            if not _is_cuda_oom(exc):
                raise
            if batch_size <= min_batch_size:
                break
            next_bs = max(min_batch_size, batch_size // 2)
            print(f"  [OOM] Retrying with batch_size={next_bs} (was {batch_size})")
            batch_size = next_bs
            _release_cuda_memory()
    _release_cuda_memory()
    raise CudaOomExhaustedError(
        f"CUDA OOM persisted at batch_size={batch_size} (min={min_batch_size})"
    ) from last_exc


def _infer_combined_pass(model_with_head, model_backbone, raw_images, raw_labels,
                         subset_idx, transform, num_workers, seed):
    def infer_fn(batch_size):
        loader = _build_corrupt_dataloader(
            raw_images, raw_labels, subset_idx, transform,
            batch_size, num_workers, seed,
        )
        try:
            return _run_combined_pass(model_with_head, model_backbone, loader)
        finally:
            del loader

    return infer_fn


# =============================================================================
# 5) CHECKPOINT / RESUME HELPERS
# =============================================================================

def _checkpoint_csv_path(output_dir, dataset_name, corruption, severity, seed):
    """Deterministic path for the incremental checkpoint file."""
    return Path(output_dir) / f".ckpt_{dataset_name}_{corruption}_sev{severity}_seed{seed}.csv"


def _load_checkpoint(ckpt_path):
    """Load existing checkpoint CSV. Returns (DataFrame, set of completed model names)."""
    if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
        try:
            df = pd.read_csv(ckpt_path)
            done = set(df['Model'].tolist())
            return df, done
        except Exception as exc:
            print(f"  [Resume] Checkpoint corrupt, starting fresh: {exc}")
    return pd.DataFrame(), set()


def _append_checkpoint(ckpt_path, row_dict):
    """Append a single model result row to the checkpoint CSV."""
    df_row = pd.DataFrame([row_dict])
    write_header = not ckpt_path.exists() or ckpt_path.stat().st_size == 0
    df_row.to_csv(ckpt_path, mode='a', header=write_header, index=False)


# =============================================================================
# 6) SINGLE-SEED CORRUPT BENCHMARK
# =============================================================================

def run_single_seed_corrupt_benchmark(
    dataset_name: str,
    seed: int,
    corruption: str,
    severity: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
    resume: bool = True,
):
    """
    Run benchmark for one (dataset, corruption, severity, seed) combination.

    Saves progress after every model to an incremental checkpoint CSV so that
    a crashed session can resume without re-evaluating completed models.

    Parameters
    ----------
    dataset_name : 'cifar10c' or 'cifar100c'
    seed         : random seed
    corruption   : one of ALL_CORRUPTIONS (e.g. 'gaussian_noise')
    severity     : 1–5
    resume       : if True, load checkpoint and skip already-done models
    """
    set_seed(seed)

    dataset_name = dataset_name.lower()
    config = DATASET_CONFIG.get(dataset_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIG.keys())}")

    model_names = get_strategic_model_list()
    tag = f"{dataset_name.upper()} | {corruption} | sev{severity} | seed{seed}"
    print(f"\n{'='*70}")
    print(f"Running Corrupt Benchmark: {tag}")
    print(f"{'='*70}")
    print(f"  Device: {DEVICE}, Batch size: {batch_size}, Workers: {num_workers}")
    print(f"  Models: {len(model_names)}")

    raw_images, raw_labels = load_cifar_c(dataset_name, corruption, severity)

    n_samples = min(config['n_samples'], len(raw_labels))
    print(f"  Dataset size: {len(raw_labels)}, Using: {n_samples} samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    idx_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{corruption}_sev{severity}_seed{seed}_subset_idx.npy")

    # Resume: reuse the same subset_idx from a previous run if available
    if resume and os.path.exists(idx_path):
        subset_idx = np.load(idx_path)
        print(f"  [Resume] Loaded subset_idx from {idx_path}")
    else:
        subset_idx = np.random.choice(len(raw_labels), n_samples, replace=False)
        np.save(idx_path, subset_idx)

    y_target = raw_labels[subset_idx]
    subset_hash = hashlib.sha256(subset_idx.tobytes()).hexdigest()[:12]
    print(f"  Subset SHA: {subset_hash}")

    # Resume: load checkpoint and figure out which models are already done
    ckpt_path = _checkpoint_csv_path(OUTPUT_DIR, dataset_name, corruption, severity, seed)
    prior_df, done_models = pd.DataFrame(), set()
    if resume:
        prior_df, done_models = _load_checkpoint(ckpt_path)
        if done_models:
            print(f"  [Resume] {len(done_models)}/{len(model_names)} models already done — skipping them")

    results = list(prior_df.to_dict('records')) if not prior_df.empty else []
    leep_status_counts = {}
    n_oom_skipped = 0

    def _make_oom_skip_row(model_name):
        return {
            "Model": model_name,
            "Dataset": dataset_name,
            "Corruption": corruption,
            "Severity": severity,
            "Seed": seed,
            "LEEP_Real": np.nan,
            "LEEP_Status": "oom_skip",
            "Logits_C": np.nan,
            "Had_Any_Ok_Batches": False,
            "LogME": np.nan,
            "SHESHA_Var": np.nan,
            "SHESHA_FS": np.nan,
            "Dim": np.nan,
            "N_Samples": n_samples,
            "Subset_Hash": subset_hash,
            "Inference_Batch_Size": np.nan,
        }

    models_todo = [m for m in model_names if m not in done_models]
    n_skipped = len(done_models)
    interrupted = False
    for m_name in tqdm(models_todo, desc=tag, initial=len(done_models), total=len(model_names)):
        tqdm.write(f"  -> {m_name}")
        model_l = None
        model_f = None
        try:
            model_l = timm.create_model(m_name, pretrained=True).to(DEVICE).eval()
            config_data = timm.data.resolve_data_config({}, model=model_l)
            transform = timm.data.create_transform(**config_data, is_training=False)
            model_f = timm.create_model(m_name, pretrained=True, num_classes=0).to(DEVICE).eval()

            (all_feats, all_logits, batch_statuses), infer_bs = _run_with_batch_retry(
                _infer_combined_pass(
                    model_l, model_f, raw_images, raw_labels, subset_idx,
                    transform, num_workers, seed,
                ),
                batch_size,
            )

            del model_l, model_f
            model_l = model_f = None
            _release_cuda_memory()

            X = np.concatenate(all_feats)
            L = np.concatenate(all_logits) if all_logits else None
            del all_feats, all_logits

            had_any_ok = 'ok' in batch_statuses

            if L is None:
                status_counts = Counter(batch_statuses)
                status_counts.pop('ok', None)
                if status_counts:
                    leep_status = status_counts.most_common(1)[0][0]
                else:
                    leep_status = 'unsupported_format'
                logits_c = np.nan
            elif L.shape[0] != len(y_target):
                leep_status = 'partial_logits'
                logits_c = np.nan
                L = None
            else:
                leep_status = 'ok'
                logits_c = L.shape[1]

            leep_val = compute_leep(L, y_target)
            leep_status_counts[leep_status] = leep_status_counts.get(leep_status, 0) + 1

            row = {
                "Model": m_name,
                "Dataset": dataset_name,
                "Corruption": corruption,
                "Severity": severity,
                "Seed": seed,
                "LEEP_Real": leep_val,
                "LEEP_Status": leep_status,
                "Logits_C": logits_c,
                "Had_Any_Ok_Batches": had_any_ok,
                "LogME": LogME(regression=False).fit(X, y_target),
                "SHESHA_Var": compute_shesha_variance(X, y_target),
                "SHESHA_FS": compute_shesha_feature_split(X, seed=seed),
                "Dim": X.shape[1],
                "N_Samples": n_samples,
                "Subset_Hash": subset_hash,
                "Inference_Batch_Size": infer_bs,
            }
            results.append(row)
            _append_checkpoint(ckpt_path, row)

            del X, L

        except KeyboardInterrupt:
            print(f"\n  [Interrupted] Saving progress ({len(results)}/{len(model_names)} models)...")
            del model_l, model_f
            _release_cuda_memory()
            interrupted = True
            break

        except CudaOomExhaustedError as e:
            print(f"  [OOM] Skipping {m_name}: {e}")
            n_oom_skipped += 1
            leep_status_counts["oom_skip"] = leep_status_counts.get("oom_skip", 0) + 1
            row = _make_oom_skip_row(m_name)
            results.append(row)
            _append_checkpoint(ckpt_path, row)
            del model_l, model_f
            _release_cuda_memory()

        except RuntimeError as e:
            if _is_cuda_oom(e):
                print(f"  [OOM] Skipping {m_name}: {e}")
                n_oom_skipped += 1
                leep_status_counts["oom_skip"] = leep_status_counts.get("oom_skip", 0) + 1
                row = _make_oom_skip_row(m_name)
                results.append(row)
                _append_checkpoint(ckpt_path, row)
            else:
                print(f"Error {m_name}: {e}")
            del model_l, model_f
            _release_cuda_memory()

        except Exception as e:
            print(f"Error {m_name}: {e}")
            del model_l, model_f
            _release_cuda_memory()

    df = pd.DataFrame(results)

    n_new = len(results) - len(done_models)
    if interrupted:
        print(f"\nSeed {seed} INTERRUPTED: {len(results)}/{len(model_names)} models saved"
              f" ({n_skipped} resumed, {n_new} new, {n_oom_skipped} OOM-skipped)")
        print(f"  Checkpoint kept at {ckpt_path.name} — resume will continue from here")
    else:
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"  [Checkpoint] Removed {ckpt_path.name} (seed complete)")
        print(f"\nSeed {seed} Complete: {len(results)}/{len(model_names)} models"
              f" ({n_skipped} resumed, {n_new} new, {n_oom_skipped} OOM-skipped)")

    print(f"  LEEP status breakdown:")
    for status, count in sorted(leep_status_counts.items()):
        print(f"    {status}: {count}")

    return df


# =============================================================================
# 7) MULTI-SEED CORRUPT BENCHMARK WITH AVERAGING
# =============================================================================

def run_multi_seed_corrupt_benchmark(
    dataset_name: str,
    corruption: str,
    severity: int = 5,
    seeds=SEEDS,
    batch_size: int = 64,
    num_workers: int = 4,
    resume: bool = True,
):
    """Run corrupt benchmark across multiple seeds and compute averages."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = dataset_name.lower()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_seed_dfs = []

    for seed in seeds:
        df_seed = run_single_seed_corrupt_benchmark(
            dataset_name=dataset_name,
            seed=seed,
            corruption=corruption,
            severity=severity,
            batch_size=batch_size,
            num_workers=num_workers,
            resume=resume,
        )

        seed_filename = f"{dataset_name.upper()}_{corruption}_sev{severity}_SEED{seed}_{timestamp}.csv"
        df_seed.to_csv(os.path.join(OUTPUT_DIR, seed_filename), index=False)
        print(f"  Saved: {seed_filename}")
        per_seed_dfs.append(df_seed)

    df_all = pd.concat(per_seed_dfs, ignore_index=True)

    all_seeds_filename = f"{dataset_name.upper()}_{corruption}_sev{severity}_ALL_SEEDS_{timestamp}.csv"
    df_all.to_csv(os.path.join(OUTPUT_DIR, all_seeds_filename), index=False)
    print(f"\nSaved combined: {all_seeds_filename}")

    df_avg = df_all.groupby('Model').agg({
        'LEEP_Real': ['mean', 'std'],
        'LogME': ['mean', 'std'],
        'SHESHA_Var': ['mean', 'std'],
        'SHESHA_FS': ['mean', 'std'],
        'Dim': 'first',
        'Dataset': 'first',
        'Corruption': 'first',
        'Severity': 'first',
        'N_Samples': 'first',
    }).reset_index()

    df_avg.columns = [
        'Model',
        'LEEP_Real_Mean', 'LEEP_Real_Std',
        'LogME_Mean', 'LogME_Std',
        'SHESHA_Var_Mean', 'SHESHA_Var_Std',
        'SHESHA_FS_Mean', 'SHESHA_FS_Std',
        'Dim', 'Dataset', 'Corruption', 'Severity', 'N_Samples',
    ]

    df_avg['Seeds'] = str(seeds)
    df_avg['N_Seeds'] = len(seeds)

    avg_filename = f"{dataset_name.upper()}_{corruption}_sev{severity}_AVERAGED_{timestamp}.csv"
    df_avg.to_csv(os.path.join(OUTPUT_DIR, avg_filename), index=False)
    print(f"Saved averaged: {avg_filename}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name.upper()} | {corruption} | sev{severity}")
    print(f"{'='*70}")
    print(f"  Seeds: {seeds}")
    print(f"  Models evaluated: {len(df_avg)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\nMetric Means (averaged across seeds):")
    for col in ['LEEP_Real_Mean', 'LogME_Mean', 'SHESHA_Var_Mean', 'SHESHA_FS_Mean']:
        mean_val = df_avg[col].mean()
        std_val = df_avg[col].std()
        print(f"  {col}: {mean_val:.4f} +/- {std_val:.4f}")

    return per_seed_dfs, df_avg


# =============================================================================
# 8) RUN ALL CORRUPTIONS FOR A DATASET
# =============================================================================

def run_all_corruptions(
    dataset_name: str,
    severities=(5,),
    corruptions=None,
    seeds=SEEDS,
    batch_size: int = 64,
    num_workers: int = 4,
    resume: bool = True,
):
    """
    Iterate over (corruption × severity) pairs for a given CIFAR-C dataset.

    Parameters
    ----------
    dataset_name : 'cifar10c' or 'cifar100c'
    severities   : tuple of severity levels to run, e.g. (1, 3, 5)
    corruptions  : list of corruption names; defaults to all available on disk
    resume       : if True, skip already-evaluated models within each seed
    """
    if corruptions is None:
        corruptions = list_available_corruptions(dataset_name)
        if not corruptions:
            raise FileNotFoundError(
                f"No corruption .npy files found in {DATASET_CONFIG[dataset_name]['data_dir']}"
            )

    all_results = {}

    for corruption in corruptions:
        for severity in severities:
            key = (corruption, severity)
            print(f"\n{'#'*70}")
            print(f"# {dataset_name.upper()} | {corruption} | severity {severity}")
            print(f"{'#'*70}")

            per_seed_dfs, df_avg = run_multi_seed_corrupt_benchmark(
                dataset_name=dataset_name,
                corruption=corruption,
                severity=severity,
                seeds=seeds,
                batch_size=batch_size,
                num_workers=num_workers,
                resume=resume,
            )
            all_results[key] = (per_seed_dfs, df_avg)

    return all_results


# =============================================================================
# 9) MAIN — shardable CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CIFAR-C corrupt benchmark (shardable for limited compute)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List what would run (no GPU work)
  python vision_architecture_corrupt.py --dry-run

  # CIFAR-10-C, noise corruptions only, severity 5, seed 320
  python vision_architecture_corrupt.py --datasets cifar10c --group noise

  # Single corruption — auto-resumes if the session crashed mid-run
  python vision_architecture_corrupt.py --datasets cifar10c \\
      --corruptions shot_noise --severities 5 --seeds 320

  # Force a fresh run (ignore any checkpoint from a previous crash)
  python vision_architecture_corrupt.py --datasets cifar10c \\
      --corruptions shot_noise --no-resume

  # Split full CIFAR-10-C (19 corruptions) across 4 parallel jobs
  python vision_architecture_corrupt.py --datasets cifar10c --shard 0 --num-shards 4
  python vision_architecture_corrupt.py --datasets cifar10c --shard 1 --num-shards 4
  ...

  # Original full sweep (both datasets, all corruptions) — long run
  python vision_architecture_corrupt.py --full

  # Show corruption files present on disk
  python vision_architecture_corrupt.py --list-available --datasets cifar10c,cifar100c
        """,
    )
    parser.add_argument(
        "--datasets", type=str, default="cifar10c",
        help="Comma-separated: cifar10c, cifar100c (default: cifar10c only)",
    )
    parser.add_argument(
        "--corruptions", type=str, default=None,
        help="Comma-separated corruption names (overrides --group)",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        choices=["noise", "blur", "weather", "digital", "all"],
        help="Preset corruption batch (ignored if --corruptions is set)",
    )
    parser.add_argument(
        "--severities", type=str, default="5",
        help="Comma-separated severity levels 1-5 (default: 5)",
    )
    parser.add_argument(
        "--seeds", type=str, default="320",
        help="Comma-separated seeds (default: 320)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Initial inference batch size; halves on OOM down to 8 (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers; 0 = in-process, avoids fork OOM (default: 0)",
    )
    parser.add_argument(
        "--shard", type=int, default=0,
        help="Which shard to run (0-indexed); use with --num-shards",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Split jobs into N deterministic shards (default: 1 = no split)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run legacy full sweep: both datasets, all corruptions, sev5, seed320",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print job plan only; do not evaluate models",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show completed vs remaining jobs from AVERAGED CSVs on disk",
    )
    parser.add_argument(
        "--list-available", action="store_true",
        help="List corruption .npy files on disk and exit",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh; ignore checkpoint files from previous crashed runs",
    )
    args = parser.parse_args()

    if args.full:
        datasets = ["cifar10c", "cifar100c"]
        severities = (5,)
        seeds = [320]
        corruption_group = "all"
        explicit_corruptions = None
    else:
        datasets = [d.lower() for d in parse_csv_list(args.datasets)]
        severities = tuple(parse_int_list(args.severities))
        seeds = parse_int_list(args.seeds)
        corruption_group = args.group
        explicit_corruptions = parse_csv_list(args.corruptions)

    for ds in datasets:
        if ds not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset '{ds}'. Use: cifar10c, cifar100c")

    if args.list_available:
        for ds in datasets:
            avail = list_available_corruptions(ds)
            print(f"\n{ds.upper()} ({DATASET_CONFIG[ds]['data_dir']}):")
            for c in avail:
                print(f"  - {c}")
            print(f"  ({len(avail)} corruptions)")
        raise SystemExit(0)

    corruptions_by_dataset = {}
    for ds in datasets:
        corruptions_by_dataset[ds] = resolve_corruptions(
            ds,
            corruptions=explicit_corruptions,
            corruption_group=corruption_group if explicit_corruptions is None else None,
        )

    if args.status:
        print_run_status(datasets, severities, corruptions_by_dataset)
        raise SystemExit(0)

    jobs = build_corrupt_jobs(datasets, severities, corruptions_by_dataset)
    jobs = shard_jobs(jobs, args.shard, args.num_shards)

    if not jobs:
        print("No jobs scheduled for this shard/configuration.")
        raise SystemExit(0)

    print_job_plan(jobs, shard_index=args.shard, num_shards=args.num_shards)

    if args.dry_run:
        raise SystemExit(0)

    do_resume = not args.no_resume
    completed = 0
    for dataset_name, corruption, severity in jobs:
        print(f"\n{'#'*70}")
        print(f"# {dataset_name.upper()} | {corruption} | severity {severity}")
        print(f"{'#'*70}")
        run_multi_seed_corrupt_benchmark(
            dataset_name=dataset_name,
            corruption=corruption,
            severity=severity,
            seeds=seeds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resume=do_resume,
        )
        completed += 1

    print("\n" + "=" * 70)
    print(f"SHARD COMPLETE: {completed}/{len(jobs)} jobs")
    if args.num_shards > 1:
        print(f"  shard {args.shard} of {args.num_shards}")
    print(f"  output: {OUTPUT_DIR}")
    print("=" * 70)
