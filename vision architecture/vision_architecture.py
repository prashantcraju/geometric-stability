"""
Shesha Vision Architecture Experiment
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from LogME import LogME


# =============================================================================
# 0) CONFIGURATION
# =============================================================================
SEEDS = [320, 1991, 9]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
# Start high and let OOM backoff (down to 8) find the limit per model.
DEFAULT_BATCH_SIZE = 2048

# Models excluded from benchmarking (persistent OOM / unusable for this paper).
EXCLUDED_MODELS = {
    'vitamin_base_224.datacomp1b_clip',
}

# Valid source label spaces for LEEP
VALID_LEEP_SOURCE_CLASSES = {1000, 21841, 21843, 11821, 11221, 10450, 12000}

# Dataset configurations
DATASET_CONFIG = {
    'cifar10': {
        'n_samples': 5000,
        'n_classes': 10,
    },
    'cifar100': {
        'n_samples': 5000,
        'n_classes': 100,
    },
    'flowers102': {
        'n_samples': 5000,  # Will use min(n_samples, len(dataset))
        'n_classes': 102,
    },
    'dtd': {
        'n_samples': 1600,  # DTD test set is small (~1880)
        'n_classes': 47,
    },
    'pets': {
        'n_samples': 1500,  # Oxford Pets test set
        'n_classes': 37,
    },
    'eurosat': {
        'n_samples': 5000,
        'n_classes': 10,
    },
}

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
GDRIVE_FOLDER = None
DEFAULT_OUTPUT_DIR = "./shesha-vision_architecture"


def _resolve_output_dir(gdrive_folder=GDRIVE_FOLDER) -> Path:
    """Return output Path: optional override if writable, else local default."""
    if gdrive_folder:
        gdrive_path = Path(gdrive_folder)
        try:
            gdrive_path.mkdir(parents=True, exist_ok=True)
            test_file = gdrive_path / ".write_test"
            test_file.touch()
            test_file.unlink()
            print(f"[Output] Saving to: {gdrive_path}")
            return gdrive_path
        except Exception:
            pass
    local_path = Path(DEFAULT_OUTPUT_DIR)
    local_path.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Saving locally: {local_path.resolve()}")
    return local_path


OUTPUT_DIR = _resolve_output_dir()

# =============================================================================
# 1) DETERMINISM UTILITIES
# =============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn_factory(seed):
    """Create a worker init function with a specific base seed."""
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
# 3) DATASET LOADING
# =============================================================================

def get_dataset(dataset_name, transform, split='test'):
    """Load dataset by name with appropriate transform."""
    dataset_name = dataset_name.lower()

    if dataset_name == 'cifar10':
        ds = datasets.CIFAR10(root='./data', train=(split == 'train'), download=True, transform=transform)
        return ds, np.array(ds.targets)

    elif dataset_name == 'cifar100':
        ds = datasets.CIFAR100(root='./data', train=(split == 'train'), download=True, transform=transform)
        return ds, np.array(ds.targets)

    elif dataset_name == 'flowers102':
        ds = datasets.Flowers102(root='./data', split=split, download=True, transform=transform)
        return ds, np.array(ds._labels)

    elif dataset_name == 'dtd':
        ds = datasets.DTD(root='./data', split=split, download=True, transform=transform)
        return ds, np.array(ds._labels)

    elif dataset_name == 'pets':
        pet_split = 'test' if split == 'test' else 'trainval'
        ds = datasets.OxfordIIITPet(root='./data', split=pet_split, download=True, transform=transform)
        return ds, np.array(ds._labels)

    elif dataset_name == 'eurosat':
        ds = datasets.EuroSAT(root='./data', download=True, transform=transform)
        return ds, np.array(ds.targets)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_raw_dataset(dataset_name):
    """Get raw dataset (no transform) for label extraction."""
    dataset_name = dataset_name.lower()
    minimal_transform = transforms.ToTensor()

    if dataset_name == 'cifar10':
        ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=minimal_transform)
        return ds, np.array(ds.targets)

    elif dataset_name == 'cifar100':
        ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=minimal_transform)
        return ds, np.array(ds.targets)

    elif dataset_name == 'flowers102':
        ds = datasets.Flowers102(root='./data', split='test', download=True, transform=minimal_transform)
        return ds, np.array(ds._labels)

    elif dataset_name == 'dtd':
        ds = datasets.DTD(root='./data', split='test', download=True, transform=minimal_transform)
        return ds, np.array(ds._labels)

    elif dataset_name == 'pets':
        ds = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=minimal_transform)
        return ds, np.array(ds._labels)

    elif dataset_name == 'eurosat':
        ds = datasets.EuroSAT(root='./data', download=True, transform=minimal_transform)
        return ds, np.array(ds.targets)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


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


def _build_benchmark_dataloader(full_ds, subset_idx, batch_size, num_workers, seed):
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


def _infer_combined_pass(model_with_head, model_backbone, full_ds, subset_idx,
                         num_workers, seed):
    def infer_fn(batch_size):
        loader = _build_benchmark_dataloader(full_ds, subset_idx, batch_size, num_workers, seed)
        try:
            return _run_combined_pass(model_with_head, model_backbone, loader)
        finally:
            del loader

    return infer_fn


# =============================================================================
# 5) SINGLE-SEED BENCHMARK
# =============================================================================

def run_single_seed_benchmark(dataset_name, seed, batch_size=64, num_workers=4):
    """Run benchmark for a single dataset and seed."""
    set_seed(seed)

    dataset_name = dataset_name.lower()
    config = DATASET_CONFIG.get(dataset_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIG.keys())}")

    model_names = get_strategic_model_list()
    print(f"\n{'='*70}")
    print(f"Running Benchmark: {dataset_name.upper()} | Seed: {seed}")
    print(f"{'='*70}")
    print(f"  Device: {DEVICE}, Batch size: {batch_size}, Workers: {num_workers}")
    print(f"  Models: {len(model_names)}")

    raw_ds, all_labels = get_raw_dataset(dataset_name)

    n_samples = min(config['n_samples'], len(raw_ds))
    print(f"  Dataset size: {len(raw_ds)}, Using: {n_samples} samples")

    subset_idx = np.random.choice(len(raw_ds), n_samples, replace=False)
    y_target = all_labels[subset_idx]

    subset_hash = hashlib.sha256(subset_idx.tobytes()).hexdigest()[:12]
    print(f"  Subset SHA: {subset_hash}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    idx_filename = f"{dataset_name}_seed{seed}_subset_idx.npy"
    np.save(os.path.join(OUTPUT_DIR, idx_filename), subset_idx)

    results = []
    leep_status_counts = {}
    n_oom_skipped = 0

    def _make_oom_skip_row(model_name):
        return {
            "Model": model_name,
            "Dataset": dataset_name,
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

    for m_name in tqdm(model_names, desc=f"{dataset_name}/seed{seed}"):
        tqdm.write(f"  -> {m_name}")
        model_l = None
        model_f = None
        full_ds = None
        try:
            model_l = timm.create_model(m_name, pretrained=True).to(DEVICE).eval()
            config_data = timm.data.resolve_data_config({}, model=model_l)
            transform = timm.data.create_transform(**config_data, is_training=False)
            full_ds, _ = get_dataset(dataset_name, transform, split='test')
            model_f = timm.create_model(m_name, pretrained=True, num_classes=0).to(DEVICE).eval()

            (all_feats, all_logits, batch_statuses), infer_bs = _run_with_batch_retry(
                _infer_combined_pass(
                    model_l, model_f, full_ds, subset_idx, num_workers, seed,
                ),
                batch_size,
            )

            del model_l, model_f, full_ds
            model_l = model_f = full_ds = None
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

            results.append({
                "Model": m_name,
                "Dataset": dataset_name,
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
            })

            del X, L

        except CudaOomExhaustedError as e:
            print(f"  [OOM] Skipping {m_name}: {e}")
            n_oom_skipped += 1
            leep_status_counts["oom_skip"] = leep_status_counts.get("oom_skip", 0) + 1
            results.append(_make_oom_skip_row(m_name))
            del model_l, model_f, full_ds
            _release_cuda_memory()

        except RuntimeError as e:
            if _is_cuda_oom(e):
                print(f"  [OOM] Skipping {m_name}: {e}")
                n_oom_skipped += 1
                leep_status_counts["oom_skip"] = leep_status_counts.get("oom_skip", 0) + 1
                results.append(_make_oom_skip_row(m_name))
            else:
                print(f"Error {m_name}: {e}")
            del model_l, model_f, full_ds
            _release_cuda_memory()

        except Exception as e:
            print(f"Error {m_name}: {e}")
            del model_l, model_f, full_ds
            _release_cuda_memory()

    df = pd.DataFrame(results)

    print(f"\nSeed {seed} Complete: {len(results)}/{len(model_names)} models"
          f" ({n_oom_skipped} OOM-skipped)")
    print(f"  LEEP status breakdown:")
    for status, count in sorted(leep_status_counts.items()):
        print(f"    {status}: {count}")

    return df


# =============================================================================
# 6) MULTI-SEED BENCHMARK WITH AVERAGING
# =============================================================================

def run_multi_seed_benchmark(dataset_name, seeds=SEEDS, batch_size=64, num_workers=4):
    """Run benchmark across multiple seeds and compute averages."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = dataset_name.lower()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_seed_dfs = []

    for seed in seeds:
        df_seed = run_single_seed_benchmark(
            dataset_name=dataset_name,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers
        )

        seed_filename = f"{dataset_name.upper()}_SEED{seed}_{timestamp}.csv"
        seed_path = os.path.join(OUTPUT_DIR, seed_filename)
        df_seed.to_csv(seed_path, index=False)
        print(f"  Saved: {seed_filename}")

        per_seed_dfs.append(df_seed)

    df_all = pd.concat(per_seed_dfs, ignore_index=True)

    all_seeds_filename = f"{dataset_name.upper()}_ALL_SEEDS_{timestamp}.csv"
    all_seeds_path = os.path.join(OUTPUT_DIR, all_seeds_filename)
    df_all.to_csv(all_seeds_path, index=False)
    print(f"\nSaved combined: {all_seeds_filename}")

    df_avg = df_all.groupby('Model').agg({
        'LEEP_Real': ['mean', 'std'],
        'LogME': ['mean', 'std'],
        'SHESHA_Var': ['mean', 'std'],
        'SHESHA_FS': ['mean', 'std'],
        'Dim': 'first',
        'Dataset': 'first',
        'N_Samples': 'first',
    }).reset_index()

    df_avg.columns = [
        'Model',
        'LEEP_Real_Mean', 'LEEP_Real_Std',
        'LogME_Mean', 'LogME_Std',
        'SHESHA_Var_Mean', 'SHESHA_Var_Std',
        'SHESHA_FS_Mean', 'SHESHA_FS_Std',
        'Dim', 'Dataset', 'N_Samples'
    ]

    df_avg['Seeds'] = str(seeds)
    df_avg['N_Seeds'] = len(seeds)

    avg_filename = f"{dataset_name.upper()}_AVERAGED_{timestamp}.csv"
    avg_path = os.path.join(OUTPUT_DIR, avg_filename)
    df_avg.to_csv(avg_path, index=False)
    print(f"Saved averaged: {avg_filename}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name.upper()}")
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
# 7) RUN ALL DATASETS
# =============================================================================

def run_all_datasets(seeds=SEEDS, datasets=None, batch_size=64, num_workers=4):
    """Run benchmarks on all (or specified) datasets."""
    if datasets is None:
        datasets = list(DATASET_CONFIG.keys())

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*70}")

        per_seed_dfs, df_avg = run_multi_seed_benchmark(
            dataset_name=dataset_name,
            seeds=seeds,
            batch_size=batch_size,
            num_workers=num_workers
        )

        all_results[dataset_name] = (per_seed_dfs, df_avg)

    return all_results


# =============================================================================
# 8) MAIN
# =============================================================================

if __name__ == "__main__":
    all_results = run_all_datasets(
        seeds=SEEDS,
        datasets=['cifar10', 'cifar100', 'flowers102', 'dtd', 'eurosat', 'pets'],
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=4,
    )


    print("\n" + "="*70)
    print("ALL BENCHMARKS COMPLETE")
    print("="*70)