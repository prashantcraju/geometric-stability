"""
Shesha Vision Architecture Experiment
"""
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

# Output directory
OUTPUT_DIR = Path("./shesha-vision_architecture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    """Get curated list of 94 pretrained models."""
    foundation = [
        'vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
        'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
        'vit_base_patch32_clip_224.openai', 'vit_base_patch16_clip_224.openai',
        'vit_large_patch14_clip_224.openai', 'vit_base_patch16_224.mae',
        'eva02_base_patch14_224.mim_in22k', 'eva02_large_patch14_224.mim_in22k',
        'vit_base_patch16_224.dino', 'beit_base_patch16_224.in22k_ft_in22k',
    ]

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
        'efficientnet_b0.ra_in1k', 'efficientnet_b1.ra_in1k', 'efficientnet_b2.ra_in1k', 'efficientnet_b3.ra_in1k',
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
    combined = foundation + transformers + cnns + robust

    if len(combined) < 90:
        combined += [m for m in all_timm if 'mobilenetv3' in m or 'densenet' in m][:30]

    return [m for m in combined if m in all_timm]


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

    for m_name in tqdm(model_names, desc=f"{dataset_name}/seed{seed}"):
        try:
            model_l = timm.create_model(m_name, pretrained=True).to(DEVICE).eval()
            model_f = timm.create_model(m_name, pretrained=True, num_classes=0).to(DEVICE).eval()

            config_data = timm.data.resolve_data_config({}, model=model_l)
            transform = timm.data.create_transform(**config_data, is_training=False)

            full_ds, _ = get_dataset(dataset_name, transform, split='test')

            loader_kwargs = dict(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(DEVICE.type == 'cuda'),
            )
            if num_workers > 0:
                loader_kwargs.update(dict(
                    persistent_workers=True,
                    prefetch_factor=2,
                    worker_init_fn=worker_init_fn_factory(seed),
                ))

            loader = DataLoader(
                torch.utils.data.Subset(full_ds, subset_idx),
                **loader_kwargs
            )

            all_feats, all_logits = [], []
            batch_statuses = []

            with torch.no_grad():
                for imgs, _ in loader:
                    imgs = imgs.to(DEVICE, non_blocking=True)

                    l_out, status = get_robust_logits(model_l(imgs))
                    f_out = get_pooled_features(model_f(imgs))

                    all_feats.append(f_out.cpu().numpy())
                    batch_statuses.append(status)
                    if l_out is not None:
                        all_logits.append(l_out.cpu().numpy())

            X = np.concatenate(all_feats)
            L = np.concatenate(all_logits) if all_logits else None

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
            })

            del model_l, model_f, X, L
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error {m_name}: {e}")
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    print(f"\nSeed {seed} Complete: {len(results)}/{len(model_names)} models")
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
    # USAGE EXAMPLES
    # 1. Run single dataset with single seed:
    # df = run_single_seed_benchmark('cifar100', seed=320)

    # 2. Run single dataset with multiple seeds:
    # per_seed_dfs, df_avg = run_multi_seed_benchmark('cifar10', seeds=[320, 1991, 9])

    # 3. Run all datasets with all seeds:
    # all_results = run_all_datasets(seeds=[320, 1991, 9])

    # 4. Run specific datasets:
    # all_results = run_all_datasets(
    #     seeds=[320, 1991, 9],
    #     datasets=['cifar10', 'cifar100', 'flowers102']
    # )


    all_results = run_all_datasets(
        seeds=[320],
        datasets=['cifar10', 'cifar100', 'flowers102', 'dtd', 'eurosat', 'pets'],
        batch_size=256,
        num_workers=4
    )


    print("\n" + "="*70)
    print("ALL BENCHMARKS COMPLETE")
    print("="*70)