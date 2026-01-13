"""
Shesha Metric Validation - Embeddings
"""

import os
import sys

# =============================================================================
# 0) DETERMINISM GUARDS (MATCHING VALIDATION SUITE)
# =============================================================================
SEED = 320
REQUIRED_ENV = {
    "PYTHONHASHSEED": str(SEED),
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

def require_env(k: str, v: str) -> None:
    got = os.environ.get(k)
    if got != v:
        print(f"[FATAL] Environment variable {k} must be set to '{v}' before starting Python.")
        print(f"Run:\n  export {k}={v}\n  python {sys.argv[0]}")
        sys.exit(1)

def check_required_env() -> None:
    require_env("PYTHONHASHSEED", REQUIRED_ENV["PYTHONHASHSEED"])
    require_env("CUBLAS_WORKSPACE_CONFIG", REQUIRED_ENV["CUBLAS_WORKSPACE_CONFIG"])
    require_env("OMP_NUM_THREADS", REQUIRED_ENV["OMP_NUM_THREADS"])
    require_env("MKL_NUM_THREADS", REQUIRED_ENV["MKL_NUM_THREADS"])

check_required_env()
os.environ.update(REQUIRED_ENV)

# =============================================================================
# 1) IMPORTS & CONFIGURATION
# =============================================================================
import torch
import timm
from timm.data import resolve_data_config, create_transform
import numpy as np
import torchvision
from pathlib import Path
from tqdm import tqdm
import random

# 15 Models x 2 Datasets
MODELS_TO_EXTRACT = [
    # --- ResNets ---
    "resnet18",
    "resnet34",
    "resnet50",
    "seresnet50",
    # --- Efficiency ---
    "densenet121",
    "mobilenetv3_large_100",
    # --- EfficientNets ---
    "efficientnet_b0",
    "efficientnet_b2",
    # --- ConvNeXt ---
    "convnext_tiny",
    "convnext_small",
    # --- Transformers ---
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "deit_small_patch16_224",
]

DATASETS = ["cifar10", "cifar100"]

OUTPUT_DIR = Path("./shesha-validation/embeds")
LABEL_DIR = Path("./shesha-validation")
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Strict Torch Determinism
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Global Seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =============================================================================
# 2) EXTRACTION LOGIC
# =============================================================================

def get_dataset(name, transform):
    """Returns dataset with the exact transform required by the specific model."""
    if name == 'cifar10':
        return torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif name == 'cifar100':
        return torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")

def extract_features(model_name, dataset_name):
    dest_path = OUTPUT_DIR / f"{dataset_name}_{model_name}.npy"
    if dest_path.exists():
        print(f"[SKIP] {dest_path} already exists.")
        return

    print(f"[{dataset_name.upper()}] Extracting {model_name}...")
    
    try:
        # 1. Load Model
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model = model.to(DEVICE)
        model.eval()
        
        # 2. Resolve Canonical Data Config (The "Right" Way)
        config = resolve_data_config({}, model=model)
        # Force is_training=False to ensure deterministic CenterCrop/Resize behavior
        transform = create_transform(**config, is_training=False)
        
        # 3. Setup Dataset & Loader
        ds = get_dataset(dataset_name, transform)
        
        # Generator for DataLoader determinism
        g = torch.Generator()
        g.manual_seed(SEED)
        
        loader = torch.utils.data.DataLoader(
            ds, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        # 4. Save Labels (Once per dataset)
        lbl_path = LABEL_DIR / f"{dataset_name}_labels.npy"
        if not lbl_path.exists():
            # ds.targets is a list in torchvision datasets
            np.save(lbl_path, np.array(ds.targets))
            print(f"Saved labels to {lbl_path}")

        # 5. Extract
        embeds = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=f"{model_name}", leave=False):
                images = images.to(DEVICE)
                features = model(images)
                embeds.append(features.cpu().numpy())
                
        all_features = np.concatenate(embeds)
        
        # Save raw (Validation suite handles normalization if needed)
        np.save(dest_path, all_features.astype(np.float32))
        print(f" -> Saved {all_features.shape} to {dest_path}")
        
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f" -> FAILED {model_name}: {e}")

if __name__ == "__main__":
    print(f"Starting STRICT Extraction (Seed={SEED})...")
    print(f"Models: {len(MODELS_TO_EXTRACT)} | Datasets: {DATASETS}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    for ds_name in DATASETS:
        for m_name in MODELS_TO_EXTRACT:
            extract_features(m_name, ds_name)
            
    print("\nDONE. Ready for shesha_validation.py")