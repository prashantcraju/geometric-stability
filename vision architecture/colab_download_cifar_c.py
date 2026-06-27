"""
Download CIFAR-10-C / CIFAR-100-C to local data directory.

Usage:
    python colab_download_cifar_c.py

    python colab_download_cifar_c.py --datasets cifar10c

    python colab_download_cifar_c.py --datasets cifar10c,cifar100c

Data layout (matches vision_architecture_corrupt.py):
    ./data/CIFAR-10-C/
    ./data/CIFAR-100-C/
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = ROOT / "data"

DATASETS = {
    "cifar10c": {
        "folder": "CIFAR-10-C",
        "url": "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1",
        "tar": "CIFAR-10-C.tar",
        "expected_npy_min": 20,
    },
    "cifar100c": {
        "folder": "CIFAR-100-C",
        "url": "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1",
        "tar": "CIFAR-100-C.tar",
        "expected_npy_min": 20,
    },
}


def run(cmd):
    print(f"\n$ {cmd}")
    subprocess.check_call(cmd, shell=True)


def download_one(output_base: Path, key: str, force: bool = False) -> Path:
    meta = DATASETS[key]
    data_root = output_base
    dest = data_root / meta["folder"]
    tar_path = data_root / meta["tar"]

    data_root.mkdir(parents=True, exist_ok=True)

    npy_count = len(glob.glob(str(dest / "*.npy"))) if dest.exists() else 0
    if not force and npy_count >= meta["expected_npy_min"]:
        print(f"[SKIP] {dest} already has {npy_count} .npy files")
        return dest

    if dest.exists() and force:
        print(f"[WARN] Re-downloading {key}; existing files may be overwritten by tar extract")

    run(f'wget -q --show-progress -O "{tar_path}" "{meta["url"]}"')
    run(f'tar -xf "{tar_path}" -C "{data_root}"')
    run(f'rm -f "{tar_path}"')

    npy_count = len(glob.glob(str(dest / "*.npy")))
    print(f"[OK] {dest}: {npy_count} .npy files")
    if npy_count < meta["expected_npy_min"]:
        print(f"[WARN] Expected at least {meta['expected_npy_min']} .npy files")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Download CIFAR-C corruption benchmarks")
    parser.add_argument(
        "--datasets",
        type=str,
        default="cifar10c",
        help="Comma-separated: cifar10c, cifar100c",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument(
        "--output-base",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Directory for downloaded CIFAR-C data (default: ./data)",
    )
    args = parser.parse_args()

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    keys = [k.strip().lower() for k in args.datasets.split(",") if k.strip()]
    for key in keys:
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset {key}. Use: cifar10c, cifar100c")
        download_one(output_base, key, force=args.force)

    print("\n--- Verify ---")
    for key in keys:
        folder = DATASETS[key]["folder"]
        path = output_base / folder
        print(f"{folder} files: {len(glob.glob(str(path / '*.npy')))}  ({path})")


if __name__ == "__main__":
    main()
