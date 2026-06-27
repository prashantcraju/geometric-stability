# Geometric Stability: The Missing Axis of Representations
<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2601.09173" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-blue" /></a>
    <a style="text-decoration:none !important;" href="https://huggingface.co/papers/2601.09173" alt="Hugging Face Papers"><img src="https://img.shields.io/badge/paper-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" /></a>
</p>


## Overview

This repository contains code to reproduce the metric validation, distinction, and vision-architecture experiments from the paper. Each experiment folder is self-contained with its own dependencies and executable scripts.

## Installation

**Note on Dependencies:** Different experiments require different library versions (e.g., conflicting versions of PyTorch or SciPy). We strongly recommend creating a fresh virtual environment (Conda or venv) for each experiment folder to avoid conflicts.

Each experiment folder contains its own `requirements.txt`. Install dependencies for the specific experiment you want to run:
```bash
cd "<folder_name>"
pip install -r requirements.txt
```

For GPU-accelerated experiments (e.g., `distinction/`), also install:
```bash
pip install -r requirements-gpu.txt
```

For `vision architecture/`, install PyTorch with the CUDA wheel that matches your system before running the scripts. See the comment at the top of `vision architecture/requirements.txt`.

## Additional Dependencies

`vision architecture/` includes `LogME.py` for the LogME scoring function, sourced from the [official implementation](https://github.com/thuml/LogME) (You et al., ICML 2021; You et al., JMLR 2022). No separate download is required.

For corrupted-benchmark experiments, download CIFAR-10-C and CIFAR-100-C first:
```bash
cd "vision architecture"
python colab_download_cifar_c.py
```

## Experiments

| Folder | Description | Notes |
|--------|-------------|-------|
| `metric validation/` | Shesha metric validation on synthetic and real embeddings | Run `shesha_validation_embeddings.py` before `shesha_validation.py` |
| `distinction/` | Ground-truth validation, cross-domain encoder tests, and metric dissociation | GPU recommended; run encoder tests before `combined_analysis.py` |
| `vision architecture/` | Vision model architecture benchmarks and follow-on analyses | Includes `LogME.py`; run `vision_architecture.py` before downstream analysis scripts |

### `metric validation/`

| Script | Purpose |
|--------|---------|
| `shesha_validation_embeddings.py` | Extract embeddings for the validation suite |
| `shesha_validation.py` | Run the full Shesha metric validation battery |
| `proof_ortho_rotation.py` | Verify CKA invariance under orthogonal rotation |

### `distinction/`

| Script | Purpose |
|--------|---------|
| `distinction_ground_truth.py` | Synthetic ground-truth distinction tests |
| `distinction_encoder_test.py` | Cross-domain encoder benchmark (7 domains, 15 seeds) |
| `distinction_encoder_test_ucf101.py` | Same benchmark with UCF-101 video domain |
| `combined_analysis.py` | Aggregate cross-domain analysis from encoder-test outputs |
| `distinction_encoders_robustness.py` | Metric robustness checks in the language domain |
| `distinction-spectral-test-extended.py` | Spectral sensitivity vs. multiple similarity metrics |
| `spectral_cka_vs_shesha.py` | Focused CKA vs. Shesha under spectral deletion |

### `vision architecture/`

| Script | Purpose |
|--------|---------|
| `vision_architecture.py` | Main clean-benchmark run (CIFAR-10 / CIFAR-100) |
| `vision_architecture_corrupt.py` | Corrupted benchmarks (CIFAR-10-C / CIFAR-100-C) |
| `colab_download_cifar_c.py` | Download CIFAR-C archives into `./data/` |
| `compute_robustness_table.py` | Build the corruption-robustness correlation table |
| `extreme_groups_robustness.py` | Top/bottom decile robustness comparison |
| `training_objective_stability.py` | Training-objective group analysis |
| `cifar10_seed_stability.py` | Seed-to-seed variability analysis |
| `probe_subset_sensitivity.py` | Probe subset-sensitivity experiment |
| `sam-ablation.py` | SAM rho sweep ablation |

## Usage

Each folder contains standalone scripts. Typical entry points:

```bash
cd "metric validation"
python shesha_validation_embeddings.py
python shesha_validation.py
```

```bash
cd distinction
python distinction_ground_truth.py
python distinction_encoder_test.py
python combined_analysis.py
```

```bash
cd "vision architecture"
python vision_architecture.py
python compute_robustness_table.py
```

Many analysis scripts in `vision architecture/` expect benchmark outputs under `./shesha-vision_architecture/` and, for corruption analyses, `./shesha-vision_architecture-corrupt/`. Pass `--clean-dir` and `--corrupt-dir` to override these defaults.

## 🚀 Quick Start (For Practitioners)

**Looking to use Geometric Stability (Shesha) in your own research or production models?**

You do not need to clone this repository. We maintain a production-ready, optimized Python library for that:

| **Repository** | **Purpose** | **Link** |
| :--- | :--- | :--- |
| **`shesha` (Recommended)** | 📦 **The Library.** Use this to measure stability in your own models (LLMs, Bio, Vision). | [**View on GitHub**](https://github.com/prashantcraju/shesha) |
| `geometric-stability` | 📄 **The Paper.** Use this only to reproduce the specific figures/experiments from our arXiv paper. | *You are here* |

### Installation
```bash
pip install shesha-geometry
```

### Citation

If you use `shesha-geometry`, please cite:
```bibtex
@software{shesha2026,
  title = {Shesha: Self-Consistency Metrics for Representational Stability},
  author = {Raju, Prashant C.},
  year = {2026},
  howpublished = {Zenodo},
  doi = {10.5281/zenodo.18227453},
  url = {https://doi.org/10.5281/zenodo.18227453},
  copyright = {MIT License}
}

@article{raju2026geometric,
  title = {Geometric Stability: The Missing Axis of Representations},
  author = {Raju, Prashant C.},
  journal = {arXiv preprint arXiv:2601.09173},
  year = {2026}
}
```

## License

This repository is released under the [MIT License](LICENSE).
