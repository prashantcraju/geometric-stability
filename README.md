# Geometric Stability: The Missing Axis of Representations

## Overview

This repository contains code to reproduce all experiments in the paper. Each experiment is self-contained with its own dependencies and executable scripts.

## Installation

**Note on Dependencies:** Different experiments require different library versions (e.g., conflicting versions of PyTorch or SciPy). We strongly recommend creating a fresh virtual environment (Conda or venv) for each experiment folder to avoid conflicts.

Each experiment folder contains its own `requirements.txt`. Install dependencies for the specific experiment you want to run:
```bash
cd <folder_name>
pip install -r requirements.txt
```

For GPU-accelerated experiments (e.g., `distinction/`), also install:
```bash
pip install -r requirements-gpu.txt
```

## Hugging Face Configuration

Some experiments (specifically in `drift/`, `steering/`, and `transfer_learning/`) rely on **gated models** hosted on Hugging Face (e.g., Llama-3, Gemma). To run these scripts, you must provide a Hugging Face authentication token with the correct permissions.

### 1. Prerequisite: License Acceptance
Before generating a token, ensure your Hugging Face account has accepted the license terms for the following models. You must visit each link and click "Agree" on the model card:
* [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
* [google/gemma-7b](https://huggingface.co/google/gemma-7b)
* [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

### 2. Generate an Access Token
1. Log in to [Hugging Face](https://huggingface.co/).
2. Go to **[Settings > Access Tokens](https://huggingface.co/settings/tokens)**.
3. Create a new token with **READ** permissions.

### 3. Set Environment Variable
**Do not hardcode your token.** Instead, export it as an environment variable before running the scripts. The code is configured to automatically detect `HF_TOKEN`.

```bash
# Linux/macOS
export HF_TOKEN="your_huggingface_token_here"

# Windows PowerShell
$env:HF_TOKEN = "your_huggingface_token_here"
```

## Additional Dependencies

`transfer_learning/` and `vision_architecture/` require additional file `LogME.py` for the LogME scoring function sourced from the [official implementation](https://github.com/thuml/LogME) (You et al., ICML 2021, You et al., JMLR 2022).
```bash
wget https://raw.githubusercontent.com/thuml/LogME/main/LogME.py
```


## Experiments
| Folder | Description | Paper Section | Notes |
|--------|-------------|---------------| ---------------|
| `metric_validation/` | Shesha metric validation on embeddings | Appendix B | must run `shesha_validation_embeddings.py` before running `shesha_validation.py`|
| `distinction/` | Ground truth validation and metric dissociation | Section 2, Appendix C | |
| `steering/` | Representation steering (synthetic and real tasks) | Section 3.1 | |
| `vision_architecture/` | Vision model architecture comparisons | Section 3.2 | requires LogME (see Additional Dependencies)|
| `drift/` | Representational drift in language models | Section 3.3 | requires Hugging Face token (see Hugging Face Configuration)|
| `crispr/` | CRISPR perturbation coherence analysis | Section 3.4 | |
| `transfer_learning/` | Transfer learning benchmarks | Appendix G | requires LogME (see Additional Dependencies)|
| `neuroscience/` | Neural population stability analysis | Appendix I | |



## Usage

Each folder contains standalone scripts. For example:
```bash
cd distinction
python distinction_ground_truth.py
```

Results are saved to local output directories within each folder.

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
  title = {Shesha: Self-consistency Metrics for Representational Stability},
  author = {Raju, Prashant C.},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18227453},
  url = {https://doi.org/10.5281/zenodo.18227453},
  copyright = {MIT License}
}

@article{raju2026geometric,
  title={Geometric Stability: The Missing Axis of Representations},
  author={Raju, Prashant C.},
  journal={arXiv preprint arXiv:2601.09173},
  year={2026}
}
```

