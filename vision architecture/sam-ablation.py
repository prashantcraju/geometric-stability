"""
SAM Rho Sweep: Geometric Stability vs Sharpness-Aware Minimization
===================================================================
Trains ResNet-18 on CIFAR-10 and CIFAR-100 with SGD (rho=0) and SAM
across rho in [0, 0.5] in steps of 0.025, repeated over 15 seeds.
Plots Shesha geometric stability and accuracy (mean ± std across seeds)
as a function of SAM perturbation radius.

Requirements:
    pip install torch torchvision shesha-geometry sam-pytorch numpy matplotlib

Usage:
    python sam-ablation.py                          # full run (GPU recommended)
    python sam-ablation.py --epochs 5               # quick test
    python sam-ablation.py --quick                  # minimal smoke test
    python sam-ablation.py --datasets cifar10       # single dataset
"""

import argparse
import csv
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sam import SAM
import shesha

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]

DEFAULT_RHOS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]

DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
        "loader": torchvision.datasets.CIFAR10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
        "loader": torchvision.datasets.CIFAR100,
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="SAM rho sweep -- Shesha vs sharpness-aware minimization")
    p.add_argument("--epochs", type=int, default=100,
                   help="Training epochs per model (default: 100)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05,
                   help="Learning rate for SGD base optimizer")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--rhos", type=float, nargs="+", default=None,
                   help="SAM rho values to sweep (default: 0 to 0.5 step 0.025)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds to run (default: 15-seed list)")
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["cifar10", "cifar100"],
                   choices=list(DATASET_STATS.keys()),
                   help="Datasets to evaluate (default: cifar10 cifar100)")
    p.add_argument("--quick", action="store_true",
                   help="Quick smoke test: 5 epochs, 3 rhos, 2 seeds, small subset")
    p.add_argument("--device", type=str, default=None,
                   help="Force device (default: auto-detect)")
    p.add_argument("--output", type=str, default="sam_sweep_results.json",
                   help="Output JSON path")
    p.add_argument("--plot_prefix", type=str, default="sam_sweep",
                   help="Prefix for output plot files (one per dataset)")
    p.add_argument("--checkpoint-dir", type=str, default="./sam_sweep_results",
                   help="Directory for incremental checkpoints and CSV exports")
    p.add_argument("--n_shesha_splits", type=int, default=30,
                   help="Bootstrap splits for Shesha (default: 30)")
    p.add_argument("--max_eval_samples", type=int, default=2000,
                   help="Max test samples for Shesha eval (default: 2000)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size, dataset_name, quick=False):
    stats = DATASET_STATS[dataset_name]
    mean, std = stats["mean"], stats["std"]
    ds_cls = stats["loader"]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = ds_cls(root="./data", train=True,  download=True, transform=train_transform)
    test_set  = ds_cls(root="./data", train=False, download=True, transform=test_transform)

    if quick:
        train_set = torch.utils.data.Subset(train_set, range(2000))
        test_set  = torch.utils.data.Subset(test_set,  range(1000))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_resnet18(num_classes):
    """ResNet-18 adapted for CIFAR (small 32x32 input)."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_loader, epochs, lr, momentum, weight_decay,
                rho, device):
    """Train with SGD (rho=0) or SAM (rho>0)."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    use_sam = rho > 0

    base_opt = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay)

    if use_sam:
        optimizer = SAM(model.parameters(), base_opt, rho=rho)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=epochs)
    else:
        optimizer = base_opt
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    label = f"SAM(rho={rho})" if use_sam else "SGD"

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if use_sam:
                def closure():
                    optimizer.zero_grad()
                    out  = model(inputs)
                    loss = criterion(out, targets)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                with torch.no_grad():
                    outputs = model(inputs)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total   += targets.size(0)
        scheduler.step()

        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            acc = 100.0 * correct / total
            print(f"  [{label}] Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={running_loss/total:.4f}  train_acc={acc:.1f}%")

    return model


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

def extract_penultimate(model, dataloader, device, max_samples=None):
    """Extract penultimate-layer (avgpool, 512-d) representations."""
    model.eval()
    model.to(device)

    representations, labels_list = [], []
    total = 0
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output["feat"] = output

    handle = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            feat = hook_output["feat"].squeeze(-1).squeeze(-1)
            representations.append(feat.cpu().numpy())
            labels_list.append(targets.numpy())
            total += inputs.size(0)
            if max_samples and total >= max_samples:
                break

    handle.remove()

    X = np.concatenate(representations, axis=0)
    y = np.concatenate(labels_list, axis=0)
    if max_samples and X.shape[0] > max_samples:
        X, y = X[:max_samples], y[:max_samples]
    return X, y


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def linear_CKA(X, Y):
    """Linear Centered Kernel Alignment."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def checkpoint_path(ckpt_dir, dataset_name, rho, seed):
    """Standard checkpoint filename for a (dataset, rho, seed) run."""
    return Path(ckpt_dir) / f"resnet18_{dataset_name}_rho{rho:.3f}_seed{seed}.pt"


def resolve_checkpoint(stored_path, ckpt_dir):
    """Find a saved model checkpoint on disk (cwd or Drive ckpt_dir)."""
    candidates = []
    if stored_path:
        candidates.append(Path(stored_path))
    if stored_path:
        candidates.append(Path(ckpt_dir) / Path(stored_path).name)
    for path in candidates:
        if path.exists():
            return path
    return None


def load_features_from_checkpoint(ckpt_path, num_classes, test_loader, device,
                                  max_samples):
    """Reload a trained model and extract penultimate features."""
    model = make_resnet18(num_classes)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return extract_penultimate(model, test_loader, device, max_samples=max_samples)


def ensure_features_for_cka(res, rho, dataset_name, num_classes,
                            train_loader, test_loader, device, max_samples,
                            ckpt_dir, epochs, lr, momentum, weight_decay):
    """
    Return penultimate features for a saved run. If the checkpoint is missing
    (e.g. from an old run before Drive checkpointing), retrain only that
    dataset/rho/seed checkpoint, save it, and preserve existing metrics.
    """
    seed = res["seed"]
    ckpt = resolve_checkpoint(res.get("checkpoint"), ckpt_dir)
    if ckpt is not None:
        res["checkpoint"] = str(ckpt)
        return load_features_from_checkpoint(
            ckpt, num_classes, test_loader, device, max_samples)

    print(f"    [Recover] Missing checkpoint for rho={rho}, seed={seed}; "
          "retraining weights for CKA only")
    seed_everything(seed)
    model = make_resnet18(num_classes)
    t0 = time.time()
    model = train_model(
        model, train_loader, epochs, lr, momentum, weight_decay, rho, device)
    res["cka_retrain_time_s"] = time.time() - t0

    ckpt = checkpoint_path(ckpt_dir, dataset_name, rho, seed)
    torch.save(model.state_dict(), ckpt)
    res["checkpoint"] = str(ckpt)
    res["checkpoint_retrained_for_cka"] = True

    return extract_penultimate(
        model, test_loader, device, max_samples=max_samples)


def cka_is_missing(res):
    val = res.get("cka_vs_sgd")
    if val is None:
        return True
    try:
        return not np.isfinite(float(val))
    except (TypeError, ValueError):
        return True


def save_progress(progress_file, rho_seed_results, rhos):
    """Write per-rho/per-seed results (including CKA) to the progress JSON."""
    progress_data = {str(r): rho_seed_results[r] for r in rhos if rho_seed_results[r]}
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)


def save_final_json(out_path, config, all_dataset_results):
    output = {"config": config, "results": all_dataset_results}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


def export_dataset_csvs(dataset_name, rho_seed_results, rhos, aggregated, out_dir,
                        prefix):
    """Export raw per-seed rows and aggregated-by-rho tables (with CKA)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{prefix}_{dataset_name}_raw.csv"
    agg_path = out_dir / f"{prefix}_{dataset_name}_aggregated.csv"
    cka_path = out_dir / f"{prefix}_{dataset_name}_cka.csv"

    raw_rows = []
    for rho in rhos:
        for res in rho_seed_results.get(rho, []):
            row = {
                "dataset": dataset_name,
                "rho": rho,
                "seed": res.get("seed"),
                "label": res.get("label"),
                "accuracy": res.get("accuracy"),
                "cka_vs_sgd": res.get("cka_vs_sgd"),
                "training_time_s": res.get("training_time_s"),
                "checkpoint": res.get("checkpoint"),
            }
            for key, val in (res.get("shesha") or {}).items():
                row[f"shesha_{key}"] = val
            raw_rows.append(row)

    if raw_rows:
        fieldnames = list(raw_rows[0].keys())
        with open(raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_rows)

    if aggregated:
        agg_rows = []
        for agg in aggregated:
            row = {
                "dataset": dataset_name,
                "rho": agg["rho"],
                "n_seeds": len(rho_seed_results.get(agg["rho"], [])),
                "accuracy_mean": agg["accuracy_mean"],
                "accuracy_std": agg["accuracy_std"],
                "cka_mean": agg["cka_mean"],
                "cka_std": agg["cka_std"],
            }
            for key, val in agg.get("shesha_mean", {}).items():
                row[f"shesha_{key}_mean"] = val
            for key, val in agg.get("shesha_std", {}).items():
                row[f"shesha_{key}_std"] = val
            agg_rows.append(row)

        fieldnames = list(agg_rows[0].keys())
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(agg_rows)

        cka_rows = [
            {
                "dataset": dataset_name,
                "rho": agg["rho"],
                "n_seeds": len(rho_seed_results.get(agg["rho"], [])),
                "cka_mean": agg["cka_mean"],
                "cka_std": agg["cka_std"],
            }
            for agg in aggregated
        ]
        with open(cka_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["dataset", "rho", "n_seeds", "cka_mean", "cka_std"])
            writer.writeheader()
            writer.writerows(cka_rows)

        print(f"  CSV saved: {raw_path}, {agg_path}, {cka_path}")
    elif raw_rows:
        print(f"  CSV saved: {raw_path}")


def compute_cka_vs_sgd_baseline(rho_seed_results, rhos, dataset_name, num_classes,
                                train_loader, test_loader, device, max_samples,
                                ckpt_dir, epochs, lr, momentum, weight_decay,
                                progress_file=None):
    """
    For each (rho, seed), compute linear CKA between penultimate features and the
    SGD baseline (rho=0) for the same seed. Saves progress after each value.
    """
    sgd_rho = 0.0 if 0.0 in rhos else rhos[0]
    sgd_by_seed = {}

    print(f"  Baseline: rho={sgd_rho} (SGD)")
    for res in rho_seed_results[sgd_rho]:
        seed = res["seed"]
        if not cka_is_missing(res):
            res["cka_vs_sgd"] = 1.0
            continue
        print(f"    Loading SGD baseline, seed {seed}...")
        X_sgd, _ = ensure_features_for_cka(
            res, sgd_rho, dataset_name, num_classes,
            train_loader, test_loader, device, max_samples, ckpt_dir,
            epochs, lr, momentum, weight_decay)
        sgd_by_seed[seed] = X_sgd
        res["cka_vs_sgd"] = 1.0
        if progress_file is not None:
            save_progress(progress_file, rho_seed_results, rhos)

    for rho in rhos:
        for res in rho_seed_results[rho]:
            seed = res["seed"]
            if rho == sgd_rho:
                if cka_is_missing(res):
                    res["cka_vs_sgd"] = 1.0
                    if progress_file is not None:
                        save_progress(progress_file, rho_seed_results, rhos)
                continue

            if not cka_is_missing(res):
                continue

            if seed not in sgd_by_seed:
                sgd_res = next(
                    (r for r in rho_seed_results[sgd_rho] if r["seed"] == seed), None)
                if sgd_res is None:
                    print(f"    [WARN] No SGD baseline row for seed {seed}; CKA=NaN")
                    res["cka_vs_sgd"] = float("nan")
                    if progress_file is not None:
                        save_progress(progress_file, rho_seed_results, rhos)
                    continue
                print(f"    Loading SGD baseline, seed {seed}...")
                X_sgd, _ = ensure_features_for_cka(
                    sgd_res, sgd_rho, dataset_name, num_classes,
                    train_loader, test_loader, device, max_samples, ckpt_dir,
                    epochs, lr, momentum, weight_decay)
                sgd_by_seed[seed] = X_sgd
                sgd_res["cka_vs_sgd"] = 1.0

            X_rho, _ = ensure_features_for_cka(
                res, rho, dataset_name, num_classes,
                train_loader, test_loader, device, max_samples, ckpt_dir,
                epochs, lr, momentum, weight_decay)
            res["cka_vs_sgd"] = linear_CKA(X_rho, sgd_by_seed[seed])
            print(f"    rho={rho:.3f} seed={seed}: CKA={res['cka_vs_sgd']:.4f}")
            if progress_file is not None:
                save_progress(progress_file, rho_seed_results, rhos)

    return sgd_rho


def evaluate_accuracy(model, dataloader, device):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            correct += (model(inputs).argmax(1) == targets).sum().item()
            total   += targets.size(0)
    return 100.0 * correct / total


def compute_shesha_metrics(X, y, n_splits, seed):
    """Compute key Shesha variants."""
    return {
        "feature_split":        shesha.feature_split(X, n_splits=n_splits, seed=seed),
        "sample_split":         shesha.sample_split(X, n_splits=n_splits, seed=seed),
        "anchor":               shesha.anchor_stability(X, n_splits=n_splits, seed=seed),
        "variance_ratio":       shesha.variance_ratio(X, y),
        "supervised_alignment": shesha.supervised_alignment(X, y, seed=seed),
        "class_separation":     shesha.class_separation_ratio(X, y, n_bootstrap=n_splits, seed=seed),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_seed_results(seed_results):
    """
    Given a list of per-seed result dicts (each having 'accuracy', 'shesha', etc.),
    return mean and std dicts.
    """
    accs   = np.array([r["accuracy"] for r in seed_results])
    ckas   = np.array([
        np.nan if r.get("cka_vs_sgd", np.nan) is None else r.get("cka_vs_sgd", np.nan)
        for r in seed_results
    ], dtype=float)
    shesha_keys = list(seed_results[0]["shesha"].keys())
    cka_finite = ckas[np.isfinite(ckas)]

    agg = {
        "accuracy_mean": float(accs.mean()),
        "accuracy_std":  float(accs.std()),
        "cka_mean":      float(cka_finite.mean()) if len(cka_finite) else float("nan"),
        "cka_std":       float(cka_finite.std()) if len(cka_finite) else float("nan"),
        "shesha_mean":   {},
        "shesha_std":    {},
    }
    for k in shesha_keys:
        vals = np.array([r["shesha"][k] for r in seed_results])
        agg["shesha_mean"][k] = float(vals.mean())
        agg["shesha_std"][k]  = float(vals.std())
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(rhos, aggregated, dataset_name, plot_path):
    """
    Two-panel figure (mean ± std across seeds):
      Left:  Shesha variants vs SAM rho
      Right: Test accuracy vs SAM rho
    """
    shesha_keys = list(aggregated[0]["shesha_mean"].keys())
    accs_mean = np.array([a["accuracy_mean"] for a in aggregated])
    accs_std  = np.array([a["accuracy_std"]  for a in aggregated])

    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    cmap    = plt.cm.Set2
    colors  = {k: cmap(i) for i, k in enumerate(shesha_keys)}
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    rhos_arr = np.array(rhos)

    # --- Left: Shesha vs rho (mean ± std) ---
    for idx, key in enumerate(shesha_keys):
        vals_mean = np.array([a["shesha_mean"][key] for a in aggregated])
        vals_std  = np.array([a["shesha_std"][key]  for a in aggregated])
        ax1.plot(rhos_arr, vals_mean,
                 marker=markers[idx % len(markers)],
                 label=key.replace("_", " "),
                 color=colors[key],
                 markersize=6, linewidth=2, alpha=0.9)
        ax1.fill_between(rhos_arr,
                         vals_mean - vals_std,
                         vals_mean + vals_std,
                         alpha=0.12, color=colors[key])

    ax1.set_xlabel(r"SAM $\rho$ (0 = SGD)", fontsize=13)
    ax1.set_ylabel("Shesha Score", fontsize=13)
    ax1.set_title(
        f"[{dataset_name.upper()}] Geometric Stability vs SAM $\\rho$"
        f"\n(mean ± std, {len(SEEDS)} seeds)",
        fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="best", framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.set_xlim(rhos_arr[0] - 0.005, rhos_arr[-1] * 1.02 + 0.005)

    # --- Right: Accuracy vs rho (mean ± std) ---
    ax2.plot(rhos_arr, accs_mean, "s-", color="#2c3e50", markersize=7, linewidth=2.5)
    ax2.fill_between(rhos_arr,
                     accs_mean - accs_std,
                     accs_mean + accs_std,
                     alpha=0.15, color="#2c3e50", label="±1 std")

    ax2.set_xlabel(r"SAM $\rho$ (0 = SGD)", fontsize=13)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax2.set_title(
        f"[{dataset_name.upper()}] Test Accuracy vs SAM $\\rho$"
        f"\n(mean ± std, {len(SEEDS)} seeds)",
        fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.set_xlim(rhos_arr[0] - 0.005, rhos_arr[-1] * 1.02 + 0.005)

    fig.tight_layout(pad=2.0)

    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    png_path = str(plot_path).replace(".pdf", ".png")
    if png_path != str(plot_path):
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"  Plot saved: {plot_path} and {png_path}")
    else:
        print(f"  Plot saved: {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_output_dir(output_path):
    """Create output directory for checkpoints and exported results."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Checkpoints and exports: {out_dir.resolve()}")
    return out_dir


def main():
    args = parse_args()

    rhos  = args.rhos  if args.rhos  else DEFAULT_RHOS
    seeds = args.seeds if args.seeds else SEEDS

    if args.quick:
        args.epochs          = 5
        args.n_shesha_splits = 10
        args.max_eval_samples = 500
        rhos  = [0.0, 0.1, 0.5]
        seeds = seeds[:2]

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Device   : {device}")
    print(f"Epochs   : {args.epochs}")
    print(f"Datasets : {args.datasets}")
    print(f"SAM rhos : {rhos}")
    print(f"Seeds    : {seeds}")
    print(f"Shesha splits: {args.n_shesha_splits}")
    print()

    all_dataset_results = {}

    ckpt_dir = _setup_output_dir(args.checkpoint_dir)

    for dataset_name in args.datasets:
        num_classes = DATASET_STATS[dataset_name]["num_classes"]
        print("=" * 70)
        print(f"DATASET: {dataset_name.upper()}  ({num_classes} classes)")
        print("=" * 70)

        train_loader, test_loader = get_dataloaders(
            args.batch_size, dataset_name, quick=args.quick)

        # rho -> list of per-seed result dicts
        rho_seed_results = {rho: [] for rho in rhos}

        # --- Resume: load any previously saved per-rho checkpoints ---
        progress_file = ckpt_dir / f"_progress_{dataset_name}.json"
        if progress_file.exists():
            with open(progress_file) as f:
                saved_progress = json.load(f)
            for rho_key, seed_list in saved_progress.items():
                rho_val = float(rho_key)
                if rho_val in rho_seed_results:
                    rho_seed_results[rho_val] = seed_list
            n_done = sum(1 for r in rhos if len(rho_seed_results[r]) == len(seeds))
            print(f"  [Resume] Loaded progress: {n_done}/{len(rhos)} rhos complete")

        for i, rho in enumerate(rhos):
            # Skip if all seeds already done for this rho
            if len(rho_seed_results[rho]) >= len(seeds):
                print(f"\n[{dataset_name}] [{i+1}/{len(rhos)}] rho={rho} — already done, skipping")
                continue

            label = "SGD" if rho == 0 else f"SAM(rho={rho})"
            print(f"\n[{dataset_name}] [{i+1}/{len(rhos)}] {label}")
            print("-" * 50)

            # Determine which seeds are already done for this rho
            done_seeds = {r["seed"] for r in rho_seed_results[rho]}

            for s_idx, seed in enumerate(seeds):
                if seed in done_seeds:
                    print(f"  Seed {seed}  ({s_idx+1}/{len(seeds)}) — cached, skipping")
                    continue

                print(f"  Seed {seed}  ({s_idx+1}/{len(seeds)})")
                seed_everything(seed)
                model = make_resnet18(num_classes)

                t0    = time.time()
                model = train_model(
                    model, train_loader, args.epochs,
                    args.lr, args.momentum, args.weight_decay,
                    rho, device)
                train_time = time.time() - t0

                acc = evaluate_accuracy(model, test_loader, device)
                print(f"    acc={acc:.2f}%  time={train_time:.0f}s")

                X, y = extract_penultimate(
                    model, test_loader, device,
                    max_samples=args.max_eval_samples)

                shesha_scores = compute_shesha_metrics(
                    X, y, args.n_shesha_splits, seed=seed)

                ckpt = checkpoint_path(ckpt_dir, dataset_name, rho, seed)
                torch.save(model.state_dict(), ckpt)

                rho_seed_results[rho].append({
                    "seed":            seed,
                    "rho":             rho,
                    "label":           label,
                    "accuracy":        acc,
                    "shesha":          shesha_scores,
                    "training_time_s": train_time,
                    "checkpoint":      str(ckpt),
                })

            # --- Incremental save after each rho completes ---
            save_progress(progress_file, rho_seed_results, rhos)
            print(f"  [Checkpoint] Progress saved ({i+1}/{len(rhos)} rhos)")

        # --- CKA pass: reload checkpoints, compare each rho to SGD (rho=0) ---
        need_cka = any(
            cka_is_missing(res)
            for rho in rhos for res in rho_seed_results[rho]
        )
        if need_cka:
            print(f"\n[{dataset_name}] Computing CKA vs SGD baseline...")
            compute_cka_vs_sgd_baseline(
                rho_seed_results, rhos, dataset_name, num_classes,
                train_loader, test_loader, device, args.max_eval_samples,
                ckpt_dir, args.epochs, args.lr, args.momentum,
                args.weight_decay, progress_file=progress_file)
            save_progress(progress_file, rho_seed_results, rhos)
            print(f"  [Checkpoint] CKA results saved to {progress_file}")
        else:
            print(f"\n[{dataset_name}] CKA vs SGD already computed — skipping reload")

        # --- Aggregate across seeds per rho ---
        aggregated = []
        for rho in rhos:
            agg = aggregate_seed_results(rho_seed_results[rho])
            agg["rho"] = rho
            aggregated.append(agg)

        # --- Summary table ---
        print(f"\n{'='*100}")
        print(f"SUMMARY: {dataset_name.upper()}")
        print(f"{'='*100}")
        shesha_keys = list(aggregated[0]["shesha_mean"].keys())
        col = 10
        header = (f"{'rho':>6s} | {'Acc mean':>{col}s} | {'Acc std':>{col}s} | "
                  f"{'CKA mean':>{col}s} | {'CKA std':>{col}s}")
        for k in shesha_keys:
            header += f" | {k[:10]:>{col}s}"
        print(header)
        print("-" * len(header))
        for agg in aggregated:
            row = (f"{agg['rho']:>6.3f} | "
                   f"{agg['accuracy_mean']:>{col}.2f} | "
                   f"{agg['accuracy_std']:>{col}.2f} | "
                   f"{agg['cka_mean']:>{col}.4f} | "
                   f"{agg['cka_std']:>{col}.4f}")
            for k in shesha_keys:
                row += f" | {agg['shesha_mean'][k]:>{col}.4f}"
            print(row)
        print()

        # --- Plot ---
        plot_path = f"{args.plot_prefix}_{dataset_name}.pdf"
        make_plot(rhos, aggregated, dataset_name, plot_path)

        all_dataset_results[dataset_name] = {
            "rho_seed_results": {
                str(rho): rho_seed_results[rho] for rho in rhos
            },
            "aggregated": aggregated,
        }

        export_dataset_csvs(
            dataset_name, rho_seed_results, rhos, aggregated,
            ckpt_dir, args.plot_prefix)

        run_config = {
            "seeds":             seeds,
            "epochs":            args.epochs,
            "lr":                args.lr,
            "momentum":          args.momentum,
            "weight_decay":      args.weight_decay,
            "rhos":              rhos,
            "batch_size":        args.batch_size,
            "datasets":          args.datasets,
            "n_shesha_splits":   args.n_shesha_splits,
            "max_eval_samples":  args.max_eval_samples,
            "device":            str(device),
            "quick":             args.quick,
        }
        save_final_json(Path(args.output), run_config, all_dataset_results)
        save_final_json(ckpt_dir / Path(args.output).name, run_config, all_dataset_results)
        print(f"  Results JSON saved ({dataset_name} complete)")

    # --- Save JSON ---
    run_config = {
        "seeds":             seeds,
        "epochs":            args.epochs,
        "lr":                args.lr,
        "momentum":          args.momentum,
        "weight_decay":      args.weight_decay,
        "rhos":              rhos,
        "batch_size":        args.batch_size,
        "datasets":          args.datasets,
        "n_shesha_splits":   args.n_shesha_splits,
        "max_eval_samples":  args.max_eval_samples,
        "device":            str(device),
        "quick":             args.quick,
    }
    out_path = Path(args.output)
    save_final_json(out_path, run_config, all_dataset_results)
    save_final_json(ckpt_dir / out_path.name, run_config, all_dataset_results)
    print(f"Results saved to {out_path}")
    print(f"Checkpoint copy saved to {ckpt_dir / out_path.name}")


if __name__ == "__main__":
    main()
