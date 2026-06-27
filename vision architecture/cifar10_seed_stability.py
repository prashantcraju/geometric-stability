"""
cifar10_seed_stability.py
==========================
Analyse seed-to-seed variability for the CIFAR-10 clean benchmark
(3 seeds, typically 320 / 321 / 322).

For every model with data in at least 2 seeds the script computes:
  • Per-seed raw values of LogME, Shesha-FS, Shesha-Var, LEEP_Real
  • Mean, SD, and coefficient of variation (CV = SD/|mean|) per metric
  • Kendall rank per seed and the max rank-shift across seeds
  • Flags models whose FS or LogME CV exceeds a threshold (default 5 %)

Outputs
-------
  cifar10_seed_stability_output.txt     — full console log
  cifar10_seed_per_model.csv            — per-model summary table
  cifar10_seed_high_variance.csv        — models with CV > threshold
  cifar10_seed_rank_changes.csv         — models sorted by max rank shift

Usage
-----
  python cifar10_seed_stability.py \\
      --clean-dir ./shesha-vision_architecture \\
      --out-dir   ./seed_stability_out \\
      --cv-thresh 0.05
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEAN_DIR  = "./shesha-vision_architecture"
OUT_DIR    = "./seed_stability_out"
DATASET    = "cifar10"
CV_THRESH  = 0.05   # flag if CV exceeds this for FS or LogME

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, friedmanchisquare

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# TEE
# =============================================================================

class _Tee:
    def __init__(self, fh, orig):
        self._f, self._o = fh, orig
    def write(self, m):
        self._o.write(m); self._f.write(m)
    def flush(self):
        self._o.flush(); self._f.flush()
    def isatty(self):
        return False

# =============================================================================
# FAMILY
# =============================================================================

def get_family(m: str) -> str:
    m = m.lower()
    if 'dinov3' in m:                       return 'DINOv3'
    if 'dinov2' in m:                       return 'DINOv2'
    if '.dino' in m or '_dino' in m:        return 'DINO'
    if 'siglip' in m:                       return 'SigLIP'
    if 'clip' in m:                         return 'CLIP'
    if 'eva02' in m or 'eva_' in m:         return 'EVA'
    if 'beit3' in m:                        return 'BEiT3'
    if 'beitv2' in m:                       return 'BEiTv2'
    if 'beit' in m:                         return 'BEiT'
    if 'hiera' in m:                        return 'Hiera'
    if '.mae' in m:                         return 'MAE'
    if 'samvit' in m or '.sam_' in m:       return 'SAM'
    if 'ijepa' in m:                        return 'I-JEPA'
    if 'vitamin' in m:                      return 'ViTamin'
    if 'resmlp' in m and 'dino' in m:       return 'DINO'
    if 'swinv2' in m:                       return 'SwinV2'
    if 'swin' in m:                         return 'Swin'
    if 'pvt_v2' in m:                       return 'PVTv2'
    if 'poolformer' in m:                   return 'PoolFormer'
    if 'deit3' in m:                        return 'DeiT3'
    if 'deit' in m:                         return 'DeiT'
    if 'maxvit' in m:                       return 'MaxViT'
    if 'coatnet' in m:                      return 'CoAtNet'
    if 'convnextv2' in m:                   return 'ConvNeXtV2'
    if 'convnext' in m:                     return 'ConvNeXt'
    if 'efficientnetv2' in m:               return 'EfficientNetV2'
    if 'efficientnet' in m:                 return 'EfficientNet'
    if 'regnety' in m:                      return 'RegNetY'
    if 'regnetx' in m:                      return 'RegNetX'
    if 'resnext' in m:                      return 'ResNeXt'
    if 'resnetrs' in m:                     return 'ResNetRS'
    if 'resnetv2' in m:                     return 'ResNetV2'
    if 'wide_resnet' in m:                  return 'WideResNet'
    if 'resnet' in m:                       return 'ResNet'
    if 'densenet' in m:                     return 'DenseNet'
    if 'mobilenetv3' in m:                  return 'MobileNetV3'
    if 'inception' in m:                    return 'Inception'
    if 'vit_' in m:                         return 'ViT'
    return 'Other'

# =============================================================================
# DATA LOADING
# =============================================================================

METRICS = ['LogME', 'SHESHA_FS', 'SHESHA_Var', 'LEEP_Real']

def load_per_seed(dataset_name: str, clean_dir: Path) -> pd.DataFrame | None:
    """
    Return a DataFrame with one row per (Model, Seed) containing raw metric
    values.  Prioritises individual SEED_<n> CSVs; falls back to ALL_SEEDS.
    OOM rows are kept with NaN metrics.
    """
    ds_upper = dataset_name.upper()
    d = Path(clean_dir)
    if not d.exists():
        print(f'[ERROR] clean_dir not found: {clean_dir}')
        return None

    EXCLUDE = ['SPECTRAL', 'PROBE_SENSITIVITY', 'AVERAGED']

    # Prefer per-seed files (CIFAR10_SEED320_*.csv)
    seed_files = sorted(
        [f for f in d.glob('*.csv')
         if re.search(rf'^{ds_upper}_SEED\d+_', f.name, re.IGNORECASE)
         and not any(p in f.name.upper() for p in EXCLUDE)],
        key=lambda f: f.stat().st_mtime,
    )

    # Fall back to ALL_SEEDS aggregate if no individual files
    if not seed_files:
        all_seeds = sorted(
            [f for f in d.glob('*.csv')
             if re.search(rf'^{ds_upper}_ALL_SEEDS_', f.name, re.IGNORECASE)],
            key=lambda f: f.stat().st_mtime,
        )
        if all_seeds:
            seed_files = all_seeds
            print(f'  No per-seed files found; using ALL_SEEDS: '
                  f'{[f.name for f in all_seeds]}')

    if not seed_files:
        print(f'[ERROR] No seed CSV files for {dataset_name} in {clean_dir}')
        return None

    print(f'  Found {len(seed_files)} seed file(s):')
    dfs = []
    for f in seed_files:
        try:
            tmp = pd.read_csv(f, on_bad_lines='skip')
            if 'Model' not in tmp.columns:
                print(f'    [skip] {f.name}: no Model column')
                continue
            # Try to extract seed from filename if not in data
            if 'Seed' not in tmp.columns:
                m_seed = re.search(r'SEED(\d+)', f.name, re.IGNORECASE)
                if m_seed:
                    tmp['Seed'] = int(m_seed.group(1))
            print(f'    {f.name}: {len(tmp)} rows')
            dfs.append(tmp)
        except Exception as e:
            print(f'    [skip] {f.name}: {e}')

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Coerce metrics
    for c in METRICS + ['Dim']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # NaN-out OOM rows
    if 'LEEP_Status' in df.columns:
        oom = df['LEEP_Status'] == 'oom_skip'
        n_oom = oom.sum()
        for c in METRICS:
            if c in df.columns:
                df.loc[oom, c] = np.nan
        if n_oom:
            print(f'  {n_oom} oom_skip rows → NaN metrics')

    # Deduplicate (Model, Seed) — keep last (most recent run wins)
    if 'Seed' in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=['Model', 'Seed'], keep='last')
        if len(df) < n_before:
            print(f'  Deduped {n_before - len(df)} duplicate (Model, Seed) rows')
    else:
        print('  [WARN] No Seed column; treating all rows as single seed')
        df['Seed'] = 0

    df['Family'] = df['Model'].map(get_family)
    seeds = sorted(df['Seed'].unique())
    print(f'  Seeds found: {seeds}')
    print(f'  Models: {df["Model"].nunique()}  total rows: {len(df)}')
    return df

# =============================================================================
# ANALYSIS
# =============================================================================

def _cv(vals: np.ndarray) -> float:
    """Coefficient of variation = SD / |mean|, or NaN if fewer than 2 points."""
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2 or abs(np.mean(vals)) < 1e-12:
        return np.nan
    return float(np.std(vals, ddof=1) / abs(np.mean(vals)))


def build_per_model_table(df: pd.DataFrame, seeds: list) -> pd.DataFrame:
    """
    For each model, compute per-seed values and cross-seed statistics.
    Returns a wide DataFrame indexed by Model.
    """
    available_metrics = [c for c in METRICS if c in df.columns]
    rows = []

    for model, grp in df.groupby('Model'):
        row = {'Model': model, 'Family': grp['Family'].iloc[0]}
        row['N_Seeds'] = grp['Seed'].nunique()

        for metric in available_metrics:
            vals_by_seed = {}
            for seed in seeds:
                sv = grp.loc[grp['Seed'] == seed, metric]
                row[f'{metric}_S{seed}'] = sv.values[0] if len(sv) == 1 else np.nan
                vals_by_seed[seed] = row[f'{metric}_S{seed}']

            arr = np.array(list(vals_by_seed.values()), dtype=float)
            finite = arr[np.isfinite(arr)]
            row[f'{metric}_mean'] = np.mean(finite) if len(finite) else np.nan
            row[f'{metric}_std']  = np.std(finite, ddof=1) if len(finite) >= 2 else np.nan
            row[f'{metric}_cv']   = _cv(arr)

        rows.append(row)

    wide = pd.DataFrame(rows)

    # Add per-seed ranks for LogME and SHESHA_FS
    for metric in ['LogME', 'SHESHA_FS']:
        if metric not in available_metrics:
            continue
        for seed in seeds:
            col = f'{metric}_S{seed}'
            if col in wide.columns:
                # rank 1 = highest value = best; NaN last
                wide[f'{metric}_rank_S{seed}'] = (
                    wide[col].rank(ascending=False, na_option='bottom')
                             .round(0).astype('Int64')
                )

        # Max rank shift across seeds
        rank_cols = [f'{metric}_rank_S{s}' for s in seeds if f'{metric}_rank_S{s}' in wide.columns]
        if len(rank_cols) >= 2:
            rank_mat = wide[rank_cols].astype(float)
            wide[f'{metric}_rank_range'] = rank_mat.max(axis=1) - rank_mat.min(axis=1)

    return wide


def print_overview(wide: pd.DataFrame, seeds: list, cv_thresh: float) -> None:
    available_metrics = [c for c in METRICS if f'{c}_mean' in wide.columns]
    n_models = len(wide)
    n_full   = (wide['N_Seeds'] == len(seeds)).sum()

    print(f'\n  Models total:           {n_models}')
    print(f'  Models with all {len(seeds)} seeds: {n_full}')
    print(f'  Seeds:                  {seeds}')
    print()

    print(f'  {"Metric":<14}  {"Grand mean":>11}  {"Grand SD":>9}  '
          f'{"Median CV":>10}  {"Max CV":>8}  {"N CV>thresh":>12}')
    print('  ' + '-'*72)
    for metric in available_metrics:
        mc = wide[f'{metric}_mean'].dropna()
        cv_col = wide[f'{metric}_cv'].dropna()
        if mc.empty:
            continue
        n_high = (cv_col > cv_thresh).sum()
        print(f'  {metric:<14}  {mc.mean():>11.4f}  {mc.std():>9.4f}  '
              f'{cv_col.median():>10.4f}  {cv_col.max():>8.4f}  '
              f'{n_high:>8} ({n_high/len(wide)*100:.1f}%)')


def print_high_cv_models(wide: pd.DataFrame, seeds: list,
                         cv_thresh: float, metric: str = 'SHESHA_FS',
                         top_n: int = 30) -> None:
    cv_col = f'{metric}_cv'
    if cv_col not in wide.columns:
        return
    high = wide.dropna(subset=[cv_col]).nlargest(top_n, cv_col)

    print(f'\n  Top {top_n} models by {metric} CV (threshold = {cv_thresh:.1%}):')
    seed_val_cols = [f'{metric}_S{s}' for s in seeds if f'{metric}_S{s}' in wide.columns]
    rank_cols     = [f'{metric}_rank_S{s}' for s in seeds
                     if f'{metric}_rank_S{s}' in wide.columns]

    header = (f'  {"Model":<50}  {"Family":<12}  {"Mean":>7}  {"SD":>7}  {"CV":>7}'
              + ''.join(f'  S{s:>4}' for s in seeds)
              + ''.join(f'  R{s:>4}' for s in seeds))
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for _, row in high.iterrows():
        vals   = ''.join(f'  {row[c]:>6.3f}' if pd.notna(row[c]) else '     —'
                         for c in seed_val_cols)
        ranks  = ''.join(f'  {int(row[c]):>5}' if pd.notna(row[c]) else '     —'
                         for c in rank_cols)
        flag = ' !' if row[cv_col] > cv_thresh else '  '
        print(f'  {flag}{row["Model"]:<50}  {row["Family"]:<12}  '
              f'{row[f"{metric}_mean"]:>7.4f}  '
              f'{row[f"{metric}_std"]:>7.4f}  '
              f'{row[cv_col]:>7.4f}'
              + vals + ranks)


def print_rank_changes(wide: pd.DataFrame, seeds: list,
                       metric: str = 'SHESHA_FS', top_n: int = 30) -> None:
    rc_col = f'{metric}_rank_range'
    if rc_col not in wide.columns:
        return
    top = wide.dropna(subset=[rc_col]).nlargest(top_n, rc_col)

    print(f'\n  Top {top_n} models by {metric} rank shift (max − min rank across seeds):')
    rank_cols = [f'{metric}_rank_S{s}' for s in seeds
                 if f'{metric}_rank_S{s}' in wide.columns]
    val_cols  = [f'{metric}_S{s}' for s in seeds
                 if f'{metric}_S{s}' in wide.columns]

    print(f'  {"Model":<50}  {"Family":<12}  {"Range":>6}'
          + ''.join(f'  R{s:>4}' for s in seeds)
          + ''.join(f'  V{s:>5}' for s in seeds))
    print('  ' + '-'*100)

    for _, row in top.iterrows():
        ranks = ''.join(f'  {int(row[c]):>5}' if pd.notna(row[c]) else '     —'
                        for c in rank_cols)
        vals  = ''.join(f'  {row[c]:>6.3f}' if pd.notna(row[c]) else '      —'
                        for c in val_cols)
        print(f'  {row["Model"]:<50}  {row["Family"]:<12}  '
              f'{int(row[rc_col]):>6}' + ranks + vals)


def print_family_stability(wide: pd.DataFrame, metric: str = 'SHESHA_FS') -> None:
    cv_col = f'{metric}_cv'
    if cv_col not in wide.columns:
        return
    fam_stats = (wide.dropna(subset=[cv_col, f'{metric}_mean'])
                     .groupby('Family')
                     .agg(n=('Model', 'count'),
                          mean_val=(f'{metric}_mean', 'mean'),
                          mean_cv=(cv_col, 'mean'),
                          max_cv=(cv_col, 'max'))
                     .sort_values('mean_cv', ascending=False)
                     .reset_index())

    print(f'\n  Family-level {metric} stability (sorted by mean CV):')
    print(f'  {"Family":<18}  {"n":>4}  {"Mean FS":>8}  {"Mean CV":>8}  {"Max CV":>8}')
    print('  ' + '-'*54)
    for _, r in fam_stats.iterrows():
        print(f'  {r["Family"]:<18}  {r["n"]:>4}  {r["mean_val"]:>8.4f}  '
              f'{r["mean_cv"]:>8.4f}  {r["max_cv"]:>8.4f}')


def friedman_test(df: pd.DataFrame, seeds: list, metric: str = 'SHESHA_FS') -> None:
    """Friedman test: are seed distributions significantly different?"""
    cols = [f'{metric}_S{s}' for s in seeds]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 3:
        print(f'\n  Friedman test skipped (need ≥3 seeds; have {len(cols)})')
        return

    sub = df[cols].dropna()
    if len(sub) < 5:
        print(f'\n  Friedman test skipped (only {len(sub)} complete rows)')
        return

    arrays = [sub[c].values for c in cols]
    stat, p = friedmanchisquare(*arrays)
    print(f'\n  Friedman test ({metric}, {len(sub)} models with all seeds):')
    print(f'    χ² = {stat:.4f},  p = {p:.6f}')
    if p < 0.05:
        print('    → Seeds produce significantly different distributions (p < 0.05)')
    else:
        print('    → No significant difference across seeds (results are stable)')


def kendall_cross_seed(wide: pd.DataFrame, seeds: list, metric: str = 'SHESHA_FS') -> None:
    """Kendall τ between every pair of seeds on their rank orderings."""
    rank_cols = {s: f'{metric}_rank_S{s}' for s in seeds
                 if f'{metric}_rank_S{s}' in wide.columns}
    valid_seeds = list(rank_cols.keys())
    if len(valid_seeds) < 2:
        return

    print(f'\n  Kendall τ between seed rankings ({metric}):')
    print(f'  {"Pair":<18}  {"τ":>7}  {"p":>10}')
    print('  ' + '-'*38)
    for i, s1 in enumerate(valid_seeds):
        for s2 in valid_seeds[i+1:]:
            merged = wide[[rank_cols[s1], rank_cols[s2]]].dropna()
            if len(merged) < 5:
                continue
            tau, p = kendalltau(merged[rank_cols[s1]], merged[rank_cols[s2]])
            print(f'  S{s1} vs S{s2}          {tau:>7.4f}  {p:>10.6f}')

    # Also Spearman on raw values
    val_cols = {s: f'{metric}_S{s}' for s in seeds if f'{metric}_S{s}' in wide.columns}
    print(f'\n  Spearman ρ between seed raw values ({metric}):')
    print(f'  {"Pair":<18}  {"ρ":>7}  {"p":>10}')
    print('  ' + '-'*38)
    for i, s1 in enumerate(list(val_cols.keys())):
        for s2 in list(val_cols.keys())[i+1:]:
            merged = wide[[val_cols[s1], val_cols[s2]]].dropna()
            if len(merged) < 5:
                continue
            rho, p = spearmanr(merged[val_cols[s1]], merged[val_cols[s2]])
            print(f'  S{s1} vs S{s2}          {rho:>7.4f}  {p:>10.6f}')


# =============================================================================
# FIGURES
# =============================================================================

def fig_seed_scatter(wide: pd.DataFrame, seeds: list,
                     metric: str, out_dir: Path) -> None:
    """Pairwise scatter of per-seed values; identity line = perfect agreement."""
    cols = [f'{metric}_S{s}' for s in seeds if f'{metric}_S{s}' in wide.columns]
    valid_seeds = [s for s in seeds if f'{metric}_S{s}' in wide.columns]
    n = len(valid_seeds)
    if n < 2:
        return

    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
    if n == 1:
        axes = [[axes]]

    for i, si in enumerate(valid_seeds):
        for j, sj in enumerate(valid_seeds):
            ax = axes[i][j]
            if i == j:
                vals = wide[f'{metric}_S{si}'].dropna()
                ax.hist(vals, bins=20, color='#4878CF', alpha=0.7, edgecolor='white')
                ax.set_title(f'Seed {si}', fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                x = wide[f'{metric}_S{sj}']
                y = wide[f'{metric}_S{si}']
                merged = pd.DataFrame({'x': x, 'y': y}).dropna()
                ax.scatter(merged['x'], merged['y'], s=7, alpha=0.4,
                           color='#4878CF', linewidths=0)
                lim = [min(merged['x'].min(), merged['y'].min()) * 0.97,
                       max(merged['x'].max(), merged['y'].max()) * 1.03]
                ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.4)
                rho, _ = spearmanr(merged['x'], merged['y'])
                ax.text(0.05, 0.95, f'ρ = {rho:.3f}', transform=ax.transAxes,
                        fontsize=7.5, va='top')
                ax.set_xlabel(f'Seed {sj}', fontsize=8)
                ax.set_ylabel(f'Seed {si}', fontsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = out_dir / f'cifar10_seed_scatter_{metric.lower()}.pdf'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


def fig_cv_distribution(wide: pd.DataFrame, out_dir: Path,
                        cv_thresh: float) -> None:
    """CV distributions for LogME and Shesha-FS side by side."""
    metrics_to_plot = [m for m in ['LogME', 'SHESHA_FS']
                       if f'{m}_cv' in wide.columns]
    if not metrics_to_plot:
        return

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        cv = wide[f'{metric}_cv'].dropna()
        ax.hist(cv, bins=30, color='#4878CF', alpha=0.75, edgecolor='white')
        ax.axvline(cv_thresh, color='#D62728', lw=1.2, ls='--',
                   label=f'CV = {cv_thresh:.0%}')
        n_high = (cv > cv_thresh).sum()
        ax.text(0.97, 0.97,
                f'{n_high}/{len(cv)} models\nabove threshold',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color='#D62728')
        ax.set_xlabel('Coefficient of variation  (SD / |mean|)', fontsize=9)
        ax.set_ylabel('Number of models', fontsize=9)
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = out_dir / 'cifar10_seed_cv_distribution.pdf'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


def fig_rank_change_bars(wide: pd.DataFrame, seeds: list,
                         metric: str, out_dir: Path, top_n: int = 25) -> None:
    """Horizontal bar chart of the top-N models by rank shift."""
    rc_col = f'{metric}_rank_range'
    if rc_col not in wide.columns:
        return

    top = (wide.dropna(subset=[rc_col])
               .nlargest(top_n, rc_col)
               .sort_values(rc_col))

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.32)))
    labels = [m.split('.')[-2] if '.' in m else m for m in top['Model']]
    ax.barh(range(len(top)), top[rc_col], color='#4878CF', alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f'{metric} rank shift (max − min rank across {len(seeds)} seeds)',
                  fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    path = out_dir / f'cifar10_seed_rank_change_{metric.lower()}.pdf'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


def fig_mean_vs_sd(wide: pd.DataFrame, seeds: list,
                   metric: str, out_dir: Path) -> None:
    """Mean vs SD scatter — shows heteroskedasticity if any."""
    mc = f'{metric}_mean'
    sc = f'{metric}_std'
    if mc not in wide.columns or sc not in wide.columns:
        return

    merged = wide[[mc, sc, 'Family']].dropna()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(merged[mc], merged[sc], s=10, alpha=0.5, color='#4878CF',
               linewidths=0)
    ax.set_xlabel(f'Mean {metric} across {len(seeds)} seeds', fontsize=9)
    ax.set_ylabel(f'SD {metric} across {len(seeds)} seeds', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    rho, p = spearmanr(merged[mc], merged[sc])
    ax.text(0.03, 0.97, f'ρ(mean, SD) = {rho:+.3f}  p = {p:.4f}',
            transform=ax.transAxes, va='top', fontsize=8)
    fig.tight_layout()
    path = out_dir / f'cifar10_seed_mean_vs_sd_{metric.lower()}.pdf'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


# =============================================================================
# LATEX TABLE: seed stability summary
# =============================================================================

def latex_seed_summary(wide: pd.DataFrame, seeds: list) -> None:
    """Per-seed median Shesha-FS and LogME across all models + cross-seed ρ."""
    print('\n' + '='*70)
    print('LaTeX: Seed stability summary')
    print('='*70)
    print(r'\begin{table}[t!]')
    print(r'\caption{CIFAR-10 seed stability. '
          r'Median Shesha-FS and LogME per seed across all models, '
          r'and Spearman $\rho$ between seeds on raw values. '
          r'High $\rho$ and low inter-seed variation confirm metric stability.}')
    print(r'\label{tab:seed_stability}')
    print(r'\centering\small')
    print(r'\begin{tabular}{l' + 'r' * len(seeds) + '}')
    print(r'\hline')
    print('Metric & ' + ' & '.join(f'Seed {s}' for s in seeds) + r' \\')
    print(r'\hline')

    for metric in ['LogME', 'SHESHA_FS']:
        cols = [f'{metric}_S{s}' for s in seeds if f'{metric}_S{s}' in wide.columns]
        if not cols:
            continue
        medians = [f'{wide[c].median():.4f}' for c in cols]
        print(f'Median {metric} & ' + ' & '.join(medians) + r' \\')

    print(r'\hline')

    # Cross-seed Spearman
    for metric in ['LogME', 'SHESHA_FS']:
        valid_seeds = [s for s in seeds if f'{metric}_S{s}' in wide.columns]
        if len(valid_seeds) < 2:
            continue
        pairs = [(s1, s2) for i, s1 in enumerate(valid_seeds)
                 for s2 in valid_seeds[i+1:]]
        rho_strs = []
        for s1, s2 in pairs:
            m = wide[[f'{metric}_S{s1}', f'{metric}_S{s2}']].dropna()
            rho, _ = spearmanr(m[f'{metric}_S{s1}'], m[f'{metric}_S{s2}'])
            rho_strs.append(f'S{s1}–S{s2}: {rho:.4f}')
        print(f'Spearman $\\rho$ ({metric}) & \\multicolumn{{{len(seeds)}}}{{c}}'
              f'{{{"; ".join(rho_strs)}}} \\\\')

    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-dir', default=CLEAN_DIR)
    parser.add_argument('--out-dir',   default=OUT_DIR)
    parser.add_argument('--cv-thresh', type=float, default=CV_THRESH,
                        help='Flag models whose CV exceeds this (default 0.05 = 5%%)')
    parser.add_argument('--top-n', type=int, default=25,
                        help='Number of models to show in rank-change tables')
    parser.add_argument('--out-file', default='cifar10_seed_stability_output.txt',
                        help='Filename for saved log (placed in --out-dir)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.out_file

    log_fh = open(log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(log_fh, sys.__stdout__)

    try:
        print('CIFAR-10 Seed Stability Analysis')
        print('='*70)
        print(f'  clean_dir : {args.clean_dir}')
        print(f'  out_dir   : {out_dir.resolve()}')
        print(f'  cv_thresh : {args.cv_thresh:.1%}')
        print()

        # ── Load ──────────────────────────────────────────────────────────────
        df = load_per_seed(DATASET, Path(args.clean_dir))
        if df is None:
            print('[FATAL] Could not load data. Exiting.')
            return

        seeds = sorted(df['Seed'].unique())
        available_metrics = [c for c in METRICS if c in df.columns]
        print(f'\n  Metrics available: {available_metrics}')

        # ── Build per-model table ─────────────────────────────────────────────
        print('\nBuilding per-model summary...')
        wide = build_per_model_table(df, seeds)

        # ── Overview ──────────────────────────────────────────────────────────
        print('\n' + '='*70)
        print('Overview: cross-seed variability')
        print('='*70)
        print_overview(wide, seeds, args.cv_thresh)

        # ── Friedman test ─────────────────────────────────────────────────────
        print('\n' + '='*70)
        print('Friedman test (are seeds different?)')
        print('='*70)
        for metric in ['SHESHA_FS', 'LogME']:
            if metric in available_metrics:
                friedman_test(wide, seeds, metric)

        # ── Kendall / Spearman concordance ────────────────────────────────────
        print('\n' + '='*70)
        print('Rank concordance between seeds')
        print('='*70)
        for metric in ['SHESHA_FS', 'LogME']:
            if metric in available_metrics:
                print(f'\n  — {metric} —')
                kendall_cross_seed(wide, seeds, metric)

        # ── High-CV models ────────────────────────────────────────────────────
        print('\n' + '='*70)
        print(f'High-CV models (CV > {args.cv_thresh:.1%})')
        print('='*70)
        for metric in ['SHESHA_FS', 'LogME']:
            if metric in available_metrics:
                print(f'\n  — {metric} —')
                print_high_cv_models(wide, seeds, args.cv_thresh, metric, args.top_n)

        # ── Rank changes ──────────────────────────────────────────────────────
        print('\n' + '='*70)
        print('Rank changes across seeds')
        print('='*70)
        for metric in ['SHESHA_FS', 'LogME']:
            if metric in available_metrics:
                print(f'\n  — {metric} —')
                print_rank_changes(wide, seeds, metric, args.top_n)

        # ── Family stability ──────────────────────────────────────────────────
        print('\n' + '='*70)
        print('Family-level stability')
        print('='*70)
        for metric in ['SHESHA_FS', 'LogME']:
            if metric in available_metrics:
                print(f'\n  — {metric} —')
                print_family_stability(wide, metric)

        # ── LaTeX ─────────────────────────────────────────────────────────────
        latex_seed_summary(wide, seeds)

        # ── Save CSVs ─────────────────────────────────────────────────────────
        csv_all = out_dir / 'cifar10_seed_per_model.csv'
        wide.to_csv(csv_all, index=False)
        print(f'\n  saved: {csv_all}')

        # High-CV CSV (either metric)
        cv_mask = pd.Series(False, index=wide.index)
        for metric in ['SHESHA_FS', 'LogME']:
            cc = f'{metric}_cv'
            if cc in wide.columns:
                cv_mask = cv_mask | (wide[cc].fillna(0) > args.cv_thresh)
        csv_hv = out_dir / 'cifar10_seed_high_variance.csv'
        wide[cv_mask].to_csv(csv_hv, index=False)
        print(f'  saved: {csv_hv}  ({cv_mask.sum()} models)')

        # Rank-change CSV
        for metric in ['SHESHA_FS', 'LogME']:
            rc_col = f'{metric}_rank_range'
            if rc_col in wide.columns:
                csv_rc = out_dir / f'cifar10_seed_rank_changes_{metric.lower()}.csv'
                (wide.dropna(subset=[rc_col])
                     .sort_values(rc_col, ascending=False)
                     .to_csv(csv_rc, index=False))
                print(f'  saved: {csv_rc}')

        # ── Figures ───────────────────────────────────────────────────────────
        print('\nGenerating figures...')
        for metric in ['SHESHA_FS', 'LogME']:
            if metric in available_metrics:
                fig_seed_scatter(wide, seeds, metric, out_dir)
                fig_rank_change_bars(wide, seeds, metric, out_dir, args.top_n)
                fig_mean_vs_sd(wide, seeds, metric, out_dir)
        fig_cv_distribution(wide, out_dir, args.cv_thresh)

        print('\nDone.')

    finally:
        sys.stdout = sys.__stdout__
        log_fh.close()
        print(f'Log saved to: {log_path.resolve()}')


if __name__ == '__main__':
    main()
