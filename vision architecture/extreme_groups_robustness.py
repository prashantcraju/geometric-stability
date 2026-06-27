"""
extreme_groups_robustness.py
=============================
Pre-specified extreme-groups analysis: do the least geometrically stable
models (bottom decile by clean Shesha-FS) degrade more under corruption
than the most stable models (top decile)?

Design choices that prevent cherry-picking
------------------------------------------
* Cutoff is fixed before looking: TOP_K = BOTTOM_K = 17  (≈ top/bottom decile
  of 170 models).  A secondary window of 15 is also shown.
* Test: Mann-Whitney U (two-sided) on mean ΔLogME within each corruption
  category.  The pre-specified replication bar: result must hold on BOTH
  CIFAR-10-C and CIFAR-100-C at α = 0.05.
* ΔLogME = LogME_clean − LogME_corrupt  (positive = more vulnerable).
* Per-model average within category before comparing groups (consistent with
  compute_robustness_table.py).
* All 19 corruptions are used; categories mirror compute_robustness_table.py.

Outputs
-------
  Console: group means, Mann-Whitney U statistics, verdict per category.
  Figures: scatter of ΔLogME vs clean FS (one per dataset × category),
           box/strip comparison of top vs bottom group.
  LaTeX: table of group means and test statistics.

Usage
-----
    python extreme_groups_robustness.py

    python extreme_groups_robustness.py \\
        --clean-dir  ./shesha-vision_architecture \\
        --corrupt-dir ./shesha-vision_architecture-corrupt \\
        --k 17
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEAN_DIR   = "./shesha-vision_architecture"
CORRUPT_DIR = "./shesha-vision_architecture-corrupt"
FIG_DIR     = "./extreme_groups_figs"

# Pre-specified group size (top-K and bottom-K by clean Shesha-FS)
TOP_K = BOTTOM_K = 17     # ≈ decile of 170 models

# Secondary window (also shown, for sensitivity)
TOP_K2 = BOTTOM_K2 = 15

FS_COL    = "SHESHA_FS"
LOGME_COL = "LogME"
CORR_LOGME_COL = "LogME"

CATEGORIES = {
    "Noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "Extra":   ["speckle_noise", "gaussian_blur", "spatter", "saturate"],
}
ALL19 = sum(CATEGORIES.values(), [])

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import re
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING  (mirrors compute_robustness_table.py)
# =============================================================================

def _load_clean(clean_dir: Path, dataset: str) -> pd.DataFrame | None:
    d = clean_dir
    ds_upper = dataset.upper()
    for pat in [rf'^{ds_upper}_SEED320_', rf'^{ds_upper}_SEED\d+_',
                rf'^{ds_upper}_ALL_SEEDS_', rf'^{ds_upper}_AVERAGED_']:
        cands = sorted(
            [f for f in d.glob('*.csv') if re.search(pat, f.name, re.IGNORECASE)],
            key=lambda f: f.stat().st_mtime, reverse=True)
        if not cands:
            continue
        try:
            df = pd.read_csv(cands[0], on_bad_lines='skip')
        except Exception as e:
            print(f'  [warn] {cands[0].name}: {e}')
            continue
        fs_col  = FS_COL    if FS_COL    in df.columns else ('SHESHA_FS_Mean'  if 'SHESHA_FS_Mean'  in df.columns else None)
        lme_col = LOGME_COL if LOGME_COL in df.columns else ('LogME_Mean'      if 'LogME_Mean'      in df.columns else None)
        if fs_col is None or lme_col is None or 'Model' not in df.columns:
            continue
        df = (df[['Model', fs_col, lme_col]]
              .rename(columns={fs_col: 'fs', lme_col: 'logme_clean'})
              .assign(fs=lambda x: pd.to_numeric(x['fs'], errors='coerce'),
                      logme_clean=lambda x: pd.to_numeric(x['logme_clean'], errors='coerce'))
              .drop_duplicates('Model', keep='last'))
        n_valid = df[['fs','logme_clean']].notna().all(axis=1).sum()
        print(f'  clean {dataset}: {n_valid} models with fs+logme  ← {cands[0].name}')
        return df
    print(f'  [WARN] No clean CSV for {dataset}')
    return None


def _load_corrupt_long(corrupt_dir: Path, corrupt_ds: str) -> pd.DataFrame | None:
    d = corrupt_dir
    ds_upper = corrupt_ds.upper()
    all_files = sorted(
        [f for f in d.glob('*.csv') if re.search(rf'^{ds_upper}_', f.name, re.IGNORECASE)],
        key=lambda f: f.name)

    buckets: dict[str, dict] = {}
    for f in all_files:
        m = re.match(rf'^{ds_upper}_(.+?)_sev\d+_(ALL_SEEDS|SEED\d+|AVERAGED)_',
                     f.name, re.IGNORECASE)
        if not m:
            continue
        corr  = m.group(1).lower()
        ftype = m.group(2).upper()
        b     = buckets.setdefault(corr, {'ALL_SEEDS':[], 'SEED':[], 'AVERAGED':[]})
        key   = 'ALL_SEEDS' if 'ALL_SEEDS' in ftype else ('AVERAGED' if 'AVERAGED' in ftype else 'SEED')
        b[key].append(f)

    dfs = []
    for corr in sorted(buckets):
        b = buckets[corr]
        chosen = None
        for priority in ['ALL_SEEDS','SEED','AVERAGED']:
            cands = sorted(b[priority], key=lambda f: f.stat().st_mtime, reverse=True)
            if cands:
                chosen = cands[0]; break
        if chosen is None:
            continue
        try:
            df = pd.read_csv(chosen, on_bad_lines='skip')
        except Exception:
            continue
        if 'Model' not in df.columns or CORR_LOGME_COL not in df.columns:
            continue
        df[CORR_LOGME_COL] = pd.to_numeric(df[CORR_LOGME_COL], errors='coerce')
        if 'LEEP_Status' in df.columns:
            df = df[df['LEEP_Status'] != 'oom_skip']
        if 'Corruption' not in df.columns:
            df['Corruption'] = corr
        df = (df[['Model','Corruption', CORR_LOGME_COL]]
              .rename(columns={CORR_LOGME_COL: 'logme_corr'})
              .drop_duplicates(subset=['Model','Corruption'], keep='last'))
        dfs.append(df)

    if not dfs:
        print(f'  [WARN] No corrupt data for {corrupt_ds}')
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined['Corruption'] = combined['Corruption'].str.lower()
    combined = combined.drop_duplicates(subset=['Model','Corruption'], keep='last')
    print(f'  corrupt {corrupt_ds}: {combined["Model"].nunique()} models  '
          f'{combined["Corruption"].nunique()} corruptions')
    return combined


def build(clean_dir, corrupt_dir, clean_ds, corrupt_ds):
    clean = _load_clean(clean_dir, clean_ds)
    if clean is None:
        return None, None
    raw = _load_corrupt_long(corrupt_dir, corrupt_ds)
    if raw is None:
        return clean, None
    raw = raw.merge(clean[['Model','logme_clean']], on='Model', how='inner')
    raw['delta'] = raw['logme_clean'] - raw['logme_corr']
    raw = raw.dropna(subset=['logme_clean','logme_corr','delta'])
    return clean, raw


# =============================================================================
# GROUP ASSIGNMENT
# =============================================================================

def assign_groups(clean: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Return a DataFrame with an extra 'group' column:
      'top'    — top-k by clean FS (most stable)
      'bottom' — bottom-k by clean FS (least stable)
      'middle' — everyone else
    Only models with valid fs are included.
    """
    df = clean.dropna(subset=['fs']).copy()
    df = df.sort_values('fs', ascending=False).reset_index(drop=True)
    df['group'] = 'middle'
    df.loc[:k-1, 'group'] = 'top'
    df.loc[len(df)-k:, 'group'] = 'bottom'
    return df


# =============================================================================
# CORE ANALYSIS: per category, per group
# =============================================================================

def analyse_category(clean_groups: pd.DataFrame, corrupt: pd.DataFrame,
                     corruptions: list[str]) -> dict | None:
    """
    Compute per-group mean ΔLogME for a corruption category.
    Returns dict with top/bottom/all group stats and Mann-Whitney result.
    """
    present = [c for c in corruptions if c in corrupt['Corruption'].unique()]
    if not present:
        return None

    sub = corrupt[corrupt['Corruption'].isin(present)]
    mdelta = sub.groupby('Model')['delta'].mean().rename('mdelta')

    df = (clean_groups[['Model','fs','logme_clean','group']]
          .merge(mdelta, on='Model', how='inner')
          .dropna(subset=['fs','logme_clean','mdelta']))

    if len(df) < 10:
        return None

    top_vals    = df.loc[df['group']=='top',    'mdelta'].values
    bottom_vals = df.loc[df['group']=='bottom', 'mdelta'].values
    all_vals    = df['mdelta'].values

    result = {
        'n_corr':       len(present),
        'n_models':     len(df),
        'n_top':        len(top_vals),
        'n_bottom':     len(bottom_vals),
        'mean_top':     float(np.mean(top_vals))    if len(top_vals)    > 0 else np.nan,
        'mean_bottom':  float(np.mean(bottom_vals)) if len(bottom_vals) > 0 else np.nan,
        'mean_all':     float(np.mean(all_vals)),
        'df':           df,
    }

    if len(top_vals) >= 3 and len(bottom_vals) >= 3:
        stat, p = mannwhitneyu(bottom_vals, top_vals, alternative='two-sided')
        result['mw_stat'] = stat
        result['mw_p']    = p
        result['mw_direction'] = (
            'bottom > top' if result['mean_bottom'] > result['mean_top']
            else 'top > bottom')
    else:
        result['mw_stat'] = np.nan
        result['mw_p']    = np.nan
        result['mw_direction'] = '—'

    rho, p_rho = spearmanr(df['fs'], df['mdelta'])
    result['rho']   = rho
    result['p_rho'] = p_rho

    return result


def pstar(p):
    if not np.isfinite(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# =============================================================================
# PLOTTING
# =============================================================================

def _group_color(g):
    return {'top': '#2980B9', 'bottom': '#C0392B', 'middle': '#BDC3C7'}[g]


def fig_scatter(clean_groups, corrupt, corrupt_ds_label, cat_name, corruptions, k):
    """ΔLogME vs clean FS scatter, coloured by group."""
    present = [c for c in corruptions if c in corrupt['Corruption'].unique()]
    if not present:
        return

    sub    = corrupt[corrupt['Corruption'].isin(present)]
    mdelta = sub.groupby('Model')['delta'].mean().rename('mdelta')
    df     = (clean_groups[['Model','fs','group']]
              .merge(mdelta, on='Model', how='inner')
              .dropna())

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for grp in ['middle','top','bottom']:
        g = df[df['group'] == grp]
        ax.scatter(g['fs'], g['mdelta'],
                   c=_group_color(grp), s=28 if grp == 'middle' else 55,
                   alpha=0.55 if grp == 'middle' else 0.9,
                   edgecolors='none' if grp == 'middle' else 'white',
                   linewidths=0.5,
                   label=f'{grp.capitalize()} (N={len(g)})', zorder=3 if grp != 'middle' else 2)

    # Best-fit line
    if len(df) >= 5:
        z = np.polyfit(df['fs'], df['mdelta'], 1)
        xs = np.linspace(df['fs'].min(), df['fs'].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color='#2C3E50', lw=1.2, ls='--', alpha=0.7, zorder=4)

    rho, p_rho = spearmanr(df['fs'], df['mdelta'])
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)

    ax.set_xlabel('Clean Shesha-FS', fontsize=11)
    ax.set_ylabel('Mean ΔLogME (clean − corrupt)', fontsize=11)
    ax.set_title(f'{corrupt_ds_label} / {cat_name}  '
                 f'ρ={rho:+.3f}{pstar(p_rho)}  (top-{k} vs bottom-{k})',
                 fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    fname = f"{corrupt_ds_label.replace('-','').replace(' ','_')}_{cat_name}_scatter_k{k}.pdf"
    path  = os.path.join(FIG_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    saved: {path}')


def fig_group_boxplot(results_by_ds: dict, k: int):
    """
    Side-by-side box plots: top vs bottom group ΔLogME per category,
    one subplot per corrupt dataset.
    """
    cat_names = list(CATEGORIES.keys()) + ['All-19']
    ds_labels = list(results_by_ds.keys())
    if not ds_labels:
        return

    fig, axes = plt.subplots(1, len(ds_labels),
                              figsize=(5.5 * len(ds_labels), 5), sharey=False)
    if len(ds_labels) == 1:
        axes = [axes]

    for ax, ds_label in zip(axes, ds_labels):
        cat_results = results_by_ds[ds_label]
        positions_top    = []
        positions_bottom = []
        data_top    = []
        data_bottom = []
        labels      = []

        for i, cat in enumerate(cat_names):
            r = cat_results.get(cat)
            if r is None or r['df'] is None:
                continue
            df  = r['df']
            top = df.loc[df['group']=='top',    'mdelta'].values
            bot = df.loc[df['group']=='bottom', 'mdelta'].values
            if len(top) < 2 or len(bot) < 2:
                continue
            pos = len(labels)
            data_top.append(top)
            data_bottom.append(bot)
            positions_top.append(pos * 3)
            positions_bottom.append(pos * 3 + 1)
            labels.append(cat)

        if not labels:
            ax.set_visible(False)
            continue

        bp_top = ax.boxplot(data_top, positions=positions_top, widths=0.7,
                            patch_artist=True,
                            boxprops=dict(facecolor='#2980B9', alpha=0.7),
                            medianprops=dict(color='white', lw=2),
                            whiskerprops=dict(color='#2980B9'),
                            capprops=dict(color='#2980B9'),
                            flierprops=dict(marker='o', color='#2980B9', alpha=0.4, ms=4))
        bp_bot = ax.boxplot(data_bottom, positions=positions_bottom, widths=0.7,
                            patch_artist=True,
                            boxprops=dict(facecolor='#C0392B', alpha=0.7),
                            medianprops=dict(color='white', lw=2),
                            whiskerprops=dict(color='#C0392B'),
                            capprops=dict(color='#C0392B'),
                            flierprops=dict(marker='o', color='#C0392B', alpha=0.4, ms=4))

        # Significance markers
        for pos_t, pos_b, cat in zip(positions_top, positions_bottom, labels):
            r = cat_results.get(cat)
            if r and np.isfinite(r.get('mw_p', np.nan)):
                star = pstar(r['mw_p'])
                if star not in ('ns', ''):
                    mid = (pos_t + pos_b) / 2
                    ymax = max(
                        np.max(data_top[labels.index(cat)]),
                        np.max(data_bottom[labels.index(cat)]))
                    ax.text(mid, ymax * 1.05, star, ha='center', va='bottom',
                            fontsize=11, color='#2C3E50')

        tick_pos = [(t + b) / 2 for t, b in zip(positions_top, positions_bottom)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color='black', lw=0.7, alpha=0.4)
        ax.set_ylabel('Mean ΔLogME', fontsize=10)
        ax.set_title(f'{ds_label}  (top-{k} vs bottom-{k})', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#2980B9', alpha=0.7, label=f'Top-{k} FS (stable)'),
            Patch(facecolor='#C0392B', alpha=0.7, label=f'Bottom-{k} FS (fragile?)')],
            fontsize=9, frameon=False, loc='upper right')

    fig.suptitle(f'Extreme-groups comparison: ΔLogME by corruption category',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f'extreme_groups_boxplot_k{k}.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run(clean_dir: Path, corrupt_dir: Path, k: int = TOP_K):

    def _fl(msg=''):
        print(msg); sys.stdout.flush()

    PAIRS = [
        ('cifar10',  'cifar10c',  'CIFAR-10-C'),
        ('cifar100', 'cifar100c', 'CIFAR-100-C'),
    ]

    all_verdicts   = []           # (ds_label, cat, holds_α05)
    results_by_ds  = {}           # ds_label -> {cat: result_dict}

    for clean_ds, corrupt_ds, ds_label in PAIRS:
        _fl('\n' + '='*72)
        _fl(f'  {ds_label}')
        _fl(f'  Pre-specified cutoff: top-{k} vs bottom-{k} by clean Shesha-FS')
        _fl('='*72)

        clean, corrupt = build(clean_dir, corrupt_dir, clean_ds, corrupt_ds)
        if clean is None or corrupt is None:
            _fl('  [skip] Data unavailable.')
            continue

        # ── Assign groups ────────────────────────────────────────────────
        clean_groups = assign_groups(clean, k)
        top_models    = clean_groups[clean_groups['group']=='top']
        bottom_models = clean_groups[clean_groups['group']=='bottom']

        _fl(f'\n  Group summary (N = {len(clean_groups)} models with valid FS):')
        _fl(f"  Top-{k}    (most stable):  "
            f"FS mean={top_models['fs'].mean():.4f}  "
            f"min={top_models['fs'].min():.4f}  max={top_models['fs'].max():.4f}")
        _fl(f"  Bottom-{k} (least stable): "
            f"FS mean={bottom_models['fs'].mean():.4f}  "
            f"min={bottom_models['fs'].min():.4f}  max={bottom_models['fs'].max():.4f}")

        _fl(f"\n  Top-{k} models:")
        for _, row in top_models.sort_values('fs', ascending=False).iterrows():
            _fl(f"    {row['fs']:>7.4f}  {row['Model']}")
        _fl(f"\n  Bottom-{k} models:")
        for _, row in bottom_models.sort_values('fs').iterrows():
            _fl(f"    {row['fs']:>7.4f}  {row['Model']}")

        # ── Per-category analysis ────────────────────────────────────────
        _fl(f'\n  {"Category":<12}  {"N_corr":>6}  {"N_mod":>6}  '
            f'{"mean ΔLogME(top)":>18}  {"mean ΔLogME(bot)":>18}  '
            f'{"Diff(bot-top)":>15}  {"MW p":>10}  {"sig":>5}  {"direction"}')
        _fl('  ' + '-'*120)

        cat_results = {}
        for cat_name, corr_list in list(CATEGORIES.items()) + [('All-19', ALL19)]:
            r = analyse_category(clean_groups, corrupt, corr_list)
            if r is None:
                _fl(f'  {cat_name:<12}  [no data]')
                continue
            cat_results[cat_name] = r

            diff = r['mean_bottom'] - r['mean_top']
            sig  = pstar(r.get('mw_p', np.nan))
            holds = (np.isfinite(r.get('mw_p', np.nan))
                     and r['mw_p'] < 0.05
                     and r['mean_bottom'] > r['mean_top'])
            all_verdicts.append((ds_label, cat_name, holds))

            _fl(f"  {cat_name:<12}  {r['n_corr']:>6}  {r['n_models']:>6}  "
                f"{r['mean_top']:>+18.4f}  {r['mean_bottom']:>+18.4f}  "
                f"{diff:>+15.4f}  "
                f"{r['mw_p']:>10.4f}  {sig:>5}  {r['mw_direction']}")

            # Scatter figures (only for named categories, not All-19)
            if cat_name != 'All-19':
                fig_scatter(clean_groups, corrupt, ds_label,
                            cat_name, corr_list, k)

        results_by_ds[ds_label] = cat_results

        # ── Spearman as context ──────────────────────────────────────────
        _fl(f'\n  Spearman ρ(clean FS, mean ΔLogME) for context:')
        for cat_name, r in cat_results.items():
            rho_s = f"{r['rho']:+.3f}" if np.isfinite(r['rho']) else 'N/A'
            p_s   = f"{r['p_rho']:.4f}" if np.isfinite(r.get('p_rho', np.nan)) else 'N/A'
            _fl(f"    {cat_name:<12}  ρ={rho_s}  p={p_s}{pstar(r.get('p_rho', np.nan))}")

        # ── LaTeX table ──────────────────────────────────────────────────
        _fl(f'\n  --- LaTeX (extreme-groups table, {ds_label}) ---')
        _fl(r'  \begin{tabular}{lrrrrrr}')
        _fl(r'  \hline')
        _fl(f'  Category & $N_{{\\text{{corr}}}}$ & $N_{{\\text{{mod}}}}$ '
            f'& Mean $\\Delta$LogME (top-{k}) & Mean $\\Delta$LogME (bot-{k}) '
            r'& $\Delta$(bot$-$top) & MW $p$ \\')
        _fl(r'  \hline')
        for cat_name, r in cat_results.items():
            diff = r['mean_bottom'] - r['mean_top']
            p_s  = f"{r['mw_p']:.4f}" if np.isfinite(r.get('mw_p', np.nan)) else '—'
            sig  = pstar(r.get('mw_p', np.nan))
            _fl(f"  {cat_name} & {r['n_corr']} & {r['n_models']} "
                f"& ${r['mean_top']:+.3f}$ & ${r['mean_bottom']:+.3f}$ "
                f"& ${diff:+.3f}$ & {p_s}{sig} \\\\")
        _fl(r'  \hline')
        _fl(r'  \end{tabular}')

    # ── Box plots ────────────────────────────────────────────────────────
    if results_by_ds:
        fig_group_boxplot(results_by_ds, k)

    # ── Sensitivity check: k=15 ──────────────────────────────────────────
    if k != TOP_K2:
        _fl(f'\n\n{"#"*72}')
        _fl(f'  SENSITIVITY CHECK: k={TOP_K2} (instead of k={k})')
        _fl(f'{"#"*72}')
        for clean_ds, corrupt_ds, ds_label in PAIRS:
            _fl(f'\n  {ds_label}  (k={TOP_K2})')
            clean, corrupt = build(clean_dir, corrupt_dir, clean_ds, corrupt_ds)
            if clean is None or corrupt is None:
                continue
            cg2 = assign_groups(clean, TOP_K2)
            _fl(f"  {'Category':<12}  {'Diff(bot-top)':>15}  {'MW p':>10}  {'sig'}")
            _fl('  ' + '-'*50)
            for cat_name, corr_list in list(CATEGORIES.items()) + [('All-19', ALL19)]:
                r2 = analyse_category(cg2, corrupt, corr_list)
                if r2 is None:
                    continue
                diff2 = r2['mean_bottom'] - r2['mean_top']
                sig2  = pstar(r2.get('mw_p', np.nan))
                _fl(f"  {cat_name:<12}  {diff2:>+15.4f}  "
                    f"{r2['mw_p']:>10.4f}  {sig2}")

    # ── Pre-specified verdict ────────────────────────────────────────────
    _fl('\n\n' + '='*72)
    _fl('  PRE-SPECIFIED VERDICT')
    _fl('  Criterion: bottom group degrades MORE (mean ΔLogME(bottom) > mean ΔLogME(top))')
    _fl('             AND Mann-Whitney p < 0.05  on BOTH datasets.')
    _fl('='*72)

    categories = list(CATEGORIES.keys()) + ['All-19']
    datasets   = [label for _, _, label in PAIRS]

    _fl(f"\n  {'Category':<12}" + ''.join(f"  {ds:>15}" for ds in datasets) +
        f"  {'Replicates both?':>18}")
    _fl('  ' + '-'*80)

    for cat in categories:
        row_verdicts = {}
        for ds_label in datasets:
            matches = [(ds, c, h) for ds, c, h in all_verdicts
                       if ds == ds_label and c == cat]
            row_verdicts[ds_label] = matches[0][2] if matches else None

        both = (all(v is True for v in row_verdicts.values())
                if all(v is not None for v in row_verdicts.values()) else None)

        def _vstr(v):
            if v is None: return '      no data'
            return '✓ holds  p<.05' if v else '✗ fails'

        row_s = f"  {cat:<12}" + ''.join(f"  {_vstr(row_verdicts.get(ds)):>15}"
                                          for ds in datasets)
        if both is True:
            row_s += f"  {'YES — CLAIM SURVIVES':>18}"
        elif both is False:
            row_s += f"  {'NO  — orthogonal':>18}"
        else:
            row_s += f"  {'? (missing data)':>18}"
        _fl(row_s)

    # Count how many categories survive
    survivors = [cat for cat in categories
                 if all((ds, cat, True) in [(d,c,h) for d,c,h in all_verdicts]
                        for ds in datasets)]
    if survivors:
        _fl(f'\n  Categories surviving the pre-specified bar: {survivors}')
        _fl(f'  → Defensible claim: "the least geometrically stable models are '
            f'disproportionately fragile under {", ".join(survivors)} corruptions"')
    else:
        _fl('\n  No category survives the pre-specified bar on BOTH datasets.')
        _fl('  → The extreme-groups analysis confirms the orthogonality result.')
        _fl('  → Recommended framing: "clean Shesha-FS is orthogonal to '
            'corruption-induced LogME degradation across all corruption categories."')


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pre-specified extreme-groups robustness analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--clean-dir',   default=CLEAN_DIR)
    parser.add_argument('--corrupt-dir', default=CORRUPT_DIR)
    parser.add_argument('--k', type=int, default=TOP_K,
                        help=f'Group size (default: {TOP_K} ≈ top/bottom decile)')
    args, _ = parser.parse_known_args()

    run(Path(args.clean_dir), Path(args.corrupt_dir), k=args.k)


if __name__ == '__main__':
    main()
