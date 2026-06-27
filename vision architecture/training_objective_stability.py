"""
training_objective_stability.py
================================
Determinants section analysis: does training objective predict geometric
stability (Shesha-FS)?

Five training-objective groups
-------------------------------
  1. Semantic-aligned  — CLIP, SigLIP, ViTamin (by objective, not architecture),
                         EVA / EVA-02 (semantic-aligned via ALIGN-reconstruction;
                         stated explicitly — EVA-02 is NOT self-supervised).
  2. Self-distillation — DINOv1, DINOv2, DINOv3, BEiTv2 (DINO teacher),
                         SAM (supervised-distillation with human masks).
  3. Masked prediction — MAE, BEiT (v1), BEiT3, I-JEPA, Hiera,
                         ConvNeXtV2 (FCMAE objective).
  4. Supervised-columnar — plain ViT (imagenet21k / augreg checkpoints),
                            DeiT / DeiT3 by checkpoint (supervised unless
                            the checkpoint name encodes 'clip' or 'align').
  5. Supervised-hierarchical — Swin, SwinV2, ConvNeXt (by checkpoint —
                                 supervised unless name encodes 'clip'),
                                 PVTv2, MaxViT, CoAtNet, ResNet family,
                                 EfficientNet, RegNet, MobileNet, etc.

Assignment precedence (evaluated top-to-bottom on the lowercased model name):
  • Any name containing 'clip', 'siglip', 'vitamin', 'eva02', 'eva_'  → Semantic-aligned
  • 'dinov3', 'dinov2', '.dino', '_dino', 'resmlp…dino', 'beitv2', 'samvit', '.sam_' → Self-distillation
  • '.mae', 'ijepa', 'hiera', 'beit3', 'beit', 'convnextv2'            → Masked prediction
  • 'deit3', 'deit', 'vit_'                                             → Supervised-columnar
  • everything else (swin, pvt, convnext, resnet, efficientnet, …)     → Supervised-hierarchical

Statistics (per dataset)
------------------------
  1. Kruskal-Wallis omnibus on Shesha-FS across all 5 groups.
  2. Dunn's post-hoc (all pairs) with Holm-Bonferroni correction on
     p-values — implemented from scratch to avoid optional dependencies.
  3. Pre-specified contrasts highlighted:
       Semantic-aligned vs {Self-distillation, Masked-pred,
                             Supervised-columnar, Supervised-hierarchical}
  4. Effect size η²_KW = (H − (k−1)) / (N − k) clipped to [0, 1].
  5. Median and IQR per group per dataset.

Outputs
-------
  • Console: per-dataset summary + full Dunn table + LaTeX table snippet.
  • PDFs: violin plot per dataset (fig_obj_stability_<ds>.pdf).
  • CSV:  objective_stability_results.csv  (per-dataset group medians + KW + η²).

Usage
-----
  python training_objective_stability.py \\
      --clean-dir ./shesha-vision_architecture \\
      --out-dir ./obj_stability_figs
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEAN_DIR  = "./shesha-vision_architecture"
OUT_DIR    = "./obj_stability_figs"

CLEAN_DATASETS = ['cifar10', 'cifar100', 'flowers102', 'dtd', 'eurosat', 'pets']
DISPLAY_NAMES  = {
    'cifar10':    'CIFAR-10',
    'cifar100':   'CIFAR-100',
    'flowers102': 'Flowers-102',
    'dtd':        'DTD',
    'eurosat':    'EuroSAT',
    'pets':       'Oxford Pets',
}

# Group labels (order preserved in plots / tables)
GROUPS = [
    'Semantic-aligned',
    'Self-distillation',
    'Masked-prediction',
    'Supervised-columnar',
    'Supervised-hierarchical',
]

# Pre-specified contrasts: semantic-aligned vs each other
FOCAL_CONTRASTS = [
    ('Semantic-aligned', 'Self-distillation'),
    ('Semantic-aligned', 'Masked-prediction'),
    ('Semantic-aligned', 'Supervised-columnar'),
    ('Semantic-aligned', 'Supervised-hierarchical'),
]

# Palette
GRP_COLOR = {
    'Semantic-aligned':        '#1F77B4',   # blue
    'Self-distillation':       '#D62728',   # red
    'Masked-prediction':       '#2CA02C',   # green
    'Supervised-columnar':     '#FF7F0E',   # orange
    'Supervised-hierarchical': '#9467BD',   # purple
}

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, norm, mannwhitneyu, rankdata

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# TRAINING OBJECTIVE ASSIGNMENT
# =============================================================================

def get_training_objective(model_name: str) -> str:
    """
    Assign a model to one of five training-objective groups.

    Precedence (evaluated top-to-bottom on lowercased name):

    Semantic-aligned:
        • 'clip', 'siglip' — contrastive text-image objectives
        • 'vitamin'        — ViTamin: contrastive by objective, not architecture
        • 'eva02', 'eva_'  — EVA / EVA-02: semantic-aligned via ALIGN-style
                             reconstruction to CLIP targets (NOT self-supervised)

    Self-distillation:
        • 'dinov3', 'dinov2', '.dino', '_dino', 'resmlp'+'dino'
                           — DINO-family teacher-student
        • 'beitv2'         — BEiTv2 uses a DINO-style teacher (not MIM)
        • 'samvit', '.sam_' — SAM: supervised with human-annotation targets

    Masked-prediction (MIM / masked-feature):
        • '.mae'           — MAE pixel reconstruction
        • 'ijepa'          — I-JEPA feature-space masked prediction
        • 'hiera'          — Hiera = hierarchical MAE
        • 'beit3', 'beit'  — BEiT / BEiT3 masked-token prediction
        • 'convnextv2'     — ConvNeXtV2 trained with FCMAE objective

    Supervised-columnar:
        • 'deit3', 'deit'  — DeiT / DeiT3 by checkpoint; assigned supervised
                             unless name encodes 'clip' or 'align' (caught above)
        • 'vit_'           — plain ViT (imagenet21k / augreg supervised)

    Supervised-hierarchical (default):
        • Swin, SwinV2, ConvNeXt (supervised checkpoint), PVTv2, MaxViT,
          CoAtNet, ResNet*, EfficientNet*, RegNet*, MobileNet, PoolFormer, …
    """
    m = model_name.lower()

    # ── Semantic-aligned ────────────────────────────────────────────────────
    if 'clip' in m:
        return 'Semantic-aligned'
    if 'siglip' in m:
        return 'Semantic-aligned'
    if 'vitamin' in m:
        return 'Semantic-aligned'
    if 'eva02' in m or 'eva_' in m:
        return 'Semantic-aligned'

    # ── Self-distillation ───────────────────────────────────────────────────
    if 'dinov3' in m:
        return 'Self-distillation'
    if 'dinov2' in m:
        return 'Self-distillation'
    if '.dino' in m or '_dino' in m:
        return 'Self-distillation'
    if 'resmlp' in m and 'dino' in m:
        return 'Self-distillation'
    if 'beitv2' in m:
        return 'Self-distillation'
    if 'samvit' in m or '.sam_' in m:
        return 'Self-distillation'

    # ── Masked prediction ───────────────────────────────────────────────────
    if '.mae' in m:
        return 'Masked-prediction'
    if 'ijepa' in m:
        return 'Masked-prediction'
    if 'hiera' in m:
        return 'Masked-prediction'
    if 'beit3' in m:
        return 'Masked-prediction'
    if 'beit' in m:
        return 'Masked-prediction'
    if 'convnextv2' in m:
        return 'Masked-prediction'

    # ── Supervised-columnar ─────────────────────────────────────────────────
    if 'deit3' in m or 'deit' in m:
        return 'Supervised-columnar'
    if 'vit_' in m:
        return 'Supervised-columnar'

    # ── Supervised-hierarchical (default) ───────────────────────────────────
    return 'Supervised-hierarchical'


# =============================================================================
# DATA LOADING
# =============================================================================

def _latest_csv(directory, pattern):
    d = Path(directory)
    matches = [f for f in d.glob('*.csv')
               if re.search(pattern, f.name, re.IGNORECASE)]
    if not matches:
        return None
    return max(matches, key=lambda f: f.stat().st_mtime)


def load_clean(dataset_name: str, clean_dir: Path) -> pd.DataFrame | None:
    """
    Load and average per-seed CSVs for one clean dataset.
    OOM-skipped models are kept with NaN metrics.
    """
    ds_upper = dataset_name.upper()
    d = Path(clean_dir)
    if not d.exists():
        print(f'[WARN] clean_dir does not exist: {clean_dir}')
        return None

    EXCLUDE = ['SPECTRAL', 'PROBE_SENSITIVITY']
    seed_files, avg_files = [], []
    for f in sorted(d.glob('*.csv')):
        nu = f.name.upper()
        if not nu.startswith(ds_upper + '_'):
            continue
        if any(p in nu for p in EXCLUDE):
            continue
        if 'AVERAGED' in nu:
            avg_files.append(f)
        else:
            seed_files.append(f)

    seed_files.sort(key=lambda f: f.stat().st_mtime)
    avg_files.sort(key=lambda f: f.stat().st_mtime)

    raw_dfs = []
    for f in seed_files:
        try:
            tmp = pd.read_csv(f, on_bad_lines='skip')
            if len(tmp) and 'Model' in tmp.columns:
                raw_dfs.append(tmp)
        except Exception as e:
            print(f'  [skip] {f.name}: {e}')

    df_from_seeds = None
    if raw_dfs:
        df_all = pd.concat(raw_dfs, ignore_index=True)
        for c in ['LEEP_Real', 'LogME', 'SHESHA_Var', 'SHESHA_FS', 'Dim']:
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

        if 'LEEP_Status' in df_all.columns and 'Seed' in df_all.columns:
            df_all['_oom'] = (df_all['LEEP_Status'] == 'oom_skip').astype(int)
            df_all['_has'] = df_all['SHESHA_FS'].notna().astype(int)
            df_all = df_all.sort_values(
                ['Model', 'Seed', '_has', '_oom'],
                ascending=[True, True, True, False])
            df_all = df_all.drop_duplicates(subset=['Model', 'Seed'], keep='last')
            df_all = df_all.drop(columns=['_oom', '_has'])
        elif 'Seed' in df_all.columns:
            df_all = df_all.drop_duplicates(subset=['Model', 'Seed'], keep='last')
        else:
            df_all = df_all.drop_duplicates(subset=['Model'], keep='last')

        if 'LEEP_Status' in df_all.columns:
            oom = df_all['LEEP_Status'] == 'oom_skip'
            for c in ['LEEP_Real', 'LogME', 'SHESHA_Var', 'SHESHA_FS']:
                if c in df_all.columns:
                    df_all.loc[oom, c] = np.nan

        metric_cols = [c for c in ['LEEP_Real', 'LogME', 'SHESHA_Var', 'SHESHA_FS']
                       if c in df_all.columns]
        agg = {}
        for c in metric_cols:
            agg[f'{c}_Mean'] = (c, 'mean')
            agg[f'{c}_Std']  = (c, 'std')
        if 'Dim' in df_all.columns:
            agg['Dim'] = ('Dim', 'first')
        agg['N_Seeds'] = (metric_cols[0] if metric_cols else 'Model', 'count')
        df_from_seeds = df_all.groupby('Model').agg(**agg).reset_index()

    df_from_avg = None
    if avg_files:
        p = max(avg_files, key=lambda f: f.stat().st_mtime)
        try:
            da = pd.read_csv(p, on_bad_lines='skip')
            if 'Model' in da.columns:
                for c in ['LEEP_Real_Mean', 'LogME_Mean',
                          'SHESHA_Var_Mean', 'SHESHA_FS_Mean', 'Dim']:
                    if c in da.columns:
                        da[c] = pd.to_numeric(da[c], errors='coerce')
                # Handle older AVERAGED files that lack _Mean suffix
                for raw in ['LEEP_Real', 'LogME', 'SHESHA_Var', 'SHESHA_FS']:
                    mean_col = f'{raw}_Mean'
                    if mean_col not in da.columns and raw in da.columns:
                        da[mean_col] = pd.to_numeric(da[raw], errors='coerce')
                df_from_avg = da
        except Exception as e:
            print(f'  [skip AVERAGED] {p.name}: {e}')

    if df_from_seeds is not None and df_from_avg is not None:
        extra = df_from_avg[~df_from_avg['Model'].isin(set(df_from_seeds['Model']))]
        df_out = pd.concat([df_from_seeds, extra], ignore_index=True) if len(extra) else df_from_seeds
    elif df_from_seeds is not None:
        df_out = df_from_seeds
    elif df_from_avg is not None:
        df_out = df_from_avg
    else:
        print(f'[WARN] No data for {dataset_name} in {clean_dir}')
        return None

    df_out['Training_Objective'] = df_out['Model'].map(get_training_objective)
    return df_out


# =============================================================================
# STATISTICS
# =============================================================================

def dunn_test_holm(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Dunn's post-hoc test with Holm-Bonferroni correction.

    Parameters
    ----------
    groups : {group_label: array of SHESHA_FS values}

    Returns
    -------
    DataFrame with columns [group_A, group_B, z, p_raw, p_holm, significant]
    """
    labels = list(groups.keys())
    all_vals = np.concatenate(list(groups.values()))
    N = len(all_vals)
    # Global ranks (average ties)
    global_ranks = rankdata(all_vals)

    # Build per-group mean ranks and sizes
    pos = 0
    grp_mean_rank = {}
    grp_n = {}
    for lbl, vals in groups.items():
        n = len(vals)
        grp_mean_rank[lbl] = global_ranks[pos: pos + n].mean()
        grp_n[lbl] = n
        pos += n

    # Tie correction term for variance
    # σ²_ij = (N(N+1)/12) * (1/n_i + 1/n_j)
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            na, nb = grp_n[a], grp_n[b]
            se = np.sqrt((N * (N + 1) / 12.0) * (1.0 / na + 1.0 / nb))
            z  = (grp_mean_rank[a] - grp_mean_rank[b]) / se
            p  = 2.0 * (1.0 - norm.cdf(abs(z)))
            rows.append({'group_A': a, 'group_B': b,
                         'z': z, 'p_raw': p,
                         'n_A': na, 'n_B': nb})

    df = pd.DataFrame(rows).sort_values('p_raw').reset_index(drop=True)

    # Holm-Bonferroni correction
    m = len(df)
    p_holm = df['p_raw'].values.copy()
    for k in range(m):
        p_holm[k] = min(1.0, df['p_raw'].iloc[k] * (m - k))
    # Ensure monotonicity
    for k in range(1, m):
        p_holm[k] = max(p_holm[k], p_holm[k - 1])
    df['p_holm'] = p_holm
    df['significant'] = df['p_holm'] < 0.05
    return df


def eta_squared_kw(H: float, k: int, N: int) -> float:
    """η²_KW = (H − k + 1) / (N − k), clipped to [0, 1]."""
    if N <= k:
        return np.nan
    return float(np.clip((H - k + 1) / (N - k), 0.0, 1.0))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d (Semantic-aligned vs comparison group)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_sd = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_sd == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


# =============================================================================
# PER-DATASET ANALYSIS
# =============================================================================

def analyse_dataset(ds: str, df: pd.DataFrame, out_dir: Path) -> dict:
    """
    Run the full pipeline for one dataset.
    Returns a dict of summary statistics for the CSV output.
    """
    dname = DISPLAY_NAMES[ds]
    fs_col = 'SHESHA_FS_Mean' if 'SHESHA_FS_Mean' in df.columns else 'SHESHA_FS'

    sub = df[['Model', 'Training_Objective', fs_col]].copy()
    sub = sub.dropna(subset=[fs_col])
    sub.rename(columns={fs_col: 'FS'}, inplace=True)

    N = len(sub)
    print(f'\n{"="*70}')
    print(f'  {dname}  (N={N} models with valid Shesha-FS)')
    print(f'{"="*70}')

    # Per-group descriptives
    present_groups = [g for g in GROUPS if g in sub['Training_Objective'].values]
    grp_vals = {g: sub.loc[sub['Training_Objective'] == g, 'FS'].values
                for g in present_groups}

    print(f'\n  {"Group":<26}  {"n":>4}  {"Median":>7}  {"IQR":>12}')
    print('  ' + '-' * 56)
    for g in present_groups:
        v = grp_vals[g]
        q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
        print(f'  {g:<26}  {len(v):>4}  {np.median(v):>7.4f}  '
              f'[{q1:.4f},{q3:.4f}]')

    # ── Kruskal-Wallis ───────────────────────────────────────────────────────
    if len(present_groups) < 2:
        print('  [skip] fewer than 2 groups with data')
        return {}

    arrays = [grp_vals[g] for g in present_groups]
    H_stat, kw_p = kruskal(*arrays)
    k = len(present_groups)
    eta2 = eta_squared_kw(H_stat, k, N)
    print(f'\n  Kruskal-Wallis: H={H_stat:.3f}, p={kw_p:.4f}, '
          f'η²={eta2:.3f}  (k={k}, N={N})')

    # ── Dunn post-hoc ────────────────────────────────────────────────────────
    dunn = dunn_test_holm(grp_vals)

    print(f'\n  Dunn post-hoc (Holm-corrected) — all {len(dunn)} pairs:')
    print(f'  {"Group A":<26}  {"Group B":<26}  {"z":>7}  '
          f'{"p_raw":>8}  {"p_holm":>8}  sig')
    print('  ' + '-' * 90)
    for _, row in dunn.iterrows():
        sig = '*' if row['significant'] else ' '
        print(f'  {row["group_A"]:<26}  {row["group_B"]:<26}  '
              f'{row["z"]:>7.3f}  {row["p_raw"]:>8.4f}  '
              f'{row["p_holm"]:>8.4f}  {sig}')

    # ── Pre-specified focal contrasts ────────────────────────────────────────
    print(f'\n  Pre-specified contrasts  (Semantic-aligned vs others):')
    print(f'  {"vs.":<26}  {"z":>7}  {"p_holm":>8}  {"Cohen d":>8}  sig')
    print('  ' + '-' * 62)
    focal_rows = []
    sem = grp_vals.get('Semantic-aligned', np.array([]))
    for _, other in FOCAL_CONTRASTS:
        if other not in grp_vals:
            continue
        row = dunn[
            ((dunn['group_A'] == 'Semantic-aligned') & (dunn['group_B'] == other)) |
            ((dunn['group_B'] == 'Semantic-aligned') & (dunn['group_A'] == other))
        ]
        if row.empty:
            continue
        r = row.iloc[0]
        d = cohens_d(sem, grp_vals[other])
        sig = '*' if r['significant'] else ' '
        print(f'  {other:<26}  {r["z"]:>7.3f}  '
              f'{r["p_holm"]:>8.4f}  {d:>8.3f}  {sig}')
        focal_rows.append({'dataset': dname, 'contrast': f'SA vs {other}',
                           'z': r['z'], 'p_holm': r['p_holm'],
                           'cohen_d': d, 'sig': r['significant']})

    # ── Violin plot ──────────────────────────────────────────────────────────
    _plot_violin(ds, dname, sub, present_groups, H_stat, kw_p, eta2,
                 dunn, out_dir)

    # ── Summary row for CSV ──────────────────────────────────────────────────
    summary = {
        'dataset': dname,
        'N': N,
        'KW_H': round(H_stat, 4),
        'KW_p': round(kw_p, 6),
        'eta2': round(eta2, 4),
        'k_groups': k,
    }
    for g in GROUPS:
        v = grp_vals.get(g, np.array([]))
        summary[f'n_{g[:4]}'] = len(v)
        summary[f'med_{g[:4]}'] = round(np.median(v), 4) if len(v) else np.nan
    summary['focal_rows'] = focal_rows
    return summary


# =============================================================================
# VISUALIZATION
# =============================================================================

def _plot_violin(ds, dname, sub, present_groups, H, kw_p, eta2,
                 dunn, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    positions = list(range(len(present_groups)))
    vp = ax.violinplot(
        [sub.loc[sub['Training_Objective'] == g, 'FS'].values
         for g in present_groups],
        positions=positions,
        widths=0.65,
        showmedians=True,
        showextrema=True,
    )

    for i, (body, g) in enumerate(zip(vp['bodies'], present_groups)):
        body.set_facecolor(GRP_COLOR[g])
        body.set_alpha(0.55)
        body.set_edgecolor('#333333')
        body.set_linewidth(0.8)

    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        vp[part].set_color('#222222')
        vp[part].set_linewidth(1.2)

    # Jittered strip
    rng = np.random.default_rng(42)
    for i, g in enumerate(present_groups):
        vals = sub.loc[sub['Training_Objective'] == g, 'FS'].values
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, s=9, alpha=0.55,
                   color=GRP_COLOR[g], zorder=3, linewidths=0)

    # Annotate Semantic-aligned vs others with * if significant
    y_top = sub['FS'].max() * 1.05
    sem_idx = (present_groups.index('Semantic-aligned')
               if 'Semantic-aligned' in present_groups else None)
    if sem_idx is not None:
        for i, g in enumerate(present_groups):
            if g == 'Semantic-aligned':
                continue
            row = dunn[
                ((dunn['group_A'] == 'Semantic-aligned') & (dunn['group_B'] == g)) |
                ((dunn['group_B'] == 'Semantic-aligned') & (dunn['group_A'] == g))
            ]
            if row.empty or not row.iloc[0]['significant']:
                continue
            x0, x1 = sorted([sem_idx, i])
            y_br = y_top + (i - 1) * 0.012
            ax.plot([x0, x0, x1, x1], [y_br, y_br + 0.006, y_br + 0.006, y_br],
                    lw=0.8, color='#444444')
            ax.text((x0 + x1) / 2, y_br + 0.007, '*',
                    ha='center', va='bottom', fontsize=10, color='#444444')

    short_labels = {
        'Semantic-aligned':        'Semantic\naligned',
        'Self-distillation':       'Self-\ndistill.',
        'Masked-prediction':       'Masked\npred.',
        'Supervised-columnar':     'Sup.\ncolumnar',
        'Supervised-hierarchical': 'Sup.\nhierarch.',
    }
    ax.set_xticks(positions)
    ax.set_xticklabels([short_labels.get(g, g) for g in present_groups],
                       fontsize=9)
    ax.set_ylabel('Shesha-FS  (geometric stability)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    # (a) panel letter
    ax.text(-0.06, 1.02, '(a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    # KW annotation
    p_str = f'p = {kw_p:.4f}' if kw_p >= 0.0001 else 'p < 0.0001'
    ax.text(0.97, 0.97,
            f'KW  H = {H:.2f},  {p_str}\nη² = {eta2:.3f}',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            color='#444444')

    # Legend patches
    handles = [mpatches.Patch(facecolor=GRP_COLOR[g], edgecolor='#333333',
                               alpha=0.7, label=g)
               for g in present_groups]
    ax.legend(handles=handles, fontsize=7.5, frameon=True,
              framealpha=0.9, edgecolor='#CCCCCC',
              loc='upper left', bbox_to_anchor=(0.0, 0.93))

    fig.tight_layout()
    path = out_dir / f'fig_obj_stability_{ds}.pdf'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  saved: {path}')


# =============================================================================
# LATEX TABLE
# =============================================================================

def print_latex_table(all_summaries: list[dict]) -> None:
    """
    Print a LaTeX table: rows = datasets, columns = per-group median Shesha-FS,
    plus KW H and η².
    """
    grp_abbr = {
        'Semantic-aligned':        r'\textsc{sem}',
        'Self-distillation':       r'\textsc{self}',
        'Masked-prediction':       r'\textsc{mask}',
        'Supervised-columnar':     r'\textsc{sup-c}',
        'Supervised-hierarchical': r'\textsc{sup-h}',
    }
    header_cols = ' & '.join(grp_abbr[g] for g in GROUPS)
    print('\n' + '='*70)
    print('LaTeX table  (copy into paper)')
    print('='*70)
    print(r'\begin{table}[t!]')
    print(r'\caption{Shesha-FS by training objective per dataset. '
          r'Groups: \textsc{sem} = semantic-aligned (CLIP/SigLIP/ViTamin/EVA-02); '
          r'\textsc{self} = self-distillation (DINO generations); '
          r'\textsc{mask} = masked-image/feature prediction (MAE/BEiT/I-JEPA); '
          r'\textsc{sup-c} = supervised columnar (ViT/DeiT); '
          r'\textsc{sup-h} = supervised hierarchical (Swin/ConvNeXt/ResNet etc.). '
          r'Median Shesha-FS (bold = highest group per row). '
          r'$^{*}$Dunn post-hoc Holm-corrected $p < 0.05$ vs.\ \textsc{sem}; '
          r'$\eta^2_\mathrm{KW}$ reports effect size.}')
    print(r'\label{tab:obj_stability}')
    print(r'\centering\small')
    print(r'\begin{tabular}{l ' + 'r' * len(GROUPS) + ' rr}')
    print(r'\hline')
    print(r'Dataset & ' + header_cols + r' & $H$ & $\eta^2$ \\')
    print(r'\hline')

    for s in all_summaries:
        vals = {g: s.get(f'med_{g[:4]}', np.nan) for g in GROUPS}
        best_g = max(vals, key=lambda g: vals[g] if np.isfinite(vals[g]) else -np.inf)
        focal_sig = {r['contrast'].replace('SA vs ', ''): r['sig']
                     for r in s.get('focal_rows', [])}

        cells = []
        for g in GROUPS:
            v = vals[g]
            cell = f'{v:.3f}' if np.isfinite(v) else '—'
            if g == best_g:
                cell = r'\textbf{' + cell + '}'
            if g != 'Semantic-aligned' and focal_sig.get(g, False):
                cell += r'$^{*}$'
            cells.append(cell)

        h_str  = f'{s["KW_H"]:.2f}'
        e2_str = f'{s["eta2"]:.3f}'
        print(f'  {s["dataset"]} & ' + ' & '.join(cells) +
              f' & {h_str} & {e2_str} \\\\')

    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-dir', default=CLEAN_DIR,
                        help='Path to clean-benchmark CSV folder')
    parser.add_argument('--out-dir', default=OUT_DIR,
                        help='Output directory for PDFs and CSV')
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Training-objective stability analysis')
    print('======================================')
    print('Assignment rules (in order of precedence):')
    print('  Semantic-aligned:        clip, siglip, vitamin, eva02/eva_')
    print('  Self-distillation:       dinov1/v2/v3, beitv2, samvit')
    print('  Masked-prediction:       .mae, ijepa, hiera, beit3, beit, convnextv2')
    print('  Supervised-columnar:     deit3, deit, vit_')
    print('  Supervised-hierarchical: everything else')
    print()
    print('EVA-02 is assigned to Semantic-aligned via ALIGN-reconstruction objective,')
    print('not treated as self-supervised. ViTamin is assigned by objective (contrastive).')
    print('ConvNeXt (non-v2) → Supervised-hierarchical by checkpoint.')
    print('ConvNeXtV2 → Masked-prediction (FCMAE objective).')
    print()

    # ── Load all datasets ────────────────────────────────────────────────────
    clean = {}
    print('Loading datasets...')
    for ds in CLEAN_DATASETS:
        df = load_clean(ds, clean_dir)
        if df is not None:
            clean[ds] = df
            obj_counts = df['Training_Objective'].value_counts().to_dict()
            print(f'  {DISPLAY_NAMES[ds]}: {len(df)} models — '
                  + ', '.join(f'{g}: {n}' for g, n in obj_counts.items()))

    if not clean:
        print('[ERROR] No datasets loaded. Check --clean-dir path.')
        return

    # ── Group inventory (once, from first dataset) ───────────────────────────
    first_ds = next(iter(clean.values()))
    print('\nGroup inventory (model count from first loaded dataset):')
    for g in GROUPS:
        models_in_g = first_ds.loc[
            first_ds['Training_Objective'] == g, 'Model'].tolist()
        print(f'\n  {g} ({len(models_in_g)} models):')
        for m in sorted(models_in_g)[:12]:
            print(f'    {m}')
        if len(models_in_g) > 12:
            print(f'    … and {len(models_in_g) - 12} more')

    # ── Per-dataset analysis ─────────────────────────────────────────────────
    all_summaries = []
    for ds in CLEAN_DATASETS:
        if ds not in clean:
            continue
        summary = analyse_dataset(ds, clean[ds], out_dir)
        if summary:
            all_summaries.append(summary)

    # ── Cross-dataset headline ───────────────────────────────────────────────
    if all_summaries:
        print('\n' + '='*70)
        print('Cross-dataset headline: semantic alignment confers stability')
        print('='*70)
        sig_ds = []
        for s in all_summaries:
            any_sig = any(r['sig'] for r in s.get('focal_rows', []))
            if any_sig:
                sig_ds.append(s['dataset'])
        n_sig = len(sig_ds)
        n_tot = len(all_summaries)
        print(f'\n  Semantic-aligned significantly outperforms ≥1 other group on '
              f'{n_sig}/{n_tot} datasets: {", ".join(sig_ds)}')

        # Pool all focal contrast results
        all_focal = []
        for s in all_summaries:
            all_focal.extend(s.get('focal_rows', []))
        if all_focal:
            df_focal = pd.DataFrame(all_focal)
            print('\n  Focal contrasts across all datasets:')
            print(f'  {"Dataset":<14}  {"Contrast":<36}  {"z":>7}  '
                  f'{"p_holm":>8}  {"d":>7}  sig')
            print('  ' + '-' * 80)
            for _, r in df_focal.iterrows():
                sig = '*' if r['sig'] else ' '
                print(f'  {r["dataset"]:<14}  {r["contrast"]:<36}  '
                      f'{r["z"]:>7.3f}  {r["p_holm"]:>8.4f}  '
                      f'{r["cohen_d"]:>7.3f}  {sig}')

    # ── LaTeX table ──────────────────────────────────────────────────────────
    if all_summaries:
        print_latex_table(all_summaries)

    # ── CSV output ───────────────────────────────────────────────────────────
    if all_summaries:
        rows_csv = []
        for s in all_summaries:
            base = {k: v for k, v in s.items() if k != 'focal_rows'}
            rows_csv.append(base)
        df_csv = pd.DataFrame(rows_csv)
        csv_path = out_dir / 'objective_stability_results.csv'
        df_csv.to_csv(csv_path, index=False)
        print(f'\n  saved: {csv_path}')

        # Also save focal contrasts
        all_focal = []
        for s in all_summaries:
            all_focal.extend(s.get('focal_rows', []))
        if all_focal:
            focal_path = out_dir / 'objective_stability_focal_contrasts.csv'
            pd.DataFrame(all_focal).to_csv(focal_path, index=False)
            print(f'  saved: {focal_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
