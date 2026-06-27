"""
compute_robustness_table.py
============================
Standalone script that prints every cell of tab:corruption_robustness.

Computes, for each corruption category and for each of CIFAR-10-C / CIFAR-100-C:

    ρ(clean Shesha-FS,  mean ΔLogME per model)
    partial ρ  controlling for clean LogME

where  ΔLogME = LogME_clean − LogME_corrupt  (positive = more vulnerable).

Design decisions that match the table caption exactly
-----------------------------------------------------
* Deduplication: clean files: drop_duplicates("Model").
                 corrupt files: drop_duplicates(["Model","Corruption"]) so
                 DINOv3-qkvb variants are not double-counted.
* Per-model average within a category (not a pooled stack), matching the
  "across 170 models" caption.
* Ragged N: some corruptions have <170 models (e.g. CIFAR-100-C snow=168).
  Missing models are silently skipped; N reflects the actual overlap.
* All 19 corruptions are supported (see CATEGORIES below).
* Bootstrap 95% CIs (1000 draws) on both ρ and partial ρ.
* Output: console table + ready-to-paste LaTeX.

Usage
-----
    python compute_robustness_table.py

    python compute_robustness_table.py \\
        --clean-dir ./shesha-vision_architecture \\
        --corrupt-dir ./shesha-vision_architecture-corrupt
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEAN_DIR   = "./shesha-vision_architecture"
CORRUPT_DIR = "./shesha-vision_architecture-corrupt"

# Column names in the clean per-seed CSVs  (SEED320 files, 170 models)
FS_COL    = "SHESHA_FS"    # also tries SHESHA_FS_Mean if this is absent
LOGME_COL = "LogME"        # also tries LogME_Mean

# Column name for LogME in the corrupt CSVs
CORR_LOGME_COL = "LogME"

# Corruption categories — 5 groups covering all 19 types
CATEGORIES = {
    "Noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "Extra":   ["speckle_noise", "gaussian_blur", "spatter", "saturate"],
}
ALL19 = sum(CATEGORIES.values(), [])

# Bootstrap CIs
N_BOOT = 1000
BOOT_SEED = 320

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
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')


# =============================================================================
# PARTIAL SPEARMAN (user-provided formula — rank-transform then Pearson)
# =============================================================================

def partial_spearman(x, y, z):
    """
    Partial Spearman ρ of x,y controlling for z.
    Applies Pearson partial-correlation formula to the three rank vectors.
    """
    rx = pd.Series(x).rank().values.astype(float)
    ry = pd.Series(y).rank().values.astype(float)
    rz = pd.Series(z).rank().values.astype(float)

    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]

    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom < 1e-12:
        return np.nan
    return (rxy - rxz * ryz) / denom


# =============================================================================
# BOOTSTRAP CI
# =============================================================================

def _bootstrap(x, y, z, n_boot=N_BOOT, seed=BOOT_SEED):
    """Return (lo_rho, hi_rho, lo_partial, hi_partial) 95% CIs."""
    rng = np.random.default_rng(seed)
    n   = len(x)
    rho_b, part_b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            r, _ = spearmanr(x[idx], y[idx])
            p    = partial_spearman(x[idx], y[idx], z[idx])
            if np.isfinite(r):  rho_b.append(r)
            if np.isfinite(p):  part_b.append(p)
        except Exception:
            continue

    def ci(boots):
        if len(boots) < 50:
            return np.nan, np.nan
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    return ci(rho_b) + ci(part_b)


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_clean_csv(clean_dir: Path, dataset: str) -> pd.DataFrame | None:
    """
    Load the best clean CSV for a dataset (prefer SEED320 > ALL_SEEDS).
    Returns a DataFrame with columns [Model, fs, logme_clean].
    """
    d = clean_dir
    ds_upper = dataset.upper()

    for pat in [
        rf'^{ds_upper}_SEED320_',
        rf'^{ds_upper}_SEED\d+_',
        rf'^{ds_upper}_ALL_SEEDS_',
        rf'^{ds_upper}_AVERAGED_',
    ]:
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

        # Resolve FS column
        fs_col = FS_COL if FS_COL in df.columns else (
            'SHESHA_FS_Mean' if 'SHESHA_FS_Mean' in df.columns else None)
        lme_col = LOGME_COL if LOGME_COL in df.columns else (
            'LogME_Mean' if 'LogME_Mean' in df.columns else None)

        if fs_col is None or lme_col is None or 'Model' not in df.columns:
            continue

        df = df[['Model', fs_col, lme_col]].copy()
        df = df.rename(columns={fs_col: 'fs', lme_col: 'logme_clean'})
        df['fs']          = pd.to_numeric(df['fs'],          errors='coerce')
        df['logme_clean'] = pd.to_numeric(df['logme_clean'], errors='coerce')

        # Deduplicate: one row per model, keep last
        df = df.drop_duplicates('Model', keep='last')

        n_valid = df[['fs','logme_clean']].notna().all(axis=1).sum()
        print(f'  clean {dataset}: {len(df)} models ({n_valid} with fs+logme)  '
              f'← {cands[0].name}')
        return df

    print(f'  [WARN] No clean CSV found for {dataset} in {clean_dir}')
    return None


def _load_corrupt_long(corrupt_dir: Path, corrupt_ds: str) -> pd.DataFrame | None:
    """
    Load all corruption CSVs for a corrupt dataset (e.g. cifar10c) and
    return a long-format DataFrame with columns:
        Model | Corruption | logme_corr
    Priority per corruption: ALL_SEEDS > SEED > AVERAGED > checkpoint.
    """
    d = corrupt_dir
    ds_upper = corrupt_ds.upper()

    all_files = sorted(
        [f for f in d.glob('*.csv')
         if re.search(rf'^{ds_upper}_', f.name, re.IGNORECASE)],
        key=lambda f: f.name)

    # Group by corruption name
    buckets: dict[str, dict[str, list]] = {}
    for f in all_files:
        m = re.match(rf'^{ds_upper}_(.+?)_sev\d+_(ALL_SEEDS|SEED\d+|AVERAGED)_',
                     f.name, re.IGNORECASE)
        if not m:
            continue
        corr    = m.group(1).lower()
        ftype   = m.group(2).upper()
        bucket  = buckets.setdefault(corr, {'ALL_SEEDS':[], 'SEED':[], 'AVERAGED':[]})
        key     = 'ALL_SEEDS' if 'ALL_SEEDS' in ftype else ('AVERAGED' if 'AVERAGED' in ftype else 'SEED')
        bucket[key].append(f)

    print(f'  corrupt {corrupt_ds}: {len(buckets)} corruption types found')

    dfs = []
    for corr in sorted(buckets):
        b = buckets[corr]
        chosen = None
        for priority in ['ALL_SEEDS', 'SEED', 'AVERAGED']:
            cands = sorted(b[priority], key=lambda f: f.stat().st_mtime, reverse=True)
            if cands:
                chosen = cands[0]
                break
        if chosen is None:
            continue

        try:
            df = pd.read_csv(chosen, on_bad_lines='skip')
        except Exception as e:
            print(f'    [skip] {chosen.name}: {e}')
            continue

        if 'Model' not in df.columns or CORR_LOGME_COL not in df.columns:
            continue

        df[CORR_LOGME_COL] = pd.to_numeric(df[CORR_LOGME_COL], errors='coerce')

        # Status filter: drop oom_skip rows
        if 'LEEP_Status' in df.columns:
            df = df[df['LEEP_Status'] != 'oom_skip']

        # Deduplicate: one row per (Model, Corruption) — guards against DINOv3-qkvb
        # double-counting from duplicate rows in the same file
        df = df.drop_duplicates(subset=['Model', 'Corruption'], keep='last') \
               if 'Corruption' in df.columns else df.drop_duplicates('Model', keep='last')

        # If Corruption column is missing, inject it from the filename
        if 'Corruption' not in df.columns:
            df['Corruption'] = corr

        df = df[['Model', 'Corruption', CORR_LOGME_COL]].rename(
            columns={CORR_LOGME_COL: 'logme_corr'})
        dfs.append(df)

    if not dfs:
        print(f'  [WARN] No corrupt data loaded for {corrupt_ds}')
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined['Corruption'] = combined['Corruption'].str.lower()
    # Final global dedup per (Model, Corruption) across files
    combined = combined.drop_duplicates(subset=['Model','Corruption'], keep='last')
    n_models = combined['Model'].nunique()
    n_corrs  = combined['Corruption'].nunique()
    print(f'    → {len(combined)} rows  |  {n_models} models  |  {n_corrs} corruptions')
    return combined


# =============================================================================
# CORE COMPUTATION (user-provided logic, annotated)
# =============================================================================

def compute_row(clean: pd.DataFrame, corr: pd.DataFrame,
                corruptions: list[str], boot: bool = True):
    """
    Compute one table row for the given list of corruption names.

    Steps (matching the caption: "per-model average within category"):
      1. Filter corrupt rows to the requested corruptions.
      2. For each model compute the mean ΔLogME across those corruptions.
      3. Merge with clean FS and clean LogME (inner join = only models
         present in both clean and corrupt data).
      4. Compute ρ and partial ρ.
    Returns: rho, partial_rho, N, (lo_rho, hi_rho), (lo_part, hi_part)
    """
    present = [c for c in corruptions if c in corr['Corruption'].unique()]
    if not present:
        return np.nan, np.nan, 0, (np.nan, np.nan), (np.nan, np.nan)

    sub  = corr[corr['Corruption'].isin(present)]
    # Per-model average ΔLogME (step 2)
    mdelta = sub.groupby('Model')['delta'].mean().rename('mdelta')

    df = (clean[['Model','fs','logme_clean']]
          .merge(mdelta, on='Model', how='inner')
          .dropna(subset=['fs','logme_clean','mdelta']))

    n = len(df)
    if n < 5:
        return np.nan, np.nan, n, (np.nan, np.nan), (np.nan, np.nan)

    x = df['fs'].values
    y = df['mdelta'].values
    z = df['logme_clean'].values

    rho, _  = spearmanr(x, y)
    pr      = partial_spearman(x, y, z)

    if boot:
        lo_r, hi_r, lo_p, hi_p = _bootstrap(x, y, z)
    else:
        lo_r = hi_r = lo_p = hi_p = np.nan

    return rho, pr, n, (lo_r, hi_r), (lo_p, hi_p)


def build_dataset(clean_dir: Path, corrupt_dir: Path,
                  clean_ds: str, corrupt_ds: str):
    """Load and join clean + corrupt for one dataset pair."""
    clean = _load_clean_csv(clean_dir, clean_ds)
    if clean is None:
        return None, None

    corrupt_raw = _load_corrupt_long(corrupt_dir, corrupt_ds)
    if corrupt_raw is None:
        return clean, None

    # Merge clean LogME into corrupt table for ΔLogME
    corrupt_raw = corrupt_raw.merge(
        clean[['Model','logme_clean']], on='Model', how='inner')
    corrupt_raw['delta'] = corrupt_raw['logme_clean'] - corrupt_raw['logme_corr']
    # Drop rows where either LogME is missing
    corrupt_raw = corrupt_raw.dropna(subset=['logme_clean','logme_corr','delta'])

    return clean, corrupt_raw


# =============================================================================
# PRINT HELPERS
# =============================================================================

def _pstar(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _fmt(v, lo=None, hi=None):
    """Format value ± CI."""
    if not np.isfinite(v):
        return 'N/A'
    s = f'{v:+.3f}'
    if lo is not None and np.isfinite(lo):
        s += f'  [{lo:+.3f}, {hi:+.3f}]'
    return s


def _latex_row(cat, n, rho, ci_rho, pr, ci_pr, mean_delta):
    lo_r, hi_r = ci_rho
    lo_p, hi_p = ci_pr

    def lv(v, lo, hi):
        if not np.isfinite(v):
            return r'\text{N/A}'
        if np.isfinite(lo):
            return f'${v:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$]'
        return f'${v:+.2f}$'

    return (f'    {cat} & {n} & {lv(rho, lo_r, hi_r)} '
            f'& {lv(pr, lo_p, hi_p)} '
            f'& ${mean_delta:+.3f}$ \\\\')


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run(clean_dir: Path, corrupt_dir: Path, boot: bool = True):
    import sys

    def _fl(msg=''):
        print(msg); sys.stdout.flush()

    PAIRS = [
        ('cifar10',  'cifar10c',  'CIFAR-10 / CIFAR-10-C'),
        ('cifar100', 'cifar100c', 'CIFAR-100 / CIFAR-100-C'),
    ]

    for clean_ds, corrupt_ds, label in PAIRS:
        _fl('\n' + '='*72)
        _fl(f'  {label}')
        _fl('='*72)

        clean, corrupt = build_dataset(clean_dir, corrupt_dir, clean_ds, corrupt_ds)
        if clean is None or corrupt is None:
            _fl('  [skip] Data unavailable.')
            continue

        present_all = set(corrupt['Corruption'].unique())
        _fl(f'\n  Corruptions present: {len(present_all)}')
        _fl(f'  {sorted(present_all)}')

        # ── Console table ────────────────────────────────────────────────
        _fl(f'\n  {"Category":<10}  {"Corrs":>5}  {"N":>5}  '
            f'{"ρ(FS, ΔLogME)":>14}  {"[95% CI]":>18}  '
            f'{"partial ρ":>12}  {"[95% CI]":>18}  {"mean ΔLogME":>12}')
        _fl('  ' + '-'*110)

        rows_for_latex = []
        for name, corr_list in list(CATEGORIES.items()) + [('All-19', ALL19)]:
            present = [c for c in corr_list if c in present_all]
            rho, pr, n, ci_rho, ci_pr = compute_row(clean, corrupt, present, boot=boot)

            # Mean ΔLogME (weighted equally across models in the merged set)
            sub = corrupt[corrupt['Corruption'].isin(present)]
            mdelta = sub.groupby('Model')['delta'].mean()
            merged = clean[['Model','fs','logme_clean']].merge(
                mdelta.rename('mdelta'), on='Model', how='inner').dropna()
            mean_d = merged['mdelta'].mean() if len(merged) > 0 else np.nan

            lo_r, hi_r = ci_rho
            lo_p, hi_p = ci_pr
            ci_r_s  = f'[{lo_r:+.3f}, {hi_r:+.3f}]' if np.isfinite(lo_r) else '[?, ?]'
            ci_p_s  = f'[{lo_p:+.3f}, {hi_p:+.3f}]' if np.isfinite(lo_p) else '[?, ?]'
            rho_s   = f'{rho:+.3f}' if np.isfinite(rho) else 'N/A'
            pr_s    = f'{pr:+.3f}'  if np.isfinite(pr)  else 'N/A'
            md_s    = f'{mean_d:+.4f}' if np.isfinite(mean_d) else 'N/A'

            _fl(f'  {name:<10}  ({len(present):>2})  {n:>5}  '
                f'{rho_s:>14}  {ci_r_s:>18}  '
                f'{pr_s:>12}  {ci_p_s:>18}  {md_s:>12}')

            rows_for_latex.append(
                (name, len(present), n, rho, ci_rho, pr, ci_pr, mean_d))

        # ── Per-corruption detail ────────────────────────────────────────
        _fl(f'\n  Per-corruption (ρ only, no bootstrap):')
        _fl(f'  {"Corruption":<25}  {"N":>5}  {"ρ(FS,Δ)":>10}  '
            f'{"partial ρ":>12}  {"mean ΔLogME":>12}  {"Category"}')
        _fl('  ' + '-'*85)
        corr_to_cat = {c: cat for cat, cs in CATEGORIES.items() for c in cs}
        for corr_name in sorted(present_all):
            rho2, pr2, n2, _, _ = compute_row(
                clean, corrupt, [corr_name], boot=False)
            sub   = corrupt[corrupt['Corruption'] == corr_name]
            mdelta2 = sub.groupby('Model')['delta'].mean()
            merged2 = clean[['Model','fs','logme_clean']].merge(
                mdelta2.rename('mdelta'), on='Model').dropna()
            md2 = merged2['mdelta'].mean() if len(merged2) > 0 else np.nan
            cat = corr_to_cat.get(corr_name, '—')
            rho_s2 = f'{rho2:+.3f}' if np.isfinite(rho2) else 'N/A'
            pr_s2  = f'{pr2:+.3f}'  if np.isfinite(pr2)  else 'N/A'
            md_s2  = f'{md2:+.4f}'  if np.isfinite(md2)  else 'N/A'
            _fl(f'  {corr_name:<25}  {n2:>5}  {rho_s2:>10}  '
                f'{pr_s2:>12}  {md_s2:>12}  {cat}')

        # ── LaTeX (tab:corruption_robustness) ────────────────────────────
        _fl(f'\n  --- LaTeX (tab:corruption_robustness) — {label} ---')
        _fl(r'  \begin{tabular}{lrcccc}')
        _fl(r'  \hline')
        _fl(r'  Category & $N_{\text{corr}}$ & $N_{\text{models}}$ '
            r'& $\rho$ [95\% CI] & $\rho_{\mathrm{partial}}$ [95\% CI] '
            r'& $\overline{\Delta\mathrm{LogME}}$ \\')
        _fl(r'  \hline')
        for (name, n_corrs, n_models, rho, ci_rho, pr, ci_pr, mean_d) in rows_for_latex:
            lo_r, hi_r = ci_rho
            lo_p, hi_p = ci_pr
            def lv(v, lo, hi):
                if not np.isfinite(v): return r'\text{N/A}'
                return (f'${v:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$]'
                        if np.isfinite(lo) else f'${v:+.2f}$')
            md_s = f'${mean_d:+.3f}$' if np.isfinite(mean_d) else r'\text{N/A}'
            _fl(f'  {name} & {n_corrs} & {n_models} '
                f'& {lv(rho, lo_r, hi_r)} '
                f'& {lv(pr, lo_p, hi_p)} '
                f'& {md_s} \\\\')
        _fl(r'  \hline')
        _fl(r'  \end{tabular}')

        # ── Interpretation note ──────────────────────────────────────────
        _fl(f'\n  Interpretation note:')
        _fl(f'  ΔLogME > 0 means corruption degraded the metric (model was more vulnerable).')
        _fl(f'  Hypothesis (Geirhos): low clean FS → large ΔLogME → ρ < 0, partial ρ < 0.')
        # Check Noise vs Blur ordering
        all_cats = {name: (rho, pr) for name, _, _, rho, _, pr, _, _ in rows_for_latex}
        rho_noise = all_cats.get('Noise', (np.nan, np.nan))[0]
        rho_blur  = all_cats.get('Blur',  (np.nan, np.nan))[0]
        if np.isfinite(rho_noise) and np.isfinite(rho_blur):
            if rho_noise < rho_blur:
                _fl(f'  ✓ Noise ρ ({rho_noise:+.3f}) < Blur ρ ({rho_blur:+.3f}) — '
                    f'noise-vs-blur paragraph supported.')
            else:
                _fl(f'  ✗ Noise ρ ({rho_noise:+.3f}) ≥ Blur ρ ({rho_blur:+.3f}) — '
                    f'noise-vs-blur ordering does NOT hold; revise that paragraph.')


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute tab:corruption_robustness from clean + corrupt CSVs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--clean-dir',   default=CLEAN_DIR)
    parser.add_argument('--corrupt-dir', default=CORRUPT_DIR)
    parser.add_argument('--no-boot', action='store_true',
                        help='Skip bootstrap CIs (faster, for quick checks)')
    args, _ = parser.parse_known_args()

    run(Path(args.clean_dir), Path(args.corrupt_dir), boot=not args.no_boot)


if __name__ == '__main__':
    main()
