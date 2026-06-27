"""
Combined cross-domain analysis
================================
Uses the merged dataset (original 6 domains + UCF-101 video).

Input:
  merged_aggregated_by_encoder.csv — one row per (domain, base_model, encoder),
  seed-averaged. Expected at ./shesha-distinction/merged_aggregated_by_encoder.csv
  (override with --input).

Outputs (written to OUT_DIR):
  combined_domain_ci.csv       — per-domain bootstrap CI table
  combined_aggregate_ci.csv    — aggregate across all domains
  combined_domain_scatter.png
  combined_domain_bar.png
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent
DEFAULT_MERGED_CSV = ROOT / "shesha-distinction" / "merged_aggregated_by_encoder.csv"
OUT_DIR = ROOT / "combined_analysis"
N_BOOTSTRAP    = 10_000
ALPHA          = 0.05
RNG_SEED       = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=ALPHA,
                 statistic=np.mean, seed=RNG_SEED):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng  = np.random.default_rng(seed)
    boot = np.array([statistic(rng.choice(vals, size=len(vals), replace=True))
                     for _ in range(n_boot)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(statistic(vals)), float(lo), float(hi)


def bootstrap_ci_rho(x, y, n_boot=N_BOOTSTRAP, alpha=ALPHA, seed=RNG_SEED):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        rho, _ = spearmanr(x[idx], y[idx])
        if np.isfinite(rho):
            boot_rhos.append(rho)
    if not boot_rhos:
        return np.nan, np.nan, np.nan, np.nan
    boot_rhos = np.array(boot_rhos)
    lo = np.percentile(boot_rhos, 100 * alpha / 2)
    hi = np.percentile(boot_rhos, 100 * (1 - alpha / 2))
    point_rho, pval = spearmanr(x, y)
    return float(point_rho), float(lo), float(hi), float(pval)


def _video_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower() == "video"


def load_merged(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"[ERROR] Merged file not found: {path}\n"
            "  Place merged_aggregated_by_encoder.csv in ./shesha-distinction/ "
            "or pass --input."
        )
    df = pd.read_csv(path)
    required = {"domain", "base_model", "encoder", "SHESHA", "CKA"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns: {sorted(missing)}")

    for col in ["SHESHA", "CKA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["SHESHA", "CKA"])

    print(f"\n[merged] Loaded {path}")
    print(f"[merged] Shape: {df.shape}  columns: {list(df.columns)}")
    for dom in sorted(df["domain"].unique()):
        n = len(df[df["domain"] == dom])
        print(f"  {dom:20s}  N={n}")

    video_n = len(df[_video_mask(df["domain"])])
    if video_n == 0:
        sys.exit("[ERROR] No Video rows in merged file.")
    print(f"\n[merged] Video encoder configs: {video_n}")
    return df


def main(merged_csv: Path = DEFAULT_MERGED_CSV, out_dir: Path = OUT_DIR):
    print("\n" + "=" * 60)
    print("COMBINED CROSS-DOMAIN ANALYSIS")
    print(f"Source: {merged_csv}")
    print(f"Bootstrap resamples: {N_BOOTSTRAP:,}   α = {ALPHA}   (95 % CI)")
    print("=" * 60)

    df_all = load_merged(merged_csv)
    keep = ["domain", "SHESHA", "CKA"]
    df_all[keep].to_csv(out_dir / "combined_encoder_agg.csv", index=False)

    print(f"\nTotal encoder configs: {len(df_all)}")
    print(f"Domains: {sorted(df_all['domain'].unique())}\n")

    # Per-domain CI
    print("=" * 60)
    print("PER-DOMAIN RESULTS  (95 % bootstrap CI)")
    print("=" * 60)

    ci_rows = []
    for domain in sorted(df_all["domain"].unique()):
        sub = df_all[df_all["domain"] == domain].dropna(subset=["SHESHA", "CKA"])
        if len(sub) < 5:
            print(f"\n{domain}: too few rows ({len(sub)}), skipping.")
            continue

        s_est, s_lo, s_hi       = bootstrap_ci(sub["SHESHA"].values)
        c_est, c_lo, c_hi       = bootstrap_ci(sub["CKA"].values)
        r_est, r_lo, r_hi, pval = bootstrap_ci_rho(sub["SHESHA"].values, sub["CKA"].values)

        print(f"\n{domain}  (N={len(sub)}):")
        print(f"  SHESHA : {s_est:+.4f}  95% CI [{s_lo:+.4f}, {s_hi:+.4f}]")
        print(f"  CKA    : {c_est:+.4f}  95% CI [{c_lo:+.4f}, {c_hi:+.4f}]")
        if np.isfinite(r_est):
            print(f"  rho    : {r_est:+.4f}  95% CI [{r_lo:+.4f}, {r_hi:+.4f}]"
                  f"  (p={pval:.4f})")

        ci_rows.append({
            "domain":       domain,
            "N":            len(sub),
            "SHESHA_mean":  s_est,  "SHESHA_ci_lo": s_lo,  "SHESHA_ci_hi": s_hi,
            "CKA_mean":     c_est,  "CKA_ci_lo":    c_lo,  "CKA_ci_hi":    c_hi,
            "rho":          r_est,  "rho_ci_lo":    r_lo,  "rho_ci_hi":    r_hi,
            "rho_pval":     pval,   "n_bootstrap":  N_BOOTSTRAP,
        })

    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(out_dir / "combined_domain_ci.csv", index=False)
    print(f"\nPer-domain CI table → {out_dir / 'combined_domain_ci.csv'}")

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATE  (all domains pooled)")
    print("=" * 60)

    mask  = np.isfinite(df_all["SHESHA"]) & np.isfinite(df_all["CKA"])
    agg_s = df_all.loc[mask, "SHESHA"].values
    agg_c = df_all.loc[mask, "CKA"].values

    s_est, s_lo, s_hi       = bootstrap_ci(agg_s)
    c_est, c_lo, c_hi       = bootstrap_ci(agg_c)
    r_est, r_lo, r_hi, pval = bootstrap_ci_rho(agg_s, agg_c)

    print(f"\n  N      = {len(agg_s)}")
    print(f"  SHESHA : {s_est:+.4f}  95% CI [{s_lo:+.4f}, {s_hi:+.4f}]")
    print(f"  CKA    : {c_est:+.4f}  95% CI [{c_lo:+.4f}, {c_hi:+.4f}]")
    if np.isfinite(r_est):
        print(f"  rho    : {r_est:+.4f}  95% CI [{r_lo:+.4f}, {r_hi:+.4f}]"
              f"  (p={pval:.4f})")

    pd.DataFrame([{
        "N":            len(agg_s),
        "SHESHA_mean":  s_est,  "SHESHA_ci_lo": s_lo,  "SHESHA_ci_hi": s_hi,
        "CKA_mean":     c_est,  "CKA_ci_lo":    c_lo,  "CKA_ci_hi":    c_hi,
        "rho":          r_est,  "rho_ci_lo":    r_lo,  "rho_ci_hi":    r_hi,
        "rho_pval":     pval,   "n_bootstrap":  N_BOOTSTRAP,
    }]).to_csv(out_dir / "combined_aggregate_ci.csv", index=False)

    _plot_scatter(df_all, df_ci, out_dir)
    _plot_rho_bars(df_ci, r_est, r_lo, r_hi, out_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"All outputs in: {out_dir.resolve()}")
    print("=" * 60)


DOMAIN_COLORS = {
    "Language": "#4C72B0", "Vision": "#DD8452", "Audio": "#55A868",
    "Video": "#C44E52", "Neuroscience": "#8172B2",
    "Protein": "#937860", "Molecular": "#DA8BC3",
}


def _color(name):
    for k, v in DOMAIN_COLORS.items():
        if k.lower() in name.lower():
            return v
    return "#999999"


def _plot_scatter(df_all, df_ci, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    handles = []
    for domain in sorted(df_all["domain"].unique()):
        sub   = df_all[df_all["domain"] == domain].dropna(subset=["SHESHA", "CKA"])
        color = _color(domain)
        ax.scatter(sub["SHESHA"], sub["CKA"],
                   alpha=0.20, s=14, color=color, rasterized=True)
        row = df_ci[df_ci["domain"] == domain]
        if not row.empty:
            sm  = float(row["SHESHA_mean"].iloc[0])
            cm  = float(row["CKA_mean"].iloc[0])
            slo = max(sm - float(row["SHESHA_ci_lo"].iloc[0]), 0)
            shi = max(float(row["SHESHA_ci_hi"].iloc[0]) - sm, 0)
            clo = max(cm - float(row["CKA_ci_lo"].iloc[0]), 0)
            chi = max(float(row["CKA_ci_hi"].iloc[0]) - cm, 0)
            ax.scatter([sm], [cm], s=120, color=color,
                       edgecolors="black", linewidths=0.8, zorder=5)
            ax.errorbar([sm], [cm],
                        xerr=[[slo], [shi]], yerr=[[clo], [chi]],
                        fmt="none", color=color, capsize=4,
                        linewidth=1.2, zorder=4)
        handles.append(mpatches.Patch(color=color, label=domain))

    ax.set_xlabel("SHESHA", fontsize=11)
    ax.set_ylabel("CKA", fontsize=11)
    ax.set_title("SHESHA vs CKA — all domains\n"
                 "(large markers = domain mean ± 95 % CI)", fontsize=11)
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "combined_domain_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  Scatter plot  → {p}")


def _plot_rho_bars(df_ci, agg_rho, agg_lo, agg_hi, out_dir):
    df_v = df_ci.dropna(subset=["rho"])
    if df_v.empty:
        return

    domains   = df_v["domain"].tolist()
    rhos      = df_v["rho"].values
    errs_lo   = np.clip(rhos - df_v["rho_ci_lo"].values, 0, None)
    errs_hi   = np.clip(df_v["rho_ci_hi"].values - rhos, 0, None)
    colors    = [_color(d) for d in domains]
    pvals     = df_v["rho_pval"].values

    domains_p = domains + ["AGGREGATE"]
    rhos_p    = list(rhos) + [agg_rho if np.isfinite(agg_rho) else 0]
    errs_lo_p = list(errs_lo) + [max(agg_rho - agg_lo, 0) if np.isfinite(agg_lo) else 0]
    errs_hi_p = list(errs_hi) + [max(agg_hi - agg_rho, 0) if np.isfinite(agg_hi) else 0]
    colors_p  = colors + ["#333333"]
    pvals_p   = list(pvals) + [np.nan]

    x = np.arange(len(domains_p))
    fig, ax = plt.subplots(figsize=(max(8, len(domains_p) * 1.3), 5))
    ax.bar(x, rhos_p, color=colors_p, edgecolor="black", linewidth=0.6, zorder=3)
    ax.errorbar(x, rhos_p, yerr=[errs_lo_p, errs_hi_p],
                fmt="none", color="black", capsize=5, linewidth=1.2, zorder=4)

    for xi, (rho, pv) in enumerate(zip(rhos_p, pvals_p)):
        label = f"{rho:+.3f}"
        if np.isfinite(pv):
            label += f"\np={pv:.3f}"
        offset = 0.03 if rho >= 0 else -0.07
        ax.text(xi, rho + offset, label, ha="center",
                va="bottom" if rho >= 0 else "top", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(domains_p, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Spearman ρ  (SHESHA vs CKA)", fontsize=11)
    ax.set_title("Per-domain & aggregate Spearman ρ  (95 % bootstrap CI)", fontsize=11)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    p = out_dir / "combined_domain_bar.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Bar chart     → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined cross-domain Shesha vs CKA analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_MERGED_CSV,
        help="Path to merged_aggregated_by_encoder.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for figures and summary CSVs",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main(args.input, args.out_dir)
