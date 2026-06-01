"""
Generate AD-MIGA comparison figures from results/20_ad_miga.json.

Produces:
  fig6_ad_miga_fr.png   — grouped bar chart: Fr mean ± std across 4 datasets × 4 variants
  fig6_ad_miga_rmse.png — same for RMSE
  fig6_ad_miga_conv.png — convergence generation bar chart (speed comparison, 3 MIGA variants)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).parent.parent
IN_FILE  = ROOT / "results" / "20_ad_miga.json"
OUT_DIR  = ROOT / "thesis_template" / "thesis_template_0.1" / "Figures"

DATASETS  = ["Iris", "Haberman", "Wholesale"]
VARIANTS  = ["STD", "L", "E", "MICE"]
LABELS    = {"STD": "MIGA", "L": "AD-MIGA-L", "E": "AD-MIGA-E", "MICE": "MICE"}
COLORS    = {"STD": "#4C72B0", "L": "#DD8452", "E": "#55A868", "MICE": "#C44E52"}

def load():
    with open(IN_FILE) as f:
        return json.load(f)

def bar_group(ax, data, metric_mean, metric_std, title, ylabel):
    x = np.arange(len(DATASETS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(VARIANTS))
    for i, v in enumerate(VARIANTS):
        means = [data[ds][v][metric_mean] for ds in DATASETS]
        stds  = [data[ds][v][metric_std]  for ds in DATASETS]
        ax.bar(x + offsets[i], means, width, yerr=stds,
               label=LABELS[v], color=COLORS[v], capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

def plot_fr_rmse(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bar_group(axes[0], data, "fr_mean",   "fr_std",   "$F_r$ (lower = better)", "$F_r$")
    bar_group(axes[1], data, "rmse_mean", "rmse_std", "RMSE (lower = better)", "NRMSE")
    fig.suptitle("AD-MIGA vs MIGA vs MICE — 30% MAR, 10 seeds", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "fig6_ad_miga.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def plot_convergence(data):
    """Bar chart of convergence generation for STD, L, E (MICE has no gen history)."""
    miga_variants = ["STD", "L", "E"]
    miga_labels   = [LABELS[v] for v in miga_variants]
    miga_colors   = [COLORS[v] for v in miga_variants]

    x = np.arange(len(DATASETS))
    width = 0.22
    offsets = np.linspace(-width, width, len(miga_variants))

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, v in enumerate(miga_variants):
        means = [data[ds][v]["conv_gen_mean"] or 0 for ds in DATASETS]
        stds  = [data[ds][v]["conv_gen_std"]  or 0 for ds in DATASETS]
        ax.bar(x + offsets[i], means, width, yerr=stds,
               label=miga_labels[i], color=miga_colors[i], capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Convergence generation (lower = faster)")
    ax.set_title("Convergence speed: generations to reach 110% of final $F_r$")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "fig7_ad_miga_conv.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def print_table(data):
    print("\n── AD-MIGA Results Table ──────────────────────────────────────────")
    print(f"{'Dataset':12s} {'Variant':12s} {'Fr mean':>9s} {'Fr std':>7s} "
          f"{'RMSE':>8s} {'Conv gen':>9s}")
    print("─" * 65)
    for ds in DATASETS:
        for v in VARIANTS:
            r = data[ds][v]
            cg = f"{r['conv_gen_mean']:.1f}" if r["conv_gen_mean"] is not None else "  n/a"
            print(f"{ds:12s} {LABELS[v]:12s} {r['fr_mean']:9.4f} {r['fr_std']:7.4f} "
                  f"{r['rmse_mean']:8.4f} {cg:>9s}")
        print()

if __name__ == "__main__":
    data = load()
    print_table(data)
    plot_fr_rmse(data)
    plot_convergence(data)
    print("Done.")
