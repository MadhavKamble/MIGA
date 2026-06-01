"""
Generate figures for GAIN vs MIGA vs MICE comparison.

Produces:
  fig8_gain_fr_rmse.png   — grouped bar chart: Fr and RMSE across 5 methods × 3 datasets
  fig8_gain_scatter.png   — Fr vs RMSE scatter (one point per method per dataset)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent.parent
IN_FILE = ROOT / "results" / "21_gain_comparison.json"
OUT_DIR = ROOT / "thesis_template" / "thesis_template_0.1" / "Figures"

DATASETS = ["Iris", "Haberman", "Wholesale"]
METHODS  = ["MEAN", "KNN", "MICE", "GAIN", "MIGA"]
LABELS   = {"MEAN": "Mean", "KNN": "KNN", "MICE": "MICE",
            "GAIN": "GAIN", "MIGA": "MIGA"}
COLORS   = {"MEAN": "#999999", "KNN": "#4DBBDF", "MICE": "#C44E52",
            "GAIN": "#8172B2", "MIGA": "#4C72B0"}
MARKERS  = {"MEAN": "s", "KNN": "^", "MICE": "D", "GAIN": "P", "MIGA": "o"}


def load():
    with open(IN_FILE) as f:
        return json.load(f)


def plot_bar(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(DATASETS))
    width = 0.14
    n = len(METHODS)
    offsets = np.linspace(-(n-1)/2 * width, (n-1)/2 * width, n)

    for ax, (metric_m, metric_s, title, ylabel) in zip(axes, [
        ("fr_mean",   "fr_std",   "$F_r$ (lower = better distributional fit)", "$F_r$"),
        ("rmse_mean", "rmse_std", "RMSE (lower = better accuracy)", "NRMSE"),
    ]):
        for i, m in enumerate(METHODS):
            means = [data[ds][m][metric_m] for ds in DATASETS]
            stds  = [data[ds][m][metric_s]  for ds in DATASETS]
            ax.bar(x + offsets[i], means, width, yerr=stds,
                   label=LABELS[m], color=COLORS[m], capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("GAIN vs MIGA vs MICE — 30% MAR, 5 seeds", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "fig8_gain_fr_rmse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_scatter(data):
    """Fr vs RMSE scatter: shows the Fr-RMSE tradeoff across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, ds in zip(axes, DATASETS):
        for m in METHODS:
            r = data[ds][m]
            ax.scatter(r["rmse_mean"], r["fr_mean"],
                       color=COLORS[m], marker=MARKERS[m],
                       s=120, label=LABELS[m], zorder=3)
            ax.errorbar(r["rmse_mean"], r["fr_mean"],
                        xerr=r["rmse_std"], yerr=r["fr_std"],
                        color=COLORS[m], alpha=0.4, fmt="none", capsize=3)
        ax.set_xlabel("RMSE (lower = accurate)")
        ax.set_ylabel("$F_r$ (lower = distributional)")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
        if ds == DATASETS[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Fr–RMSE tradeoff: all 5 methods (lower-left = best on both)\n"
                 "MIGA→low Fr; MICE/MEAN→low RMSE; GAIN→between",
                 fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "fig9_gain_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_table(data):
    print("\n── GAIN Comparison Table ───────────────────────────────────────────────")
    print(f"{'Dataset':12s} {'Method':8s} {'Fr mean':>9s} {'Fr std':>7s} {'RMSE mean':>10s}")
    print("─" * 55)
    for ds in DATASETS:
        for m in METHODS:
            r = data[ds][m]
            print(f"{ds:12s} {LABELS[m]:8s} {r['fr_mean']:9.4f} {r['fr_std']:7.4f} {r['rmse_mean']:10.4f}")
        print()


if __name__ == "__main__":
    data = load()
    print_table(data)
    plot_bar(data)
    plot_scatter(data)
    print("Done.")
