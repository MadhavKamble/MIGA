"""
Plot GAIN α-sweep results: Fr vs RMSE trajectory per dataset.

Generates:
  fig10_gain_alpha_trajectory.png  — Fr vs RMSE scatter with α-sweep path
  fig11_gain_alpha_bars.png        — Fr and RMSE vs log(α) line plots
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

ROOT    = Path(__file__).parent.parent
IN_FILE = ROOT / "results" / "22_gain_alpha_sweep.json"
FIG_DIR = ROOT / "thesis_template" / "thesis_template_0.1" / "Figures"
FIG_DIR.mkdir(exist_ok=True)

DATASETS = ["Iris", "Wine", "Glass", "Haberman", "Wholesale"]
ALPHA_LABELS = ["0", "1", "10", "100", "1000", "10000"]
ALPHA_NUMS   = [0, 1, 10, 100, 1000, 10000]

with open(IN_FILE) as f:
    data = json.load(f)

# ── Fig 10: Fr vs RMSE trajectory (one panel per dataset) ──────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

cmap   = cm.plasma
alphas = np.array(ALPHA_NUMS, dtype=float)
# Replace 0 with 0.5 for log-colour mapping
alpha_log = np.where(alphas == 0, 0.5, alphas)
norm  = plt.Normalize(vmin=np.log10(0.5), vmax=np.log10(10000))

for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    ds_data = data[ds]

    fr_gain   = [ds_data["alphas"][a]["fr_mean"]   for a in ALPHA_LABELS]
    rmse_gain = [ds_data["alphas"][a]["rmse_mean"] for a in ALPHA_LABELS]
    fr_mice   = ds_data["mice"]["fr_mean"]
    rmse_mice = ds_data["mice"]["rmse_mean"]
    fr_miga   = ds_data["miga"]["fr_mean"]
    rmse_miga = ds_data["miga"]["rmse_mean"]

    # Draw path
    ax.plot(rmse_gain, fr_gain, color="grey", linewidth=1, zorder=1,
            linestyle="--", alpha=0.6)

    # Colour-coded α points
    for i, (rmse, fr, alph) in enumerate(zip(rmse_gain, fr_gain, alpha_log)):
        c = cmap(norm(np.log10(alph)))
        ax.scatter(rmse, fr, color=c, s=90, zorder=3,
                   edgecolors="black", linewidths=0.5)
        label = ALPHA_LABELS[i]
        ax.annotate(f"α={label}", (rmse, fr),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color="black")

    # Anchor points
    ax.scatter(rmse_mice, fr_mice, marker="^", color="steelblue",
               s=130, zorder=4, label="MICE", edgecolors="black", linewidths=0.7)
    ax.scatter(rmse_miga, fr_miga, marker="*", color="darkorange",
               s=180, zorder=4, label="MIGA", edgecolors="black", linewidths=0.7)

    # Ideal-point arrow hint
    ax.annotate("", xy=(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0,
                         ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0),
                xytext=(0.15, 0.15), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
    ax.text(0.03, 0.03, "ideal", transform=ax.transAxes,
            fontsize=7, color="green", va="bottom")

    ax.set_xlabel("RMSE", fontsize=10)
    ax.set_ylabel("$F_r$", fontsize=10)
    ax.set_title(ds, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

# Colourbar for α
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[:5], shrink=0.6, pad=0.02)
cbar.set_label("log$_{10}$(α)  [reconstruction weight]", fontsize=10)
cbar.set_ticks([np.log10(v) for v in [0.5, 1, 10, 100, 1000, 10000]])
cbar.set_ticklabels(["0", "1", "10", "100", "1000", "10000"])

axes[5].set_visible(False)

fig.suptitle("GAIN α-sweep: $F_r$ vs RMSE trajectory\n"
             "MICE (▲) and MIGA (★) are fixed reference anchors",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out10 = FIG_DIR / "fig10_gain_alpha_trajectory.png"
plt.savefig(out10, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out10}")

# ── Fig 11: Fr and RMSE vs α (line plots, one panel per dataset) ───────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

x_ticks  = list(range(len(ALPHA_LABELS)))
x_labels = ALPHA_LABELS

for idx, ds in enumerate(DATASETS):
    ax  = axes[idx]
    ax2 = ax.twinx()
    ds_data = data[ds]

    fr_gain   = [ds_data["alphas"][a]["fr_mean"]   for a in ALPHA_LABELS]
    rmse_gain = [ds_data["alphas"][a]["rmse_mean"] for a in ALPHA_LABELS]
    fr_std    = [ds_data["alphas"][a]["fr_std"]    for a in ALPHA_LABELS]
    rmse_std  = [ds_data["alphas"][a]["rmse_std"]  for a in ALPHA_LABELS]
    fr_mice   = ds_data["mice"]["fr_mean"]
    rmse_mice = ds_data["mice"]["rmse_mean"]
    fr_miga   = ds_data["miga"]["fr_mean"]
    rmse_miga = ds_data["miga"]["rmse_mean"]

    l1, = ax.plot(x_ticks, fr_gain, "o-", color="darkorange", label="GAIN $F_r$")
    ax.fill_between(x_ticks,
                    [f - s for f, s in zip(fr_gain, fr_std)],
                    [f + s for f, s in zip(fr_gain, fr_std)],
                    color="darkorange", alpha=0.15)
    ax.axhline(fr_mice, color="steelblue", linestyle="--", linewidth=1.2,
               label="MICE $F_r$")
    ax.axhline(fr_miga, color="darkorange", linestyle=":", linewidth=1.5,
               label="MIGA $F_r$")

    l2, = ax2.plot(x_ticks, rmse_gain, "s--", color="forestgreen", label="GAIN RMSE")
    ax2.fill_between(x_ticks,
                     [r - s for r, s in zip(rmse_gain, rmse_std)],
                     [r + s for r, s in zip(rmse_gain, rmse_std)],
                     color="forestgreen", alpha=0.15)
    ax2.axhline(rmse_mice, color="steelblue", linestyle="-.", linewidth=1.2,
                label="MICE RMSE")
    ax2.axhline(rmse_miga, color="purple", linestyle=":", linewidth=1.2,
                label="MIGA RMSE")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel("α  (reconstruction weight)", fontsize=9)
    ax.set_ylabel("$F_r$", color="darkorange", fontsize=10)
    ax2.set_ylabel("RMSE", color="forestgreen", fontsize=10)
    ax.set_title(ds, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    lines  = [l1, l2]
    labels = ["GAIN $F_r$", "GAIN RMSE"]
    ax.legend(lines, labels, fontsize=7, loc="upper right")

axes[5].set_visible(False)
fig.suptitle("GAIN α-sweep: $F_r$ and RMSE vs reconstruction weight\n"
             "Dashed/dotted lines = MICE and MIGA reference values",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out11 = FIG_DIR / "fig11_gain_alpha_lines.png"
plt.savefig(out11, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out11}")
