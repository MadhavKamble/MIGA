"""
Generate convergence plots: Fixed c3 vs Adaptive c3 (15→3) vs Adaptive c3 (3→15).

Runs MIGA on Iris, Glass, Haberman at 30% MAR with G=200, Q=5, seed=42.
Saves results/convergence_data.json and figures/convergence.png.

Usage:
    .venv/bin/python scripts/plot_convergence.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from miga import MIGA
from miga.data_utils import apply_mar, auto_generators, load_iris, load_glass, load_haberman

os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

DATASETS = {
    "Iris":     (load_iris,     30, []),
    "Glass":    (load_glass,    30, [9]),
    "Haberman": (load_haberman, 30, []),
}

G    = 200
Q    = 5
L    = 200
SEED = 42

CONFIGS = [
    ("Fixed c3=5",      dict(c3=5)),
    ("Adaptive 15→3",   dict(c3=5, c3_schedule=(15, 3))),
    ("Adaptive 3→15",   dict(c3=5, c3_schedule=(3, 15))),
]
COLORS = ["#2196F3", "#F44336", "#4CAF50"]
STYLES = ["-", "--", ":"]

all_data = {}

for ds_name, (loader, pct, excl) in DATASETS.items():
    print(f"\n=== {ds_name} ===")
    X, _ = loader()
    X_miss = apply_mar(X, pct=pct, seed=30, exclude_cols=excl)
    gens = auto_generators(X_miss)

    ds_data = {}
    for label, kwargs in CONFIGS:
        miga = MIGA(l=L, G=G, Q=Q, seed=SEED, verbose=False, **kwargs)
        miga.fit_transform(X_miss.copy(), generators=gens)
        # generation_history_: list of Q lists, each of length G (best Fr per gen)
        gh = miga.generation_history_   # Q × G
        mean_curve = np.mean(gh, axis=0).tolist()
        std_curve  = np.std(gh,  axis=0).tolist()
        ds_data[label] = {"mean": mean_curve, "std": std_curve,
                          "final_fr": mean_curve[-1]}
        print(f"  {label:<20}  final Fr={mean_curve[-1]:.4f}")

    all_data[ds_name] = ds_data

# ── Save data ──────────────────────────────────────────────────────────────────
with open(os.path.join(ROOT, "results", "convergence_data.json"), "w") as f:
    json.dump(all_data, f, indent=2)
print("\nSaved → results/convergence_data.json")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
gens_x = np.arange(1, G + 1)

for ax, (ds_name, ds_data) in zip(axes, all_data.items()):
    for (label, _), color, ls in zip(CONFIGS, COLORS, STYLES):
        curve = ds_data[label]
        mean  = np.array(curve["mean"])
        std   = np.array(curve["std"])
        ax.plot(gens_x, mean, color=color, ls=ls, lw=1.8, label=label)
        ax.fill_between(gens_x, mean - std, mean + std,
                        color=color, alpha=0.12)

    ax.set_title(ds_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Fr (best in population)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle(
    f"Convergence: Fixed vs Adaptive c3 mutation schedule\n"
    f"(l={L}, G={G}, Q={Q}, 30% MAR, mean ± 1 std over {Q} runs)",
    fontsize=11, y=1.02
)
fig.tight_layout()
out_fig = os.path.join(ROOT, "figures", "convergence.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"Saved → figures/convergence.png")
