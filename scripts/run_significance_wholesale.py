"""
Significance tests for Wholesale (p=8) to narrow the dimension-effect threshold.

Existing results:
  Iris     (p=4):  MIGA wins Fr (p=0.001)
  Glass    (p=10): MICE wins Fr (p=1.000)
  Wholesale (p=8): ← this script determines which side of the threshold p=8 falls on

Usage (from the MIGA repo root):
    .venv/bin/python scripts/run_significance_wholesale.py

Saves results/13_significance_wholesale.json
"""

import sys, os, json, time, warnings
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from miga import MIGA
from miga.data_utils import apply_mar, auto_generators, compute_metrics, EXCLUDE_COLS, load_wholesale
from miga.fitness import FitnessEvaluator

N_SEEDS  = 5
SEEDS    = list(range(10, 60, 10))
L, G, Q  = 200, 300, 3
PCT      = 30
MISS_SEED = 30
EXCL     = EXCLUDE_COLS.get("Wholesale", [0, 1])
R        = float("inf")


def compute_fr(X_miss, X_imp):
    complete_mask = ~np.any(np.isnan(X_miss), axis=1)
    X_A = X_miss[complete_mask]
    X_C = X_imp[~complete_mask]
    if len(X_A) < 2 or len(X_C) < 2:
        return float("nan")
    try:
        return FitnessEvaluator(X_A, r=R).evaluate(X_C)
    except Exception:
        return float("nan")


def main():
    X, cols = load_wholesale()
    n, p = X.shape
    print(f"Wholesale  n={n}, p={p}, exclude={EXCL}")
    print(f"MIGA params: l={L}, G={G}, Q={Q}, {N_SEEDS} seeds\n")

    X_miss = apply_mar(X, pct=PCT, seed=MISS_SEED, exclude_cols=EXCL)
    missing_mask = np.isnan(X_miss)
    print(f"Missing cells: {missing_mask.sum()}")

    # baselines
    baselines = {}
    for name, imp in [
        ("Mean", SimpleImputer(strategy="mean")),
        ("KNN",  KNNImputer(n_neighbors=5)),
        ("MICE", IterativeImputer(max_iter=20, random_state=42, tol=1e-3)),
    ]:
        X_imp = imp.fit_transform(X_miss)
        m  = compute_metrics(X, X_imp, missing_mask)
        fr = compute_fr(X_miss, X_imp)
        baselines[name] = {"rmse": float(m["rmse"]), "fr": float(fr)}
        print(f"  {name:<5}  RMSE={m['rmse']:.4f}  Fr={fr:.4f}")

    # MIGA across seeds
    gens = auto_generators(X_miss, seed=42)
    miga_rmse, miga_fr = [], []

    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        miga = MIGA(l=L, G=G, Q=Q, r=R, seed=seed, verbose=False)
        X_imp = miga.fit_transform(X_miss.copy(), generators=gens)
        elapsed = time.time() - t0

        m  = compute_metrics(X, X_imp, missing_mask)
        fr = compute_fr(X_miss, X_imp)
        miga_rmse.append(float(m["rmse"]))
        miga_fr.append(float(fr))
        print(f"  seed={seed:3d}  RMSE={m['rmse']:.4f}  Fr={fr:.4f}  ({elapsed:.0f}s)")

    rmse_arr = np.array(miga_rmse)
    fr_arr   = np.array([f for f in miga_fr if np.isfinite(f)])

    print(f"\nMIGA  RMSE: {rmse_arr.mean():.4f} ± {rmse_arr.std():.4f}")
    print(f"MIGA  Fr:   {fr_arr.mean():.4f} ± {fr_arr.std():.4f}")

    # Wilcoxon vs MICE on Fr
    mice_fr = baselines["MICE"]["fr"]
    diffs   = fr_arr - mice_fr
    stat, p = stats.wilcoxon(diffs, alternative="less")
    miga_wins = bool(fr_arr.mean() < mice_fr)
    direction = "MIGA wins Fr" if miga_wins else "MICE wins Fr"
    print(f"\nWilcoxon MIGA Fr vs MICE Fr:")
    print(f"  MIGA={fr_arr.mean():.4f}  MICE={mice_fr:.4f}  p={p:.3f}  → {direction}")

    result = {
        "dataset": "Wholesale", "p": p, "n": n,
        "l": L, "G": G, "Q": Q, "n_seeds": N_SEEDS,
        "baselines": baselines,
        "miga_rmse": miga_rmse, "miga_fr": miga_fr,
        "miga_rmse_mean": float(rmse_arr.mean()), "miga_rmse_std": float(rmse_arr.std()),
        "miga_fr_mean":   float(fr_arr.mean()),   "miga_fr_std":   float(fr_arr.std()),
        "wilcoxon_vs_mice_fr": {"p": float(p), "miga_wins": miga_wins,
                                "miga_fr": float(fr_arr.mean()), "mice_fr": float(mice_fr)},
    }

    out_path = os.path.join(ROOT, "results", "13_significance_wholesale.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
