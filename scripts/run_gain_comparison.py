"""
GAIN vs MIGA vs MICE: Universal Fr-RMSE Orthogonality Comparison.

Tests whether neural imputers trained with reconstruction loss (GAIN)
also exhibit Fr > 0, extending the Fr-RMSE orthogonality theorem (C2)
beyond MICE to neural methods.

Methods compared:
  MEAN   — column mean imputation (pure MSE baseline)
  KNN    — 5-NN weighted imputation
  MICE   — iterative conditional regression (conditional mean → Fr > 0)
  GAIN   — GAN imputer trained with MSE + adversarial loss
  MIGA   — distributional GA imputer (explicit Fr minimisation)

Datasets: Iris (p=4), Haberman (p=3), Wholesale (p=8)
Missing:  30% MAR, 5 seeds
Metrics:  Fr (lower = better distributional fit), RMSE (lower = better accuracy)

Output: results/21_gain_comparison.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from miga.core import MIGA
from miga.data_utils import load_dataset, apply_mar
from miga.fitness import FitnessEvaluator
from miga.gain_numpy import GAIN_NP as GAIN

# ─── Config ────────────────────────────────────────────────────────────────

DATASETS   = ["Iris", "Haberman", "Wholesale"]
MISS_PCT   = 30
N_SEEDS    = 5
MIGA_G     = 200
MIGA_L     = 200
MIGA_Q     = 2

GAIN_ITERATIONS = 5000     # network converges well before 10000
GAIN_ALPHA      = 100      # paper default: reconstruction weight
GAIN_HINT_RATE  = 0.9      # paper default

OUT_FILE = ROOT / "results" / "21_gain_comparison.json"

# ─── Helpers ───────────────────────────────────────────────────────────────

def nrmse(X_true: np.ndarray, X_imp: np.ndarray, mask: np.ndarray) -> float:
    errors = []
    for j in range(X_true.shape[1]):
        col_mask = mask[:, j]
        if not col_mask.any():
            continue
        true_vals = X_true[col_mask, j]
        imp_vals  = X_imp[col_mask, j]
        rng = X_true[:, j].max() - X_true[:, j].min()
        if rng < 1e-10:
            continue
        errors.append(np.sqrt(np.mean((true_vals - imp_vals) ** 2)) / rng)
    return float(np.mean(errors)) if errors else float("nan")


def compute_fr(X_full: np.ndarray, X_missing: np.ndarray,
               X_imputed: np.ndarray) -> float:
    complete_mask = ~np.any(np.isnan(X_missing), axis=1)
    X_A = X_full[complete_mask]
    X_C_rows = ~complete_mask
    if len(X_A) == 0 or not X_C_rows.any():
        return float("nan")
    evaluator = FitnessEvaluator(X_A, r=np.inf)
    return float(evaluator.evaluate(X_imputed[X_C_rows]))


def run_method(name: str, X_missing: np.ndarray, seed: int) -> np.ndarray:
    if name == "MEAN":
        imp = SimpleImputer(strategy="mean")
        return imp.fit_transform(X_missing)
    elif name == "KNN":
        imp = KNNImputer(n_neighbors=5)
        return imp.fit_transform(X_missing)
    elif name == "MICE":
        imp = IterativeImputer(random_state=seed, max_iter=10)
        return imp.fit_transform(X_missing)
    elif name == "GAIN":
        imp = GAIN(
            hint_rate=GAIN_HINT_RATE, alpha=GAIN_ALPHA,
            iterations=GAIN_ITERATIONS, seed=seed, verbose=False,
        )
        return imp.fit_transform(X_missing.copy())
    elif name == "MIGA":
        miga = MIGA(l=MIGA_L, G=MIGA_G, Q=MIGA_Q, r=np.inf,
                    seed=seed, verbose=False)
        return miga.fit_transform(X_missing.copy())
    else:
        raise ValueError(f"Unknown method: {name}")


# ─── Main ──────────────────────────────────────────────────────────────────

METHODS = ["MEAN", "KNN", "MICE", "GAIN", "MIGA"]

def main():
    results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        X_full, _ = load_dataset(ds_name)
        results[ds_name] = {}

        for method in METHODS:
            fr_list, rmse_list = [], []
            t0 = time.time()

            for seed in range(N_SEEDS):
                X_missing = apply_mar(X_full.copy(), MISS_PCT, seed=seed)
                miss_mask = np.isnan(X_missing)

                X_imp = run_method(method, X_missing, seed)
                fr   = compute_fr(X_full, X_missing, X_imp)
                rmse = nrmse(X_full, X_imp, miss_mask)

                fr_list.append(fr)
                rmse_list.append(rmse)
                print(f"  {method:6s}  seed={seed}  Fr={fr:.4f}  RMSE={rmse:.4f}")

            wall = time.time() - t0
            results[ds_name][method] = {
                "fr_mean":   float(np.mean(fr_list)),
                "fr_std":    float(np.std(fr_list)),
                "rmse_mean": float(np.mean(rmse_list)),
                "rmse_std":  float(np.std(rmse_list)),
                "wall_sec":  wall,
                "fr_list":   fr_list,
                "rmse_list": rmse_list,
            }

        # Summary table
        print(f"\n  Summary — {ds_name}:")
        print(f"  {'Method':8s}  {'Fr mean':>9s}  {'Fr std':>7s}  {'RMSE mean':>10s}")
        for m in METHODS:
            r = results[ds_name][m]
            print(f"  {m:8s}  {r['fr_mean']:9.4f}  {r['fr_std']:7.4f}  {r['rmse_mean']:10.4f}")

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
