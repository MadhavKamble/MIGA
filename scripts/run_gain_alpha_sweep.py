"""
GAIN α-sweep: Fr–RMSE tradeoff as reconstruction weight varies.

Runs GAIN at α ∈ {0, 1, 10, 100, 1000, ∞} on 5 datasets.
α=0   → pure adversarial (distributional)
α=∞   → pure MSE reconstruction (MICE-like)
Also records MICE and MIGA as fixed reference points.

Datasets: Iris (p=4), Wine (p=13), Glass (p=10), Haberman (p=3), Wholesale (p=8)
Missing:  30% MAR, 5 seeds

Output: results/22_gain_alpha_sweep.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from miga.core import MIGA
from miga.data_utils import load_dataset, apply_mar
from miga.fitness import FitnessEvaluator
from miga.gain_numpy import GAIN_NP as GAIN

# ─── Config ────────────────────────────────────────────────────────────────

DATASETS   = ["Iris", "Wine", "Glass", "Haberman", "Wholesale"]
MISS_PCT   = 30
N_SEEDS    = 5
MIGA_G     = 200
MIGA_L     = 200
MIGA_Q     = 2

GAIN_ITERATIONS = 5000
GAIN_HINT_RATE  = 0.9
# α=inf represented as a very large number (pure MSE reconstruction)
ALPHA_VALUES = [0, 1, 10, 100, 1000, 10000]
ALPHA_LABELS = ["0", "1", "10", "100", "1000", "10000"]

OUT_FILE = ROOT / "results" / "22_gain_alpha_sweep.json"

# ─── Helpers ───────────────────────────────────────────────────────────────

def nrmse(X_true, X_imp, mask):
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


def compute_fr(X_full, X_missing, X_imputed):
    complete_mask = ~np.any(np.isnan(X_missing), axis=1)
    X_A = X_full[complete_mask]
    X_C_rows = ~complete_mask
    if len(X_A) == 0 or not X_C_rows.any():
        return float("nan")
    evaluator = FitnessEvaluator(X_A, r=np.inf)
    return float(evaluator.evaluate(X_imputed[X_C_rows]))


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        X_full, _ = load_dataset(ds_name)
        results[ds_name] = {"alphas": {}, "mice": {}, "miga": {}}

        # ── MICE reference ──────────────────────────────────────────────
        fr_list, rmse_list = [], []
        for seed in range(N_SEEDS):
            X_missing = apply_mar(X_full.copy(), MISS_PCT, seed=seed)
            miss_mask = np.isnan(X_missing)
            imp = IterativeImputer(random_state=seed, max_iter=10)
            X_imp = imp.fit_transform(X_missing)
            fr_list.append(compute_fr(X_full, X_missing, X_imp))
            rmse_list.append(nrmse(X_full, X_imp, miss_mask))
        results[ds_name]["mice"] = {
            "fr_mean": float(np.mean(fr_list)), "fr_std": float(np.std(fr_list)),
            "rmse_mean": float(np.mean(rmse_list)), "rmse_std": float(np.std(rmse_list)),
        }
        print(f"  MICE    Fr={results[ds_name]['mice']['fr_mean']:.4f}  "
              f"RMSE={results[ds_name]['mice']['rmse_mean']:.4f}", flush=True)

        # ── MIGA reference ──────────────────────────────────────────────
        fr_list, rmse_list = [], []
        for seed in range(N_SEEDS):
            X_missing = apply_mar(X_full.copy(), MISS_PCT, seed=seed)
            miss_mask = np.isnan(X_missing)
            miga = MIGA(l=MIGA_L, G=MIGA_G, Q=MIGA_Q, r=np.inf,
                        seed=seed, verbose=False)
            X_imp = miga.fit_transform(X_missing.copy())
            fr_list.append(compute_fr(X_full, X_missing, X_imp))
            rmse_list.append(nrmse(X_full, X_imp, miss_mask))
        results[ds_name]["miga"] = {
            "fr_mean": float(np.mean(fr_list)), "fr_std": float(np.std(fr_list)),
            "rmse_mean": float(np.mean(rmse_list)), "rmse_std": float(np.std(rmse_list)),
        }
        print(f"  MIGA    Fr={results[ds_name]['miga']['fr_mean']:.4f}  "
              f"RMSE={results[ds_name]['miga']['rmse_mean']:.4f}", flush=True)

        # ── GAIN α-sweep ────────────────────────────────────────────────
        for alpha, label in zip(ALPHA_VALUES, ALPHA_LABELS):
            fr_list, rmse_list = [], []
            t0 = time.time()
            for seed in range(N_SEEDS):
                X_missing = apply_mar(X_full.copy(), MISS_PCT, seed=seed)
                miss_mask = np.isnan(X_missing)
                imp = GAIN(
                    hint_rate=GAIN_HINT_RATE, alpha=alpha,
                    iterations=GAIN_ITERATIONS, seed=seed, verbose=False,
                )
                X_imp = imp.fit_transform(X_missing.copy())
                fr_list.append(compute_fr(X_full, X_missing, X_imp))
                rmse_list.append(nrmse(X_full, X_imp, miss_mask))
            wall = time.time() - t0
            results[ds_name]["alphas"][label] = {
                "alpha": alpha,
                "fr_mean": float(np.mean(fr_list)), "fr_std": float(np.std(fr_list)),
                "rmse_mean": float(np.mean(rmse_list)), "rmse_std": float(np.std(rmse_list)),
                "wall_sec": wall,
                "fr_list": fr_list, "rmse_list": rmse_list,
            }
            print(f"  α={label:>6s}  Fr={np.mean(fr_list):.4f}  "
                  f"RMSE={np.mean(rmse_list):.4f}  ({wall:.0f}s)", flush=True)

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
