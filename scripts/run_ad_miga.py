"""
AD-MIGA Experiment — Adaptive Diversity Injection comparison.

Compares three variants:
  MIGA-STD  : standard MIGA, fixed 90% diversity injection (decay_mode='none')
  AD-MIGA-L : linear decay  (decay_mode='linear',      diversity_decay=1.0)
  AD-MIGA-E : exponential decay (decay_mode='exponential', decay_lambda=2.0)

Against MICE as the external baseline.

Metrics:
  - Final Fr  (mean ± std over N_SEEDS seeds)
  - Final RMSE
  - Convergence generation: first gen where Fr ≤ 1.10 × final Fr  (speed)
  - Fraction of diversity slots used at final generation (exploitation level)

Datasets: Iris, Haberman, Wholesale (MIGA-competitive), Glass (MIGA loses — sanity check)
Missing rate: 30% MCAR
Seeds: 10
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

# ─── Config ────────────────────────────────────────────────────────────────

DATASETS   = ["Iris", "Haberman", "Wholesale"]
MISS_PCT   = 30          # passed to apply_mar as integer percentage
N_SEEDS    = 5
G          = 300          # generations — enough to see convergence plateau
L          = 200
Q          = 2
DECAY_LAMBDA = 2.0        # exponential rate: frac=exp(-2g/G) → ~13.5% at g=G

OUT_FILE = ROOT / "results" / "20_ad_miga.json"

# ─── Helpers ───────────────────────────────────────────────────────────────

def nrmse(X_true, X_imputed, mask):
    """Range-normalised RMSE over masked (originally missing) cells."""
    errors = []
    for j in range(X_true.shape[1]):
        col_mask = mask[:, j]
        if not col_mask.any():
            continue
        true_vals = X_true[col_mask, j]
        imp_vals  = X_imputed[col_mask, j]
        rng = X_true[:, j].max() - X_true[:, j].min()
        if rng < 1e-10:
            continue
        errors.append(np.sqrt(np.mean((true_vals - imp_vals) ** 2)) / rng)
    return float(np.mean(errors)) if errors else float("nan")


def compute_fr(X_full, X_missing, X_imputed):
    """Compute Fr: X_A = complete rows under missingness pattern; X_C = imputed rows."""
    complete_mask = ~np.any(np.isnan(X_missing), axis=1)
    X_A = X_full[complete_mask]
    X_C_rows = ~complete_mask
    if len(X_A) == 0 or not X_C_rows.any():
        return float("nan")
    evaluator = FitnessEvaluator(X_A, r=np.inf)
    return float(evaluator.evaluate(X_imputed[X_C_rows]))


def convergence_gen(gen_history, final_fr, threshold=1.10):
    """First generation where Fr ≤ threshold × final_fr."""
    target = threshold * final_fr
    for g, fr in enumerate(gen_history):
        if fr <= target:
            return g
    return len(gen_history) - 1


def run_mice(X_missing, seed):
    imp = IterativeImputer(random_state=seed, max_iter=10)
    return imp.fit_transform(X_missing)


def run_variant(X_missing, variant, seed):
    """Run one MIGA variant; return (X_imputed, gen_history_best_run)."""
    kwargs = dict(l=L, G=G, Q=Q, r=np.inf, seed=seed, verbose=False)
    if variant == "STD":
        kwargs["decay_mode"] = "none"
    elif variant == "L":
        kwargs["decay_mode"] = "linear"
        kwargs["diversity_decay"] = 1.0
    elif variant == "E":
        kwargs["decay_mode"] = "exponential"
        kwargs["decay_lambda"] = DECAY_LAMBDA
    else:
        raise ValueError(variant)

    miga = MIGA(**kwargs)
    X_imp = miga.fit_transform(X_missing.copy())
    # Return gen history of the best run (lowest final Fr run)
    best_run_idx = int(np.argmin(miga.history_))
    return X_imp, miga.generation_history_[best_run_idx]


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        X_full, _ = load_dataset(ds_name)
        results[ds_name] = {}

        for variant in ["STD", "L", "E", "MICE"]:
            fr_list, rmse_list, conv_list = [], [], []
            t0 = time.time()

            for seed in range(N_SEEDS):
                X_missing = apply_mar(X_full.copy(), MISS_PCT, seed=seed)
                miss_mask = np.isnan(X_missing)

                if variant == "MICE":
                    X_imp = run_mice(X_missing, seed=seed)
                    gen_hist = []
                    conv = None
                else:
                    X_imp, gen_hist = run_variant(X_missing, variant, seed=seed)
                    final_fr = compute_fr(X_full, X_missing, X_imp)
                    conv = convergence_gen(gen_hist, final_fr) if gen_hist else None

                fr   = compute_fr(X_full, X_missing, X_imp)
                rmse = nrmse(X_full, X_imp, miss_mask)

                fr_list.append(fr)
                rmse_list.append(rmse)
                if conv is not None:
                    conv_list.append(conv)

                print(f"  {variant:5s}  seed={seed}  Fr={fr:.4f}  RMSE={rmse:.4f}"
                      + (f"  conv_gen={conv}" if conv is not None else ""))

            wall = time.time() - t0
            results[ds_name][variant] = {
                "fr_mean":   float(np.mean(fr_list)),
                "fr_std":    float(np.std(fr_list)),
                "rmse_mean": float(np.mean(rmse_list)),
                "rmse_std":  float(np.std(rmse_list)),
                "conv_gen_mean": float(np.mean(conv_list)) if conv_list else None,
                "conv_gen_std":  float(np.std(conv_list))  if conv_list else None,
                "wall_sec":  wall,
                "fr_list":   fr_list,
                "rmse_list": rmse_list,
                "conv_list": conv_list,
            }

        # Print summary table for this dataset
        print(f"\n  Summary — {ds_name}:")
        print(f"  {'Variant':8s}  {'Fr mean':>10s}  {'Fr std':>8s}  "
              f"{'RMSE mean':>10s}  {'Conv gen':>9s}")
        for v in ["STD", "L", "E", "MICE"]:
            r = results[ds_name][v]
            cg = f"{r['conv_gen_mean']:.1f}" if r["conv_gen_mean"] is not None else "  n/a"
            print(f"  {v:8s}  {r['fr_mean']:10.4f}  {r['fr_std']:8.4f}  "
                  f"{r['rmse_mean']:10.4f}  {cg:>9s}")

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
