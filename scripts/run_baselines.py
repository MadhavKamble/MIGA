"""
Run Mean, KNN, and MICE baseline imputation on all benchmark datasets.
Also computes Fr (MIGA's fitness function) for each baseline — this is the
key comparison that shows MIGA's actual win domain.

Usage (from the MIGA repo root):
    .venv/bin/python scripts/run_baselines.py

Saves results/12_baselines.json with RMSE/MAD/CoD/Fr for each
method × dataset × pct.  Notebook 12 loads this alongside the existing
MIGA JSONs for the full comparison.
"""

import sys, os, json, time, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from miga.data_utils import (
    apply_mar, compute_metrics, EXCLUDE_COLS,
    load_iris, load_wine, load_glass, load_haberman,
    load_wholesale, load_cardio, load_adult,
)
from miga.fitness import FitnessEvaluator
from miga.paper_results import TABLE3_PARAMS

PERCENTAGES = [30, 40, 50, 60]

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "Iris":      load_iris,
    "Wine":      load_wine,
    "Glass":     load_glass,
    "Haberman":  load_haberman,
    "Wholesale": load_wholesale,
    "Cardio":    load_cardio,
    "Adult":     load_adult,
}

# ── Baseline methods ──────────────────────────────────────────────────────────

def get_methods():
    return {
        "Mean": SimpleImputer(strategy="mean"),
        "KNN":  KNNImputer(n_neighbors=5),
        "MICE": IterativeImputer(max_iter=20, random_state=42, tol=1e-3),
    }

# ── Fr evaluation for a completed matrix ──────────────────────────────────────

def compute_fr(X_miss, X_imp, r):
    """Compute MIGA fitness Fr for an imputed matrix."""
    complete_mask = ~np.any(np.isnan(X_miss), axis=1)
    incomplete_mask = np.any(np.isnan(X_miss), axis=1)
    X_A = X_miss[complete_mask]
    X_C = X_imp[incomplete_mask]
    if len(X_A) < 2 or len(X_C) < 2:
        return {"F_r": float("nan"), "d_means": float("nan"),
                "d_cov": float("nan"), "d_skew": float("nan")}
    try:
        ev = FitnessEvaluator(X_A, r=r)
        return ev.decompose(X_C)
    except Exception:
        return {"F_r": float("nan"), "d_means": float("nan"),
                "d_cov": float("nan"), "d_skew": float("nan")}

# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    out = {}

    for ds_name, loader_fn in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            X, cols = loader_fn()
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            continue

        n, p = X.shape
        excl = EXCLUDE_COLS.get(ds_name, [])
        r = TABLE3_PARAMS[ds_name]["r"]
        out[ds_name] = {"n": n, "p": p, "cols": cols, "r": float(r)}

        for pct in PERCENTAGES:
            X_miss = apply_mar(X, pct=pct, seed=pct, exclude_cols=excl)
            missing_mask = np.isnan(X_miss)
            n_miss = int(missing_mask.sum())
            print(f"\n  {pct}% missing — {n_miss} cells")
            print(f"  {'Method':<6}  {'RMSE':>8}  {'MAD':>8}  {'CoD':>7}  {'Fr':>10}  {'d_means':>9}  {'d_cov':>9}  {'d_skew':>9}")
            print(f"  {'-'*75}")

            out[ds_name][pct] = {}

            for method_name, imputer in get_methods().items():
                t0 = time.time()
                try:
                    X_imp = imputer.fit_transform(X_miss)
                except Exception as e:
                    print(f"    {method_name:6s}: ERROR — {e}")
                    out[ds_name][pct][method_name] = {
                        "rmse": None, "mad": None, "cod": None,
                        "F_r": None, "d_means": None, "d_cov": None, "d_skew": None,
                    }
                    continue
                elapsed = time.time() - t0

                m  = compute_metrics(X, X_imp, missing_mask)
                fr = compute_fr(X_miss, X_imp, r)

                print(f"  {method_name:<6}  {m['rmse']:>8.4f}  {m['mad']:>8.4f}  "
                      f"{m['cod']:>7.4f}  {fr['F_r']:>10.4f}  "
                      f"{fr['d_means']:>9.4f}  {fr['d_cov']:>9.4f}  "
                      f"{fr['d_skew']:>9.4f}  ({elapsed:.1f}s)")

                out[ds_name][pct][method_name] = {
                    "rmse":   m["rmse"],
                    "mad":    m["mad"],
                    "cod":    m["cod"],
                    "F_r":    fr["F_r"],
                    "d_means": fr["d_means"],
                    "d_cov":  fr["d_cov"],
                    "d_skew": fr["d_skew"],
                }

    out_path = os.path.join(ROOT, "results", "12_baselines.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    run_all()
