"""
Variance preservation analysis: does MIGA preserve marginal variances better than MICE?

Usage:
    .venv/bin/python scripts/run_variance_preservation.py [Dataset] [pct]

Examples:
    .venv/bin/python scripts/run_variance_preservation.py Wine 30
    .venv/bin/python scripts/run_variance_preservation.py Iris 40

Saves results/14_variance_<dataset>_<pct>.json. Skips if already exists.
Omit both args to run all datasets at 30%.
"""

import sys, os, json, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from miga import MIGA
from miga.data_utils import (
    apply_mar, auto_generators, EXCLUDE_COLS,
    load_iris, load_wine, load_glass, load_haberman, load_wholesale,
)
from miga.paper_results import TABLE3_PARAMS

DATASETS = {
    "Iris":      (load_iris,      30, []),
    "Wine":      (load_wine,      30, []),
    "Glass":     (load_glass,     30, [9]),
    "Haberman":  (load_haberman,  30, []),
    "Wholesale": (load_wholesale, 30, [0, 1]),
}

L, G, Q = 200, 300, 5
SEED = 42

DS_ARG  = sys.argv[1] if len(sys.argv) > 1 else None
PCT_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 30

if DS_ARG and DS_ARG not in DATASETS:
    print(f"Unknown dataset '{DS_ARG}'. Choose from: {list(DATASETS)}")
    sys.exit(1)


def variance_ratio(X_true, X_imp, missing_mask):
    """
    Per-feature variance ratio: Var(imputed) / Var(true).
    Closer to 1.0 = better variance preservation.
    Only for features that had missing values.
    """
    ratios = {}
    n, p = X_true.shape
    for j in range(p):
        if not missing_mask[:, j].any():
            continue
        var_true = np.var(X_true[:, j], ddof=1)
        var_imp  = np.var(X_imp[:, j],  ddof=1)
        if var_true > 1e-10:
            ratios[j] = float(var_imp / var_true)
    return ratios


def _already_done(ds, pct):
    out_path = os.path.join(ROOT, "results", f"14_variance_{ds.lower()}_{pct}.json")
    if os.path.exists(out_path):
        return True, out_path
    if pct == 30:
        old_path = os.path.join(ROOT, "results", "14_variance_preservation.json")
        try:
            import json as _j
            existing = _j.load(open(old_path))
            if ds in existing:
                return True, old_path
        except Exception:
            pass
    return False, out_path


def run_all():
    out = {}
    if DS_ARG:
        done, path = _already_done(DS_ARG, PCT_ARG)
        if done:
            print(f"Already exists, skipping: {path}")
            return
        datasets_to_run = {DS_ARG: (DATASETS[DS_ARG][0], PCT_ARG, DATASETS[DS_ARG][2])}
    else:
        datasets_to_run = {k: (fn, PCT_ARG, excl) for k, (fn, _, excl) in DATASETS.items()}

    for ds_name, (loader_fn, pct, excl) in datasets_to_run.items():
        print(f"\n{'='*55}")
        print(f"Dataset: {ds_name}  ({pct}% missing)")
        print(f"{'='*55}")

        X, cols = loader_fn()
        n, p = X.shape
        X_miss = apply_mar(X, pct=pct, seed=30, exclude_cols=excl)
        missing_mask = np.isnan(X_miss)
        miss_cols = [j for j in range(p) if missing_mask[:, j].any()]
        gens = auto_generators(X_miss, seed=SEED)

        # True variance (reference)
        true_var = {j: float(np.var(X[:, j], ddof=1)) for j in miss_cols}

        print(f"  True variance (reference): "
              f"{', '.join(f'{cols[j]}={true_var[j]:.3f}' for j in miss_cols)}")

        results = {}

        # Baselines
        for bl_name, imp in [
            ("Mean", SimpleImputer(strategy="mean")),
            ("KNN",  KNNImputer(n_neighbors=5)),
            ("MICE", IterativeImputer(max_iter=20, random_state=42)),
        ]:
            X_imp = imp.fit_transform(X_miss)
            ratios = variance_ratio(X, X_imp, missing_mask)
            mean_ratio = float(np.mean(list(ratios.values())))
            print(f"  {bl_name:<5}  var ratios: "
                  f"{', '.join(f'{cols[j]}={ratios[j]:.3f}' for j in miss_cols)}"
                  f"  → mean={mean_ratio:.3f}")
            results[bl_name] = {"ratios": {cols[j]: ratios[j] for j in miss_cols},
                                 "mean_ratio": mean_ratio}

        # MIGA
        miga = MIGA(l=L, G=G, Q=Q, seed=SEED, verbose=False)
        X_imp_miga = miga.fit_transform(X_miss.copy(), generators=gens)
        ratios_miga = variance_ratio(X, X_imp_miga, missing_mask)
        mean_ratio_miga = float(np.mean(list(ratios_miga.values())))
        print(f"  {'MIGA':<5}  var ratios: "
              f"{', '.join(f'{cols[j]}={ratios_miga[j]:.3f}' for j in miss_cols)}"
              f"  → mean={mean_ratio_miga:.3f}")
        results["MIGA"] = {"ratios": {cols[j]: ratios_miga[j] for j in miss_cols},
                           "mean_ratio": mean_ratio_miga}

        # Summary: how close to 1.0 is each method?
        print(f"\n  |ratio - 1.0| (lower = better variance preservation):")
        for method, r in results.items():
            dev = abs(r["mean_ratio"] - 1.0)
            best = "★" if all(
                abs(r["mean_ratio"] - 1.0) <= abs(results[m]["mean_ratio"] - 1.0)
                for m in results
            ) else ""
            print(f"    {method:<5}: {dev:.4f} {best}")

        out[ds_name] = {
            "n": n, "p": p, "pct": pct, "cols": cols,
            "miss_cols": [cols[j] for j in miss_cols],
            "true_var": {cols[j]: true_var[j] for j in miss_cols},
            **results,
        }

    if DS_ARG:
        out_path = os.path.join(ROOT, "results",
                                f"14_variance_{DS_ARG.lower()}_{PCT_ARG}.json")
        old_path = os.path.join(ROOT, "results", "14_variance_preservation.json")
        if os.path.exists(out_path):
            print(f"Already exists, skipping: {out_path}")
            return
        # For 30%, also check the old combined file
        if PCT_ARG == 30:
            try:
                import json as _json
                existing = _json.load(open(old_path))
                if DS_ARG in existing:
                    print(f"Already exists in old combined file, skipping: {DS_ARG} @ 30%")
                    return
            except Exception:
                pass
    else:
        out_path = os.path.join(ROOT, "results", "14_variance_preservation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    run_all()
