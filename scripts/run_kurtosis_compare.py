"""
Kurtosis extension comparison: Fr  vs  Fr+  (Fr with 4th-moment kurtosis term).

Usage (from MIGA repo root):
    .venv/bin/python scripts/run_kurtosis_compare.py <Dataset> [pct]

Examples:
    .venv/bin/python scripts/run_kurtosis_compare.py Wine 30
    .venv/bin/python scripts/run_kurtosis_compare.py Iris 40

Saves results/15_kurtosis_<dataset>_<pct>.json. Skips if already exists.
"""

import sys, os, json, time, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from scipy.stats import kurtosis as sp_kurtosis

from miga import MIGA
from miga.data_utils import (
    apply_mar, auto_generators, compute_metrics, EXCLUDE_COLS,
    load_iris, load_wine, load_glass, load_haberman, load_wholesale,
)
from miga.fitness import FitnessEvaluator
from miga.statistics import compute_kurtosis
from miga.paper_results import TABLE3_PARAMS

# ── Configuration ──────────────────────────────────────────────────────────────

LOADERS = {
    "Iris":      (load_iris,      {"pct": 30, "miss_seed": 30, "exclude_cols": []}),
    "Wine":      (load_wine,      {"pct": 30, "miss_seed": 30, "exclude_cols": []}),
    "Glass":     (load_glass,     {"pct": 30, "miss_seed": 30, "exclude_cols": [9]}),
    "Haberman":  (load_haberman,  {"pct": 30, "miss_seed": 30, "exclude_cols": []}),
    "Wholesale": (load_wholesale, {"pct": 30, "miss_seed": 30, "exclude_cols": [0, 1]}),
}

MIGA_SEEDS  = list(range(10, 110, 10))   # 10 seeds: 10, 20, …, 100
L, G, Q     = 200, 300, 3

if len(sys.argv) < 2:
    print("Usage: python scripts/run_kurtosis_compare.py <Dataset> [pct]")
    print(f"       datasets: {list(LOADERS)}")
    sys.exit(1)

ds_name = sys.argv[1]
pct_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 30

if ds_name not in LOADERS:
    print(f"Unknown dataset '{ds_name}'. Choose from: {list(LOADERS)}")
    sys.exit(1)

loader_fn, loader_kwargs = LOADERS[ds_name]
loader_kwargs = dict(loader_kwargs, pct=pct_arg)   # override pct
r_val = TABLE3_PARAMS.get(ds_name, {}).get("r", np.inf)

results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, f"15_kurtosis_{ds_name.lower()}_{pct_arg}.json")
old_path = os.path.join(results_dir, f"15_kurtosis_{ds_name.lower()}.json")

if os.path.exists(out_path):
    print(f"Already exists, skipping: {out_path}")
    sys.exit(0)
if pct_arg == 30 and os.path.exists(old_path):
    print(f"Already exists (old format), skipping: {old_path}")
    sys.exit(0)

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_kurt_per_feature(X_miss, X_imp):
    """Excess kurtosis of imputed values for each missing feature."""
    complete_mask    = ~np.any(np.isnan(X_miss), axis=1)
    incomplete_mask  = ~complete_mask
    X_C_imp = X_imp[incomplete_mask]
    scale = np.maximum(np.std(X_imp[complete_mask], axis=0, ddof=1), 1e-8)
    return compute_kurtosis(X_C_imp / scale).tolist()

def compute_fr_both(X_miss, X_imp, r):
    """Return Fr (3-term) and Fr+ (4-term) for a given imputed matrix."""
    complete_mask   = ~np.any(np.isnan(X_miss), axis=1)
    incomplete_mask = ~complete_mask
    X_A = X_miss[complete_mask]
    X_C = X_imp[incomplete_mask]
    try:
        ev3 = FitnessEvaluator(X_A, r=r, use_kurtosis=False)
        ev4 = FitnessEvaluator(X_A, r=r, use_kurtosis=True)
        dec3 = ev3.decompose(X_C)
        dec4 = ev4.decompose(X_C)
        return dec3, dec4
    except Exception as e:
        nan = float("nan")
        empty = {"F_r": nan, "d_means": nan, "d_cov": nan, "d_skew": nan}
        return empty, {**empty, "d_kurt": nan}

def baseline_imputations(X_miss):
    results = {}
    for name, imp in [
        ("Mean", SimpleImputer(strategy="mean")),
        ("KNN",  KNNImputer(n_neighbors=5)),
        ("MICE", IterativeImputer(random_state=0, max_iter=10)),
    ]:
        X_imp = imp.fit_transform(X_miss)
        results[name] = X_imp
    return results

# ── Main ───────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Kurtosis extension: Fr vs Fr+  |  Dataset: {ds_name}  |  30% MAR")
print(f"{'='*60}\n")

X_true, _ = loader_fn()
exclude_cols = loader_kwargs["exclude_cols"]
X_miss = apply_mar(X_true, pct=loader_kwargs["pct"],
                   seed=loader_kwargs["miss_seed"],
                   exclude_cols=exclude_cols)
missing_mask = np.isnan(X_miss)

n, p = X_true.shape
print(f"  n={n}, p={p}, missing cells={int(np.sum(np.isnan(X_miss)))}")

# X_A kurtosis (reference)
complete_mask = ~np.any(np.isnan(X_miss), axis=1)
X_A = X_miss[complete_mask]
scale_A = np.maximum(np.std(X_A, axis=0, ddof=1), 1e-8)
kurt_A = compute_kurtosis(X_A / scale_A).tolist()
print(f"  X_A kurtosis (reference): {[round(k,3) for k in kurt_A]}\n")

# ── Baselines ──────────────────────────────────────────────────────────────────
print("--- Baselines ---")
bl_imps = baseline_imputations(X_miss)
bl_results = {}
for bl_name, X_imp in bl_imps.items():
    metrics = compute_metrics(X_true, X_imp, missing_mask)
    dec3, dec4 = compute_fr_both(X_miss, X_imp, r_val)
    kurt_imp   = get_kurt_per_feature(X_miss, X_imp)
    bl_results[bl_name] = {
        "rmse": metrics["rmse"],
        "Fr":   dec3["F_r"],
        "Fr+":  dec4["F_r"],
        "d_kurt": dec4["d_kurt"],
        "kurt_C": kurt_imp,
    }
    print(f"  {bl_name:<5}  RMSE={metrics['rmse']:.4f}  "
          f"Fr={dec3['F_r']:.4f}  d_kurt={dec4['d_kurt']:.4f}  Fr+={dec4['F_r']:.4f}")

# ── MIGA Fr (3-term) ───────────────────────────────────────────────────────────
print("\n--- MIGA  Fr  (standard, 3 terms) ---")
gens = auto_generators(X_miss)
fr3_rmse, fr3_fr, fr3_frplus, fr3_dkurt = [], [], [], []

for seed in MIGA_SEEDS:
    t0 = time.time()
    miga = MIGA(l=L, G=G, Q=Q, r=r_val, use_kurtosis=False, seed=seed, verbose=False)
    X_imp = miga.fit_transform(X_miss, generators=gens)
    metrics = compute_metrics(X_true, X_imp, missing_mask)
    dec3, dec4 = compute_fr_both(X_miss, X_imp, r_val)
    fr3_rmse.append(metrics["rmse"])
    fr3_fr.append(dec3["F_r"])
    fr3_frplus.append(dec4["F_r"])
    fr3_dkurt.append(dec4["d_kurt"])
    elapsed = int(time.time() - t0)
    print(f"  seed={seed:3d}  RMSE={metrics['rmse']:.4f}  "
          f"Fr={dec3['F_r']:.4f}  d_kurt={dec4['d_kurt']:.4f}  "
          f"Fr+={dec4['F_r']:.4f}  ({elapsed}s)")

print(f"  Fr  summary: RMSE={np.mean(fr3_rmse):.4f}±{np.std(fr3_rmse):.4f}  "
      f"Fr={np.mean(fr3_fr):.4f}±{np.std(fr3_fr):.4f}  "
      f"Fr+={np.mean(fr3_frplus):.4f}±{np.std(fr3_frplus):.4f}")

# ── MIGA Fr+ (4-term) ──────────────────────────────────────────────────────────
print("\n--- MIGA  Fr+ (extended, 4 terms, optimises kurtosis too) ---")
fr4_rmse, fr4_fr, fr4_frplus, fr4_dkurt = [], [], [], []

for seed in MIGA_SEEDS:
    t0 = time.time()
    miga = MIGA(l=L, G=G, Q=Q, r=r_val, use_kurtosis=True, seed=seed, verbose=False)
    X_imp = miga.fit_transform(X_miss, generators=gens)
    metrics = compute_metrics(X_true, X_imp, missing_mask)
    dec3, dec4 = compute_fr_both(X_miss, X_imp, r_val)
    fr4_rmse.append(metrics["rmse"])
    fr4_fr.append(dec3["F_r"])
    fr4_frplus.append(dec4["F_r"])
    fr4_dkurt.append(dec4["d_kurt"])
    elapsed = int(time.time() - t0)
    print(f"  seed={seed:3d}  RMSE={metrics['rmse']:.4f}  "
          f"Fr={dec3['F_r']:.4f}  d_kurt={dec4['d_kurt']:.4f}  "
          f"Fr+={dec4['F_r']:.4f}  ({elapsed}s)")

print(f"  Fr+ summary: RMSE={np.mean(fr4_rmse):.4f}±{np.std(fr4_rmse):.4f}  "
      f"Fr={np.mean(fr4_fr):.4f}±{np.std(fr4_fr):.4f}  "
      f"Fr+={np.mean(fr4_frplus):.4f}±{np.std(fr4_frplus):.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
output = {
    ds_name: {
        "dataset": ds_name, "n": n, "p": p, "n_seeds": len(MIGA_SEEDS),
        "l": L, "G": G, "Q": Q, "r": str(r_val),
        "kurt_A": kurt_A,
        "baselines": bl_results,
        "miga_fr": {
            "rmse":   fr3_rmse,  "fr":    fr3_fr,
            "frplus": fr3_frplus, "dkurt": fr3_dkurt,
            "rmse_mean":   float(np.mean(fr3_rmse)),
            "fr_mean":     float(np.mean(fr3_fr)),
            "frplus_mean": float(np.mean(fr3_frplus)),
            "dkurt_mean":  float(np.mean(fr3_dkurt)),
        },
        "miga_frplus": {
            "rmse":   fr4_rmse,  "fr":    fr4_fr,
            "frplus": fr4_frplus, "dkurt": fr4_dkurt,
            "rmse_mean":   float(np.mean(fr4_rmse)),
            "fr_mean":     float(np.mean(fr4_fr)),
            "frplus_mean": float(np.mean(fr4_frplus)),
            "dkurt_mean":  float(np.mean(fr4_dkurt)),
        },
    }
}

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved → {out_path}")
