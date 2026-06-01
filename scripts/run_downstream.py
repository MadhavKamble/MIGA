"""
Downstream task evaluation: does distributional accuracy (Fr) translate to
better real-world outcomes compared to pointwise accuracy (RMSE)?

Three downstream evaluations:
  1. Classification accuracy  — train on imputed features, predict class label
                                 (skipped for Wholesale — no natural label)
  2. KS test pass rate        — Kolmogorov-Smirnov test on each imputed feature
  3. Bootstrap CI coverage    — does 95% CI from imputed data contain true mean?

Datasets: Iris, Wine, Glass, Haberman, Wholesale
Missing rate: 30% MAR, 10 seeds

Usage (from the MIGA repo root):
    .venv/bin/python scripts/run_downstream.py Iris
    .venv/bin/python scripts/run_downstream.py Wine
    .venv/bin/python scripts/run_downstream.py Glass
    .venv/bin/python scripts/run_downstream.py Haberman
    .venv/bin/python scripts/run_downstream.py Wholesale

Each saves results/16_downstream_<dataset>.json independently.
"""

import sys, os, json, warnings
import numpy as np
from scipy.stats import ks_2samp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from miga.data_utils import apply_mar, EXCLUDE_COLS
from miga.core import MIGA

SEEDS  = list(range(10))
PCT    = 30
N_BOOT = 500
ALPHA  = 0.05


# ── Dataset registry with label handling ──────────────────────────────────────

def load_dataset_with_label(name):
    """
    Returns (X_feat, y, feat_names).
    y is None for datasets without a natural classification label.
    """
    if name == "Iris":
        from sklearn.datasets import load_iris as _li
        d = _li()
        return d.data.astype(float), d.target.astype(int), list(d.feature_names), []

    elif name == "Wine":
        from sklearn.datasets import load_wine as _lw
        d = _lw()
        return d.data.astype(float), d.target.astype(int), list(d.feature_names), []

    elif name == "Glass":
        from miga.data_utils import load_glass
        X, cols = load_glass()
        # Type is the last column (class label 1–7)
        return X[:, :-1], X[:, -1].astype(int), cols[:-1], []

    elif name == "Haberman":
        import pandas as pd
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/haberman/haberman.data")
        cols = ["Age", "OpYear", "Nodes", "Survival"]
        df = pd.read_csv(url, header=None, names=cols)
        X = df[["Age", "OpYear", "Nodes"]].values.astype(float)
        y = df["Survival"].values.astype(int)
        return X, y, ["Age", "OpYear", "Nodes"], []

    elif name == "Wholesale":
        from miga.data_utils import load_wholesale
        X, cols = load_wholesale()
        # Exclude Channel (0) and Region (1) — categorical identifiers
        feat_idx = [i for i in range(len(cols)) if i not in [0, 1]]
        X_feat = X[:, feat_idx]
        feat_names = [cols[i] for i in feat_idx]
        # No natural classification label for Wholesale
        return X_feat, None, feat_names, []

    else:
        raise ValueError(f"Unknown dataset '{name}'")


DATASETS = ["Iris", "Wine", "Glass", "Haberman", "Wholesale"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_methods(seed):
    return {
        "Mean": SimpleImputer(strategy="mean"),
        "KNN":  KNNImputer(n_neighbors=5),
        "MICE": IterativeImputer(max_iter=20, random_state=seed, tol=1e-3),
    }


def impute_miga(X_miss, seed):
    miga = MIGA(l=200, G=300, Q=3, r=np.inf, seed=seed,
                cov_estimator="ledoit_wolf")
    return miga.fit_transform(X_miss)


def classification_accuracy(X_feat, y, X_imp, seed):
    """5-fold CV: train on imputed, test on true features."""
    scaler = StandardScaler()
    clf    = LogisticRegression(max_iter=1000, random_state=seed)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs   = []
    for train_idx, test_idx in skf.split(X_feat, y):
        X_train = scaler.fit_transform(X_imp[train_idx])
        X_test  = scaler.transform(X_feat[test_idx])
        clf.fit(X_train, y[train_idx])
        accs.append(clf.score(X_test, y[test_idx]))
    return float(np.mean(accs))


def ks_pass_rate(X_true, X_imp, missing_mask):
    passes = []
    for j in range(X_true.shape[1]):
        mask_j = missing_mask[:, j]
        if mask_j.sum() < 5:
            continue
        _, p = ks_2samp(X_true[mask_j, j], X_imp[mask_j, j])
        passes.append(1 if p > ALPHA else 0)
    return float(np.mean(passes)) if passes else float("nan")


def bootstrap_ci_coverage(X_true, X_imp, missing_mask):
    coverages = []
    rng = np.random.default_rng(0)
    for j in range(X_true.shape[1]):
        mask_j = missing_mask[:, j]
        if mask_j.sum() < 5:
            continue
        imp_vals  = X_imp[mask_j, j]
        true_mean = X_true[mask_j, j].mean()
        boot_means = [rng.choice(imp_vals, size=len(imp_vals), replace=True).mean()
                      for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        coverages.append(1 if lo <= true_mean <= hi else 0)
    return float(np.mean(coverages)) if coverages else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print(f"Usage: python scripts/run_downstream.py <Dataset> [pct]")
    print(f"       datasets: {DATASETS}")
    sys.exit(1)

ds_name = sys.argv[1]
PCT = int(sys.argv[2]) if len(sys.argv) > 2 else PCT  # allow pct override

if ds_name not in DATASETS:
    print(f"Unknown dataset '{ds_name}'. Choose from: {DATASETS}")
    sys.exit(1)

out_path = os.path.join(ROOT, "results", f"16_downstream_{ds_name.lower()}_{PCT}.json")
if os.path.exists(out_path):
    print(f"Already exists, skipping: {out_path}")
    sys.exit(0)

print(f"\n=== Downstream evaluation: {ds_name} ===")
X_feat, y, feat_names, _ = load_dataset_with_label(ds_name)
has_label = y is not None
print(f"  n={X_feat.shape[0]}, p={X_feat.shape[1]}, features={feat_names}")
print(f"  classification: {'yes' if has_label else 'skipped (no label)'}")

results = {}

for seed in SEEDS:
    print(f"  seed {seed}...", end=" ", flush=True)
    X_miss = apply_mar(X_feat, pct=PCT, seed=seed + 100)
    missing_mask = np.isnan(X_miss)

    methods = get_methods(seed)
    seed_results = {}

    for method_name, imputer in methods.items():
        X_imp = imputer.fit_transform(X_miss)
        seed_results[method_name] = {
            "accuracy":    classification_accuracy(X_feat, y, X_imp, seed) if has_label else None,
            "ks_pass":     ks_pass_rate(X_feat, X_imp, missing_mask),
            "ci_coverage": bootstrap_ci_coverage(X_feat, X_imp, missing_mask),
        }

    X_imp_miga = impute_miga(X_miss, seed)
    seed_results["MIGA"] = {
        "accuracy":    classification_accuracy(X_feat, y, X_imp_miga, seed) if has_label else None,
        "ks_pass":     ks_pass_rate(X_feat, X_imp_miga, missing_mask),
        "ci_coverage": bootstrap_ci_coverage(X_feat, X_imp_miga, missing_mask),
    }

    results[seed] = seed_results
    print("done")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"SUMMARY — {ds_name} (mean across {len(SEEDS)} seeds)")
print(f"{'='*65}")
header = f"  {'Method':<8}  {'Accuracy':>10}  {'KS pass%':>10}  {'CI cov%':>10}"
print(header)
print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

summary = {}
for method in ["Mean", "KNN", "MICE", "MIGA"]:
    acc_vals = [results[s][method]["accuracy"] for s in SEEDS if results[s][method]["accuracy"] is not None]
    ks_vals  = [results[s][method]["ks_pass"]     for s in SEEDS]
    ci_vals  = [results[s][method]["ci_coverage"] for s in SEEDS]
    acc = float(np.mean(acc_vals)) if acc_vals else None
    ks  = float(np.mean(ks_vals))
    ci  = float(np.mean(ci_vals))
    summary[method] = {"accuracy": acc, "ks_pass": ks, "ci_coverage": ci}
    acc_str = f"{acc:>10.4f}" if acc is not None else f"{'N/A':>10}"
    print(f"  {method:<8}  {acc_str}  {ks*100:>9.1f}%  {ci*100:>9.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump({"dataset": ds_name, "has_label": has_label,
               "per_seed": results, "summary": summary}, f, indent=2)
print(f"\nSaved → {out_path}")
