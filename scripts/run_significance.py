"""
Statistical significance tests + Fr comparison across all methods under MAR and MNAR.

For each dataset at a given missing %:
  1. Run MIGA with 10 different seeds → distribution of RMSE and Fr values
  2. Run Mean/KNN/MICE once (deterministic) → single RMSE and Fr values
  3. Wilcoxon signed-rank test: is MIGA Fr significantly lower than each baseline?
  4. Under MNAR (top/tails mechanisms): same comparison

Usage (from the MIGA repo root):
    .venv/bin/python scripts/run_significance.py <Dataset> [pct]

Examples:
    .venv/bin/python scripts/run_significance.py Iris 30
    .venv/bin/python scripts/run_significance.py Wine 40
    .venv/bin/python scripts/run_significance.py Wholesale 50

Saves results/13_significance_<dataset>_<pct>.json.
Skips if output file already exists.
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
from miga.data_utils import (
    apply_mar, apply_mnar, auto_generators, compute_metrics, EXCLUDE_COLS,
    load_iris, load_wine, load_glass, load_haberman, load_wholesale,
)
from miga.fitness import FitnessEvaluator
from miga.paper_results import TABLE3_PARAMS

# ── Configuration ─────────────────────────────────────────────────────────────

DATASETS = {
    "Iris":      (load_iris,      {"pct": 30, "miss_seed": 30, "exclude_cols": []}),
    "Wine":      (load_wine,      {"pct": 30, "miss_seed": 30, "exclude_cols": []}),
    "Glass":     (load_glass,     {"pct": 30, "miss_seed": 30, "exclude_cols": [9]}),
    "Haberman":  (load_haberman,  {"pct": 30, "miss_seed": 30, "exclude_cols": []}),
    "Wholesale": (load_wholesale, {"pct": 30, "miss_seed": 30, "exclude_cols": [0, 1]}),
}

N_SEEDS = 10                        # independent MIGA runs for significance test
MIGA_SEEDS = list(range(10, 110, 10))  # seeds: 10,20,...,100

L, G, Q = 200, 300, 3               # reduced params — enough for significance testing

MECHANISMS = ["mar", "top", "tails"]  # mechanisms to evaluate

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_fr(X_miss, X_imp, r):
    complete_mask = ~np.any(np.isnan(X_miss), axis=1)
    incomplete_mask = ~complete_mask
    X_A = X_miss[complete_mask]
    X_C = X_imp[incomplete_mask]
    if len(X_A) < 2 or len(X_C) < 2:
        return float("nan")
    try:
        return FitnessEvaluator(X_A, r=r).evaluate(X_C)
    except Exception:
        return float("nan")


def apply_mechanism(X, mech, pct, seed, excl):
    if mech == "mar":
        return apply_mar(X, pct=pct, seed=seed, exclude_cols=excl)
    return apply_mnar(X, pct=pct, mechanism=mech, seed=seed, exclude_cols=excl)


def run_baselines(X_miss, X, missing_mask, r):
    results = {}
    for name, imp in [
        ("Mean", SimpleImputer(strategy="mean")),
        ("KNN",  KNNImputer(n_neighbors=5)),
        ("MICE", IterativeImputer(max_iter=20, random_state=42, tol=1e-3)),
    ]:
        try:
            X_imp = imp.fit_transform(X_miss)
            m = compute_metrics(X, X_imp, missing_mask)
            fr = compute_fr(X_miss, X_imp, r)
            results[name] = {"rmse": m["rmse"], "mad": m["mad"], "Fr": fr}
        except Exception as e:
            results[name] = {"rmse": None, "mad": None, "Fr": None, "error": str(e)}
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_dataset(ds_name):
    loader_fn, cfg = DATASETS[ds_name]
    X, cols = loader_fn()
    n, p = X.shape
    pct       = cfg["pct"]
    miss_seed = cfg["miss_seed"]
    excl      = cfg["exclude_cols"]
    r         = TABLE3_PARAMS[ds_name]["r"]

    print(f"\n{'='*60}")
    print(f"{ds_name}  (n={n}, p={p}, {pct}% missing, l={L}, G={G}, Q={Q})")
    print(f"{'='*60}")

    out = {"dataset": ds_name, "n": n, "p": p, "n_seeds": N_SEEDS,
           "l": L, "G": G, "Q": Q}

    for mech in MECHANISMS:
        print(f"\n--- {mech.upper()} ---")
        X_miss = apply_mechanism(X, mech, pct, miss_seed, excl)
        missing_mask = np.isnan(X_miss)
        n_miss = int(missing_mask.sum())
        print(f"  missing cells: {n_miss}")

        # Baselines (deterministic — run once)
        bl = run_baselines(X_miss, X, missing_mask, r)
        for name, vals in bl.items():
            if vals.get("rmse") is not None:
                print(f"  {name:<5}  RMSE={vals['rmse']:.4f}  Fr={vals['Fr']:.4f}")

        # MIGA across N_SEEDS seeds
        miga_rmse, miga_fr = [], []
        gens_dict = auto_generators(X_miss, seed=42)

        for i, seed in enumerate(MIGA_SEEDS):
            t0 = time.time()
            miga = MIGA(l=L, G=G, Q=Q, seed=seed, verbose=False)
            X_imp = miga.fit_transform(X_miss.copy(), generators=gens_dict)
            elapsed = time.time() - t0

            m  = compute_metrics(X, X_imp, missing_mask)
            fr = compute_fr(X_miss, X_imp, r)
            miga_rmse.append(m["rmse"])
            miga_fr.append(fr)
            print(f"  MIGA seed={seed:3d}  RMSE={m['rmse']:.4f}  Fr={fr:.4f}  ({elapsed:.0f}s)")

        miga_rmse_arr = np.array(miga_rmse)
        miga_fr_arr   = np.array([f for f in miga_fr if f == f])  # drop NaN

        print(f"\n  MIGA summary ({N_SEEDS} seeds):")
        print(f"    RMSE: mean={miga_rmse_arr.mean():.4f}  std={miga_rmse_arr.std():.4f}")
        print(f"    Fr:   mean={miga_fr_arr.mean():.4f}  std={miga_fr_arr.std():.4f}")

        # Wilcoxon vs each baseline on Fr
        print(f"\n  Wilcoxon: MIGA Fr vs baseline Fr (H1: MIGA < baseline)")
        wilcoxon = {}
        for bl_name, bl_vals in bl.items():
            bl_fr = bl_vals.get("Fr")
            if bl_fr is None or bl_fr != bl_fr:
                continue
            # Compare MIGA Fr distribution against a constant (baseline is deterministic)
            # Use one-sample Wilcoxon against the baseline value
            diffs = miga_fr_arr - bl_fr
            if len(diffs) < 4 or np.all(diffs == diffs[0]):
                wilcoxon[bl_name] = {"p": float("nan"), "miga_wins": None}
                continue
            try:
                stat, p = stats.wilcoxon(diffs, alternative="less")
                miga_wins = bool(miga_fr_arr.mean() < bl_fr)
                wilcoxon[bl_name] = {"p": float(p), "miga_wins": miga_wins,
                                     "miga_mean_fr": float(miga_fr_arr.mean()),
                                     "baseline_fr": float(bl_fr)}
                sig = "✓ p<0.05" if p < 0.05 else ("p<0.10" if p < 0.10 else "n.s.")
                direction = "MIGA<" if miga_wins else "MIGA>"
                print(f"    vs {bl_name:<5}: {direction}{bl_fr:.4f}  p={p:.3f}  {sig}")
            except Exception as e:
                wilcoxon[bl_name] = {"p": float("nan"), "miga_wins": None, "error": str(e)}

        out[mech] = {
            "baselines": bl,
            "miga_rmse": miga_rmse,
            "miga_fr":   miga_fr,
            "miga_rmse_mean": float(miga_rmse_arr.mean()),
            "miga_rmse_std":  float(miga_rmse_arr.std()),
            "miga_fr_mean":   float(miga_fr_arr.mean()) if len(miga_fr_arr) else float("nan"),
            "miga_fr_std":    float(miga_fr_arr.std())  if len(miga_fr_arr) else float("nan"),
            "wilcoxon_fr":    wilcoxon,
        }

    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_significance.py <Dataset> [pct]")
        print(f"       datasets: {list(DATASETS)}")
        sys.exit(1)

    ds_arg = sys.argv[1]
    pct_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if ds_arg not in DATASETS:
        print(f"Unknown dataset '{ds_arg}'. Choose from: {list(DATASETS)}")
        sys.exit(1)

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir,
                            f"13_significance_{ds_arg.lower()}_{pct_arg}.json")
    old_path = os.path.join(results_dir,
                            f"13_significance_{ds_arg.lower()}.json")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        sys.exit(0)
    if pct_arg == 30 and os.path.exists(old_path):
        print(f"Already exists (old format), skipping: {old_path}")
        sys.exit(0)

    # Override pct in dataset config
    loader_fn, cfg = DATASETS[ds_arg]
    cfg = dict(cfg, pct=pct_arg)
    DATASETS[ds_arg] = (loader_fn, cfg)

    result = run_dataset(ds_arg)

    with open(out_path, "w") as f:
        json.dump({ds_arg: result}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
