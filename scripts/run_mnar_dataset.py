"""
Run MNAR vs MAR comparison for ONE dataset at a given missing percentage.

Usage (from the MIGA repo root):
    .venv/bin/python scripts/run_mnar_dataset.py <Dataset> [pct]

Examples:
    .venv/bin/python scripts/run_mnar_dataset.py Iris 30
    .venv/bin/python scripts/run_mnar_dataset.py Wine 40
    .venv/bin/python scripts/run_mnar_dataset.py Wholesale 50

Saves results/11_mnar_<dataset>_<pct>.json. Skips if already exists.

Mechanisms: mar, top, bottom, tails
"""

import sys
import os
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from miga import MIGA
from miga.data_utils import (
    apply_mar, apply_mnar, auto_generators, compute_metrics, EXCLUDE_COLS,
    load_iris, load_wine, load_glass, load_haberman, load_wholesale,
)

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "Iris":      (load_iris,      {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Wine":      (load_wine,      {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Glass":     (load_glass,     {"pct": 30, "seed": 30, "exclude_cols": [9]}),
    "Haberman":  (load_haberman,  {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Wholesale": (load_wholesale, {"pct": 30, "seed": 30, "exclude_cols": [0, 1]}),
}

# ── GA parameters ─────────────────────────────────────────────────────────────

L      = 200
G_GENS = 500
Q_RUNS = 5
SEED   = 42

# ── Mechanisms to evaluate ────────────────────────────────────────────────────

MECHANISMS = ["mar", "top", "bottom", "tails"]

# ── Main ──────────────────────────────────────────────────────────────────────

def run_one(ds_name: str, out_path: str) -> None:
    loader_fn, base_kwargs = DATASETS[ds_name]
    X, cols = loader_fn()
    n, p = X.shape
    pct        = base_kwargs["pct"]
    miss_seed  = base_kwargs["seed"]
    excl       = list(base_kwargs["exclude_cols"])

    print(f"\n{'='*60}")
    print(f"Dataset : {ds_name}  (n={n}, p={p})")
    print(f"GA      : l={L}, G={G_GENS}, Q={Q_RUNS}, seed={SEED}")
    print(f"Missing : {pct}% per selected feature")
    print(f"{'='*60}\n")

    out = {"dataset": ds_name, "n": n, "p": p, "cols": cols}

    for mech in MECHANISMS:
        print(f"--- {mech.upper()} ---")

        if mech == "mar":
            X_miss = apply_mar(X, pct=pct, seed=miss_seed, exclude_cols=excl)
        else:
            X_miss = apply_mnar(
                X, pct=pct, mechanism=mech, seed=miss_seed, exclude_cols=excl
            )

        missing_mask = np.isnan(X_miss)
        n_miss = int(missing_mask.sum())
        print(f"  missing cells = {n_miss}  ({100*n_miss/(n*p):.1f}% of total)")

        gens_dict = auto_generators(X_miss, seed=SEED)

        t0 = time.time()
        miga = MIGA(l=L, G=G_GENS, Q=Q_RUNS, seed=SEED, verbose=True)
        X_imp = miga.fit_transform(X_miss.copy(), generators=gens_dict)
        elapsed = time.time() - t0

        m = compute_metrics(X, X_imp, missing_mask)
        print(f"  RMSE={m['rmse']:.4f}  MAD={m['mad']:.4f}  CoD={m['cod']:.4f}")
        print(f"  best_Fr={miga.best_score_:.4f}  elapsed={elapsed/60:.1f} min\n")

        out[mech] = {
            "rmse":        m["rmse"],
            "mad":         m["mad"],
            "cod":         m["cod"],
            "best_score":  miga.best_score_,
            "history":     miga.history_,
            "gen_history": [list(run) for run in miga.generation_history_],
            "n_missing":   n_miss,
            "elapsed_sec": elapsed,
        }

    # Save JSON
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved → {out_path}")
    for mech in MECHANISMS:
        print(f"  {mech:8s}  RMSE={out[mech]['rmse']:.4f}  MAD={out[mech]['mad']:.4f}  CoD={out[mech]['cod']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_mnar_dataset.py <Dataset> [pct]")
        print(f"       datasets: {list(DATASETS)}")
        sys.exit(1)

    ds_name = sys.argv[1]
    pct_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if ds_name not in DATASETS:
        print(f"Unknown dataset '{ds_name}'. Choose from: {list(DATASETS)}")
        sys.exit(1)

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"11_mnar_{ds_name.lower()}_{pct_arg}.json")
    old_path = os.path.join(results_dir, f"11_mnar_{ds_name.lower()}.json")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        sys.exit(0)
    if pct_arg == 30 and os.path.exists(old_path):
        print(f"Already exists (old format), skipping: {old_path}")
        sys.exit(0)

    # Override pct in dataset config
    loader_fn, kwargs = DATASETS[ds_name]
    DATASETS[ds_name] = (loader_fn, dict(kwargs, pct=pct_arg))

    run_one(ds_name, out_path)
