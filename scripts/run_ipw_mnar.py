"""
IPW-Fr experiment: compare standard MIGA vs MIGA-IPW under MNAR.

Runs both variants on Haberman and Iris across all 4 mechanisms
(MAR, top, bottom, tails) at 30% missing rate.

Usage (from MIGA repo root):
    .venv/bin/python scripts/run_ipw_mnar.py [Dataset]

    Dataset: Haberman (default), Iris

Saves results/12_ipw_<dataset>_30.json
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
    apply_mar, apply_mnar, auto_generators, compute_metrics,
    load_iris, load_wine, load_glass, load_haberman, load_wholesale,
)

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "Haberman":  (load_haberman,  {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Iris":      (load_iris,      {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Wine":      (load_wine,      {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Wholesale": (load_wholesale, {"pct": 30, "seed": 30, "exclude_cols": [0, 1]}),
    "Glass":     (load_glass,     {"pct": 30, "seed": 30, "exclude_cols": [9]}),
}

# ── GA parameters ─────────────────────────────────────────────────────────────

L      = 200
G_GENS = 500
Q_RUNS = 5
SEED   = 42

MECHANISMS = ["mar", "top", "bottom", "tails"]

# ── Main ──────────────────────────────────────────────────────────────────────

def run_one(ds_name: str, out_path: str) -> None:
    loader_fn, base_kwargs = DATASETS[ds_name]
    X, cols = loader_fn()
    n, p = X.shape
    pct       = base_kwargs["pct"]
    miss_seed = base_kwargs["seed"]
    excl      = list(base_kwargs["exclude_cols"])

    print(f"\n{'='*60}")
    print(f"Dataset : {ds_name}  (n={n}, p={p})")
    print(f"GA      : l={L}, G={G_GENS}, Q={Q_RUNS}, seed={SEED}")
    print(f"Missing : {pct}% per selected feature")
    print(f"{'='*60}\n")

    out = {"dataset": ds_name, "n": n, "p": p, "cols": cols}

    for mech in MECHANISMS:
        print(f"\n{'─'*50}")
        print(f"  Mechanism: {mech.upper()}")
        print(f"{'─'*50}")

        if mech == "mar":
            X_miss = apply_mar(X, pct=pct, seed=miss_seed, exclude_cols=excl)
        else:
            X_miss = apply_mnar(
                X, pct=pct, mechanism=mech, seed=miss_seed, exclude_cols=excl
            )

        missing_mask = np.isnan(X_miss)
        n_miss = int(missing_mask.sum())
        print(f"  missing cells = {n_miss}  ({100*n_miss/(n*p):.1f}% of total)\n")

        gens_dict = auto_generators(X_miss, seed=SEED)

        # ── Standard MIGA ────────────────────────────────────────────
        print("  [Standard MIGA]")
        t0 = time.time()
        miga_std = MIGA(l=L, G=G_GENS, Q=Q_RUNS, seed=SEED, verbose=True,
                        use_ipw=False)
        X_imp_std = miga_std.fit_transform(X_miss.copy(), generators=gens_dict)
        t_std = time.time() - t0

        m_std = compute_metrics(X, X_imp_std, missing_mask)
        print(f"  RMSE={m_std['rmse']:.4f}  Fr={miga_std.best_score_:.4f}"
              f"  elapsed={t_std/60:.1f}min")

        # ── MIGA-IPW ─────────────────────────────────────────────────
        print("\n  [MIGA-IPW]")
        t0 = time.time()
        miga_ipw = MIGA(l=L, G=G_GENS, Q=Q_RUNS, seed=SEED, verbose=True,
                        use_ipw=True)
        X_imp_ipw = miga_ipw.fit_transform(X_miss.copy(), generators=gens_dict)
        t_ipw = time.time() - t0

        m_ipw = compute_metrics(X, X_imp_ipw, missing_mask)
        print(f"  RMSE={m_ipw['rmse']:.4f}  Fr_reported={miga_ipw.best_score_:.4f}"
              f"  elapsed={t_ipw/60:.1f}min")

        out[mech] = {
            "standard": {
                "rmse":        m_std["rmse"],
                "mad":         m_std["mad"],
                "cod":         m_std["cod"],
                "best_score":  miga_std.best_score_,
                "history":     miga_std.history_,
                "elapsed_sec": t_std,
            },
            "ipw": {
                "rmse":        m_ipw["rmse"],
                "mad":         m_ipw["mad"],
                "cod":         m_ipw["cod"],
                "best_score":  miga_ipw.best_score_,
                "history":     miga_ipw.history_,
                "elapsed_sec": t_ipw,
            },
            "n_missing":   n_miss,
        }

    # Save JSON
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved → {out_path}")
    print(f"\nSummary:")
    print(f"{'Mech':8s}  {'Std-Fr':>8s}  {'IPW-Fr':>8s}  "
          f"{'Std-RMSE':>9s}  {'IPW-RMSE':>9s}")
    for mech in MECHANISMS:
        r = out[mech]
        print(f"{mech:8s}  {r['standard']['best_score']:8.4f}  "
              f"{r['ipw']['best_score']:8.4f}  "
              f"{r['standard']['rmse']:9.4f}  {r['ipw']['rmse']:9.4f}")


if __name__ == "__main__":
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "Haberman"

    if ds_name not in DATASETS:
        print(f"Unknown dataset '{ds_name}'. Choose from: {list(DATASETS)}")
        sys.exit(1)

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"12_ipw_{ds_name.lower()}_30.json")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        print("Delete it to re-run.")
        sys.exit(0)

    run_one(ds_name, out_path)
