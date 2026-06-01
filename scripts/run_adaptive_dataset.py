"""
Run adaptive vs fixed c3 comparison for ONE dataset.

Usage (from the MIGA repo root):
    .venv/bin/python scripts/run_adaptive_dataset.py Iris
    .venv/bin/python scripts/run_adaptive_dataset.py Glass
    .venv/bin/python scripts/run_adaptive_dataset.py Haberman

Run three terminals in parallel — each saves its own JSON to results/.
Notebook 10 then loads the JSONs and plots.
"""

import sys
import os
import json
import time
import numpy as np

# Resolve repo root so imports work regardless of cwd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from miga import MIGA
from miga.data_utils import apply_mar, auto_generators, compute_metrics, EXCLUDE_COLS
from miga.data_utils import load_iris, load_wine, load_glass, load_haberman, load_wholesale

# ── Dataset registry ─────────────────────────────────────────────────────────

DATASETS = {
    "Iris":      (load_iris,      {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Wine":      (load_wine,      {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Glass":     (load_glass,     {"pct": 30, "seed": 30, "exclude_cols": [9]}),
    "Haberman":  (load_haberman,  {"pct": 30, "seed": 30, "exclude_cols": []}),
    "Wholesale": (load_wholesale, {"pct": 30, "seed": 30, "exclude_cols": [0, 1]}),
}

# ── GA parameters ────────────────────────────────────────────────────────────
# Balanced for a ~20–40 min run per dataset on a mid-range laptop.
# Increase Q_RUNS / G_GENS for the final thesis numbers.

L      = 200   # population size
G_GENS = 500   # generations per run
Q_RUNS = 5     # independent runs

C3_FIXED = 5          # paper default
C3_START = 15         # adaptive schedule start (high exploration)
C3_END   = 3          # adaptive schedule end   (fine exploitation)
SEED     = 42

# ── Main ─────────────────────────────────────────────────────────────────────

def run_one(ds_name: str, out_path: str) -> None:
    loader_fn, mar_kwargs = DATASETS[ds_name]
    X, cols = loader_fn()
    n, p = X.shape
    X_miss = apply_mar(X, **mar_kwargs)
    missing_mask = np.isnan(X_miss)
    gens_dict = auto_generators(X_miss, seed=SEED)

    n_miss = int(missing_mask.sum())
    print(f"\n{'='*60}")
    print(f"Dataset : {ds_name}  (n={n}, p={p}, missing_cells={n_miss})")
    print(f"GA      : l={L}, G={G_GENS}, Q={Q_RUNS}, seed={SEED}")
    print(f"Fixed   : c3={C3_FIXED}")
    print(f"Adaptive: c3 {C3_START} → {C3_END}")
    print(f"{'='*60}\n")

    out = {"dataset": ds_name, "n": n, "p": p}

    for label, miga_kwargs in [
        ("fixed",    dict(c3=C3_FIXED)),
        ("adaptive", dict(c3=C3_FIXED, c3_schedule=(C3_START, C3_END))),
    ]:
        print(f"--- {label.upper()} ---")
        t0 = time.time()
        miga = MIGA(l=L, G=G_GENS, Q=Q_RUNS, seed=SEED, verbose=True, **miga_kwargs)
        X_imp = miga.fit_transform(X_miss.copy(), generators=gens_dict)
        elapsed = time.time() - t0

        m = compute_metrics(X, X_imp, missing_mask)
        print(f"  RMSE={m['rmse']:.4f}  MAD={m['mad']:.4f}  CoD={m['cod']:.4f}")
        print(f"  best_Fr={miga.best_score_:.4f}  elapsed={elapsed/60:.1f} min\n")

        out[label] = {
            "rmse":          m["rmse"],
            "mad":           m["mad"],
            "cod":           m["cod"],
            "best_score":    miga.best_score_,
            "history":       miga.history_,
            "gen_history":   [list(run) for run in miga.generation_history_],
            "elapsed_sec":   elapsed,
        }

    # Save JSON
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved → {out_path}")
    print(f"Fixed  RMSE = {out['fixed']['rmse']:.4f}")
    print(f"Adaptive RMSE = {out['adaptive']['rmse']:.4f}")
    delta = (out["fixed"]["rmse"] - out["adaptive"]["rmse"]) / out["fixed"]["rmse"] * 100
    print(f"Δ RMSE = {delta:+.1f}%  ({'Adaptive wins' if delta > 0 else 'Fixed wins'})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_adaptive_dataset.py <Dataset> [pct]")
        print(f"       datasets: {list(DATASETS)}")
        sys.exit(1)

    ds_arg  = sys.argv[1]
    pct_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if ds_arg not in DATASETS:
        print(f"Unknown dataset '{ds_arg}'. Choose from: {list(DATASETS)}")
        sys.exit(1)

    out_path = os.path.join(ROOT, "results",
                            f"10_adaptive_{ds_arg.lower()}_{pct_arg}.json")
    old_path = os.path.join(ROOT, "results", f"10_adaptive_{ds_arg.lower()}.json")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        sys.exit(0)
    if pct_arg == 30 and os.path.exists(old_path):
        print(f"Already exists (old format), skipping: {old_path}")
        sys.exit(0)

    # Override pct and output path
    loader_fn, kwargs = DATASETS[ds_arg]
    DATASETS[ds_arg] = (loader_fn, dict(kwargs, pct=pct_arg))
    run_one(ds_arg, out_path)
