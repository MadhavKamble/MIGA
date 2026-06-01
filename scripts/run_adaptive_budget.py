"""
C3 (revised): Budget-aware restarts with adaptive diversity decay.

Compares standard MIGA (Q=5, G=500, fixed) vs MIGA-AES (same total budget
Q*G=2500 gens, early stopping patience=50, diversity_decay=0.7) on all 5
benchmark datasets at 30% MAR.

Usage (from MIGA repo root):
    .venv/bin/python scripts/run_adaptive_budget.py

Saves results/19_adaptive_budget.json
"""

import sys, os, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from miga import MIGA
from miga.data_utils import (
    load_iris, load_wine, load_glass, load_haberman, load_wholesale,
    apply_mar, auto_generators, compute_metrics,
)

DATASETS = [
    ("Iris",      load_iris,      []),
    ("Wine",      load_wine,      []),
    ("Glass",     load_glass,     [9]),
    ("Haberman",  load_haberman,  []),
    ("Wholesale", load_wholesale, [0, 1]),
]

L, G, Q   = 200, 500, 5
PATIENCE  = 50
DECAY     = 0.7
SEED      = 42
PCT       = 30

def run_variant(X_miss, X, mask, gens, label, **kwargs):
    t0 = time.time()
    m = MIGA(l=L, G=G, Q=Q, seed=SEED, verbose=False, **kwargs)
    X_imp = m.fit_transform(X_miss.copy(), generators=gens)
    met = compute_metrics(X, X_imp, mask)
    n_runs  = len(m.history_)
    total_g = sum(len(h) for h in m.generation_history_)
    avg_g   = total_g / max(n_runs, 1)
    elapsed = time.time() - t0
    print(f"  [{label}] Fr={m.best_score_:.4f}  RMSE={met['rmse']:.4f}"
          f"  runs={n_runs}  avg_gens={avg_g:.0f}  ({elapsed/60:.1f}min)")
    return {
        "Fr": m.best_score_, "rmse": met["rmse"],
        "mad": met["mad"],   "cod":  met["cod"],
        "n_runs": n_runs,    "total_gens": total_g,
        "avg_gens_per_run": avg_g,
        "elapsed_sec": elapsed,
    }

results = {}

for name, loader, excl in DATASETS:
    X, cols = loader()
    n, p = X.shape
    X_miss = apply_mar(X, pct=PCT, seed=30, exclude_cols=excl)
    mask   = np.isnan(X_miss)
    gens   = auto_generators(X_miss, seed=SEED)

    print(f"\n{'='*55}")
    print(f"  {name}  (n={n}, p={p})")
    print(f"{'='*55}")

    std = run_variant(X_miss, X, mask, gens, "Standard",
                      diversity_decay=0.0, early_stopping_patience=None)
    aes = run_variant(X_miss, X, mask, gens, "MIGA-AES",
                      diversity_decay=DECAY,
                      early_stopping_patience=PATIENCE)

    delta_fr   = aes["Fr"]   - std["Fr"]
    delta_rmse = aes["rmse"] - std["rmse"]
    print(f"  ΔFr={delta_fr:+.4f}  ΔRMSE={delta_rmse:+.4f}"
          f"  restarts={aes['n_runs']} vs {std['n_runs']}")

    results[name] = {"standard": std, "aes": aes,
                     "n": n, "p": p, "cols": cols}

out = os.path.join(ROOT, "results", "19_adaptive_budget.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*55}")
print(f"Saved → {out}")
print(f"\n{'Dataset':10s}  {'Std-Fr':>8s}  {'AES-Fr':>8s}  {'ΔFr':>8s}"
      f"  {'Std-RMSE':>9s}  {'AES-RMSE':>9s}  {'Restarts':>8s}")
for name, r in results.items():
    s, a = r["standard"], r["aes"]
    print(f"{name:10s}  {s['Fr']:8.4f}  {a['Fr']:8.4f}"
          f"  {a['Fr']-s['Fr']:+8.4f}"
          f"  {s['rmse']:9.4f}  {a['rmse']:9.4f}"
          f"  {a['n_runs']:>8d}")
