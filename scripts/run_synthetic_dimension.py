"""
Synthetic dimension experiment: how does MIGA's Fr advantage scale with
dimensionality (p) and feature correlation (rho)?

Generates Gaussian data with a Toeplitz covariance matrix (rho^|i-j|),
applies 30% MAR, imputes with MIGA and baselines, records Fr and RMSE.

Usage (from MIGA repo root):
    .venv/bin/python scripts/run_synthetic_dimension.py

Saves results/17_synthetic_dimension.json. Skips if already exists.
Runtime: ~2-4 hours depending on hardware.
"""

import sys, os, json, time, warnings
import numpy as np
from scipy.linalg import toeplitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from miga import MIGA
from miga.data_utils import apply_mar, compute_metrics
from miga.fitness import FitnessEvaluator

# ── Experiment grid ───────────────────────────────────────────────────────────

P_VALUES   = [4, 8, 13, 20, 30]        # feature dimensionality
RHO_VALUES = [0.0, 0.3, 0.6, 0.9]     # Toeplitz correlation
N          = 200                        # samples per synthetic dataset
PCT        = 30                         # % missing (MAR)
N_SEEDS    = 5                          # MIGA seeds per cell
L, G, Q    = 100, 200, 3               # GA params (lighter than main experiments)
MISS_SEED  = 30
MIGA_SEEDS = list(range(10, 10 + 10 * N_SEEDS, 10))

out_path = os.path.join(ROOT, "results", "17_synthetic_dimension.json")

if os.path.exists(out_path):
    print(f"Already exists, skipping: {out_path}")
    sys.exit(0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_toeplitz_cov(p, rho):
    """Toeplitz covariance: Sigma_{ij} = rho^|i-j|."""
    col = np.array([rho ** k for k in range(p)])
    return toeplitz(col)


def generate_data(p, rho, n, seed):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-1, 1, p)
    cov = make_toeplitz_cov(p, rho)
    X = rng.multivariate_normal(mu, cov, size=n)
    return X.astype(np.float64)


def compute_fr(X_miss, X_imp, r=np.inf):
    complete_mask   = ~np.any(np.isnan(X_miss), axis=1)
    incomplete_mask = ~complete_mask
    X_A = X_miss[complete_mask]
    X_C = X_imp[incomplete_mask]
    if len(X_A) < 2 or len(X_C) < 2:
        return float("nan")
    try:
        return float(FitnessEvaluator(X_A, r=r).evaluate(X_C))
    except Exception:
        return float("nan")


def auto_generators_simple(X_miss, seed):
    rng = np.random.default_rng(seed)
    gens = {}
    for j in range(X_miss.shape[1]):
        obs = X_miss[:, j][~np.isnan(X_miss[:, j])]
        if len(obs) == 0:
            obs = np.zeros(1)
        gens[j] = lambda col=obs, r=rng: float(r.choice(col))
    return gens

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"Synthetic dimension experiment")
print(f"p in {P_VALUES}, rho in {RHO_VALUES}, n={N}, pct={PCT}%")
print(f"MIGA: l={L}, G={G}, Q={Q}, {N_SEEDS} seeds per cell")
print(f"Total cells: {len(P_VALUES) * len(RHO_VALUES)}")
print()

results = {}
DATA_SEED = 99

t_total = time.time()

for p in P_VALUES:
    results[p] = {}
    for rho in RHO_VALUES:
        cell_key = f"p{p}_rho{rho}"
        print(f"\n{'='*55}")
        print(f"p={p}, rho={rho}")
        print(f"{'='*55}")

        X = generate_data(p, rho, N, seed=DATA_SEED)
        X_miss = apply_mar(X, pct=PCT, seed=MISS_SEED, exclude_cols=[])
        missing_mask = np.isnan(X_miss)
        n_miss = int(missing_mask.sum())
        print(f"  missing cells: {n_miss}")

        cell = {"p": p, "rho": rho, "n": N, "n_miss": n_miss, "baselines": {}}

        # Baselines (deterministic)
        for bl_name, imp in [
            ("Mean", SimpleImputer(strategy="mean")),
            ("KNN",  KNNImputer(n_neighbors=5)),
            ("MICE", IterativeImputer(max_iter=20, random_state=42, tol=1e-3)),
        ]:
            try:
                X_imp = imp.fit_transform(X_miss.copy())
                m = compute_metrics(X, X_imp, missing_mask)
                fr = compute_fr(X_miss, X_imp)
                cell["baselines"][bl_name] = {
                    "rmse": float(m["rmse"]), "fr": fr
                }
                print(f"  {bl_name:<5}  RMSE={m['rmse']:.4f}  Fr={fr:.4f}")
            except Exception as e:
                cell["baselines"][bl_name] = {"error": str(e)}

        # MIGA across N_SEEDS seeds
        miga_rmse, miga_fr = [], []
        gens = auto_generators_simple(X_miss, seed=DATA_SEED)

        for seed in MIGA_SEEDS:
            t0 = time.time()
            try:
                miga = MIGA(l=L, G=G, Q=Q, seed=seed, verbose=False)
                X_imp = miga.fit_transform(X_miss.copy(), generators=gens)
                m = compute_metrics(X, X_imp, missing_mask)
                fr = compute_fr(X_miss, X_imp)
                miga_rmse.append(float(m["rmse"]))
                miga_fr.append(fr)
                elapsed = int(time.time() - t0)
                print(f"  MIGA seed={seed}  RMSE={m['rmse']:.4f}  Fr={fr:.4f}  ({elapsed}s)")
            except Exception as e:
                print(f"  MIGA seed={seed}  ERROR: {e}")

        if miga_rmse:
            rmse_arr = np.array(miga_rmse)
            fr_arr   = np.array([f for f in miga_fr if f == f])
            mice_fr  = cell["baselines"].get("MICE", {}).get("fr", float("nan"))
            mice_rmse = cell["baselines"].get("MICE", {}).get("rmse", float("nan"))

            fr_advantage  = float(mice_fr - np.mean(fr_arr))   if fr_arr.size  else float("nan")
            rmse_gap      = float(np.mean(rmse_arr) - mice_rmse)

            cell["miga"] = {
                "rmse_mean":   float(rmse_arr.mean()),
                "rmse_std":    float(rmse_arr.std()),
                "fr_mean":     float(fr_arr.mean()) if fr_arr.size else float("nan"),
                "fr_std":      float(fr_arr.std())  if fr_arr.size else float("nan"),
                "fr_advantage_over_mice": fr_advantage,
                "rmse_gap_vs_mice":       rmse_gap,
                "rmse_list": miga_rmse,
                "fr_list":   miga_fr,
            }
            print(f"  MIGA summary: RMSE={rmse_arr.mean():.4f}±{rmse_arr.std():.4f}  "
                  f"Fr={fr_arr.mean():.4f}±{fr_arr.std():.4f}  "
                  f"Fr advantage={fr_advantage:+.4f}")
        else:
            cell["miga"] = {"error": "all seeds failed"}

        results[p][rho] = cell

print(f"\nTotal elapsed: {(time.time()-t_total)/60:.1f} min")

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved → {out_path}")
