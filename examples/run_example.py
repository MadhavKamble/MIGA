"""
Reproduce the paper's Section 4 application example and compute
RMSE/MAD/CoD for a small synthetic dataset.

The paper uses a 311-individual, 6-variable dataset (Dataset 1) available
at https://comunidad.udistrital.edu.co/lamic/tools/.
Since we don't have that file locally, we generate a synthetic dataset that
matches the described statistical properties and variable types:
  - Variables 1, 4, 5: continuous (Normal, Normal, Exponential)
  - Variables 2, 3, 6: discrete (Discrete-Uniform, Discrete-Uniform, Poisson)

Run:
  python examples/run_example.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from miga import MIGA

# ---------------------------------------------------------------------------
# 1. Synthetic dataset matching paper's Section 4 variable types
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 311
p = 6

X_true = np.column_stack([
    rng.normal(36.002, 5.260, n),                        # var 1: Normal
    rng.integers(14, 65, n).astype(float),               # var 2: Discrete Uniform
    rng.integers(0, 26, n).astype(float),                # var 3: Discrete Uniform
    rng.normal(35.438, 4.382, n),                        # var 4: Normal
    rng.exponential(2.401, n),                           # var 5: Exponential
    rng.poisson(1.506, n).astype(float),                 # var 6: Poisson
])

# Introduce ~10% missing at random across 131 individuals
missing_frac = 0.10
n_missing = int(n * p * missing_frac)
miss_rows = rng.integers(0, n, n_missing)
miss_cols = rng.integers(0, p, n_missing)

X = X_true.copy()
for r_idx, c_idx in zip(miss_rows, miss_cols):
    X[r_idx, c_idx] = np.nan

print(f"Dataset: {n} individuals, {p} variables")
print(f"Missing values: {int(np.isnan(X).sum())} / {n*p} ({100*np.isnan(X).mean():.1f}%)")
print(f"Complete rows: {int(np.all(~np.isnan(X), axis=1).sum())}")
print()

# ---------------------------------------------------------------------------
# 2. Random variable generators R_j (paper Section 4.1)
# ---------------------------------------------------------------------------
rng2 = np.random.default_rng(0)

generators = {
    0: lambda: float(rng2.normal(36.002, 5.260)),          # R_1: Normal
    1: lambda: float(rng2.integers(14, 65)),               # R_2: Discrete Uniform [14,64]
    2: lambda: float(rng2.integers(0, 26)),                # R_3: Discrete Uniform [0,25]
    3: lambda: float(rng2.normal(35.438, 4.382)),          # R_4: Normal
    4: lambda: float(rng2.exponential(2.401)),             # R_5: Exponential
    5: lambda: float(rng2.poisson(1.506)),                 # R_6: Poisson
}

# ---------------------------------------------------------------------------
# 3. Run MIGA
# ---------------------------------------------------------------------------
print("Running MIGA (paper Section 4 parameters)...")
miga = MIGA(
    l=200,    # population size (paper uses 1000, reduced for speed)
    G=300,    # generations (paper uses 2000, reduced for speed)
    c=3,
    c1=3,
    c2=3,
    c3=10,
    Q=6,
    r=np.inf,
    seed=42,
    verbose=True,
)

X_imputed = miga.fit_transform(X, generators=generators)

print(f"\nBest overall F_inf = {miga.best_score_:.6f}")

# ---------------------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------------------
mask = np.isnan(X)
observed = X_true[mask]
predicted = X_imputed[mask]

rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
mad = float(np.mean(np.abs(observed - predicted)))
ss_res = np.sum((observed - predicted) ** 2)
ss_tot = np.sum((observed - observed.mean()) ** 2)
cod = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

print(f"\n--- Imputation quality (vs. true values) ---")
print(f"RMSE : {rmse:.4f}")
print(f"MAD  : {mad:.4f}")
print(f"CoD  : {cod:.4f}")

# ---------------------------------------------------------------------------
# 5. Compare means / covariance distances (paper's F_r decomposition)
# ---------------------------------------------------------------------------
from scipy.stats import skew
from miga.statistics import compute_stats, pooled_std, relative_cov, minkowski_distance

complete_mask_orig = ~np.any(np.isnan(X), axis=1)
X_A = X[complete_mask_orig]
X_C_filled = X_imputed[~complete_mask_orig]

mean_A, cov_A, skew_A = compute_stats(X_A)
mean_C, cov_C, skew_C = compute_stats(X_C_filled)

nu_A = len(X_A) - 1
nu_C = len(X_C_filled) - 1
S_p = pooled_std(cov_A, cov_C, nu_A, nu_C)

x_tilde_A = mean_A / S_p
x_tilde_C = mean_C / S_p
S_tilde = relative_cov(cov_A, cov_C)
I = np.eye(p)

d_means = minkowski_distance(x_tilde_A, x_tilde_C, np.inf)
d_cov   = minkowski_distance(S_tilde, I, np.inf)
d_skew  = minkowski_distance(skew_A, skew_C, np.inf)

print(f"\n--- Fitness decomposition (r=∞) ---")
print(f"D_∞(x̃_A, x̃_C) = {d_means:.4f}   (means)")
print(f"D_∞(S̃,  I)     = {d_cov:.4f}   (covariances)")
print(f"D_∞(b_A, b_C)  = {d_skew:.4f}   (skewness)")
print(f"F_∞            = {d_means + d_cov + d_skew:.4f}")

print("\nDone.")
