"""
MIGA fitness function  F_r  (Definition 5, Eq. 10).

    F_r  := D_r(x̃_A, x̃_C) + D_r(S̃, I) + D_r(b_A, b_C)
    Fr+  := F_r + D_r(k_A, k_C)   [extended, use_kurtosis=True]

Terms:
  D_r(x̃_A, x̃_C) — Minkowski distance of standardised mean vectors (1st moment)
  D_r(S̃, I)      — Minkowski distance of relative covariance and identity (2nd moment)
  D_r(b_A, b_C)  — Minkowski distance of skewness vectors (3rd moment)
  D_r(k_A, k_C)  — Minkowski distance of excess kurtosis vectors (4th moment) [optional]

All terms are dimensionless, making them additive (no unit mismatch).
F_r = 0  iff  x̄_A = x̄_C  AND  S_A = S_C  AND  b_A = b_C  (AND  k_A = k_C if enabled).
"""

import numpy as np
from .statistics import (
    compute_stats,
    compute_kurtosis,
    compute_log_kurtosis,
    ledoit_wolf_cov,
    pooled_std,
    relative_cov,
    minkowski_distance,
    weighted_mean,
    weighted_cov,
    weighted_skewness,
    _safe_cov,
)


class FitnessEvaluator:
    """
    Pre-computes available-data statistics (X_A) once and evaluates F_r
    efficiently for many completed matrices X_C.

    Parameters
    ----------
    X_A : np.ndarray  (n_A × p)  — complete rows of the original dataset
    r   : float | np.inf         — Minkowski order (paper recommends np.inf)
    """

    def __init__(self, X_A: np.ndarray, r=np.inf, cov_estimator: str = 'sample',
                 use_kurtosis: bool = False, ipw_weights: np.ndarray | None = None):
        self.r = r
        self.p = X_A.shape[1]
        self.n_A = len(X_A)
        self.nu_A = self.n_A - 1
        self.cov_estimator = cov_estimator
        self.use_kurtosis = use_kurtosis

        # Per-column scale estimated from X_A.  Dividing by this before
        # computing statistics turns the covariance into a correlation matrix,
        # which has bounded eigenvalues (condition number O(m)) regardless of
        # feature scale.  Without this, a single high-variance feature (e.g.
        # Wine's Proline: σ² ≈ 1e5 vs σ² ≈ 1 for other features) dominates
        # S_A to the point where S_A^{-1/2} is numerically meaningless.
        # S̃ = S_A^{-1/2} S_C S_A^{-1/2} is theoretically scale-invariant,
        # but requires numerical stability to realise that invariance.
        self._scale = np.maximum(np.std(X_A, axis=0, ddof=1), 1e-8)

        X_A_scaled = X_A / self._scale

        if ipw_weights is not None:
            # IPW mode: re-weight X_A to correct for MNAR selection bias.
            # Weighted statistics replace the unweighted reference distribution.
            self.mean_A = weighted_mean(X_A_scaled, ipw_weights)
            self.cov_A  = weighted_cov(X_A_scaled, ipw_weights)
            self.skew_A = weighted_skewness(X_A_scaled, ipw_weights)
            self.kurt_A = compute_kurtosis(X_A_scaled) if use_kurtosis else None
            self.lw_shrinkage_ = None
        else:
            self.mean_A, self.cov_A, self.skew_A = compute_stats(X_A_scaled)
            self.kurt_A = compute_kurtosis(X_A_scaled) if use_kurtosis else None
            if cov_estimator == 'ledoit_wolf':
                self.cov_A, self.lw_shrinkage_ = ledoit_wolf_cov(X_A_scaled)
            else:
                self.lw_shrinkage_ = None

        # Cache S_A^{-1/2} — cov_A is fixed throughout optimization,
        # so computing eigh once here saves one eigendecomposition per evaluation.
        eigvals, eigvecs = np.linalg.eigh(self.cov_A)
        min_eigval = max(float(np.max(eigvals)) * 1e-4, 1e-10)
        eigvals = np.maximum(eigvals, min_eigval)
        self._inv_sqrt_A = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self._I = np.eye(self.p)  # cached identity matrix for d_cov term

    # ------------------------------------------------------------------

    def _get_cov_C(self, X_C_scaled: np.ndarray) -> np.ndarray:
        """Return covariance estimate for X_C using the configured estimator."""
        if self.cov_estimator == 'ledoit_wolf':
            cov_C, _ = ledoit_wolf_cov(X_C_scaled)
        else:
            cov_C = _safe_cov(X_C_scaled)
        return cov_C

    def evaluate(self, X_C: np.ndarray) -> float:
        """
        Evaluate F_r for a fully-filled completed matrix X_C  (n_C × p).
        X_C must contain no NaN values.

        Returns the scalar fitness value (lower = better).
        """
        n_C = len(X_C)
        nu_C = max(n_C - 1, 1)

        X_C_scaled = X_C / self._scale
        # Single call to compute_stats covers mean, cov, skew in one pass
        mean_C, cov_C, skew_C = compute_stats(X_C_scaled)

        if self.cov_estimator == 'ledoit_wolf':
            cov_C, _ = ledoit_wolf_cov(X_C_scaled)

        # Pooled standard deviation vector  S_p  (Eq. 7)
        S_p = pooled_std(self.cov_A, cov_C, self.nu_A, nu_C)

        # Standardised means  x̃ = x̄ / S_p  (Eqs. 5, 6)
        x_tilde_A = self.mean_A / S_p
        x_tilde_C = mean_C / S_p

        # Relative covariance matrix  S̃ = S_A^{-½} S_C S_A^{-½}  (Eq. 8)
        S_tilde = self._inv_sqrt_A @ cov_C @ self._inv_sqrt_A

        # Three goal distances  G1, G2, G3  (Section 3.3)
        d_means = minkowski_distance(x_tilde_A, x_tilde_C, self.r)
        d_cov   = minkowski_distance(S_tilde, self._I, self.r)
        d_skew  = minkowski_distance(self.skew_A, skew_C, self.r)

        d_kurt = 0.0
        if self.use_kurtosis:
            kurt_C = compute_kurtosis(X_C_scaled)
            d_kurt = minkowski_distance(self.kurt_A, kurt_C, self.r)

        return d_means + d_cov + d_skew + d_kurt

    def decompose(self, X_C: np.ndarray) -> dict:
        """
        Return the three individual distances for reporting / analysis.
        Useful for reproducing Table 1 and Section 4 in the paper.
        """
        n_C = len(X_C)
        nu_C = max(n_C - 1, 1)

        X_C_scaled = X_C / self._scale
        mean_C, cov_C, skew_C = compute_stats(X_C_scaled)
        if self.cov_estimator == 'ledoit_wolf':
            cov_C, _ = ledoit_wolf_cov(X_C_scaled)
        S_p = pooled_std(self.cov_A, cov_C, self.nu_A, nu_C)

        x_tilde_A = self.mean_A / S_p
        x_tilde_C = mean_C / S_p
        S_tilde = self._inv_sqrt_A @ cov_C @ self._inv_sqrt_A

        d_means = minkowski_distance(x_tilde_A, x_tilde_C, self.r)
        d_cov   = minkowski_distance(S_tilde, self._I, self.r)
        d_skew  = minkowski_distance(self.skew_A, skew_C, self.r)

        d_kurt = 0.0
        if self.use_kurtosis:
            kurt_C = compute_kurtosis(X_C_scaled)
            d_kurt = minkowski_distance(self.kurt_A, kurt_C, self.r)

        return {
            "F_r":     d_means + d_cov + d_skew + d_kurt,
            "d_means": d_means,
            "d_cov":   d_cov,
            "d_skew":  d_skew,
            "d_kurt":  d_kurt,
        }
