"""
Statistical helpers used by the MIGA fitness function.

All definitions follow Figueroa-García et al. (2023), Definitions 1–5.
"""

import numpy as np
from scipy.stats import skew, kurtosis as sp_kurtosis
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def _fast_skewness(X: np.ndarray) -> np.ndarray:
    """Bias-corrected skewness (Fisher-Pearson g1, bias=False) via numpy.

    Avoids scipy.stats.skew which carries per-call inspection/broadcasting
    overhead of ~1.4 ms regardless of array size.
    Formula matches scipy with bias=False: n*(n-1)**0.5/(n-2) * m3/m2**1.5
    where m2,m3 are 2nd and 3rd central moments (ddof=0).
    """
    n = X.shape[0]
    if n < 3:
        return np.zeros(X.shape[1] if X.ndim > 1 else 1)
    mu = X.mean(axis=0)
    diff = X - mu
    m2 = (diff ** 2).mean(axis=0)
    m3 = (diff ** 3).mean(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        g1 = np.where(m2 > 0, m3 / (m2 ** 1.5), 0.0)
    # Small-sample correction: multiply by sqrt(n*(n-1)) / (n-2)
    correction = (n * (n - 1)) ** 0.5 / (n - 2)
    sk = correction * g1
    return np.where(np.isfinite(sk), sk, 0.0)


def compute_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (mean, covariance, skewness) for matrix X (n × p).

    Returns
    -------
    mean : (p,)  — sample mean vector x̄
    cov  : (p,p) — sample covariance matrix S  (ddof=1)
    sk   : (p,)  — sample skewness vector b    (bias-corrected)
    """
    mean = np.nanmean(X, axis=0)
    cov = _safe_cov(X)
    sk = _fast_skewness(X)
    return mean, cov, sk


def compute_kurtosis(X: np.ndarray) -> np.ndarray:
    """
    Bias-corrected excess kurtosis vector  k  (p,).

    Uses Fisher's definition: k = E[(X-μ)^4/σ^4] - 3, so k=0 for a normal
    distribution, k>0 for heavy tails, k<0 for light tails.
    bias=False applies the small-sample correction (analogous to ddof=1).
    Zero-variance or degenerate columns produce nan; these are zeroed out.
    """
    k = sp_kurtosis(X, axis=0, bias=False, fisher=True)
    return np.where(np.isfinite(k), k, 0.0)


def compute_log_kurtosis(X: np.ndarray) -> np.ndarray:
    """
    Log-compressed excess kurtosis vector  (p,).

    Transforms raw kurtosis k via  log_k = log(1 + |k|) * sign(k).
    This compresses extreme values (e.g. k=170 → 5.14, k=54 → 4.01)
    making the target achievable for the GA on heavy-tailed columns
    (Wholesale Delicassen, Glass K) while preserving ordering and sign.

    Normal distribution: log_k = 0  (same as raw).
    Monotone in |k|: larger kurtosis → larger log_k.
    No division involved: well-defined for zero-heavy sparse columns
    where Moors kurtosis (quantile-based) collapses.
    """
    k = sp_kurtosis(X, axis=0, bias=False, fisher=True)
    k = np.where(np.isfinite(k), k, 0.0)
    return np.log1p(np.abs(k)) * np.sign(k)


def _safe_cov(X: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    """Sample covariance with Tikhonov regularisation to stay positive definite."""
    cov = np.cov(X.T, ddof=1)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov += reg * np.eye(cov.shape[0])
    return cov


def ledoit_wolf_cov(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf shrinkage estimator for the covariance matrix.

    Shrinks the sample covariance toward a scaled identity target:
        S_LW = (1 - α) S_sample + α * (tr(S)/p) * I
    where α (shrinkage coefficient) is chosen analytically to minimise the
    expected Frobenius loss under a Gaussian model (Ledoit & Wolf, 2004).

    Returns
    -------
    cov       : (p×p) shrunk covariance matrix
    shrinkage : scalar α ∈ [0, 1] — 0 = full sample cov, 1 = pure diagonal
    """
    try:
        lw = LedoitWolf().fit(X)
        return lw.covariance_, float(lw.shrinkage_)
    except Exception:
        return _safe_cov(X), 0.0


# ---------------------------------------------------------------------------
# Standardisation quantities (Definition 4)
# ---------------------------------------------------------------------------

def pooled_std(
    S_A: np.ndarray, S_C: np.ndarray, nu_A: int, nu_C: int
) -> np.ndarray:
    """
    Pooled standard deviation vector  S_p  (Definition 4, Eq. 7).

    S_p²  =  ( dg(S_A)·ν_A  +  dg(S_C)·ν_C ) / ν_T

    where ν_A = n_A - 1, ν_C = n_C - 1, ν_T = n_A + n_C - 2.
    Returns the element-wise square root (standard deviations).
    """
    nu_T = nu_A + nu_C
    diag_A = np.diag(S_A) if S_A.ndim == 2 else np.atleast_1d(float(S_A))
    diag_C = np.diag(S_C) if S_C.ndim == 2 else np.atleast_1d(float(S_C))
    sp2 = (diag_A * nu_A + diag_C * nu_C) / max(nu_T, 1)
    return np.sqrt(np.maximum(sp2, 1e-12))


def relative_cov(S_A: np.ndarray, S_C: np.ndarray) -> np.ndarray:
    """
    Relative covariance matrix  S̃ = S_A^{-1/2} S_C S_A^{-1/2}  (Eq. 8).

    Uses eigendecomposition for the symmetric matrix square root.
    S̃ = I  iff  S_A = S_C   (under hypothesis H_0: S_A = S_C).

    Eigenvalue floor is set to 1% of the largest eigenvalue (condition ≤ 100).
    This bounds the noise amplification from ill-conditioned or rank-deficient
    S_A — a practical issue when n_A is close to or smaller than p (e.g. Wine
    at high missing rates).  A 0.01% floor (1e-4) allows condition numbers up
    to 1e4, which makes the inverse square root amplify covariance estimation
    noise by ~100x and produces arbitrarily large F_r values.
    """
    eigvals, eigvecs = np.linalg.eigh(S_A)
    # 0.01% floor (condition ≤ 1e4).  After FitnessEvaluator scales features
    # to unit variance, S_A is a correlation matrix (eigenvalues bounded
    # O(m)), so this floor never clips legitimate eigenvalues.  It only
    # activates for truly rank-deficient S_A (n_A < m), where eigenvalues are
    # numerically zero and need a small positive floor.
    min_eigval = max(float(np.max(eigvals)) * 1e-4, 1e-10)
    eigvals = np.maximum(eigvals, min_eigval)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return inv_sqrt @ S_C @ inv_sqrt


# ---------------------------------------------------------------------------
# IPW: weighted statistics and propensity score estimation
# ---------------------------------------------------------------------------

def weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mean vector (p,). w need not be normalised."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return (X * w[:, None]).sum(axis=0)


def weighted_cov(X: np.ndarray, w: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    """
    Weighted covariance (p×p) using analytic weights (aweights convention).

    np.cov with aweights applies the reliability-weight correction
    (denominator = sum(w) - sum(w²)/sum(w)) equivalent to ddof=1 for
    uniform weights.
    """
    cov = np.cov(X.T, aweights=w)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov += reg * np.eye(cov.shape[0])
    return cov


def weighted_skewness(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Bias-corrected weighted skewness vector (p,).

    Uses effective sample size n_eff = (Σw)² / Σw² for the small-sample
    Fisher correction, matching scipy's bias=False convention at w=1.
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    n_eff = 1.0 / float((w ** 2).sum())

    mu = (X * w[:, None]).sum(axis=0)
    diff = X - mu
    var = (w[:, None] * diff ** 2).sum(axis=0) * n_eff / max(n_eff - 1, 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    m3 = (w[:, None] * diff ** 3).sum(axis=0)

    correction = (np.sqrt(n_eff * (n_eff - 1)) / (n_eff - 2)) if n_eff > 2 else 1.0
    sk = correction * m3 / std ** 3
    return np.where(np.isfinite(sk), sk, 0.0)


def estimate_ipw_weights(X: np.ndarray, pi_floor: float = 0.05) -> np.ndarray:
    """
    Estimate IPW weights w_i = 1/π_i for the complete rows of X.

    Fits a logistic regression to predict P(row complete | observed features),
    using column-mean imputation to handle NaN in the feature matrix.
    Returns weights normalised so they sum to n_A (maintains the same scale
    as unweighted statistics).

    Parameters
    ----------
    X        : (n × p) array with NaN for missing values
    pi_floor : minimum propensity score (prevents extreme weights)

    Returns
    -------
    weights : (n_A,) array — one weight per complete row of X
    """
    complete_mask = ~np.any(np.isnan(X), axis=1)
    n_A = int(complete_mask.sum())
    n = len(X)

    if n_A == n or n_A == 0:
        return np.ones(n_A)

    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_filled = scaler.fit_transform(imp.fit_transform(X))
    y = complete_mask.astype(int)

    try:
        lr = LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs',
                                random_state=0)
        lr.fit(X_filled, y)
        pi = lr.predict_proba(X_filled)[:, 1]
    except Exception:
        return np.ones(n_A)

    pi_complete = np.maximum(pi[complete_mask], pi_floor)
    w = 1.0 / pi_complete
    w = w * n_A / w.sum()
    return w


# ---------------------------------------------------------------------------
# Minkowski distance (Definition 1)
# ---------------------------------------------------------------------------

def minkowski_distance(A: np.ndarray, B: np.ndarray, r) -> float:
    """
    Minkowski r-norm distance  D_r(A, B)  between two same-shape arrays.

    D_r(A,B) = ( Σ_i Σ_j |a_ij - b_ij|^r )^{1/r}

    r = np.inf  →  max_{i,j} |a_ij - b_ij|   (Chebyshev / L∞)
    """
    diff = np.abs(np.asarray(A).ravel() - np.asarray(B).ravel())
    if np.isinf(r):
        return float(np.max(diff))
    return float(np.sum(diff ** r) ** (1.0 / r))
