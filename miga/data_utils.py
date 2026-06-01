"""
Dataset loading and preprocessing utilities for MIGA benchmark experiments.

Datasets used in the paper (Section 5, Table 2):
  Iris             150 × 4     sklearn
  Wine             178 × 13    sklearn
  Glass            214 × 10    UCI URL
  Haberman         306 × 3     UCI URL
  Wholesale        440 × 8     UCI URL
  Cardiotocography 2126 × 23   UCI URL (Excel)
  Adult            48842 × 14  sklearn / OpenML

Missing mechanism:  MAR  (Missing At Random, paper Section 5)
  - Select floor(m/2) features at random
  - Remove pct% of observations per selected feature at random positions

Metrics are computed on z-standardised scale (using X_true statistics)
to allow scale-invariant comparison across datasets and methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable

# ------------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------------

def load_iris() -> tuple[np.ndarray, list[str]]:
    from sklearn.datasets import load_iris as _load
    d = _load()
    return d.data.astype(float), list(d.feature_names)


def load_wine() -> tuple[np.ndarray, list[str]]:
    from sklearn.datasets import load_wine as _load
    d = _load()
    return d.data.astype(float), list(d.feature_names)


def load_glass() -> tuple[np.ndarray, list[str]]:
    """Glass Identification dataset from UCI (214 × 10)."""
    local = Path(__file__).parent.parent / "data" / "glass.data"
    src = str(local) if local.exists() else (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    )
    cols = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    df = pd.read_csv(src, header=None, names=cols)
    features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    return df[features].values.astype(float), features


def load_haberman() -> tuple[np.ndarray, list[str]]:
    """Haberman's Survival dataset from UCI (306 × 3)."""
    local = Path(__file__).parent.parent / "data" / "haberman.data"
    src = str(local) if local.exists() else (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    )
    cols = ["Age", "OpYear", "Nodes", "Survival"]
    df = pd.read_csv(src, header=None, names=cols)
    features = ["Age", "OpYear", "Nodes"]
    return df[features].values.astype(float), features


def load_wholesale() -> tuple[np.ndarray, list[str]]:
    """Wholesale Customers dataset from UCI (440 × 8)."""
    local = Path(__file__).parent.parent / "data" / "wholesale.csv"
    src = str(local) if local.exists() else (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    )
    df = pd.read_csv(src)
    features = list(df.columns)
    return df.values.astype(float), features


def load_cardio() -> tuple[np.ndarray, list[str]]:
    """
    Cardiotocography dataset from UCI (2126 × 23).
    Reads from local data/CTG.xls (download once with curl or wget).
    Requires xlrd >= 2.0.1:  pip install xlrd
    """
    local = Path(__file__).parent.parent / "data" / "CTG.xls"
    src = str(local) if local.exists() else (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls"
    )
    df = pd.read_excel(src, sheet_name="Raw Data", header=0, engine="xlrd")
    # Drop header/footer artefact rows and select the 23 numeric features
    df = df.dropna(how="all").reset_index(drop=True)
    # The dataset has columns LB, AC, FM, UC, DL, DS, DP, ASTV, MSTV,
    # ALTV, MLTV, Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median,
    # Variance, Tendency, CLASS, NSP
    numeric_cols = [
        "LB", "AC", "FM", "UC", "DL", "DS", "DP",
        "ASTV", "MSTV", "ALTV", "MLTV",
        "Width", "Min", "Max", "Nmax", "Nzeros",
        "Mode", "Mean", "Median", "Variance", "Tendency",
        "CLASS", "NSP",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    df = df[available].dropna().head(2126)
    return df.values.astype(float), available


def load_adult() -> tuple[np.ndarray, list[str]]:
    """
    Adult (Census Income) dataset (48842 × 14).
    Categorical features are label-encoded so MIGA can treat them as discrete.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder
    dataset = fetch_openml("adult", version=2, as_frame=True)
    df = dataset.frame.dropna().reset_index(drop=True)
    for col in df.select_dtypes(include="category").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df.values.astype(float), list(df.columns)


# ------------------------------------------------------------------
# Registry for easy lookup by name
# ------------------------------------------------------------------

DATASET_LOADERS: dict[str, Callable] = {
    "Iris":      load_iris,
    "Wine":      load_wine,
    "Glass":     load_glass,
    "Haberman":  load_haberman,
    "Wholesale": load_wholesale,
    "Cardio":    load_cardio,
    "Adult":     load_adult,
}

# Columns that should never be made missing (class labels / categorical IDs).
# The paper applies MAR only to the continuous measurement features.
EXCLUDE_COLS: dict[str, list[int]] = {
    "Iris":      [],
    "Wine":      [],
    "Glass":     [9],          # Type (glass class label)
    "Haberman":  [],
    "Wholesale": [0, 1],       # Channel, Region (categorical identifiers)
    "Cardio":    [21, 22],     # CLASS, NSP (class labels)
    "Adult":     [],           # label-encoded; paper includes all features
}


def load_dataset(name: str) -> tuple[np.ndarray, list[str]]:
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name]()


# ------------------------------------------------------------------
# MAR missing mechanism  (paper Section 5)
# ------------------------------------------------------------------

def apply_mar(
    X: np.ndarray,
    pct: float,
    max_feat_frac: float = 0.5,
    seed: int | None = None,
    exclude_cols: list[int] | None = None,
) -> np.ndarray:
    """
    Introduce missing values using the MAR mechanism described in the paper.

    Parameters
    ----------
    X            : (n × m) complete dataset
    pct          : percentage of observations to remove per selected feature
                   (e.g. 30 means 30%)
    max_feat_frac: fraction of features allowed to have missing values (default 0.5)
    seed         : random seed
    exclude_cols : column indices that should never be made missing
                   (e.g. categorical identifiers, class labels).

    Returns
    -------
    X_miss : copy of X with NaN inserted
    """
    rng = np.random.default_rng(seed)
    n, m = X.shape
    eligible = [j for j in range(m) if j not in (exclude_cols or [])]
    n_feat = max(1, int(np.floor(len(eligible) * max_feat_frac)))
    feat_idx = rng.choice(eligible, size=n_feat, replace=False)

    X_miss = X.copy().astype(float)
    n_remove = max(1, int(round(n * pct / 100)))

    for j in feat_idx:
        row_idx = rng.choice(n, size=min(n_remove, n), replace=False)
        X_miss[row_idx, j] = np.nan

    return X_miss


# ------------------------------------------------------------------
# MNAR missing mechanism
# ------------------------------------------------------------------

def apply_mnar(
    X: np.ndarray,
    pct: float,
    mechanism: str = "top",
    max_feat_frac: float = 0.5,
    seed: int | None = None,
    exclude_cols: list[int] | None = None,
) -> np.ndarray:
    """
    Introduce missing values using an MNAR (Missing Not At Random) mechanism.

    Unlike MAR, the probability of a value being missing depends on the value
    itself — e.g. high-income individuals may not report income (van Buuren
    2018, §1.3.2; Sterne et al. 2009, BMJ).

    Parameters
    ----------
    X            : (n × m) complete dataset
    pct          : percentage of observations to make missing per selected feature
    mechanism    : one of:
                   'top'    — remove the top pct% of values (high values hidden)
                   'bottom' — remove the bottom pct% of values (low values hidden)
                   'tails'  — remove pct/2% from each tail (extremes hidden)
    max_feat_frac: fraction of features to apply MNAR to (default 0.5, same as MAR)
    seed         : random seed for feature selection and tie-breaking
    exclude_cols : column indices that should never be made missing

    Returns
    -------
    X_miss : copy of X with NaN inserted according to the MNAR rule
    """
    if mechanism not in ("top", "bottom", "tails"):
        raise ValueError(f"mechanism must be 'top', 'bottom', or 'tails'; got '{mechanism}'")

    rng = np.random.default_rng(seed)
    n, m = X.shape
    eligible = [j for j in range(m) if j not in (exclude_cols or [])]
    n_feat = max(1, int(np.floor(len(eligible) * max_feat_frac)))
    feat_idx = rng.choice(eligible, size=n_feat, replace=False)

    X_miss = X.copy().astype(float)
    n_remove = max(1, int(round(n * pct / 100)))

    for j in feat_idx:
        col = X[:, j]
        order = np.argsort(col)  # ascending sort

        if mechanism == "top":
            # Remove values in the top pct% — highest values go missing first
            row_idx = order[-n_remove:]
        elif mechanism == "bottom":
            # Remove values in the bottom pct% — lowest values go missing first
            row_idx = order[:n_remove]
        else:  # tails
            # Remove pct/2% from each end
            half = max(1, n_remove // 2)
            row_idx = np.concatenate([order[:half], order[-half:]])

        X_miss[row_idx, j] = np.nan

    return X_miss


# ------------------------------------------------------------------
# Automatic generators  (one per column, from available data)
# ------------------------------------------------------------------

def auto_generators(
    X_miss: np.ndarray,
    seed: int | None = None,
) -> dict[int, Callable]:
    """
    Build per-column random variable generators R_j from available data.

    Uses **bootstrap resampling** (empirical distribution) — consistent with
    the paper's phrase "generators R_j obtained from samples" (Section 3.4).

    Bootstrap resampling:
      - Samples directly from observed values, preserving the true shape
        (multimodal, skewed, discrete) without parametric assumptions.
      - For integer-valued columns with few unique values, samples from
        the observed integer values only.

    This is strictly better than N(μ, σ) for non-Gaussian features (e.g.,
    Iris petal length is trimodal; Wine features are skewed).
    """
    rng = np.random.default_rng(seed)
    generators = {}
    _, m = X_miss.shape

    for j in range(m):
        col = X_miss[:, j][~np.isnan(X_miss[:, j])]
        if len(col) == 0:
            generators[j] = lambda: 0.0
            continue

        is_int = np.all(col == np.floor(col)) and (len(np.unique(col)) <= 50)

        if is_int:
            # Sample from observed integer values (preserves discrete distribution)
            int_vals = col.copy()
            generators[j] = (lambda v=int_vals: float(rng.choice(v)))
        else:
            # Bootstrap: sample from empirical distribution of available values
            generators[j] = (lambda v=col: float(rng.choice(v)))

    return generators


# ------------------------------------------------------------------
# Evaluation metrics  (RMSE, MAD, CoD on z-standardised scale)
# ------------------------------------------------------------------

def compute_metrics(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    missing_mask: np.ndarray,
) -> dict[str, float]:
    """
    Compute RMSE, MAD, and CoD matching the paper's Tables 4/5/6.

      NRMSE_j = RMSE_j / (max_j - min_j),  averaged over features with missing values.
      NMAD_j  = MAD_j  / (max_j - min_j),  averaged over features with missing values.
      CoD     = 1 - SS_res(missing) / SS_tot(all cells)  — per paper's reported values.

    Parameters
    ----------
    X_true       : (n × p) original complete dataset
    X_imputed    : (n × p) dataset after imputation
    missing_mask : (n × p) boolean, True where values were missing

    Returns
    -------
    dict with keys: 'rmse', 'mad', 'cod'
    """
    ranges = X_true.max(axis=0) - X_true.min(axis=0)
    ranges = np.maximum(ranges, 1e-8)

    rmse_list, mad_list = [], []
    for j in range(X_true.shape[1]):
        mask_j = missing_mask[:, j]
        if mask_j.sum() == 0:
            continue
        diff = X_true[mask_j, j] - X_imputed[mask_j, j]
        rmse_list.append(float(np.sqrt(np.mean(diff ** 2))) / ranges[j])
        mad_list.append(float(np.mean(np.abs(diff))) / ranges[j])

    rmse = float(np.mean(rmse_list)) if rmse_list else float("nan")
    mad  = float(np.mean(mad_list))  if mad_list  else float("nan")

    # CoD = 1 - SS_res(missing) / SS_tot(all cells), matching paper Table 6.
    # SS_res: sum of squared errors over missing entries only.
    # SS_tot: total sum of squares over ALL n×p cells.
    mu_col  = np.nanmean(X_true, axis=0)
    ss_res  = float(np.sum((X_true[missing_mask] - X_imputed[missing_mask]) ** 2))
    ss_tot  = float(np.sum((X_true - mu_col) ** 2))
    cod     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return {"rmse": rmse, "mad": mad, "cod": cod}
