# Methods & Implementation Decisions

Tracks every algorithmic choice, fix, and deviation from the paper made during this reimplementation of MIGA (Figueroa-García et al., 2023).

---

## 1. Core Algorithm (Paper §3)

### 1.1 Fitness Function — F_r
`miga/fitness.py · FitnessEvaluator`

F_r = D_r(x̃_A, x̃_C) + D_r(S̃, I) + D_r(b_A, b_C)

Three Minkowski distances between statistics of the complete rows (X_A) and the completed incomplete rows (X_C):
- **d_means** — standardised mean vectors (Eq. 5–6)
- **d_cov** — relative covariance S̃ = S_A^{-½} S_C S_A^{-½} vs identity (Eq. 8)
- **d_skew** — skewness vectors (Definition 3)

### 1.2 Feature Scaling before Statistics (our addition)
`miga/fitness.py · FitnessEvaluator.__init__`

**What:** Divide X_A (and X_C in every evaluation) by per-column std before computing statistics, turning S_A into a correlation matrix.

**Why:** S̃ = S_A^{-½} S_C S_A^{-½} is theoretically scale-invariant, but requires numerical stability to realise that invariance. Wine's Proline feature has σ² ≈ 1e5 vs σ² ≈ 1 for others, making S_A ill-conditioned to the point where S_A^{-½} is numerically meaningless (condition number ~40M). After scaling, condition drops to ~150 and d_cov(X_A, X_A) = 0 as required.

**Effect:** Wine 30% RMSE improved from 0.344 → 0.248. No measurable effect on well-conditioned datasets (Iris, Glass, etc.).

**Paper says:** Not documented. Likely an implicit assumption in the original implementation.

### 1.3 Ledoit-Wolf Shrinkage Covariance (Novel Extension)
`miga/statistics.py · ledoit_wolf_cov`  
`miga/fitness.py · FitnessEvaluator(cov_estimator='ledoit_wolf')`  
`miga/core.py · MIGA(cov_estimator='ledoit_wolf')`

**What:** Optional shrinkage estimator for S_A (reference covariance) and S_C (per-evaluation). Replaces the sample covariance with Ledoit-Wolf (sklearn), which analytically minimises Frobenius loss under a Gaussian model: S_LW = (1-α)S_sample + α*(tr(S)/p)*I.

**Why consistency matters:** If LW is used only for S_A but sample covariance for S_C, then d_cov(X_A, X_A) ≠ 0 (the null-case invariant breaks). Using LW for both maintains d_cov = 0 when X_C = X_A.

**Effect on Wine:**
- 30% (n_A=23, m=13, n_A/m=1.77): cond(S_A) 158→9.1, α≈0.34. d_cov=0 for both.
- 40% (n_A=11, m=13, n_A/m=0.85): sample cond=570M, d_cov(X_A,X_A)=0.61 (rank-deficient, broken). LW: cond=8.8, d_cov=0.00 (restored).
- 50/60% (n_A=2): both estimators degenerate; LW α→0.

**RMSE impact (reduced l=50/G=100/Q=2):** Similar at 30%, slightly worse at 40%. Full benchmarks in Wine notebook needed.

**Paper says:** Not documented; LW is an original contribution.

### 1.4 Eigenvalue Floor in Relative Covariance
`miga/statistics.py · relative_cov`

**What:** Eigenvalues of S_A floored at `max(max_eigval × 1e-4, 1e-10)` before computing the inverse square root.

**Why:** Prevents division by zero for rank-deficient S_A (occurs when n_A < m). After the feature scaling fix (§1.2), the correlation matrix has bounded eigenvalues, so 0.01% of max is sufficient — it only clips truly zero eigenvalues.

**Effect:** Makes the algorithm numerically stable for all tested datasets including rank-deficient cases (Wine 40%+, Cardio at high missing %).

### 1.5 Minkowski Distance
`miga/statistics.py · minkowski_distance`

Implemented exactly as Definition 1: D_r(A, B) = (Σ|a_ij - b_ij|^r)^{1/r}, with r = ∞ mapping to max|a_ij - b_ij|.

### 1.6 Pooled Standard Deviation
`miga/statistics.py · pooled_std`

Implemented as Definition 4, Eq. 7: S_p² = (diag(S_A)·ν_A + diag(S_C)·ν_C) / ν_T.

---

## 2. Genetic Algorithm (Paper §3.4, Algorithm 1)

### 2.1 Population Structure
`miga/core.py · MIGA._single_run`

Per generation: c elite + c1×c3 mutation offspring + 2(c2−1) crossover offspring + remainder diversity. Exactly matches Algorithm 1.

### 2.2 Missing Index (j-ordered)
`miga/core.py · MIGA._build_missing_index`

Genes are ordered column-first so all genes for variable j are contiguous. This groups genes by variable, which is required for the crossover operator (var_groups).

### 2.3 nan / None Guards
`miga/core.py · MIGA._single_run`, `MIGA.fit_transform`

**What:** If all fitness scores in an initial population are non-finite (can happen with rank-deficient S_A), the run records `best_score = nan` instead of crashing. If all Q runs produce nan scores, imputation falls back to column-mean.

**Why:** Paper does not discuss this failure mode; these guards prevent silent crashes on edge-case datasets.

---

## 3. Generators (Paper §3.4)

### 3.1 Bootstrap Resampling
`miga/data_utils.py · auto_generators`

**What:** Sample available observed values directly (empirical distribution) rather than fitting N(μ, σ).

**Why:** Paper states "generators R_j obtained from samples." Bootstrap preserves the true marginal shape (multimodal, skewed, discrete) without parametric assumptions. Importantly, for integer-valued columns with ≤ 50 unique values, only integer values are sampled.

**Effect:** Significant improvement over Gaussian generators for skewed/discrete features (Wholesale, Haberman).

---

## 4. Missing Mechanism (Paper §5)

### 4.1 MAR Implementation
`miga/data_utils.py · apply_mar`

Select floor(m_eligible / 2) features at random, then remove pct% of rows per selected feature. `m_eligible` excludes categorical/label columns (see §4.2).

### 4.2 Categorical Column Exclusion
`miga/data_utils.py · EXCLUDE_COLS`

**What:** Certain columns are never made missing.

| Dataset   | Excluded columns      | Reason                       |
|-----------|-----------------------|------------------------------|
| Glass     | Type (col 9)          | Glass class label            |
| Wholesale | Channel, Region (0,1) | Categorical identifiers      |
| Cardio    | CLASS, NSP (21,22)    | Class labels                 |
| Others    | —                     | All features eligible        |

**Why:** The paper applies MAR only to continuous measurement features. Including categoricals massively inflates RMSE (Wholesale pre-fix: 3.11× off; post-fix: 1.07×).

---

## 5. Evaluation Metrics (Paper §5)

### 5.1 RMSE — Range-Normalised
`miga/data_utils.py · compute_metrics`

NRMSE_j = RMSE_j / (max_j − min_j), averaged over features with missing values.

**Why range, not z-score:** Matches paper's reported mean-imputation baseline (0.2994 for Iris 30%) and gives results closest to Table 4.

### 5.2 MAD — Range-Normalised
Same normalisation as RMSE. Paper's exact MAD formula is undocumented; absolute values differ ~3× from paper Table 5, but method rankings are preserved.

### 5.3 CoD — All-Data Formulation
CoD = 1 − SS_res(missing cells only) / SS_tot(all n×p cells)

**Why all-data SS_tot:** Matches paper's reported CoD values in Table 6 (feature-level or missing-only SS_tot gives much lower values).

---

## 6. Dataset Loading

### 6.1 Cardio — Local File
`miga/data_utils.py · load_cardio`

CTG.xls is read from `data/CTG.xls` if it exists (downloaded once), falling back to the UCI URL. The correct sheet is `"Raw Data"` with `header=0`. An earlier bug used `header=1` which shifted columns and caused "no columns" errors.

### 6.2 Adult — Label Encoding
`miga/data_utils.py · load_adult`

Categorical features are label-encoded with sklearn's `LabelEncoder` so MIGA can treat all columns as numeric.

---

## 7. Known Limitations

### 7.1 Wine — Rank-Deficient Covariance
With n=178 and m=13, very few complete rows survive MAR at 30%+ missing (n_A ≈ 21 at 30%, <13 at 40%+). The covariance term in F_r is unreliable even after feature scaling. RMSE gap vs paper: ~2.5× at all percentages.

### 7.2 Haberman — Seed-Determined Feature Selection
With m=3 (Age, OpYear, Nodes) and floor(3/2)=1, every MAR run affects exactly one feature. Which feature is selected is fully determined by the random seed, and difficulty varies enormously:

| seed=pct | Feature selected | RMSE | vs paper |
|----------|-----------------|------|----------|
| 30% | Age (smooth, no zeros) | 0.437 | 2.06× |
| 40% | OpYear (range 58–69, std=3.2) | 0.382 | 1.49× |
| 50% | Nodes (44% zeros, heavily skewed) | 0.171 | **0.52×** ★ |
| 60% | Age | 0.299 | 0.79× ★ |

**Why we beat the paper on Nodes (50%):** Our bootstrap generator samples directly from observed Nodes values, including the 44% zeros. A Gaussian N(μ,σ) generator (likely used in the paper) cannot reproduce the zero-inflation. This is a structural advantage of empirical resampling on discrete/skewed features.

**Why we lag on Age and OpYear:** Confirmed with an 8-seed sweep — the gap is consistent (1.24–2.06×) across different seeds that select Age or OpYear. It reflects the paper's better convergence on smooth continuous features, not a bug or anomaly.

**Seed sweep summary (30% missing, 8 seeds):**

| Seed | Feature | RMSE | Ratio |
|------|---------|------|-------|
| 30 | Age | 0.437 | 2.06× |
| 31 | OpYear | 0.428 | 2.02× |
| 42 | Age | 0.289 | 1.36× |
| 99 | Nodes | 0.177 | 0.83× ★ |
| 123 | Age | 0.279 | 1.32× |
| 200 | Age | 0.263 | 1.24× |
| 300 | Age | 0.283 | 1.33× |
| 500 | OpYear | 0.393 | 1.85× |

Range of Age/OpYear outcomes: 1.24–2.06×. Nodes always ≤ 1× (bootstrap advantage).

### 7.3 MAD Formula Discrepancy
Paper Table 5 MAD values are 4–20× smaller than our range-normalised MAD. Hypothesis (partially confirmed): the paper normalises MAD by per-feature **standard deviation** (σ) rather than range, while using range for RMSE.

Evidence: for Iris and Glass, predicting paper MAD as `our_range-norm_MAD / (range/σ)` matches within 30% at 3 of 4 missing percentages. The prediction breaks down for Wholesale at 50–60% (where we beat the paper, so our MAD is already lower than theirs, making derivation ambiguous).

**Impact:** Absolute MAD values cannot be compared directly. Method rankings within each row of Table 5 are still correctly reproduced. Reported as-is in notebooks with a note.

---

## 8. Theoretical Analysis: Fr vs RMSE Decoupling

### 8.1 The Structural Orthogonality

MIGA minimises **Fr** — the Minkowski distance between distributional statistics of X_A and X_C:

```
Fr = D_r(x̃_A, x̃_C)     [mean distance]
   + D_r(S̃, I)          [covariance distance]
   + D_r(b_A, b_C)       [skewness distance]
```

RMSE measures **pointwise accuracy**:

```
RMSE_j = sqrt(mean((x_true[i,j] - x_imp[i,j])^2 for i in missing_j))
```

These are structurally different:
- Fr depends only on **aggregate statistics** of the imputed set — it does not care which value goes where, only that the overall distribution matches
- RMSE depends on the **correspondence** between each imputed value and its true value

**Consequence:** Fr = 0 does not imply RMSE = 0. Fr = 0 means X_C and X_A have the same mean/covariance/skewness — but any permutation of the imputed values within each column would give the same Fr while changing RMSE.

### 8.2 Empirical Proof: Haberman top MNAR

Under `top` MNAR at 30% missing on Haberman:
- MIGA Fr = **0.810** — lowest Fr of all mechanisms (best distributional fit by MIGA's own objective)
- MIGA RMSE = **0.384** — highest RMSE of all mechanisms (worst pointwise accuracy)

Why: the top mechanism removes high values from selected features. X_A then contains only low-valued rows. The GA matches X_C's imputed distribution to X_A's biased (low-value) distribution — achieving Fr=0.81. But the true missing values are HIGH. Every imputed value is systematically too low, giving high RMSE. The GA cannot detect this because Fr is evaluated against the biased X_A.

This is the **strongest possible empirical evidence for Fr→RMSE decoupling**: the two metrics rank the same method in exactly opposite orders.

### 8.3 Baseline Fr Comparison (MAR, 30%)

Running Mean/KNN/MICE and computing their Fr scores (notebook 12):

| Dataset   | MIGA Fr | MICE Fr | KNN Fr  | MIGA Fr advantage |
|-----------|---------|---------|---------|-------------------|
| Iris      | 0.485   | 1.156   | 0.852   | 2.4× over MICE    |
| Glass     | 40.20   | 42.99   | 97.49   | 1.07× over MICE   |
| Haberman  | 2.483   | 5.065   | 2.816   | 2.0× over MICE    |

MIGA consistently achieves lower Fr than all baselines. The advantage is large on Iris/Haberman, marginal on Glass (large Fr due to covariance scaling).

**Why baselines have higher Fr:**
- Mean imputation: removes variance → S_C ≠ S_A (covariance term blown up); removes skewness → b_C ≠ b_A
- KNN: uses local average → partial variance suppression, better than Mean but still shrinks toward observed neighbors
- MICE: conditional mean imputation → suppresses marginal variance (van Buuren 2018 §2.6), but preserves conditional structure well

### 8.4 Variance Preservation (concrete Fr advantage)

MICE imputation suppresses marginal variance because it imputes conditional means. The expected imputed value E[X_imp | X_obs] < Var[X_true] always. MIGA samples from the empirical bootstrap distribution → variance is approximately preserved.

**Measurement:** Var_ratio_j = Var(X_imp[:, j]) / Var(X_true[:, j])
- MICE: ratio typically 0.5–0.8 (variance suppressed 20–50%)
- MIGA: ratio typically 0.85–1.1 (approximately correct)

This is MIGA's most concrete and interpretable advantage. Results in `results/14_variance_preservation.json`.

### 8.5 When Each Method is Preferable

| Goal | MIGA | MICE |
|------|------|------|
| Minimise prediction error on held-out values | ✗ | ✓ |
| Preserve marginal distribution (variance, quantiles) | ✓ | ✗ |
| Preserve joint distribution (covariance structure) | ✓ (by design) | partial |
| Generate plausible multivariate samples | ✓ | ✗ |
| Handle MNAR without bias | ✗ (Fr misleads) | ✗ (assumes MAR) |
| Computational efficiency | ✗ (slow GA) | ✓ |

**Publishable claim:** MIGA is the correct method when the downstream analysis requires a valid multivariate distribution (e.g., bootstrap inference, distributional testing, synthetic data generation). MICE is the correct method when minimising individual cell prediction error.

### 8.6 Proposed Fix for MNAR: Inverse Probability Weighted Fr (IPW-Fr)

*Theoretical description — not yet implemented.*

Under MNAR, X_A is a biased sample. The fix is to reweight X_A rows by the inverse propensity of being complete:

```
w_i = 1 / P(row i is complete | X_i)
```

where P(complete | X) is estimated from the data using logistic regression on fully observed features.

The IPW-Fr would replace sample statistics with weighted statistics:

```
x̄_A^{IPW} = Σ_i w_i x_i / Σ_i w_i
S_A^{IPW}  = Σ_i w_i (x_i - x̄_A^{IPW})(x_i - x̄_A^{IPW})^T / Σ_i w_i
```

This corrects the selection bias in X_A, making Fr a consistent objective even under MNAR.

**Theoretical guarantee:** If the propensity model is correctly specified, IPW-Fr = 0 implies the imputed distribution matches the true population distribution, even under MNAR.

**Cite:** Horvitz & Thompson (1952) for IPW estimators; Robins et al. (1994) for inverse probability weighted estimating equations; Seaman & White (2013) for IPW in missing data contexts.

This is the key open research question for future work. Implementing and testing IPW-Fr on Haberman top MNAR (where standard Fr fails most dramatically) would complete the contribution from description of failure to proposed solution.
