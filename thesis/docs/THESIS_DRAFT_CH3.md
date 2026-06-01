# Chapter 3 — Methodology

---

## 3.1 Overview

This chapter describes the complete implementation of MIGA from the paper specification, documents every decision made where the paper is ambiguous or silent, and presents the five novel contributions introduced in this thesis: Ledoit-Wolf shrinkage covariance (C2), adaptive mutation schedule (C3), MNAR evaluation framework (C4), kurtosis extension to the fitness function (C8), and the theoretical proposal for IPW-Fr (C4 extension). Contributions C5, C6, and C7 (baseline comparison, significance tests, variance preservation) are experimental rather than algorithmic and are described in Chapter 4.

All code is implemented in Python 3.12 using NumPy, SciPy, and scikit-learn. The implementation is structured as a Python package (`miga/`) with the following modules:

| Module | Responsibility |
|---|---|
| `miga/statistics.py` | Statistical primitives: `compute_stats`, `compute_kurtosis`, `ledoit_wolf_cov`, `pooled_std`, `relative_cov`, `minkowski_distance` |
| `miga/fitness.py` | `FitnessEvaluator`: pre-computes X_A statistics, evaluates F_r and Fr+ |
| `miga/operators.py` | `initialize_population`, `mutate`, `crossover`, `diversity`, `build_var_groups` |
| `miga/core.py` | `MIGA`: top-level estimator, orchestrates Q runs |
| `miga/data_utils.py` | Dataset loaders, `apply_mar`, `apply_mnar`, `auto_generators`, `compute_metrics` |

The full source is included in Appendix A. All experiments are reproducible via the 15 Jupyter notebooks in `notebooks/`.

---

## 3.2 Reimplementation of MIGA (C1)

### 3.2.1 Population Representation

Each individual in the genetic algorithm population represents a complete candidate imputation for all missing cells in the dataset. If the dataset has k missing cells (across all rows in X_C), an individual is a vector of k real numbers. The mapping between positions in this vector and (row, column) locations in the dataset is stored as a list `missing_index` of (local_row, col) pairs, constructed once before the GA begins.

This flat vector representation is efficient: fitness evaluation requires only reassembling X_C from the current individual and computing the three statistical distances. The population is stored as a 2D array of shape (l, k).

### 3.2.2 Fitness Evaluation

The fitness function F_r is evaluated by `FitnessEvaluator`, which pre-computes all X_A statistics at initialisation (mean, covariance, skewness, and optionally kurtosis) and evaluates each candidate X_C efficiently. Feature scaling is applied before all computations (§3.2.3).

The three terms follow Equations 5–10 of Figueroa-García et al. (2023):

**d_means** — Standardised mean distance:
$$d_{\text{means}} = D_r\!\left(\tilde{x}_A,\, \tilde{x}_C\right), \quad \tilde{x} = \frac{\bar{x}}{S_p}$$
where $S_p$ is the pooled standard deviation vector (Definition 4, Equation 7):
$$S_{p,j}^2 = \frac{\text{diag}(S_A)_j \cdot \nu_A + \text{diag}(S_C)_j \cdot \nu_C}{\nu_A + \nu_C}$$
with $\nu_A = n_A - 1$, $\nu_C = n_C - 1$.

**d_cov** — Relative covariance distance:
$$d_{\text{cov}} = D_r\!\left(\tilde{S},\, I\right), \quad \tilde{S} = S_A^{-1/2} S_C S_A^{-1/2}$$
$\tilde{S} = I$ if and only if $S_C = S_A$. The matrix square root is computed via eigendecomposition: $S_A^{-1/2} = V \Lambda^{-1/2} V^\top$ where $V, \Lambda$ are the eigenvectors and eigenvalues of $S_A$.

**d_skew** — Skewness distance:
$$d_{\text{skew}} = D_r\!\left(b_A,\, b_C\right)$$
where $b_j$ is the bias-corrected sample skewness of column $j$ (Fisher-Pearson coefficient, `scipy.stats.skew(bias=False)`).

All distances use the Minkowski order $r$ specified in Table 3 of the paper (typically $r = \infty$, the Chebyshev norm: $D_\infty(A, B) = \max_{i,j} |a_{ij} - b_{ij}|$).

### 3.2.3 Feature Scaling (Implementation Decision)

**Problem.** Wine's Proline feature has variance $\sigma^2 \approx 10^5$ while other features have $\sigma^2 \approx 1$. Without scaling, the condition number of $S_A$ is approximately $4 \times 10^7$, making $S_A^{-1/2}$ numerically meaningless and $d_{\text{cov}}(X_A, X_A) \neq 0$.

**Solution.** Before computing any statistics, divide each column of $X_A$ and $X_C$ by $\hat{\sigma}_j = \text{std}(X_A[:, j])$ (estimated from $X_A$ alone, to avoid data leakage from $X_C$). This transforms $S_A$ into a correlation matrix with eigenvalues bounded by $O(p)$, regardless of feature scales.

**Justification.** $\tilde{S} = S_A^{-1/2} S_C S_A^{-1/2}$ is theoretically scale-invariant — a global scale change $X \to cX$ leaves $\tilde{S}$ unchanged. However, per-feature scaling (which changes the relative scale of features) is not theoretically neutral. We apply it because the alternative — a numerically broken fitness function — is strictly worse. After scaling, Wine 30% condition number drops from $4 \times 10^7$ to 158 and $d_{\text{cov}}(X_A, X_A) = 0.000$.

### 3.2.4 Eigenvalue Floor in Relative Covariance

Even after feature scaling, $S_A$ can be rank-deficient when $n_A < p$ (e.g. Wine at 40%+ missing: $n_A = 11$, $p = 13$). In `relative_cov()`, eigenvalues of $S_A$ are floored at $\max(\lambda_{\max} \times 10^{-4},\; 10^{-10})$ before computing $\Lambda^{-1/2}$. This bounds the condition number of $S_A^{-1/2}$ at $10^2$, preventing catastrophic noise amplification while leaving well-conditioned covariance estimates unaffected.

### 3.2.5 Bootstrap Generators (Paper Divergence)

The paper specifies that generators $R_j$ are "obtained from samples" of the observed values of variable $j$. We interpret this as bootstrap resampling from the empirical distribution: $R_j$ draws uniformly from the set of non-missing observed values of column $j$.

For integer-valued columns (detected by checking whether all observed values are integers), the generator returns integer samples only. This is essential for datasets with discrete features: Haberman's Nodes column has 44% zeros, and a Gaussian generator $\mathcal{N}(\mu_j, \sigma_j^2)$ would never generate zero. The bootstrap generator samples zeros with probability 0.44, correctly preserving the discrete marginal distribution. This explains why our reimplementation outperforms the paper's reported result on Haberman at 30% missing (ratio 0.52× — we beat the paper — on seeds where Nodes is selected as the missing feature).

### 3.2.6 Genetic Algorithm Operators

**Selection.** The top-$c$ individuals (lowest F_r) are retained as elites unchanged.

**Mutation.** Each of the $c_1$ elites produces $c_3$ mutated copies. A mutated copy is identical to its parent except that one randomly chosen component (one missing cell value) is replaced by a fresh sample from $R_j$ for the corresponding column $j$. This is single-component mutation.

**Crossover.** $c_2$ crossover offspring are produced by uniform crossover of random pairs drawn from the elites. Each component of the offspring is taken from one parent with probability 0.5.

**Diversity injection.** The remaining $l - c - c_1 c_3 - 2(c_2 - 1)$ population slots are filled with fresh random individuals sampled entirely from the generators. Under the paper's default parameters ($l=200$, $c=3$, $c_1=3$, $c_2=2$, $c_3=5$), this is $200 - 3 - 15 - 2 = 180$ individuals, or 90% of the population. The consequence is discussed in §3.3.

### 3.2.7 Variable Group Construction

The paper applies missing data to approximately half the features per run: $\lfloor p/2 \rfloor$ features are selected uniformly at random to have values set missing. In `build_var_groups()`, this selection is fixed once before the GA begins, so all Q runs share the same missing pattern. The random seed for feature selection is inherited from the top-level `MIGA` seed.

### 3.2.8 Multi-Run Strategy

MIGA runs $Q$ independent GA instances on the same missing dataset, each with a different internal random seed derived from the top-level seed. The best individual across all $Q$ runs is returned as the final imputation. `miga.history_` stores the best F_r achieved per run; `miga.generation_history_` stores the full convergence curve (best F_r per generation) for each run, enabling convergence analysis.

---

## 3.3 Adaptive Mutation Schedule (C3)

### 3.3.1 Motivation

Standard MIGA uses a fixed $c_3$ (mutation offspring per elite) throughout all $G$ generations. In a conventional GA, the optimal strategy is to favour exploration (high mutation, high diversity) early in training and exploitation (low mutation, fine-grained search) late in training. The adaptive mutation schedule implements this intuition via a linear decay of $c_3$.

However, as noted in §3.2.6, MIGA's diversity injection already fills 90% of the population with random individuals each generation. This means $c_3$ controls only 7.5% of the population, severely limiting its impact. The adaptive schedule is most beneficial in compute-limited settings where $G$ is small, before the diversity injection has had time to dominate.

### 3.3.2 Implementation

The adaptive schedule is specified as `c3_schedule=(c3_start, c3_end)`. At generation $g \in \{0, 1, \ldots, G-1\}$:

$$c_3(g) = \text{round}\!\left(c_{3,\text{start}} + \frac{c_{3,\text{end}} - c_{3,\text{start}}}{G - 1} \cdot g\right)$$

We use $c_{3,\text{start}} = 15$, $c_{3,\text{end}} = 3$: high exploration early, fine exploitation late. The population size sanity check is applied to $c_{3,\text{start}}$ (the maximum) to ensure the population can always accommodate all operator outputs.

### 3.3.3 Results

Experiments on Iris, Glass, and Haberman at $l=200$, $G=500$, $Q=5$, seed=42, 30% MAR:

| Dataset | Fixed ($c_3=5$) RMSE | Adaptive (15→3) RMSE | Winner |
|---|---|---|---|
| Iris | 0.1270 | 0.1273 | Fixed (+0.2%) |
| Glass | 0.1656 | 0.1821 | Fixed (+10%) |
| Haberman | 0.3649 | 0.4164 | Fixed (+14%) |

At full compute ($G=500$), fixed $c_3$ wins on all three datasets. The diversity injection dominates: with 180/200 slots filled randomly each generation, the contribution of $c_3$ is marginal regardless of its value.

At reduced compute ($G=80$, Iris): adaptive 15→3 achieves RMSE=0.1198 vs fixed RMSE=0.1415 — a 15% improvement. The convergence plots (Figure 3.1) confirm that adaptive scheduling provides faster early descent in F_r, visible in the first 50 generations, before the curves converge at full compute.

**Honest framing.** The adaptive schedule is a genuine improvement under compute-limited budgets ($G \leq 100$) but offers no benefit at full training length. This reveals a structural property of MIGA: the primary exploration-exploitation control is the diversity injection rate, not the mutation rate. A more impactful future extension would be adaptive diversity injection (§5.4).

---

## 3.4 Ledoit-Wolf Shrinkage Covariance (C2)

### 3.4.1 The Rank-Deficiency Problem

The relative covariance term $d_{\text{cov}} = D_r(S_A^{-1/2} S_C S_A^{-1/2}, I)$ requires computing $S_A^{-1/2}$, which is undefined when $S_A$ is rank-deficient. $S_A$ becomes rank-deficient when $n_A < p$ — i.e., when there are fewer complete rows than features.

For Wine at 40% missing: $n_A = 11$, $p = 13$, so $S_A$ has rank at most 10. The sample covariance has condition number $5.7 \times 10^8$ and $d_{\text{cov}}(X_A, X_A) = 0.606$, a mathematical impossibility since $d_{\text{cov}}$ should equal zero when both arguments are the same set.

### 3.4.2 The Ledoit-Wolf Estimator

The Ledoit-Wolf (2004) shrinkage estimator replaces the sample covariance with a convex combination:

$$S_{\text{LW}} = (1 - \alpha)\, S_{\text{sample}} + \alpha \cdot \frac{\text{tr}(S_{\text{sample}})}{p} \cdot I$$

where $\alpha \in [0, 1]$ is the shrinkage coefficient chosen analytically to minimise the expected squared Frobenius loss under a Gaussian model (Ledoit and Wolf, 2004). The target $(\text{tr}(S)/p) \cdot I$ is always positive definite, so $S_{\text{LW}}$ is positive definite for any $\alpha > 0$.

We apply the Ledoit-Wolf estimator to **both** $S_A$ and $S_C$ when `cov_estimator='ledoit_wolf'`. Applying it to $S_A$ alone would restore positive definiteness of $S_A^{-1/2}$ but break the invariant $d_{\text{cov}}(X_A, X_A) = 0$: using LW for $S_A$ but sample covariance for $S_C$ gives $\tilde{S} = S_{\text{LW},A}^{-1/2} S_{\text{sample},C} S_{\text{LW},A}^{-1/2} \neq I$ even when $X_C = X_A$.

### 3.4.3 Results

| Setting | Condition number | $d_{\text{cov}}(X_A, X_A)$ | $\alpha$ |
|---|---|---|---|
| Sample, 30% missing ($n_A=23$) | 158 | 0.000 ✓ | — |
| Sample, 40% missing ($n_A=11$) | 570,000,000 | 0.606 ✗ | — |
| Ledoit-Wolf, 30% ($n_A=23$) | 9.1 | 0.000 ✓ | 0.34 |
| Ledoit-Wolf, 40% ($n_A=11$) | 8.8 | 0.000 ✓ | 0.42 |

Ledoit-Wolf restores mathematical consistency at 40% missing. RMSE is not improved — $n_A = 11$ is an insufficient sample size to reliably estimate the joint distribution of 13 features, regardless of the covariance estimator. The shrinkage contribution is principled (it solves a real mathematical problem) but practically limited to datasets where $n_A < p$.

---

## 3.5 MNAR Evaluation Framework (C4)

### 3.5.1 Implementation of MNAR Mechanisms

The paper evaluates MIGA only under MAR. We implement three MNAR mechanisms in `apply_mnar(X, pct, mechanism)`:

**top:** For each selected feature $j$, the $n_{\text{remove}}$ rows with the highest values of $X_{:,j}$ are set missing. Formally, $\text{row\_idx} = \text{argsort}(X_{:,j})[-n_{\text{remove}}:]$.

**bottom:** The $n_{\text{remove}}$ rows with the lowest values are set missing: $\text{row\_idx} = \text{argsort}(X_{:,j})[:n_{\text{remove}}]$.

**tails:** Half the removals come from each extreme: $\text{half} = \max(1, \lfloor n_{\text{remove}}/2 \rfloor)$, $\text{row\_idx} = \text{argsort}(X_{:,j})[:{\text{half}}] \cup \text{argsort}(X_{:,j})[-{\text{half}}:]$.

Each mechanism selects the same number of missing cells as the corresponding MAR experiment (30% of cells in the selected features). The same features are selected for missingness as under MAR (controlled by the `miss_seed` parameter).

### 3.5.2 Why MNAR Breaks MIGA's Objective

Under MAR, $X_A$ is a representative sample of the full population: $\mathbb{E}[X_A] \approx \mathbb{E}[X]$. The fitness function's goal — make $X_C$ look like $X_A$ — is equivalent to making $X_C$ look like the true population.

Under MNAR (specifically `top`), rows with high values of the selected features are removed from $X_C$ and placed in $X_A$. Wait — this is the reverse: the rows with high values go *missing*, so they leave $X_A$ and enter $X_C$ with missing values. $X_A$ now contains only rows with low values for those features. The fitness function instructs the GA to match $X_C$ to this biased $X_A$. The GA succeeds — it achieves a very low $F_r$ — but the imputed values are systematically too low, because the true missing values were the high-value rows.

This failure mode is non-detectable from within the fitness function: $F_r$ has no access to the true missing values and cannot identify that $X_A$ is a biased sample of the population.

### 3.5.3 Proposed Fix: Inverse Probability Weighted Fr (Theoretical)

The standard fix for selection bias in observational data is inverse probability weighting (IPW; Horvitz and Thompson, 1952). Let $\pi_i = P(R_i = 1 \mid X_i)$ be the probability that row $i$ is complete (i.e., included in $X_A$). Under MNAR, $\pi_i$ depends on $X_i$. If $\pi_i$ were known, we could construct an IPW version of the fitness function:

$$F_r^{\text{IPW}} = D_r\!\left(\tilde{x}_A^w,\, \tilde{x}_C\right) + D_r\!\left(\tilde{S}^w,\, I\right) + D_r\!\left(b_A^w,\, b_C\right)$$

where superscript $w$ denotes statistics computed with sample weights $w_i = 1/\pi_i$, up-weighting the rare high-value complete rows to correct for their under-representation in $X_A$. In practice, $\pi_i$ must be estimated from the data using a propensity score model (logistic regression of completeness on observed covariates). Full implementation of IPW-Fr is identified as the primary direction for future work (§5.4).

---

## 3.6 Kurtosis Extension: Fr+ (C8)

### 3.6.1 Motivation

The original fitness function F_r matches the first three moments of the distribution: mean (1st), covariance structure (2nd), and skewness (3rd). Kurtosis — the fourth standardised moment — captures tail behaviour: distributions with excess kurtosis $>0$ have heavier tails than a Gaussian; those with excess kurtosis $<0$ have lighter tails.

MICE's conditional mean imputation not only suppresses variance (2nd moment) but also collapses tail behaviour: by replacing missing values with conditional means, MICE eliminates the tails of the conditional distribution. This suppresses kurtosis, particularly for features with heavy-tailed or zero-inflated marginal distributions (e.g. Haberman's Nodes column: 44% zeros, extreme positive kurtosis).

### 3.6.2 Implementation

We add a fourth term to the fitness function:

$$F_r^+ = F_r + D_r(k_A,\, k_C)$$

where $k_j$ is the bias-corrected excess kurtosis of column $j$ (`scipy.stats.kurtosis(bias=False, fisher=True)`; $k=0$ for a normal distribution). The term is computed after feature scaling (so kurtosis is estimated from standardised columns). Zero-variance or degenerate columns (which produce non-finite kurtosis) are treated as $k_j = 0$.

The extension is controlled by `MIGA(use_kurtosis=True)` / `FitnessEvaluator(use_kurtosis=True)`, defaulting to `False` for backward compatibility with all existing notebooks and experiments.

### 3.6.3 Results

Experiments on four datasets (10 seeds, $l=200$, $G=300$, $Q=3$, 30% MAR):

| Dataset | MICE $d_{\text{kurt}}$ | MIGA-Fr $d_{\text{kurt}}$ | MIGA-Fr+ $d_{\text{kurt}}$ | Fr+ improves? | RMSE cost |
|---|---|---|---|---|---|
| Iris ($p=4$) | 0.683 | 0.716 | **0.638** ★ | Yes ($\Delta=0.078$) | −0.002 |
| Glass ($p=10$) | **45.60** | 45.81 | 45.66 | Marginal ($\Delta=0.15$) | +0.004 |
| Haberman ($p=3$) | 7.783 | **1.854** | **1.854** | No (already optimal) | +0.004 |
| Wholesale ($p=8$) | 129.08 | 129.08 | 129.08 | No (all identical) | +0.001 |

Three findings emerge. First, on Iris, Fr+ genuinely improves kurtosis matching and beats MICE on $d_{\text{kurt}}$ at negligible RMSE cost. Second, on Haberman, standard MIGA-Fr already achieves $d_{\text{kurt}} = 1.854$ versus MICE's $d_{\text{kurt}} = 7.783$ — a 4.2× advantage — without any explicit kurtosis optimisation. MICE's regression to the conditional mean collapses the heavy tail of the Nodes column more severely than it collapses variance. Adding kurtosis to the objective gives no further benefit because MIGA has already reached the kurtosis minimum. Third, on Wholesale, the X_A kurtosis is extreme (Detergents\_Paper: $k = 65.3$) due to the highly right-skewed distribution of retail spending — no imputation method can match this kurtosis from 30% missing data.

**Contribution framing.** The algorithmic improvement from Fr+ is most pronounced on low-dimensional, clean datasets. The diagnostic finding is broader: kurtosis reveals MICE's suppression of tail behaviour on heavy-tailed features, complementing the variance preservation analysis (C7) and providing a more complete characterisation of how MICE distorts the marginal distribution.

---

## 3.7 Summary of Implementation Decisions

The table below lists every decision that diverges from the paper specification, with the rationale and effect on results.

| Decision | Paper specification | Our implementation | Effect |
|---|---|---|---|
| Feature scaling | Not mentioned | Divide by $\hat{\sigma}_j$ from $X_A$ | Stabilises $d_{\text{cov}}$; required for Wine |
| Eigenvalue floor | Not mentioned | $\lambda_{\min} = \max(\lambda_{\max} \times 10^{-4}, 10^{-10})$ | Prevents division by zero |
| Generator type | "Obtained from samples" | Bootstrap from observed values | Preserves discrete/skewed marginals |
| Covariance estimator | Sample covariance | Sample (default) or Ledoit-Wolf (C2) | LW restores $d_{\text{cov}}=0$ at $n_A < p$ |
| Categorical exclusion | Not mentioned | Categorical columns never made missing | Required for Wholesale, Glass, Cardio |
| Skewness | Not specified | `scipy.stats.skew(bias=False)` | Matches bias-corrected definition |
| Kurtosis (C8) | Not in paper | `scipy.stats.kurtosis(bias=False, fisher=True)` | New 4th-moment term |
| Stopping criterion | Not specified | Fixed $G$ generations | No early stopping (future work) |

All decisions are documented in detail in `docs/METHODS.md`.

---

*[Chapter 3 draft complete. Figures: Figure 3.1 — convergence plots (figures/convergence.png). Tables to be formatted to institution style during final writing.]*
