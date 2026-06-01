# MIGA Thesis — Complete Defense Study Guide

This is a comprehensive preparation guide. It covers the mathematics, every
implementation decision with reasoning, every key result with actual numbers,
the formal theorem proof, the synthetic experiment, and 40+ committee
questions with detailed answers. Read it front to back once, then use the
Q&A section for daily revision.

---

## PART 1: THE PROBLEM — WHAT IS MISSING DATA IMPUTATION?

### What the problem is
A dataset has some cells with no observed value. You need to fill them in
before you can do downstream analysis. The question is: fill them with what?

### The three classical strategies
1. **Complete case analysis** — delete any row with a missing value.
   - Problem: loses data, introduces selection bias (the missers are not
     a random sample of the population).

2. **Single imputation** — fill each missing value with one estimate (e.g.,
   column mean).
   - Problem: treats imputed values as observed. Deflates variance.
     Standard errors are too small. Confidence intervals are too narrow.

3. **Multiple imputation** — generate M plausible complete datasets, run
   analysis on each, combine results using Rubin's rules.
   - MICE is the standard method for this.
   - Produces correct standard errors when done properly.

### What Rubin's rules say
When you have M imputed datasets and an estimate θ̂_m from each:
- Combined estimate: θ̄ = (1/M) Σ θ̂_m
- Total variance = within-imputation variance + between-imputation variance
- The between-imputation term accounts for the uncertainty due to missingness.

This only works correctly if the imputation model is "proper" — meaning it
draws from the posterior predictive distribution, not from its mean.

**MICE imputes conditional means, which is improper** — it underestimates
between-imputation variance.

### The three missingness mechanisms (formally)
Let Y = complete data, Y_obs = observed, Y_mis = missing, R = response
indicator (R_ij = 1 if observed, 0 if missing).

- **MCAR** (Missing Completely At Random): P(R | Y) = P(R)
  The missingness probability does not depend on any data value.
  Example: sensor failure at random times.

- **MAR** (Missing At Random): P(R | Y) = P(R | Y_obs)
  Missingness depends only on observed values, not on the missing values
  themselves.
  Example: income is more often missing for younger respondents, and age is
  observed.

- **MNAR** (Missing Not At Random): P(R | Y) = P(R | Y_obs, Y_mis)
  Missingness depends on the missing value itself.
  Example: high-income people are less likely to report income.

Most methods (including MICE and MIGA) assume MAR. Under MNAR, the
reference distribution X_A is biased — this is exactly what the Haberman
top MNAR experiment exposes.

---

## PART 1B: X_A AND X_C — WHAT THEY LOOK LIKE CONCRETELY

### Setup (Iris, 30% MCAR, 2 features missing)
Original Iris (150 × 4), features: sepal_length, sepal_width, petal_length, petal_width.

After applying missing mask, dataset looks like (NaN = missing):
```
row 0:   5.1,  3.5,  1.4,  0.2      ← complete → goes into X_A
row 1:   4.9,  3.0,  NaN,  0.2      ← incomplete → goes into X_C
row 2:   4.7,  3.2,  1.3,  0.2      ← complete → X_A
row 3:   NaN,  3.1,  1.5,  0.2      ← incomplete → X_C
...
row 148: 6.2,  3.4,  5.4,  2.3      ← complete → X_A
```

X_A = all rows with zero NaNs (approximately 75 rows for a well-conditioned
      30% MCAR — actual number varies by seed and which rows got masked).

X_C = all rows with ≥1 NaN — after MIGA runs, NaN cells are filled by the
      best chromosome. The rows are now fully observed numbers.

**Critical point:** X_A and X_C are disjoint by construction — no row appears
in both. They partition the dataset by whether that row had any missing value.

**Why compare X_C to X_A and not to the full dataset?**
The full dataset is not available (it has NaNs). X_A is the only clean
reference we have. MIGA's entire premise is that complete rows and incomplete
rows came from the same underlying population, so X_C should look like X_A.

This assumption holds under MCAR and MAR. It breaks under directional MNAR
(e.g., top MNAR: X_A only has low-value rows, but X_C should have high-value
rows — the premise is violated).

---

## PART 2: THE Fr FITNESS FUNCTION — DEEP DIVE

### The full formula
```
F_r = D_r(x̃_A, x̃_C) + D_r(S̃, I) + D_r(b_A, b_C)
```

Each term is a Minkowski-r distance. The algorithm minimises F_r.

### Term 1: Mean distance  D_r(x̃_A, x̃_C)
- x̃_A = standardised mean vector of complete rows X_A
- x̃_C = standardised mean vector of imputed rows X_C
- Standardised means: x̃_j = x̄_j / σ_j where σ_j is estimated from X_A
- This term penalises mean shift between the two subsets.

### Term 2: Covariance distance  D_r(S̃, I)
- S̃ = S_A^{-1/2} S_C S_A^{-1/2}   (the relative covariance matrix)
- S_A = covariance matrix of X_A
- S_C = covariance matrix of X_C (after imputation)
- I = identity matrix
- When imputed perfectly: S_C = S_A → S̃ = I → D_r(S̃, I) = 0
- The distance from S̃ to identity is the covariance mismatch.
- Why relative covariance and not just |S_A - S_C|? Because the relative
  form is scale-invariant in the correct sense — it measures the ratio of
  the two covariances, not their raw difference.
- The Minkowski distance is applied element-wise to the matrix vectorised.

### Term 3: Skewness distance  D_r(b_A, b_C)
- b_A = skewness vector of X_A (per-column skewness)
- b_C = skewness vector of X_C (per-column skewness)
- Penalises asymmetry mismatch.

### The Minkowski parameter r
- r=∞ (Chebyshev): max over all elements. Most sensitive to the worst
  component. Used for Iris, Glass, Haberman.
- r=1 (Manhattan): sum of absolute values. Used for Wine. This is why
  Wine significance takes 98 min/seed — the covariance term has O(p² n²)
  cost with r=1.
- r=2 (Euclidean): standard L2.
- The paper specifies r per dataset. We follow the paper's choice.

### Why minimise this specific F_r and not RMSE?
RMSE measures: "how close is each imputed value to its true value?"
F_r measures: "how similar are the distributional properties of
imputed rows vs complete rows?"

These are different questions. A committee may not see them as RMSE does
not use the true values during imputation (they are missing!), while F_r
uses only observed structure (X_A). F_r is computable without knowing the
truth. RMSE is not computable at imputation time — it requires knowing what
the missing values actually were.

---

## PART 3: THE GENETIC ALGORITHM — EVERY DETAIL

### Parameters
| Symbol | Meaning | Paper default |
|--------|---------|---------------|
| l | Population size | 200 |
| G | Generations per run | 300–500 |
| Q | Independent runs | 3–5 |
| c | Elites kept | floor(l * 0.1) = 20 |
| c1 | Mutation offspring | floor(l * 0.05) = 10 |
| c2 | Crossover offspring | floor(l * 0.05) = 10 |
| c3 | Diversity injection | l - c - c1 - c2 ≈ 160 |
| r | Minkowski order | Dataset-specific |

### Why Q independent runs?
Each run is seeded independently and may converge to a different local
optimum. The best of Q runs (lowest F_r) is taken. This is equivalent to
running the algorithm Q times and choosing the winner — not combining outputs.

### Chromosome representation
Each individual = a **flat numpy array of length k**, where k = total number
of missing cells across the entire dataset (not per row — global count).

**Concrete example (Iris, 30% MCAR, 2 missing features):**
- n = 150 rows. floor(p/2) = 2 features selected for missingness.
- floor(n/2) = 75 rows are made missing for each selected feature.
- k = 75 (feature 0: sepal_length) + 75 (feature 2: petal_length) = 150 cells.

**missing_index** is a list of (row, column) tuples in the order the NaNs
were scanned:
```
missing_index = [(3, 0), (7, 0), (12, 0), ..., (148, 0),   # ← 75 entries, sepal_length
                 (1, 2), (5, 2), (10, 2),  ..., (149, 2)]  # ← 75 entries, petal_length
```

One chromosome (individual) looks like this:
```
chromosome = [4.8, 5.2, 6.1, 5.5, ..., | 3.0, 4.5, 3.8, 4.1, ...]
              ←——— 75 values for sepal_length ———→←——— 75 for petal_length ———→
```
Each value is a plausible imputed value for that specific missing cell.

**To evaluate a chromosome's fitness:**
1. Start with the dataset where NaN positions are known.
2. For i in range(k): `dataset[missing_index[i]] = chromosome[i]`
3. Now the dataset is fully filled in.
4. Split into X_A (rows that were complete originally) and
   X_C (rows that had at least one NaN — now filled by this chromosome).
5. Compute F_r(X_A, X_C).
Lower F_r = better chromosome.

### Step-by-step F_r computation (what actually runs in code)
Given X_A (n_A × p) and X_C (n_C × p), both fully observed:

**Step 1 — Scale both matrices by X_A's column std:**
```
σ_j = std(X_A[:, j])  for j = 0..p-1
X_A_s = X_A / σ       (element-wise, broadcast)
X_C_s = X_C / σ
```

**Step 2 — Mean term:**
```
x̄_A = mean(X_A_s, axis=0)          # shape (p,)
x̄_C = mean(X_C_s, axis=0)          # shape (p,)
d_means = D_r(x̄_A, x̄_C)
         = (Σ_j |x̄_A_j − x̄_C_j|^r)^{1/r}   for finite r
         = max_j |x̄_A_j − x̄_C_j|            for r=∞
```

**Step 3 — Covariance term:**
```
S_A = LedoitWolf().fit(X_A_s).covariance_   # p×p, positive definite
S_C = np.cov(X_C_s.T)                        # p×p

# Compute S_A^{−1/2} via eigendecomposition:
λ, V = np.linalg.eigh(S_A)
λ_safe = np.maximum(λ, 1e-6)               # eigenvalue flooring
S_A_neghalf = V @ diag(λ_safe^{-0.5}) @ V.T

# Relative covariance:
S_tilde = S_A_neghalf @ S_C @ S_A_neghalf   # p×p
# When S_C = S_A: S_tilde = I (identity). When they differ: S_tilde ≠ I.

# Distance from identity:
d_cov = D_r(S_tilde.flatten(), I.flatten())
```
Why eigendecomposition? Because S_A is symmetric positive (semi)definite,
so it has real eigenvalues. S_A = V D V^T, so S_A^{−1/2} = V D^{−1/2} V^T.

**Step 4 — Skewness term:**
```
b_A = scipy.stats.skew(X_A_s, axis=0, bias=False)   # shape (p,)
b_C = scipy.stats.skew(X_C_s, axis=0, bias=False)   # shape (p,)
d_skew = D_r(b_A, b_C)
```
Bias-corrected skewness formula (Fisher–Pearson):
```
b_j = [n/(n-1)(n-2)] × Σ_i [(x_ij − x̄_j) / s_j]³
```
The bias correction matters for small samples (Glass n_C ≈ 107 rows).

**Step 5:**
```
F_r = d_means + d_cov + d_skew
```
This single number is the fitness of the chromosome. Lower = better.

### Population initialization (generation 0)
All l=200 individuals in generation 0 are created by the same random
generator (same as diversity injection):
- For each missing cell at position i: sample from the generator for
  column `missing_index[i][1]`.
- No structured seeding from observed values.
- The population starts fully random.

### Elite selection
After evaluating F_r for all l=200 individuals in a generation:
1. Sort individuals by F_r (ascending — lower is better).
2. Take the top c=20 individuals. These are the elites.
3. They are copied unchanged into the next generation.
No tournament selection, no proportional selection — strict top-c.

### Mutation operator (produces c1=10 offspring)
For each of the 10 mutant slots:
1. Pick one elite uniformly at random (from the 20 elites).
2. Copy its chromosome exactly (length k array).
3. Pick one position i uniformly at random from {0, ..., k−1}.
4. Find the column: `col = missing_index[i][1]`.
5. Sample a new value from the generator for column `col`.
6. Replace `chromosome[i]` with this new value.
Result: a chromosome identical to the parent on k−1 positions, with 1
position replaced by a fresh random draw from the correct column generator.

**Why only 1 position?** The paper specifies single-point mutation.
Mutating many positions simultaneously would essentially create a new random
individual (same as diversity injection). Single-point mutation creates a
small variation of an elite — it is the "fine-tuning" operator.

### Crossover operator (produces c2=10 offspring)
For each of the 10 crossover slots:
1. Pick two elites uniformly at random (parent1, parent2).
2. Pick one feature column j uniformly at random from all columns
   that have at least one missing value.
3. Find all chromosome positions belonging to column j:
   `feature_group_j = [i for i,(row,col) in enumerate(missing_index) if col == j]`
4. Copy parent1's chromosome.
5. For all i in feature_group_j, replace chromosome[i] with parent2's value.
Result: all gene positions for column j come from parent2; everything else
from parent1.

**Concrete example:** If missing_index has 75 entries for sepal_length (col 0)
and parent1 has sepal_length values [4.8, 5.2, ...] and parent2 has [5.1, 6.0, ...]:
- Crossover on column 0 → offspring has [5.1, 6.0, ...] for sepal_length,
  but parent1's values for petal_length.

**Why column-wise crossover?** It preserves the column distribution structure.
Swapping individual cells across columns would break the per-column statistical
structure that the generators are calibrated to.

### Random diversity injection (produces c3≈160 individuals)
Identical to generating a random individual from scratch:
- For each position i in range(k): sample from generator for column
  `missing_index[i][1]`.
- Completely independent of any elite.
80% of the population is replaced entirely each generation.

### Why does diversity injection dominate?
With c3 ≈ 160 out of l=200, 80% of each generation is random. The GA is
not really evolving in the traditional sense — it is more like:
"Run many random completions, keep the best 20, mix in more random ones."
This is why adaptive c3 schedule (C3) has limited impact at full compute:
reducing diversity injection slightly does not change the fundamental
search dynamics.

### Generators per column
- Continuous features: bootstrap resampling from observed X_A column values
- Binary features: Bernoulli with p estimated from X_A
- Discrete features: empirical PMF from X_A

Why bootstrap (not Gaussian)?
- Preserves the empirical distribution exactly
- Handles zero-inflation (Haberman Nodes: 44% zeros)
- Handles skewness and multimodality without assuming any parametric shape
- Gaussian sampling would smooth away discrete mass points and tails

### Termination
Always runs exactly G generations. No convergence criterion. The best
individual at the final generation is taken. The best of Q runs is the
final output.

### 10-seed significance testing — exact mechanics
The 10 seeds are used to generate 10 independent missing data masks
(different sets of rows are made missing in each seed). Both MIGA and MICE
are run on the exact same masked dataset for each seed.

For each seed s ∈ {1..10}:
1. Apply mask_s to the clean dataset → dataset_s with NaNs.
2. Run MIGA on dataset_s → Fr_MIGA_s (scalar).
3. Run MICE on dataset_s → Fr_MICE_s (computed post-hoc).
4. Record the pair (Fr_MIGA_s, Fr_MICE_s).

Wilcoxon signed-rank test is applied to the 10 differences:
```
d_s = Fr_MIGA_s − Fr_MICE_s   for s = 1..10
```
One-sided test: H0: median(d) ≥ 0, H1: median(d) < 0.
A p-value of 0.001 = 1/1024 means MIGA had lower Fr on all 10 seeds
(the most extreme possible result with 10 seeds, since min p = 1/2^10).

### Minkowski distance — what it actually computes
For two vectors u, v of length m:
- r=1 (Manhattan):  D_1(u,v) = Σ_i |u_i − v_i|
- r=2 (Euclidean):  D_2(u,v) = (Σ_i (u_i − v_i)²)^{1/2}
- r=∞ (Chebyshev):  D_∞(u,v) = max_i |u_i − v_i|

For the covariance term, u = S_tilde.flatten() and v = I.flatten(),
so u and v have length p². For the mean and skewness terms, length is p.

---

## PART 4: THE IMPLEMENTATION DECISIONS — WHY NOT SOMETHING ELSE

### 4.1 Feature scaling before computing F_r
**What we do:** Divide each column by its standard deviation (from X_A)
before computing means, covariance, and skewness.

**Why it is needed:** Wine's Proline feature has variance ~10^6× larger than
other features. Without scaling, the covariance matrix S_A has condition
number ~40,000,000. The S_A^{-1/2} computation becomes numerically
unreliable. The covariance term dominates F_r for the wrong reason.

**Why this is mathematically justified:** The original F_r is intended to
compare standardised means (x̃ already divides by std). The covariance term
in the relative form S_A^{-1/2} S_C S_A^{-1/2} is theoretically
scale-invariant, but numerically it is not when eigenvalues differ by
many orders of magnitude.

**Why not use a different normalisation?**
Column-wise std normalisation is the minimal and natural choice. Min-max
normalisation would not handle outliers properly. Whitening would change the
problem formulation. Std normalisation matches the paper's intent.

### 4.2 Ledoit-Wolf shrinkage covariance (C2)
**Problem:** When n_A (complete rows) < p (features), the sample covariance
S_A is rank-deficient. Its eigenvalues are zero or negative. S_A^{-1/2}
cannot be computed. d_cov(X_A, X_A) ≠ 0, which is mathematically wrong.

**Wine 40% example:** n_A = 11, p = 13. Sample covariance has rank ≤ 11
but is 13×13. Condition number = 570 million. d_cov(X_A, X_A) = 0.606
(should be 0.000).

**Ledoit-Wolf fix:** S_LW = (1-α) S_sample + α (tr(S_sample)/p) I
The shrinkage parameter α is estimated analytically by minimising the
expected squared Frobenius error. At Wine 40%: α = 0.42, condition number
drops to 8.8, d_cov(X_A, X_A) = 0.000. ✓

**Why not other shrinkage estimators?**
- Oracle approximating shrinkage (OAS): similar to LW, slightly better
  in some regimes, but LW is the standard sklearn choice and well understood.
- Ridge regularisation: arbitrary λ choice, no principled data-driven estimate.
- LW is the principled, automatic, and well-justified choice.

**Why does LW not improve RMSE?**
Because the fundamental problem is sample size, not estimator choice. With
n_A = 11 and p = 13, you cannot reliably estimate any 13-dimensional joint
distribution. A better covariance estimator makes F_r computable but does
not give the algorithm more information about the true missing values.

### 4.3 Eigenvalue flooring
**What:** Floor all eigenvalues of the covariance matrix to ε = 1e-6 before
computing S^{-1/2}.

**Why:** Even with LW, numerical precision can produce tiny negative
eigenvalues due to floating point. Flooring prevents sqrt of negative number.

**Why not raise an error instead?**
Because the failure mode (near-singular covariance) is expected and
documented. Raising an error would halt experiments without providing any
diagnostic value.

### 4.4 Bootstrap generators
**Already covered in Section 3. Key point for committee:**
The paper says "random generators" without specifying the distribution.
We chose empirical bootstrap because it is the most faithful to the data.
The Haberman result (beating the paper at 30% on some seeds) is direct
evidence this choice matters: the paper likely used Gaussian, which cannot
reproduce the Nodes column zero-inflation.

### 4.5 Categorical exclusion (Wholesale)
**What:** Channel and Region columns are excluded from being made missing.

**Why:** The paper's MAR setup masks floor(p/2) continuous measurement
features. Categorical identifiers are not measurement features and are not
candidates for imputation in the original design.

**Why not include them?**
Making a categorical label missing would require a categorical imputation
model, which is a different problem. Including them would also inflate the
k (missing cells count) and distort comparisons.

### 4.6 Range-normalised RMSE
**Formula:** NRMSE_j = RMSE_j / (max_j - min_j) for feature j, then
average over missing features.

**Why this normalisation (not Z-score)?**
The paper reports results in this form. Using range normalisation allows
comparison of RMSE across features with different units and scales without
assuming the data is Gaussian (which Z-score normalisation implicitly does).

**Why not use raw RMSE?**
One feature with a very large scale would dominate the average. Range
normalisation gives equal weight to each feature's contribution.

### 4.7 CoD (Coefficient of Determination) definition
**Formula:** CoD = 1 - (Σ_missing (x_true - x_imputed)²) / (Σ_all (x_true - x̄)²)

**Why denominator uses all rows?**
The paper's Table 4 values are matched more closely with this definition.
Using only the missing-row denominator would give a different scale.

### 4.8 Missing feature selection
**Rule:** floor(p/2) features are selected for missingness, applied to
floor(n/2) rows for that feature.

**Why floor(p/2)?**
The paper specifies this. It ensures a meaningful fraction of features
are missing without making the problem degenerate.

**Why apply to floor(n/2) rows?**
Again following the paper. This gives ~25% overall cell missingness
at 30% rate (not every cell of the selected features is missing, just
half the rows for those features).

---

## PART 5: THE FORMAL THEOREM — UNDERSTAND THE PROOF

### What Proposition 2.1 states
**Claim:** If you use RMSE-optimal imputation (the conditional expectation
x̂_j = E[X_j | X_{-j}]), then F_r > 0 whenever Var(X_j | X_{-j}) > 0.

**Why the conditional expectation minimises RMSE:**
For any imputer f, E[(X_j - f(X_{-j}))²] is minimised by f* = E[X_j | X_{-j}].
This is a fundamental theorem of probability. MICE approximates this f*.

**The proof sketch:**
By the **law of total variance**:
```
Var(X_j) = E[Var(X_j | X_{-j})] + Var(E[X_j | X_{-j}])
```
If Var(X_j | X_{-j}) > 0 (i.e., X_j is not perfectly determined by X_{-j}),
then:
```
Var(E[X_j | X_{-j}]) < Var(X_j)
```
because the law of total variance says the total variance is larger than
the variance of conditional means.

The RMSE-optimal imputation replaces X_j with E[X_j | X_{-j}], which has
strictly smaller variance than X_j. Therefore the variance of the imputed
column differs from the true variance. This means the marginal distribution
of the imputed column ≠ the marginal distribution of the true column.
Therefore F_r (which measures distributional distance including variances)
must be > 0.

**In one sentence:** MICE shrinks variance because it predicts means, not
draws — and shrunk variance means the distribution is wrong, so F_r > 0.

### What Proposition 2.2 states
**Claim:** If F_r = 0 (perfect distributional match), then RMSE > 0 unless
the missing values are drawn from the same marginal distribution as X_A.

**The proof sketch:**
F_r = 0 requires the imputed X_C to match X_A in means, covariances, and
skewness. This means the imputed values are drawn from a distribution that
looks like X_A's marginal.

But the true missing values were drawn from the full data distribution.
Under MAR, the marginal of X_j in X_C (the incomplete rows) is the same as
in X_A only if the data is homogeneous (no distributional difference between
complete and incomplete rows). In general, the true missing values do not
equal samples from X_A's marginal. Therefore RMSE > 0.

**The MNAR version is even cleaner:**
Under top MNAR, X_A contains only low-value rows (the high-value ones are
missing). F_r = 0 means the imputed values look like X_A's distribution
(low values). But the true missing values are high-value. RMSE is maximised.

### Corollary
No single imputation can simultaneously minimise both F_r and RMSE.
Proof: Prop 2.1 says the RMSE minimiser has F_r > 0. Prop 2.2 says the
F_r minimiser has RMSE > 0. Therefore neither objective's minimiser achieves
zero on the other objective.

### Why this theorem matters
Before this thesis, the Fr-RMSE tradeoff was only observed empirically in
the paper's results. This thesis proves it is structurally unavoidable —
not an accident of the specific datasets or parameters.

---

## PART 6: THE BASELINE METHODS — KNOW THEM DEEPLY

### Mean imputation
Replace each missing value with the column mean from observed data.
- Fastest, simplest.
- Always worst on both RMSE and F_r (except sometimes ties on CoD).
- Completely destroys variance (all imputed values are at the mean).
- Acts as the lower bound / sanity check.

### KNN imputation (k=5)
For each missing cell, find the 5 nearest neighbours (by observed features),
take the weighted average of their values for the missing feature.
- Weighted by inverse distance.
- Preserves local structure.
- Better than mean but not as good as MICE on RMSE.
- Often better than MICE on F_r for some datasets (but not always better
  than MIGA).

### MICE (Multiple Imputation by Chained Equations)
Algorithm:
1. Initialise all missing values with column means.
2. For each feature j with missing values:
   a. Train a model: X_j ~ X_{-j} using the currently imputed values.
   b. Predict missing X_j from the model.
3. Repeat step 2 for all features (one "cycle").
4. Run for multiple cycles until convergence.

**sklearn implementation:** Uses BayesianRidge as the per-feature model.
BayesianRidge is a regularised linear regression with Bayesian
hyperparameter tuning — it handles collinear features better than OLS.

**Why MICE wins on RMSE:**
Each iteration optimises E[X_j | X_{-j}] — the conditional mean. By
Proposition 2.1, this is the RMSE minimiser. The iterative refinement
converges to the fixed point of this conditional mean system.

**Why MICE loses on F_r:**
By Proposition 2.1, the conditional mean imputation must have F_r > 0.
Specifically, it deflates variance — all imputed values are more centralised
than the true values. At the distribution level, the imputed data looks
more "Gaussian-like" and narrower than the truth.

---

## PART 7: THE FIVE DATASETS — KNOW EACH ONE DEEPLY

### Iris (n=150, p=4, r=∞)
- 4 continuous features: sepal length/width, petal length/width
- 3 classes (but classification is downstream, not part of imputation)
- floor(p/2) = 2 features missing
- **Why MIGA wins here:** Low-dimensional, near-Gaussian, 4 features are
  easy to moment-match. GA has few degrees of freedom.
- **Key results:**
  - 30% MAR: MIGA Fr=0.780 vs MICE Fr=1.155, p=0.001 ✓ (MIGA 1.48× lower)
  - 30% MAR: MIGA RMSE=0.110 vs MICE RMSE=0.078 (MICE 1.41× lower)
  - 40% MAR: MIGA Fr=1.218 vs MICE Fr=1.838, p=0.001 ✓
  - 50% MAR: MIGA Fr=2.905 vs MICE Fr=2.787, p=1.000 (MIGA LOSES)
  - 50% TAILS: MIGA Fr=7.339 vs MICE Fr=7.491, p=0.001 ✓
  - Variance (30%): MIGA ratio=1.025 (deviation 0.025) vs MICE 0.024 — near tie
  - Downstream: all methods 0.957 accuracy; MIGA and KNN both 100% KS pass

### Wine (n=178, p=13, r=1)
- 13 continuous chemical features of wine
- **Why Wine is hard:**
  1. p=13, floor(p/2)=6 missing features, very high-dimensional for small n
  2. r=1 (Manhattan) makes covariance term computation O(p²n²) — very slow
  3. At 40%: n_A ≈ 11 < p=13, covariance is singular → Ledoit-Wolf needed
  4. Proline feature: variance ~10^6 larger than others → scaling needed
- **Why significance excluded:** 98 minutes per seed × 10 seeds × 3
  mechanisms ≈ 49 hours. Not feasible within thesis scope.
- **Key results:**
  - MNAR tails: RMSE=0.3012 (comparable to MAR 0.2973) — tails benign
  - MNAR top: RMSE=0.3597 (21% worse than MAR)
  - Kurtosis 30%: MICE d_kurt=12.013, MIGA-Fr=9.407, MIGA-Fr+=7.536 (20% improvement)
  - Variance 40%: MIGA ratio=1.005 (excellent), MICE=0.849

### Glass (n=214, p=10, r=2)
- 10 continuous features of glass composition
- Contains Refractive Index (RI) feature with near-zero variance (σ²≈0.00003)
- **Why MIGA loses on Fr:** p=10 crosses the dimension threshold. MICE's
  iterative regression implicitly models the joint distribution at this scale.
- **The RI problem:** Near-zero variance inflates S_A^{-1/2} in the RI
  direction, making the covariance term numerically noisy. Seed-to-seed MIGA
  Fr range: 1.6–20.9 (huge variance). GA cannot converge reliably.
- **Key results:**
  - 30% MAR: MIGA Fr=74.03 vs MICE Fr=42.99 — MICE wins by 1.72× (p=1.000)
  - 40% MAR: MIGA Fr=75.30 vs MICE Fr=29.65 — MICE wins by 2.54×
  - 40% TAILS: MIGA Fr=260.3 vs MICE Fr=275.2 — **MIGA wins** (p=0.001) ← exception
  - 50% MAR: essentially tied (423.0 vs 424.6, p=0.348)
  - MNAR top: Fr=758.8 (27× worse than MAR) — covariance collapse
  - Variance (30%): MIGA deviation=0.005 ★ vs MICE 0.063 (MIGA strongly wins)
  - Downstream (30%): MIGA accuracy=0.609 vs MICE 0.637 — MIGA loses

### Haberman (n=306, p=3, r=∞)
- Survival data: Age, Operation Year, Nodes (lymph nodes), Survival
- floor(p/2) = 1 feature missing per run
- Nodes: 44% zeros, highly zero-inflated, strong positive kurtosis
- **Why MIGA wins so strongly:** Only 1 feature missing → bootstrap
  generator matches the empirical distribution perfectly. Near-deterministic
  Fr convergence (std ≈ 0.000 across all seeds).
- **Key results:**
  - 30% MAR: MIGA Fr=2.580 vs MICE Fr=5.065 — MIGA 1.97× lower (p=0.001)
  - 40% MAR: MIGA Fr=0.593 vs MICE Fr=4.250 — MIGA 7.16× lower (p=0.001)
  - 50% MAR: MIGA Fr=0.649 vs MICE Fr=3.886 — MIGA 5.98× lower (p=0.001)
  - MNAR top (30%): MIGA Fr=0.810 (global minimum!), RMSE=0.384 (global max!)
  - Kurtosis (30%): MICE d_kurt=7.783, MIGA-Fr=1.854 (4.2× better without optimising)
  - Kurtosis floor: d_kurt=1.797 cannot be reduced by Fr+ — Nodes is the fixed point
  - Variance (30%): MIGA ratio=1.535 (poor) — single-feature degenerate case
  - Downstream: MIGA 40% KS pass (only method >0%), 70% CI coverage

### Wholesale (n=440, p=8, r=∞)
- Customer spending data: 6 product categories + Channel + Region (excluded)
- Detergents_Paper: kurtosis ≈ 65.3 (extremely right-skewed)
- **Why MIGA barely wins:** p=8, exactly at the dimension threshold. Fr
  advantage shrinks to 1.5% (vs 49% at Haberman).
- **Key results:**
  - 30% MAR: MIGA Fr=13.059 vs MICE Fr=13.252 — MIGA wins (p=0.031, barely)
  - Variance (30%): MIGA deviation=0.077 vs MICE 0.123 ★
  - Variance (50%): MIGA deviation=0.120 vs MICE deviation=0.104 — **MICE wins** ← only case
  - Kurtosis: All methods d_kurt=129.08 identically (Detergents_Paper fixed point)
  - MNAR bottom: Fr=10.702 < MAR Fr=13.016 — reference homogenisation reduces Fr
  - Downstream: MICE wins KS (40%) and CI (77%) over MIGA (20%, 53%)

---

## PART 8: THE MNAR EXPERIMENTS — MECHANISMS AND FINDINGS

### The three mechanisms
**Top quantile:** For each missing feature, remove the top 30% of values
from the observed data. These rows become X_C (forced missing). X_A contains
only lower-value rows.

**Bottom quantile:** Same but remove lowest 30%.

**Tails:** Remove top 15% + bottom 15% from each missing feature. X_A
contains only the central 70%.

### Why the paper only studied MAR
MAR is the standard assumption. MNAR evaluation requires knowing the
missingness mechanism, which is usually unverifiable in practice. The paper
focused on algorithm design, not real-world robustness.

### The Fr → RMSE inversion (Haberman top)
**Exactly what happens:**
1. Top MNAR makes all high-Nodes rows go into X_C (missing).
2. X_A now contains only low-Nodes rows.
3. The GA tries to make X_C look like X_A — it finds imputations where
   Nodes is low.
4. Fr = 0.810 (excellent distributional match to the biased X_A).
5. But the true Nodes values for those rows were HIGH.
6. RMSE = 0.384 (worst ever — imputed low, true was high).

**The committee will ask:** "Doesn't low Fr mean good imputation?"
**Answer:** Only under MAR, where X_A is an unbiased sample of the
population. Under directional MNAR, X_A is a biased reference. A low Fr
means matching the bias faithfully, which is wrong.

### Why tails MNAR is benign
Under tails, the highest and lowest observations are removed. X_A becomes
more homogeneous (less spread). The missing values are the extremes.

But the extremes are not far from the centre by definition (they are
removed symmetrically from both ends). So imputing them near the centre
(what matching X_A distribution implies) is not badly wrong.

More precisely: Var(X_A under tails) < Var(X_A under MAR), but the
bias is symmetric. Symmetric bias means the conditional expectation is
not badly shifted. Both RMSE and Fr improve or stay similar.

Iris tails result: RMSE=0.1178 (better than MAR 0.1270). Explained by
the homogeneous X_A being an easier reference to match.

### Glass top MNAR — covariance collapse
Under top MNAR on Glass: high-RI rows (and high-other-feature rows) become
missing. X_A has artificially compressed covariance. S_A has near-zero
eigenvalues in affected directions. S_A^{-1/2} amplifies those directions.
Fr = 758.8 (27× larger than MAR Fr=27.9). This is a numerical instability
caused by the combination of high dimensionality and directional MNAR.

---

## PART 9: THE SYNTHETIC DIMENSION EXPERIMENT (C10)

### Why we designed this experiment
The UCI benchmarks confound dimensionality with distributional shape,
scale, and correlation structure simultaneously. When MIGA loses on Glass
(p=10) but wins on Haberman (p=3), we cannot isolate whether the cause is:
- Dimensionality (p)
- Correlation structure (how features relate)
- Specific distributional shape (zero-inflation, skewness)
- Scale differences

The synthetic experiment holds everything constant except (p, ρ).

### Design
- Synthetic Gaussian data with Toeplitz covariance: Σ_ij = ρ^|i-j|
- p ∈ {4, 8, 13, 20, 30}, ρ ∈ {0.0, 0.3, 0.6, 0.9}
- n = 200, 30% MCAR, 5 seeds per cell
- MIGA: l=100, G=200, Q=3 (lighter than main experiments)
- Metric: ΔFr% = 100 × (MICE Fr − MIGA Fr) / MICE Fr (positive = MIGA wins)

### Results table (relative advantage %)
```
ρ     p=4    p=8    p=13   p=20†  p=30‡
0.0   +73%   +23%   +31%   +7%    FAIL
0.3   +25%   +18%   +28%   +5%    FAIL
0.6   +24%   +7%    +20%   +4%    FAIL
0.9   +24%  -39%   -11%   +16%†  FAIL
```
† = degenerate (near-zero complete rows, results unreliable)
‡ = MIGA fails completely (no complete rows)

### The three key findings

**F16 — The crossover is correlation-driven, not just dimension-driven**
MIGA does not simply lose above p=10. It loses when HIGH CORRELATION (ρ=0.9)
AND MODERATE+ DIMENSIONALITY (p≥8) coincide. At ρ≤0.6, MIGA wins even at
p=13 with +20% advantage. This refines the simple "threshold at p≈10" claim.

**F17 — Why high correlation helps MICE**
When ρ=0.9, each feature is nearly determined by the others. MICE's
conditional regression E[X_j | X_{-j}] is very accurate (nearly perfect
R²) because the regression model captures almost all the variance.
MICE essentially becomes an accurate joint distribution sampler when
correlations are very high. MIGA's fixed-moment-matching cannot exploit
this structure.

**F18 — Hard limit at p=30**
Under 30% MCAR with p=30 and n=200: expected complete rows =
200 × 0.7^30 ≈ 0.04. Essentially zero. X_A is empty. MIGA requires at
least one complete row (the reference set). This is not a bug — it is a
fundamental structural constraint of the algorithm.

### Why p=20 results are marked as degenerate
With p=20 and 30% MCAR: expected complete rows = 200 × 0.7^20 ≈ 7.8.
Very few complete rows means:
- S_A estimated from ~8 rows in 20 dimensions — always rank-deficient
- LW shrinkage helps numerically but cannot give real distributional info
- Fr values are in the thousands (6,000–133,000), unreliable

The p=20 ρ=0.9 result (+16% advantage) contradicts the trend and is likely
a numerical artifact of the degenerate fitness landscape.

---

## PART 10: THE DOWNSTREAM EVALUATION

### Why downstream evaluation matters
Fr is an internal metric — it measures how well the GA achieved its goal.
But the question practitioners care about is: does the distributional
advantage translate to better downstream analysis?

Three downstream tasks test this:

### Task 1: Classification accuracy (5-fold CV, logistic regression)
Tests whether imputation quality affects predictive modelling.
Result: All methods achieve nearly identical accuracy on every dataset
(differences < 1%). Imputation method choice is not a bottleneck for
classification in these settings.

**Key result:** Haberman ≈0.744 for all methods. MIGA's distributional
advantage does not help or hurt prediction.

### Task 2: KS test pass rate
For each missing feature: two-sample Kolmogorov-Smirnov test between
imputed values and true values (α=0.05). Report fraction passing.
Passing = the imputed distribution is statistically indistinguishable
from the truth.

**Why this tests distributional quality:** KS test is sensitive to
distributional differences in both location and shape. MICE's variance
suppression causes it to fail this test systematically.

**Key results:**
- Haberman: MIGA 40% pass, Mean/KNN/MICE all 0% pass
- Iris: KNN and MIGA both 100% pass, MICE 95%
- Glass: KNN 80%★, MIGA 20% (fitness landscape ill-conditioned)

### Task 3: Bootstrap CI coverage (95% CI of the mean)
For each missing feature: compute 500-bootstrap 95% CI of the mean from
the imputed data. Report fraction of features where CI contains true mean.
Nominal coverage should be 95%.

**Why MICE fails this:** MICE's variance suppression makes the bootstrap
CIs too narrow. The true mean often falls outside the CI.

**Key results:**
- Haberman: MIGA 70%★, MICE only 10%
- Iris: KNN 100%★, MIGA 95%, MICE 90%
- Wholesale: MICE 77%★, MIGA 53% (baseline Fr too large, GA struggles)

### Scope conditions for downstream utility
MIGA's Fr advantage propagates to downstream utility when:
1. **Moderate dimensionality:** p ≤ 8 (dimension threshold)
2. **Well-conditioned variances:** No near-zero variance features (RI in Glass)
3. **Tractable baseline Fr:** Fr ≤ ~5 at MAR baseline (Wholesale Fr≈13 is too hard)

When these conditions fail, MICE provides more reliable downstream quality.
This is a scope characterisation, not a failure — it tells practitioners
exactly when to use each method.

---

## PART 11: THE ADAPTIVE MUTATION SCHEDULE (C3)

### What it does
c3 decays linearly from start value (e.g., 15) to end value (e.g., 3)
over G generations. Early: more diversity injection (exploration). Late:
more exploitation.

### Full compute result (G=500, Q=5)
Fixed c3=5 wins on all five datasets:
- Iris: Fixed 0.1270 vs Adaptive 0.1273 (−0.2%)
- Glass: Fixed 0.1656 vs Adaptive 0.1821 (−10%)
- Haberman: Fixed 0.3649 vs Adaptive 0.4164 (−14%)
- Wine: Fixed 0.2973 vs Adaptive 0.3153 (−6.1%)
- Wholesale: Fixed 0.1514 vs Adaptive 0.1548 (−2.2%)

### Compute-limited result (G=80, Q=2, Iris)
Adaptive 15→3: RMSE 0.1198 vs Fixed 0.1415 (15% better)
Adaptive 3→15: RMSE 0.1405 (starting low then high = worse)

### Why the adaptive schedule fails at full compute
The dominant mechanism is c3 (diversity injection), not c3 schedule.
At full compute (G=500), there are enough generations for the fixed
random injection to find good solutions. The schedule's early-high/late-low
pattern only helps when there are very few generations to work with.

### Honest framing
"C3 adaptive schedule provides faster early convergence, which is relevant
in compute-constrained deployment. It does not improve results given
adequate compute budget."

---

## PART 12: THE KURTOSIS EXTENSION (C8)

### What Fr+ adds
A fourth term: D_r(k_A, k_C), where k is the excess kurtosis vector.
```
Fr+ = Fr + D_r(k_A, k_C)
```

### Results summary
| Dataset (rate) | MICE d_kurt | MIGA-Fr d_kurt | MIGA-Fr+ d_kurt | Improvement |
|----------------|-------------|----------------|-----------------|-------------|
| Iris (30%)     | 0.683       | 0.716          | 0.638 ★         | 11%         |
| Wine (30%)     | 12.013      | 9.407          | 7.536 ★         | 20%         |
| Glass (30%)    | 45.60       | 45.81          | 45.66           | Negligible  |
| Haberman (30%) | 7.783       | 1.854 ★        | 1.854 ★         | None (floor)|
| Wholesale (30%)| 129.08      | 129.08         | 129.08          | None (fixed)|
| Iris (40%)     | 0.524       | 0.641          | 0.459 ★         | 28%         |
| Haberman (40%) | 13.977      | 1.797 ★        | 1.797 ★         | None (floor)|
| Glass (40%)    | 52.645      | 52.787         | 52.637          | Negligible  |

### The Haberman kurtosis floor effect
The Nodes column has 44% zeros. The kurtosis distance is dominated by
this column. Because the bootstrap generator samples from the observed
distribution (which already includes the zero-inflation), standard MIGA-Fr
already matches kurtosis well. Fr+ cannot improve on this because the
floor is set by the column's intrinsic zero structure — kurtosis cannot
be further reduced from an already matched zero-inflated distribution.

d_kurt = 1.797 is a fixed point: both Fr and Fr+ converge to the same
value deterministically (std = 0.000 across seeds).

### The Wholesale fixed point
Detergents_Paper has kurtosis ≈ 65.3. All methods return d_kurt = 129.08.
This is not a failure of Fr+ specifically — it is an inherent property of
the reference distribution. No imputation can bridge this gap from 30%
missing data because the extreme kurtosis in X_A creates a fitness
landscape where all completions look equally bad in kurtosis space.

---

## PART 13: VARIANCE PRESERVATION (C7)

### Why variance matters
If imputed variance is too low (ratio < 1): confidence intervals computed
from imputed data are too narrow → underestimates uncertainty.
If imputed variance is too high (ratio > 1): CIs are too wide.
Target: ratio = 1.0.

### MICE always underestimates (ratio < 1)
This is predicted by the law of total variance (same mechanism as
Proposition 2.1). MICE imputes conditional means → conditional mean has
lower variance than the marginal. Always.

### MIGA's variance results across missing rates
| Dataset   | 30% MIGA | 40% MIGA | 50% MIGA | MICE wins? |
|-----------|----------|----------|----------|------------|
| Iris      | 1.025    | 0.985    | 1.006    | Never      |
| Wine      | N/A      | 1.005    | 0.900    | Never      |
| Glass     | 1.005 ★  | 0.871    | 1.069    | Never      |
| Haberman  | 1.535    | 1.080    | 1.259    | Never      |
| Wholesale | 1.077 ★  | 0.874    | 1.120    | 50% only   |

**Only exception:** Wholesale 50%: MIGA deviation=0.120 vs MICE deviation=0.104.
MICE wins here because 50% missingness leaves very few complete rows for
Wholesale (n=440, p=8, ~0.7^8×440 ≈ 25 complete rows), and the GA cannot
reliably calibrate against a small reference set.

### Why Haberman 30% is "poor" (1.535)
Single-feature degenerate case: floor(p/2)=1 feature missing. With only
one feature in X_C differing from X_A, the covariance term is essentially
a 1×1 problem. The bootstrap generator samples from Nodes' marginal
regardless of what other features say. Without multivariate constraint,
variance is over-inflated. This is documented as a known limitation.

---

## PART 14: COMMITTEE Q&A — 40 QUESTIONS

### Group A: Understanding the algorithm

**Q: What does X_A and X_C stand for?**
X_A = rows with All features observed (complete rows).
X_C = rows with at least one missing value (incomplete rows, post-imputation).
The fitness F_r compares these two sets. The subscripts come from the
paper's notation.

**Q: Why use genetic algorithm and not gradient descent?**
The fitness function F_r is not differentiable with respect to the missing
values — it involves eigendecompositions, matrix square roots, and sorting
operations (for Chebyshev distance). Gradient descent requires a
differentiable objective. GAs handle black-box functions naturally.

Also: the search space is combinatorial (each missing value is drawn from
a discrete/continuous generator) and potentially highly multimodal. GAs
are robust to multimodality.

**Q: Why not use Bayesian optimisation instead of a GA?**
Bayesian optimisation works well for low-dimensional parameter spaces
(typically < 50 dimensions). The number of missing cells k can be
hundreds or thousands. GAs scale better to high-dimensional black-box
optimisation with diversity injection providing global search.

**Q: Why not use simulated annealing?**
SA is a single-solution method. A GA maintains a population, which gives
better parallelism and diversity. The diversity injection mechanism in MIGA
is specifically designed to prevent premature convergence, which SA handles
via temperature but less explicitly.

**Q: What is the difference between MAR and MCAR in practice?**
Under MCAR, any subset of rows and features can be missing independently.
Under MAR, the probability of a value being missing depends on other
observed values. In practice, MAR is tested by examining whether the
missingness indicator R_j correlates with other observed features. MCAR
is a stricter (rarer) assumption. Both MICE and MIGA assume at most MAR.

**Q: How is X_A constructed in practice?**
A row is in X_A if and only if all its values are observed (no NaN).
After applying the missing data mask to the dataset, X_A is simply:
```python
X_A = X[~np.any(np.isnan(X), axis=1)]
```

**Q: What if X_A is empty?**
MIGA cannot run. This happens with p=30 and 30% MCAR in the synthetic
experiment. The algorithm raises an error: "No complete rows found."
This is a fundamental constraint, not a bug.

**Q: Why does MIGA run Q independent times and take the best?**
Each run can converge to a different local optimum of F_r. Taking the
best of Q runs reduces the probability of returning a poor local optimum.
The Q runs share no state — each starts with a fresh random population.

### Group B: Implementation decisions

**Q: Why did you not use the original paper's Gaussian generator?**
The paper says "random generator" without specifying the distribution.
We chose empirical bootstrap because it:
1. Preserves the exact empirical distribution
2. Handles zero-inflation (Haberman Nodes: 44% zeros)
3. Does not assume a parametric shape
Evidence: Haberman 30% MAR beats the paper (ratio 0.52) — the paper
likely used Gaussian, which would smoothe away the discrete zeros.

**Q: Why Ledoit-Wolf specifically and not Oracle Approximating Shrinkage?**
Ledoit-Wolf is the standard sklearn implementation, analytically derived,
and has well-understood convergence properties. OAS performs marginally
better in some asymptotic regimes but the difference is negligible for
our use case. LW is the principled, documented, widely-used choice.

**Q: How does the missing_index mapping work?**
Before running the GA, the code scans the dataset for all (row, column)
pairs where the value is NaN. These pairs are stored in a list (missing_index).
The chromosome of length k = len(missing_index) is indexed the same way.
To evaluate fitness, chromosome values are plugged back into the dataset
at the positions given by missing_index.

**Q: Why floor(p/2) features missing and not some other number?**
The paper specifies this. It represents a moderate level of missing-feature
density — enough to test the algorithm, not so extreme that the problem
is degenerate.

**Q: Why exclude Channel and Region from Wholesale?**
These are categorical identifiers (region: 1-3, channel: 1-2), not
continuous measurement variables. The paper's imputation protocol is
designed for continuous features. Including them would require handling
a fundamentally different variable type and would distort the comparison.

### Group C: Results interpretation

**Q: MICE has lower RMSE than MIGA on every dataset. Doesn't that mean MICE is better?**
Only if RMSE is the right metric for your use case. MICE is RMSE-optimal
by the law of total variance — it is the best possible on RMSE. But it
achieves this by imputing conditional means, which systematically
suppresses variance.

If your downstream task requires:
- Correct marginal distributions → MIGA is better (wins Fr)
- Hypothesis testing (KS test) → MIGA is better (Haberman: 40% vs 0%)
- Bootstrap confidence intervals → MIGA is better (Haberman: 70% vs 10%)
- Pointwise prediction → MICE is better

**Q: Why does MIGA lose on Fr for Glass?**
Glass has p=10. At this dimensionality, MICE's iterative conditional
regression implicitly models the joint distribution through iterated
conditional expectations. The moment-matching approach (means, covariance,
skewness) captures a coarser summary that becomes increasingly approximate
as p grows. The synthetic experiment confirms this: MIGA also loses at
p=8 when ρ=0.9.

**Q: The synthetic experiment shows MIGA wins at p=13 for low ρ. Then why does it lose on Glass (p=10)?**
Glass's RI feature has near-zero variance, which numerically ill-conditions
the covariance term. The synthetic experiment uses clean Gaussian data
without ill-conditioned features. The Glass result reflects the combined
effect of moderate dimensionality AND numerical ill-conditioning from RI.
The synthetic experiment isolates pure dimensionality and correlation effects.

**Q: Why is the Fr advantage shrinking from p=3 to p=8 (Haberman → Wholesale)?**
As p grows, the number of moments that need to be matched (means: p,
covariance: p(p+1)/2, skewness: p) grows quadratically. A GA with fixed
population size (l=200) becomes less effective at matching all these
moments simultaneously. MICE's regression, by contrast, scales gracefully
with p through iterated per-column fits.

**Q: Why does Wholesale 50% give MICE a variance advantage over MIGA?**
At 50% MCAR with n=440 and p=8: expected complete rows ≈ 440 × 0.7^8 ≈ 25.
With only ~25 reference rows, the bootstrap generator samples from a limited
empirical distribution, and the GA has a noisy fitness signal. MICE uses all
observed data for each column regression, giving it more information. The
GA is more information-efficient than the regression only when X_A is large
enough to provide a reliable reference.

**Q: What does the Fr→RMSE inversion prove, formally?**
It is the clearest possible empirical proof of Corollary 2 (the orthogonality
corollary). MIGA achieves its global minimum Fr (0.810) while simultaneously
achieving its global maximum RMSE (0.384) — the two objectives are
simultaneously at their extremes in opposite directions. The result is
confirmed across 10 independent seeds (std = 0.000005), ruling out chance.

**Q: The significance test uses Wilcoxon. Why not a t-test?**
The Wilcoxon signed-rank test is non-parametric — it does not assume the
Fr values across seeds are normally distributed. With 10 seeds, we cannot
reliably test normality. Wilcoxon is the conservative, appropriate choice
for small samples from unknown distributions. It tests whether the median
of (MIGA Fr - Baseline Fr) is < 0.

**Q: What does p=0.001 mean in the Wilcoxon test context?**
With 10 seeds, the minimum achievable p-value is 1/2^10 = 1/1024 ≈ 0.001.
A p=0.001 result means MIGA had lower Fr on all 10 seeds — a perfect record.
This is the most extreme possible result with 10 seeds.

### Group D: Theoretical content

**Q: Explain the law of total variance in one sentence.**
Var(X) = E[Var(X|Y)] + Var(E[X|Y]) — the total variance equals the
expected conditional variance plus the variance of conditional means.

**Q: How does the law of total variance prove Proposition 2.1?**
MICE imputes E[X_j | X_{-j}], which has variance = Var(E[X_j | X_{-j}]).
By the law of total variance: Var(X_j) = E[Var(X_j|X_{-j})] + Var(E[X_j|X_{-j}]).
If Var(X_j|X_{-j}) > 0 (non-trivial conditional variance):
Var(E[X_j|X_{-j}]) < Var(X_j).
So the imputed column has strictly lower variance than the true column.
Lower variance → different distribution → Fr > 0.

**Q: Is this theorem your original contribution?**
The insight is in the literature (van Buuren 2018 §2.6 states the empirical
observation). The formal statement as a theorem with proof in the imputation
context, and the exact connection to Fr, is our contribution. It is not a
deep mathematical theorem — it follows directly from the law of total
variance in 3 lines. The contribution is the formalisation and the
connection to Fr-RMSE as a structural (not empirical) orthogonality.

**Q: What is the Minkowski r=∞ distance geometrically?**
D_∞(x, y) = max_i |x_i - y_i|. It is the Chebyshev distance — the maximum
element-wise difference. It is most sensitive to the single worst-matching
component. This means F_r with r=∞ heavily penalises whichever moment
(mean, covariance, skewness) is worst-matched.

**Q: What does the relative covariance S̃ = S_A^{-1/2} S_C S_A^{-1/2} measure?**
It is the covariance of X_C after whitening by X_A's covariance. When
S_C = S_A (perfect covariance match), S̃ = I (identity). When S_C ≠ S_A,
S̃ deviates from I. The distance D_r(S̃, I) measures how far the relative
covariance is from identity — i.e., how much X_C's covariance structure
differs from X_A's, scaled by X_A's own covariance.

**Q: Why compare to the identity matrix rather than computing |S_A - S_C| directly?**
The relative form is invariant to the scale of X_A's covariance. If X_A
has all features with large variance, a small absolute difference |S_A - S_C|
could be large relative to S_A. The relative form automatically accounts
for X_A's scale — it measures the structural discrepancy, not the absolute
one.

### Group E: Scope and future work

**Q: What is the most important limitation of this thesis?**
The lack of neural baseline comparison (GAIN, GRAPE, MIWAE). These methods
can implicitly learn distributional structure and would be the natural next
comparison for Fr-type objectives. Time and compute constraints prevented
this within the thesis scope.

**Q: If you had 6 more months, what would you add?**
1. Neural baseline comparison (GAIN, GRAPE)
2. IPW-Fr: correct MNAR bias by re-weighting X_A by inverse propensity scores
3. Larger datasets (Cardiotocography n=2126, Adult n=48842)
4. Parallelised MIGA implementation (Q-fold speedup by running Q runs in parallel)

**Q: What is IPW-Fr and why would it help under MNAR?**
Inverse Probability Weighted Fr. Under MNAR, X_A is a biased sample.
IPW estimates the probability π_i that row i is complete (using logistic
regression on observed features) and weights each X_A row by 1/π_i.
Rows that are rarely complete get higher weight, correcting for selection
bias. This would prevent the Fr→RMSE inversion under directional MNAR.

**Q: Why not evaluate on Cardiotocography or Adult?**
Cardiotocography (n=2126, p=21) would likely fall in the "degenerate"
regime similar to synthetic p=20 — very few complete rows at 30% MCAR.
Adult (n=48842, mixed types) was outside the compute budget for 10-seed
significance testing. The five UCI datasets cover the key dimensionality
range (p=3 to p=13) needed to characterise the dimension threshold.

**Q: Is MIGA a proper multiple imputation method?**
Not in the Rubin's rules sense. MIGA produces a single imputed dataset
per run (the best individual from the best of Q runs). For proper multiple
imputation, you would need to generate M distinct imputed datasets. This
could be done by taking the top M individuals from the final generation,
but this is not implemented or evaluated in this thesis. It is a valid
extension.

**Q: The abstract claims the thesis delivers "the first formal proof of Fr-RMSE orthogonality." Is that claim defensible?**
Yes, with appropriate qualification. The van Buuren (2018) quote documents
the empirical observation. Muzellec et al. (2020) note the RMSE-distribution
tradeoff for Sinkhorn imputation. But neither provides a formal theorem
statement with proof specifically for Fr and RMSE as defined in the MIGA
paper. To the best of our knowledge, this specific formal result is original.
In a viva, say: "This is the first formal proof in the context of Fr as
defined by Figueroa-García et al., connecting RMSE optimality to the law
of total variance."

---

## PART 15: NUMBERS TO HAVE MEMORISED

### Core RMSE results (30% MAR, MIGA/MICE)
- Iris: MIGA 0.110 / MICE 0.078 (MICE 1.41×)
- Glass: MIGA 0.155 / MICE 0.075 (MICE 2.07×)
- Haberman: MIGA 0.365 / MICE 0.201 (MICE 1.98×)
- Wholesale: MIGA 0.144 / MICE 0.082 (MICE 1.76×)

### Core Fr results (30% MAR, 10 seeds)
- Iris: MIGA 0.780±0.005 / MICE 1.155 (MIGA 1.48× lower, p=0.001)
- Haberman: MIGA 2.580±0.004 / MICE 5.065 (MIGA 1.97× lower, p=0.001)
- Glass: MIGA 74.03±2.87 / MICE 42.99 (MICE 1.72× lower, p=1.000)
- Wholesale: MIGA 13.059±0.014 / MICE 13.252 (MIGA 1.5% lower, p=0.031)

### The inversion result
- Haberman top MNAR: Fr=0.810 (global min), RMSE=0.384 (global max)
- Confirmed across 10 seeds: Fr std = 0.000005 (deterministic)

### Variance preservation (30% MAR, deviation from 1.0)
- Glass: MIGA 0.005 ★ vs MICE 0.063
- Wholesale: MIGA 0.077 ★ vs MICE 0.123
- Haberman: MIGA 0.535 (poor, single-feature case) vs MICE 0.283
- **Only MICE win: Wholesale 50%** — MIGA 0.120 vs MICE 0.104

### Synthetic experiment key cells
- Best MIGA advantage: p=4, ρ=0 → +73%
- First MIGA loss: p=8, ρ=0.9 → −39%
- MIGA still wins at p=13, ρ=0.6 → +20%
- Hard limit: p=30 → MIGA fails completely

### Ledoit-Wolf (Wine 40%)
- Before LW: condition number = 570 million, d_cov(X_A,X_A) = 0.606
- After LW: condition number = 8.8, d_cov(X_A,X_A) = 0.000, α = 0.42

---

## PART 16: VERBAL FLOW FOR THE PRESENTATION

**One-minute version:**
"Real datasets have missing values. We evaluated MIGA — a genetic
algorithm that fills them in by matching the distribution of complete
cases — against MICE, which minimises prediction error. We prove
mathematically that these are orthogonal objectives: no method can
minimise both simultaneously. Empirically, MIGA achieves up to 2× lower
distributional distance, while MICE achieves up to 2× lower RMSE. A
controlled synthetic experiment shows MIGA's advantage depends on both
dimensionality and correlation — it holds for p≤13 at low correlation but
breaks down at high correlation. We identify when each method is
appropriate."

**How to handle "MICE beats you on RMSE":**
"Yes, and that is expected — and proved. The conditional expectation
(which MICE approximates) is the unique RMSE minimiser. We prove this
formally using the law of total variance. But MICE achieves this by
collapsing variance. Our downstream evaluation shows the consequence:
MICE passes the KS distributional test 0% of the time on Haberman, while
MIGA passes 40% of the time. The question is not which method is better —
it is which objective is right for the downstream task."

**How to handle "what's the main contribution?":**
"Three things that did not exist before this thesis: a public implementation
of MIGA, a formal proof that Fr and RMSE are structurally orthogonal, and
a controlled characterisation of where MIGA's advantage holds. The proof
is the most intellectually novel — it turns an empirical observation into
a theorem."

---

## PART 17: FINAL CHECKLIST

Before the defense, be able to:
- [ ] Write out the Fr formula and explain each term in 2 minutes
- [ ] Explain the law of total variance and how it proves Proposition 2.1
- [ ] State the three MNAR mechanisms and what each does to X_A
- [ ] Explain the Fr→RMSE inversion using the Haberman top result
- [ ] Give the key numbers: 1.48×, 1.97× Fr advantage; 1.41–2.07× RMSE disadvantage
- [ ] Explain why the synthetic experiment's finding is more precise than the UCI threshold
- [ ] Explain why MIGA fails at p=30 (expected complete rows ≈ 0.04)
- [ ] Explain the Haberman kurtosis floor (Nodes zero-inflation = fixed point)
- [ ] Explain the Wholesale 50% variance exception
- [ ] Explain why Ledoit-Wolf helps numerically but not statistically
- [ ] Explain why Wine significance was excluded (98 min/seed × 49 hours)
- [ ] Explain the Glass TAILS exception (only case MIGA wins Fr on p=10)
- [ ] Defend the "first formal proof" claim with appropriate qualification
- [ ] Give the one-sentence answer to "why not just use MICE?"
