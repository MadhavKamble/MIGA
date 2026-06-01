# Research Plan — Novelties & Future Work

Tracks what we intend to contribute beyond the baseline reimplementation of MIGA.

---

## Part A — Reproduction Checklist

Getting the baseline reimplementation complete and verified.

### Datasets
- [x] Iris (02)
- [x] Wine (03) — documented limitation (rank-deficient covariance)
- [x] Glass (04)
- [x] Haberman (05)
- [x] Wholesale (06)
- [ ] Cardio (07) — needs fresh kernel run with skewness fix
- [ ] Adult (08)
- [ ] Run notebook 09 (Results Comparison) after all datasets complete

### Implementation Fixes Applied
- [x] Range-normalised RMSE and all-data CoD (metric formula)
- [x] Bootstrap generators (empirical resampling vs Gaussian)
- [x] Categorical column exclusion via EXCLUDE_COLS
- [x] Eigenvalue floor in relative_cov (numerical stability)
- [x] Feature scaling in FitnessEvaluator (correlation matrix approach)
- [x] Skewness nan → 0 for zero-variance columns
- [x] nan/None guards in GA loop and fit_transform fallback

### Documentation
- [x] Per-dataset diagnostic cells (n_A analysis for Wine)
- [x] Ratio comparison table in notebook 09 (RMSE / MAD / CoD)
- [x] METHODS.md — implementation decisions log
- [x] Haberman seed sweep — confirmed bootstrap advantage on Nodes; Age/OpYear gap is real (1.24–2.06×)
- [x] MAD formula investigation — paper uses σ-norm (partially confirmed); documented in METHODS.md §7.3
- [ ] Thesis chapter: Methodology (describe implementation choices)
- [ ] Thesis chapter: Results & Analysis (with ratio table)
- [ ] Thesis chapter: Discussion (limitations, future work)

---

## Part B — Planned Novelties

Extensions beyond the paper that could form thesis contributions.

### B1. Ledoit-Wolf Shrinkage Covariance
**Idea:** Replace the sample covariance with a Ledoit-Wolf shrinkage estimator for both S_A (reference, computed once) and S_C (completed rows, computed per evaluation). Uses `sklearn.covariance.LedoitWolf`, which analytically minimises expected Frobenius loss: S_LW = (1-α)S_sample + α*(tr(S)/p)*I.

**Motivation:** The fixed 1e-4 eigenvalue floor is ad hoc. LW is theoretically grounded, maintains the d_cov=0 invariant even when n_A < m (rank-deficient case), and reduces S_A condition dramatically (Wine 40%: 570M→8.8).

**Exposed as:** `MIGA(cov_estimator='ledoit_wolf')` (default: `'sample'`).

**Empirical results (Wine, reduced l=50/G=100/Q=2):**
- 30%: sample=0.274, LW=0.280 (similar; n_A/m=1.77, sample already stable)
- 40%: sample=0.275, LW=0.308 (LW slightly worse; both far from paper=0.109)

**Finding:** LW fixes the rank-deficiency (d_cov=0 restored at 40%), but does not improve RMSE in quick tests. The fundamental limit at 40%+ is the tiny n_A (11 rows for 13 features) — no estimator can reliably capture a 13×13 covariance from 11 samples. Full-parameter notebook benchmarks needed.

**Status:** Implemented. Awaiting full notebook benchmark.

### B2. MNAR Extension
**Idea:** Extend the MAR generator framework to support Missing Not At Random (MNAR) mechanisms, where the probability of a value being missing depends on the value itself.

**Motivation:** Real-world data (medical, financial) is often MNAR. The paper only evaluates MAR. An MNAR version would be a genuine contribution.

**Mechanisms implemented in `miga/data_utils.apply_mnar()`:**
- `'top'`    — remove top pct% of values (self-censoring of high values; e.g. high income not reported)
- `'bottom'` — remove bottom pct% of values (floor effects, detection limits)
- `'tails'`  — remove pct/2% from each tail (censoring of extreme values)

**Datasets:** Iris, Glass, Haberman at 30% missing.

**Run scripts (parallel):**
```
.venv/bin/python scripts/run_mnar_dataset.py Iris
.venv/bin/python scripts/run_mnar_dataset.py Glass
.venv/bin/python scripts/run_mnar_dataset.py Haberman
```
Saves JSON to `results/11_mnar_<dataset>.json`. Notebook 11 aggregates and plots.

**Key hypothesis:** MIGA minimises distributional distance (Fr), never assuming MAR.
Under MNAR, X_A and X_C differ systematically in distribution; Fr may decrease
but RMSE may increase. Expected finding: MIGA degrades gracefully (better than MAR-
assuming methods), but imputation of tail-removed values is harder than random removal.

**Status:** Implemented (`apply_mnar`, `scripts/run_mnar_dataset.py`, `notebooks/11_MNAR_Extension.ipynb`). Awaiting experiment runs.

### B2b. Adaptive c3 Mutation Schedule
**Idea:** Linearly decay c3 from a high start value (broad exploration) to a low end value (fine exploitation): c3(g) = round(c3_start + (c3_end − c3_start) × g/(G−1)).

**Motivation:** Fixed c3 is a constant mutation pressure throughout all generations. Early generations benefit from many diverse offspring; late generations benefit from focused refinement of a good elite. This exploration→exploitation tradeoff is a foundational principle in evolutionary computation (Holland 1975; Goldberg 1989).

**Exposed as:** `MIGA(c3_schedule=(15, 3))` — defaults to `None` (fixed c3, backward-compatible).

**Generation history:** `miga.generation_history_` now stores best Fr per generation per run, enabling convergence plots.

**Previous evidence (PyGAD implementation):** Iris +8% RMSE, Glass +3.5% RMSE, Wilcoxon p<0.01 on all 3 tested datasets.

**Status:** Implemented. Run notebook 10 (`10_Adaptive_vs_Fixed.ipynb`) for results.

### B3. Convergence Criterion / Early Stopping
**Idea:** Stop a run early when F_r improvement across the last k generations falls below a threshold, rather than always running all G generations.

**Motivation:** Paper uses fixed G (up to 2000 for Cardio/Adult). Early stopping could drastically reduce runtime with minimal performance loss.

**Status:** Not started.

### B4. Warm-Start Population Initialisation
**Idea:** Seed the initial population with simple imputation results (mean, k-NN) rather than purely random generator samples.

**Motivation:** Could significantly accelerate convergence, especially on large datasets (Adult: 48K rows).

**Status:** Not started.

### B5. Parallel Run Execution
**Idea:** Run Q independent runs in parallel using Python multiprocessing rather than sequentially.

**Motivation:** The Q runs are fully independent. Parallelising them would give Q× speedup with no change to results.

**Status:** Not started. Infrastructure change only (no algorithmic change).

---

## Part C — Future Work (Thesis Discussion Section)

Items to mention as future directions, not planned to be implemented.

- **Non-parametric covariance:** Replace the sample covariance with a kernel-based estimator for highly non-Gaussian features.
- **Deep generator networks:** Replace bootstrap sampling with a VAE or normalising flow trained on the observed marginals, giving better-calibrated generators for complex distributions.
- **Multi-table imputation:** Extend MIGA to impute across relational tables where features span multiple entities.
- **Monotone missing patterns:** Exploit monotone missingness structure (common in longitudinal data) to reduce the search space.
- **Categorical-native fitness:** Design a fitness function that handles genuinely categorical features (ordinal, nominal) without label encoding.
- **Benchmarking on larger datasets:** Paper's largest dataset is Adult (48K × 14). Evaluate on truly large-scale datasets (millions of rows) with approximate covariance methods.
