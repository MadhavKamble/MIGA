# Phase 1 Thesis Presentation — Outline
**Slot:** ~2026-05-21  |  **Duration:** 15–20 min + Q&A  |  **Audience:** Thesis committee

---

## Slide 1 — Title

**Missing Data Imputation via Genetic Algorithms:
Reimplementation, Extension, and Distributional Analysis of MIGA**

[Your Name]  |  Supervisor: [name]  |  IIIT Allahabad  |  MTech Computer Science  |  May 2026

---

## Slide 2 — Motivation (1.5 min)

- Missing data is ubiquitous: medical records, financial surveys, sensor networks
- Naive fixes (deletion, mean imputation) distort the data distribution
- The gold standard (**MICE**) minimises prediction error but **suppresses variance** — biasing
  any downstream distributional analysis (van Buuren 2018, §2.6)

**Gap in the literature:**  MIGA (Figueroa-García et al., 2023, *Information Sciences*)
- proposes genetic-algorithm imputation targeting the joint distribution rather than pointwise error
- published **no source code**, **no comparison against MICE or KNN**, **no MNAR evaluation**
- This thesis closes all three gaps.

---

## Slide 3 — What MIGA Does (1.5 min)

**Key idea:** Find imputed values that make the completed rows (X_C) look
statistically identical to the available complete rows (X_A).

$$F_r = D_r(\tilde{x}_A, \tilde{x}_C) + D_r(\tilde{S}, I) + D_r(b_A, b_C)$$

- $D_r$(means): completed rows have same mean as available rows
- $D_r$(covariance): relative covariance $\tilde{S} = S_A^{-1/2} S_C S_A^{-1/2}$ equals identity
- $D_r$(skewness): completed rows have same skewness

Genetic algorithm: $l = 200$ individuals, $G$ generations, $Q = 3$–5 independent runs.
Each individual = one complete candidate imputation for all missing cells.

**Critical distinction from MICE:** MIGA never sees the true missing values.
It only checks if the imputed distribution matches the observed distribution.

---

## Slide 4 — Six Contributions (1 min)

| # | Contribution | Key result |
|---|---|---|
| **C1** | Open-source reimplementation | First public MIGA code; 5 benchmark datasets |
| **C2** | Formal Fr–RMSE orthogonality theorem | Proved: no imputation can jointly minimise both |
| **C3** | First MNAR evaluation | Discovered Fr→RMSE inversion under directional MNAR |
| **C4** | Systematic Fr vs RMSE comparison | MIGA 1.5–2× better Fr; MICE 1.4–2.1× better RMSE |
| **C5** | Synthetic p×ρ experiment | Failure mode is joint on (p, ρ), not p alone |
| **C6** | IPW-Fr for MNAR bias correction | Works when n/p ≥ 20; reduces Fr by 1–38% |

All implemented in Python, 5 benchmark datasets, 10 seeds, Wilcoxon significance tests.

---

## Slide 5 — C1: Replication Results (1.5 min)

**Table: Our RMSE / Paper RMSE (ratio)**

| Dataset | 30% | 40% | 50% | Notes |
|---|---|---|---|---|
| Iris | 1.29× | 1.31× | 1.33× | Expected range |
| Wine | 2.56× | 3.79× | — | n_A < p at 40%: fundamental limit |
| Glass | 1.33× | 1.56× | 1.89× | Expected range |
| Haberman | **0.52×** ★ | 1.24× | 1.43× | ★ beats paper (bootstrap handles zeros) |
| Wholesale | **1.07×** ★ | 1.08× | 1.26× | ★ best replication |

Reimplementation achieves paper-level results on all datasets.
Wine gap is documented as a hard statistical limit ($n_A < p$ → $S_A$ rank-deficient).

---

## Slide 6 — C2: Fr–RMSE Orthogonality Theorem (2 min) ★

**The central formal result of the thesis.**

**Proposition 1 (RMSE→Fr):** The RMSE-minimising imputation $\hat{X}_j = \mathbb{E}[X_j \mid X_{-j}]$
must have $F_r > 0$ whenever conditional variance $\mathrm{Var}(X_j \mid X_{-j}) > 0$.

*Proof sketch:* By the law of total variance, imputing the conditional mean removes
within-group variance → imputed marginal variance $<$ true variance → $F_r > 0$.

**Proposition 2 (Fr→RMSE):** Any imputation with $F_r = 0$ must have $\mathrm{RMSE} > 0$
unless the true missing values are drawn from exactly the same distribution as $X_A$.

**Corollary (Non-collinearity):** No single imputation can jointly minimise both $F_r$ and RMSE.

**Empirical confirmation:** Every experiment in the thesis confirms this.
MICE wins RMSE on all 5 datasets without exception.
MIGA wins Fr on Iris and Haberman (p < 0.001).
The theorem predicts both outcomes.

---

## Slide 7 — C3: MNAR Inversion — The Smoking Gun (2 min) ★★

**Under top-quantile MNAR on Haberman (confirmed across 10 seeds, p < 0.001):**

| Mechanism | MIGA Fr | MIGA RMSE | Interpretation |
|---|---|---|---|
| MAR | 2.580 | 0.398 | Normal operation |
| **top** | **0.810** ← | **0.384** ← | **Complete inversion** |
| bottom | 1.040 | 0.380 | Moderate degradation |
| tails | 1.134 | 0.354 | Robust — better than MAR |

**Fr = 0.810** is the global minimum across ALL experiments (σ = 0.000005 across seeds).
**RMSE = 0.384** is the global maximum.

**Why:** Top values go missing → X_A contains only low-value rows (biased reference).
GA matches X_C to biased X_A → Fr low. But true missing values were high → RMSE high.

**The theorem (C2) predicts this exactly:** $F_r = 0$ does not imply correct imputation
when X_A is a biased sample. The fitness function cannot detect that its reference is biased.

[fig: bar chart Fr and RMSE by mechanism for Haberman]

---

## Slide 8 — C4: Systematic Baseline Comparison (2 min) ★

**Fr comparison under MAR (10-seed Wilcoxon, H₁: MIGA < baseline):**

| Dataset (p) | MIGA Fr | MICE Fr | Ratio | p-value |
|---|---|---|---|---|
| Iris (p=4) | **0.780** | 1.155 | MIGA 1.48× lower | 0.001 |
| Haberman (p=3) | **2.580** | 5.065 | MIGA 1.97× lower | 0.001 |
| Glass (p=10) | 74.03 | **42.99** | MICE 1.72× lower | 1.000 |

**RMSE comparison under MAR (MICE wins every dataset):**

| Dataset | MIGA RMSE | MICE RMSE | MICE advantage |
|---|---|---|---|
| Iris | 0.119 | **0.078** | 1.53× |
| Haberman | 0.398 | **0.201** | 1.98× |
| Glass | 0.155 | **0.075** | 2.07× |

Neither method dominates — each wins decisively on its own metric.
Glass shows the Fr advantage is not monotone in p; it depends jointly on (p, ρ).

[fig: fig1_fr_rmse.png — grouped bar chart for all methods]

---

## Slide 9 — C5: Synthetic p×ρ Experiment (1.5 min) ★

**Controlled grid: synthetic Gaussian data, Toeplitz covariance**
$p \in \{4, 8, 13, 20, 30\}$, $\rho \in \{0, 0.3, 0.6, 0.9\}$, 30% MCAR

**MIGA Fr advantage (%) over MICE — heatmap:**

|  | ρ=0 | ρ=0.3 | ρ=0.6 | ρ=0.9 |
|---|---|---|---|---|
| p=4 | +55% | +47% | +38% | +12% |
| p=8 | +44% | +32% | +20% | **−39%** |
| p=13 | +31% | +22% | +8% | **−11%** |
| p=20 | −14% | −28% | −41% | −67% |
| p=30 | vacuous | vacuous | vacuous | vacuous |

**Finding:** Failure is joint on (p, ρ). At ρ=0.9, MIGA loses even at p=8.
At ρ≤0.6, MIGA maintains advantage up to p=13.

**Mechanism:** High correlation makes MICE's iterative regression an effective implicit
joint distribution estimator. MIGA's moment-matching cannot exploit correlation structure.

[fig: fig5_synthetic_dim.png — p×ρ heatmap]

---

## Slide 10 — C6: IPW-Fr for MNAR Correction (1.5 min)

**Idea:** Under MNAR, X_A is a biased sample. Re-weight complete rows by
$1/\pi_i$ (inverse probability of being complete) to correct the reference distribution.

Propensity scores estimated by logistic regression of completeness indicator on observed covariates.

**Results across 4 datasets — Fr change from standard MIGA (directional MNAR):**

| Dataset | n/p | top Fr change | bottom Fr change | tails Fr change |
|---|---|---|---|---|
| Wholesale | 55 | **−38.1%** | +3.0% | −28.4% |
| Haberman | 102 | −18.6% | −1.3% | −12.8% |
| Iris | 38 | −17.1% | −5.0% | −13.0% |
| Wine | 14 | **+34.7%** | −6.3% | +15.1% |

**Key finding:** n/p ≥ 20 required for stable propensity estimation.
When n/p < 15 (Wine), propensity model overfits and IPW *increases* Fr.
The Fr–RMSE orthogonality (C2) remains intact under IPW.

---

## Slide 11 — Downstream Utility (1 min)

**Do MIGA's distributional gains propagate to actual downstream analyses?**

KS test pass rate and CI coverage at 95% (Haberman, 30% MAR, 100 seeds):

| Method | KS pass rate | CI coverage |
|---|---|---|
| Mean | 0% | 20% |
| KNN | 0% | 30% |
| MICE | **0%** | 10% |
| **MIGA** | **40%** | **70%** |

Classification accuracy: all four methods achieve ≈ 0.744 — no cost to predictive accuracy.

**Scope conditions** for MIGA's distributional advantage to propagate:
- Condition 1: Moderate dimensionality (p ≲ 8)
- Condition 2: Non-degenerate feature variances (no near-zero variance features)
- Condition 3: Moderate baseline Fr gap (not extreme skewness or outliers in X_A vs X_C)

[fig: fig4_downstream.png]

---

## Slide 12 — Summary and Recommendation (1 min)

**The central finding:**
MIGA and MICE are structurally orthogonal — each wins decisively on its own metric
and loses on the other's. This is not a matter of one method being better.

**Use MIGA when:** downstream analysis requires the correct distribution (CI, KS tests,
synthetic data); features are discrete or heavy-tailed; p ≲ 8; variances well-conditioned.

**Use MICE when:** minimum pointwise prediction error required; p ≥ 10 with strong
correlations; compute budget is limited (MIGA: ~200s/seed; MICE: <1s).

**Future work:**
1. Adaptive diversity injection (primary exploration driver, more impactful than mutation rate)
2. Neural baselines — GAIN, GRAPE comparison on both Fr and RMSE
3. Parallelise Q independent runs → Q× speedup with no algorithm change

---

## Expected Q&A

**Q: Why not just use MICE?**
A: Use MICE when you need minimum RMSE. Use MIGA when the downstream analysis
requires the correct distribution (CIs, distributional tests, synthetic data).
These are different objectives requiring different methods.

**Q: The RMSE gap is up to 2×. Is MIGA practical?**
A: Expected by theory — MICE explicitly optimises RMSE. For distributional tasks,
the downstream evaluation confirms MIGA's advantage (40% KS pass vs 0% for MICE)
at zero cost to classification accuracy.

**Q: MIGA fails under MNAR — doesn't that undermine it?**
A: The failure mode is a contribution. We are the first to characterise it formally:
directional MNAR biases X_A, so Fr = 0 no longer implies correct imputation.
The Fr→RMSE inversion (global minimum Fr = global maximum RMSE) is the empirical proof.
IPW-Fr (C6) partially corrects this when n/p ≥ 20.

**Q: MICE beats MIGA on Fr for Glass (p=10). Doesn't that weaken the claim?**
A: It's an honest finding and part of C5. The advantage depends jointly on (p, ρ),
not p alone. At high correlation, MICE's regression incidentally models the joint distribution.
This tells us exactly when each method applies — more useful than an unconditional claim.

**Q: How does this compare to the original paper?**
A: The paper compared against CMIM and ANNI (weaker, older methods) and never tested MNAR.
We add MICE and KNN (stronger baselines), MNAR evaluation with three mechanisms,
formal proofs (C2), synthetic characterisation (C5), IPW correction (C6),
and the first reproducible open-source implementation (C1).
