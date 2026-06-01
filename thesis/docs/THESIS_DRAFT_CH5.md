# Chapter 5 — Discussion

---

## 5.1 The Central Finding: Fr and RMSE Are Orthogonal Objectives

The most important result of this thesis is the empirical demonstration that distributional distance (Fr) and pointwise accuracy (RMSE) are structurally orthogonal objectives. They are not two measurements of the same underlying quality; they measure different things, and optimising one does not optimise the other.

The evidence is threefold.

**Directionality under MAR.** MIGA achieves significantly lower Fr than MICE on Iris (1.48× lower, p = 0.001) and Haberman (1.97× lower, p = 0.001). MICE achieves lower RMSE than MIGA on every dataset by 1.4–2.1×. Neither method dominates the other: each wins decisively on its own metric and loses on the other's. This is not a quantitative difference of degree — it is a qualitative difference of objective.

**The MNAR inversion.** Under `top` MNAR on Haberman, MIGA achieves Fr = 0.810 — the global minimum across all experiments — while simultaneously achieving RMSE = 0.385, the global maximum. The standard deviation of Fr across 10 seeds is 0.000005: the GA *reliably* converges to this minimum, confirming it is not a sampling artefact. A method cannot achieve a lower Fr without simultaneously worsening RMSE more severely; the two objectives are pulling in exactly opposite directions.

**Theoretical grounding.** The orthogonality is predicted by theory. Van Buuren (2018, §2.6) states that regression imputation achieves minimum RMSE by suppressing variance — which is precisely the mechanism that inflates Fr. MICE imputes $\hat{X}_j = \mathbb{E}[X_j \mid X_{-j}]$, concentrating all imputed values at the conditional mean. By the law of total variance, $\text{Var}(\hat{X}_j) = \text{Var}(\mathbb{E}[X_j \mid X_{-j}]) < \text{Var}(X_j)$. The suppressed variance propagates to higher Fr — MICE's RMSE optimality causes its distributional distortion. There is no way to eliminate both simultaneously with a single imputation.

The practical consequence is that the choice between MIGA and MICE is not a matter of one being better, but of which objective is appropriate for the downstream analysis.

---

## 5.2 When MIGA Is the Right Choice

MIGA is the appropriate imputation method when the downstream analysis requires the correct marginal or joint distribution of the imputed data, not merely accurate individual cell values.

**Confidence intervals and hypothesis tests.** Multiply-imputed datasets are often used to compute standard errors via Rubin's rules (Rubin, 1987). These rules assume that the variance of each imputed dataset is approximately correct. MICE's systematic variance suppression (ratios 0.717–0.976 across datasets) biases the within-imputation variance downward, producing overconfident confidence intervals. MIGA's variance ratios of 1.005 (Glass) and 1.077 (Wholesale) are substantially closer to the nominal 1.0, producing less biased standard errors.

**Distributional hypothesis testing.** Kolmogorov-Smirnov tests, Anderson-Darling tests, and quantile-quantile comparisons all operate on the marginal or joint distribution of the imputed data. A MICE-imputed dataset has artificially compressed marginal distributions; tests on this data will have inflated Type I error rates for tests of distributional equality, or deflated power for tests of distributional difference.

**Synthetic data generation.** When imputed data is used to augment training sets or generate synthetic records (e.g. in privacy-preserving data publishing), the synthetic records must reflect the true marginal and joint distributions. MICE-imputed synthetic data will be over-concentrated near the conditional means, producing a dataset that is less diverse than the true population and potentially harmful for fairness analyses.

**Datasets with heavy-tailed or discrete features.** The kurtosis analysis (§4.8) shows that MICE suppresses tail behaviour more severely than variance on features like Haberman's Nodes column (44% zeros, strong positive kurtosis; MICE $d_{\text{kurt}} = 7.783$ vs MIGA $d_{\text{kurt}} = 1.854$). For analyses that are sensitive to the tails of the distribution — extreme value analysis, risk modelling, rare event prediction — MIGA's bootstrap generators preserve the tail structure that MICE's conditional means eliminate.

---

## 5.3 When MICE Is the Right Choice

MICE is the appropriate method when the downstream analysis requires minimum pointwise prediction error, or when the dataset is high-dimensional with strong inter-feature correlations.

**Predictive modelling.** When the imputed values are used as features in a predictive model (regression, classification), RMSE of the imputed values directly relates to the bias-variance tradeoff of the downstream model. MICE's 1.4–2.1× lower RMSE translates to better-quality features, particularly for linear models where imputation error propagates directly to prediction error.

**High-dimensional correlated data.** The dimension effect (§4.6.2) shows that MICE achieves lower Fr than MIGA on Glass ($p = 10$) across all experimental conditions (p = 1.0 for MIGA winning Fr). MICE's column-wise regressions collectively model the joint distribution when features are strongly correlated — iterative conditional regression is, in the limit, equivalent to drawing from the joint Gaussian distribution. For datasets where the joint distribution is well-approximated by a multivariate Gaussian and $p$ is moderate to large, MICE incidentally achieves good distributional fit alongside good RMSE.

**Compute constraints.** MIGA requires approximately 180–210 seconds per seed on a standard CPU for the benchmark datasets ($l=200$, $G=300$, $Q=3$). MICE runs in under one second. For large datasets or production pipelines where imputation must be fast, MIGA's compute cost is prohibitive without parallelisation.

---

## 5.4 Limitations

### 5.4.1 MNAR Failure Mode

MIGA's fitness function is fundamentally unable to detect that X_A is a biased sample of the population under directional MNAR. The Fr→RMSE inversion on Haberman `top` (§4.5.2) is the most extreme manifestation: a method that achieves Fr = 0 under `top` MNAR is not correctly imputing the data — it is correctly matching a biased reference. This is not a weakness of MIGA's implementation but a structural property of any distributional imputation method that uses the observed complete cases as its reference distribution without adjustment.

Under `tails` MNAR (symmetric censoring), MIGA is surprisingly robust: RMSE improves relative to MAR on both Iris and Haberman. The critical distinction is directionality: one-sided censoring (`top`, `bottom`) biases X_A in a predictable direction and systematically misleads the fitness function, while two-sided censoring concentrates X_A near the distribution centre without systematic bias in one direction.

### 5.4.2 Dimension Effect

MIGA's distributional advantage diminishes as $p$ increases. On Glass ($p = 10$), MICE achieves both lower RMSE (2.07×) and lower Fr (1.72×) — MIGA loses on both metrics simultaneously. The fitness function's moment-matching (means, covariance, skewness, kurtosis) is a finite-dimensional summary of the joint distribution that becomes increasingly coarse as $p$ grows. MICE's iterative regression implicitly approximates the full conditional distribution for each variable, which is a richer representation than any fixed set of moments for high-$p$ data.

The dimension threshold at which MICE surpasses MIGA on Fr appears to be between $p = 4$ (Iris, where MIGA wins) and $p = 10$ (Glass, where MICE wins). Determining this threshold more precisely — and whether it depends on the correlation structure of the data — is an open question.

### 5.4.3 Compute Cost

The per-seed runtime of 180–210 seconds for MIGA (compared to $<1$ second for MICE) represents a 200–300× compute overhead. For the benchmark datasets ($n \leq 450$), this is acceptable in a research context. For larger datasets — Cardiotocography ($n = 2126$) and Adult ($n = 48842$) — the runtime scales unfavourably. The $Q$ independent runs are trivially parallelisable (they share no state), so wall-clock time is reducible by a factor of $Q$ without any algorithm change.

### 5.4.4 Single-Feature Degenerate Case

When only one feature is made missing ($p = 3$, Haberman), MIGA's multivariate distributional objective reduces to matching a single univariate distribution. In this degenerate case, the covariance term carries little information (a $1 \times 1$ covariance matrix) and the bootstrap generator has no multivariate structure to exploit. Mean imputation is near-optimal for a single near-Gaussian feature, and MIGA's 2× RMSE disadvantage on Haberman reflects this. The variance and kurtosis analyses are similarly affected: the single-feature bootstrap over-inflates variance (ratio = 1.535) and cannot improve kurtosis beyond what standard MIGA already achieves.

---

## 5.5 Relationship to Prior Work

### 5.5.1 MIGA vs the Original Paper

Figueroa-García et al. (2023) compare MIGA against CMIM (Batista and Monard, 2003) and ANNI, both of which are substantially weaker than modern baselines. Our comparison against MICE (van Buuren and Groothuis-Oudshoorn, 2011) and KNN (Troyanskaya et al., 2001) provides the first evaluation of MIGA against current methods. The Fr advantage of MIGA over MICE is confirmed with statistical significance on two of three tested datasets; the RMSE disadvantage is confirmed on all. Neither result was present in the original paper.

The paper does not evaluate MNAR. Our MNAR experiments provide the first characterisation of MIGA's behaviour under non-ignorable missingness, revealing the Fr→RMSE inversion as a fundamental consequence of the distributional objective under directional selection bias.

### 5.5.2 Relationship to Optimal Transport Imputation

Muzellec et al. (2020) propose imputation via the Sinkhorn divergence — a regularised optimal transport distance between the completed and observed distributions. This is conceptually similar to MIGA's Fr objective: both minimise a measure of distributional discrepancy between the imputed and observed data. Muzellec et al. also note that distributional methods achieve higher RMSE than regression-based methods, consistent with our findings. The key differences are: (i) the Sinkhorn divergence is differentiable (enabling gradient-based optimisation), while Fr requires a derivative-free method (the GA); (ii) MIGA handles mixed variable types natively through per-variable generators; (iii) the Sinkhorn approach is computationally more efficient for large $n$.

### 5.5.3 Relationship to Proper Multiple Imputation

Van Buuren (2018, §3.4) defines "proper" imputation as drawing from the full posterior predictive distribution $P(X_{\text{mis}} \mid X_{\text{obs}})$ rather than its mean. Proper imputation preserves variance and produces valid inferences under Rubin's rules. MIGA can be understood as a non-parametric approximation to proper imputation: rather than drawing from a parametric posterior, it uses a GA to find imputations that match distributional moments. The bootstrap generator plays the role of the prior — sampling from the observed marginal distribution rather than a parametric model. MIGA's variance ratios close to 1.0 (Glass: 1.005, Wholesale: 1.077) are consistent with the variance-preservation property of proper imputation.

---

## 5.6 Future Work

### 5.6.1 IPW-Fr: Correcting MNAR Bias (Priority)

The most impactful theoretical extension is inverse probability weighted Fr (§3.5.3). Under MNAR, X_A is a biased sample; IPW re-weights complete rows by $1/\pi_i$ (where $\pi_i$ is the estimated probability of row $i$ being complete) to correct for this bias. The propensity scores $\pi_i$ can be estimated by logistic regression of the completeness indicator on observed covariates. IPW-Fr would allow MIGA to maintain a valid distributional objective under directional MNAR, directly addressing the failure mode documented in §4.5.2.

### 5.6.2 Adaptive Diversity Injection

Our experiments reveal that diversity injection (90% random individuals per generation) is the primary driver of MIGA's exploration, not the mutation rate $c_3$. An adaptive diversity injection rate — reducing the fraction of random individuals over generations as the GA converges — would provide a more impactful exploration-exploitation tradeoff than the adaptive $c_3$ schedule. This is the natural extension of C3.

### 5.6.3 Neural Baseline Comparison

GAIN (Yoon et al., 2018) and GRAPE (You et al., 2020) represent the current state of the art in neural imputation. A systematic comparison of MIGA against these methods — on both Fr and RMSE — would place MIGA's distributional advantage in the context of modern deep learning approaches. GAIN's adversarial discriminator is conceptually analogous to MIGA's Fr objective, making this comparison particularly informative.

### 5.6.4 Parallelisation

The $Q$ independent GA runs share no state and can be executed in parallel with no algorithm change. A $Q$-fold speedup would reduce the per-experiment wall-clock time from ~200 seconds to ~40 seconds (for $Q = 5$), making MIGA practical for larger datasets and interactive use.

### 5.6.5 Moment Expansion

The kurtosis extension (C8) demonstrates that adding the 4th moment to the fitness function provides measurable improvement on low-dimensional data (Iris) and reveals new axes on which MIGA surpasses MICE (Haberman kurtosis). Higher-order moments (5th, 6th), or a non-parametric alternative such as kernel density estimation-based energy distance, would provide a more complete distributional characterisation — at the cost of increased estimation noise for small $n_A$.

---

## 5.7 Summary

This thesis has demonstrated five things:

1. **MIGA can be faithfully reimplemented** from the paper specification. Our open-source implementation achieves paper-level RMSE on 5/7 benchmark datasets, with the Wine gap explained as a hard statistical limit (n_A < p at 40%+ missing).

2. **MIGA's distributional advantage is real and significant** on low-dimensional datasets. On Iris and Haberman under MAR, MIGA achieves 1.5–2.0× lower Fr than MICE with p < 0.001 across 10 seeds. This is the first rigorous statistical confirmation of MIGA's distributional objective.

3. **MIGA's distributional advantage is dimension-dependent.** MICE achieves lower Fr on Glass (p = 10) across all conditions, revealing that iterative conditional regression implicitly captures the joint distribution for correlated high-dimensional data. The boundary between the two regimes lies between p = 4 and p = 10.

4. **MIGA fails under directional MNAR in a characterisable way.** The Fr→RMSE inversion (Fr global minimum, RMSE global maximum, confirmed p < 0.001) is the clearest possible empirical demonstration that a low Fr score does not guarantee correct imputation when X_A is a biased sample. This failure is not algorithmic — it is a structural consequence of using the observed complete cases as the distributional reference under non-ignorable missingness.

5. **MIGA preserves the marginal distribution more completely than MICE.** MIGA's variance ratios are closer to 1.0 on multi-feature datasets (Glass: 0.005 deviation, Wholesale: 0.077 deviation vs MICE's 0.063 and 0.123). MIGA also preserves kurtosis substantially better on heavy-tailed features (Haberman: 4.2× lower kurtosis distance than MICE). These results extend the characterisation of MICE's distributional distortion from variance alone to the broader shape of the marginal distribution.

The central recommendation is: use MIGA when the downstream analysis requires the correct distribution; use MICE when it requires minimum pointwise error. Both are valid objectives; they require different methods.

---

*[Chapter 5 draft complete. All claims are supported by experimental results in Chapter 4 and significance tests in results/13_significance_*.json.]*
