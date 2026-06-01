# Chapter 2 — Background

---

## 2.1 Missing Data Mechanisms

The behaviour of any imputation method depends critically on the mechanism that produces the missing values. Rubin (1976) established the canonical taxonomy of three missing data mechanisms, which determines the conditions under which different imputation strategies are valid.

**Missing Completely At Random (MCAR).** A value is MCAR if the probability of its being missing is independent of both the observed data and the missing data. Formally, if R is the missingness indicator matrix (R_ij = 1 if X_ij is observed, 0 if missing), then MCAR requires P(R | X) = P(R). Under MCAR, complete-case analysis is unbiased, though it discards information. MCAR is the most restrictive assumption and rarely holds in practice: it would require, for example, that blood pressure readings were missing at random independent of the patient's condition.

**Missing At Random (MAR).** A value is MAR if the probability of its being missing may depend on the observed data but not on the missing data itself: P(R | X) = P(R | X_obs). Under MAR, the mechanism is ignorable (Rubin, 1976) — valid imputation can be performed using only the observed data, without modelling the missingness mechanism. Most modern imputation methods, including MICE and MIGA, assume MAR. Survey non-response that depends on respondents' other known characteristics (age, income bracket) is typically MAR.

**Missing Not At Random (MNAR).** A value is MNAR if its probability of being missing depends on the value itself, even after conditioning on observed data: P(R | X) ≠ P(R | X_obs). MNAR is the most challenging case. Examples include income values that are more likely to be missing for high earners (who do not wish to disclose), or medical test results missing for patients who did not attend follow-up appointments. Under MNAR, the missingness mechanism is non-ignorable: any imputation method that ignores the mechanism — including MIGA — will produce biased results. Chapter 4 of this thesis provides the first empirical characterisation of MIGA's failure mode under MNAR.

The three mechanisms form a hierarchy of restrictiveness. MCAR implies MAR, and MAR implies the mechanism is ignorable. The correct mechanism for a dataset is often unknown and must be assumed based on domain knowledge. Van Buuren (2018, §1.2) provides an extensive treatment of how to reason about mechanisms in practice.

---

## 2.2 Classical Imputation Methods

### 2.2.1 Complete-Case Analysis

Complete-case analysis (listwise deletion) discards any observation with at least one missing value, performing the analysis on the complete cases only. Under MCAR, this produces unbiased estimates but reduces statistical power. Under MAR or MNAR, it introduces selection bias: the complete cases are a non-representative subset of the population. With 30% missing values spread across features, the fraction of complete rows can fall below 10% even for moderate p (see §4.1 for dataset statistics). Complete-case analysis is therefore rarely appropriate for real-world datasets.

### 2.2.2 Single Imputation

Single imputation replaces each missing value with a single plausible estimate. The simplest method substitutes the column mean (for continuous features) or the column mode (for categorical features). Mean imputation is computationally trivial and introduces no bias in the mean itself, but it severely distorts the variance structure: replacing 30% of a column with its mean compresses the variance by approximately (1 − 0.3)² ≈ 0.49, i.e., nearly halving it. This distortion compounds across features, making mean imputation inappropriate for any analysis that uses variance, correlation, or distributional properties.

KNN imputation (Troyanskaya et al., 2001) replaces each missing value with a weighted average of the k nearest complete observations in the feature space. The intuition is that similar observations tend to have similar values for a missing feature. KNN preserves local structure better than mean imputation and is non-parametric, but its quality depends on the choice of k and the distance metric, and it does not account for the uncertainty introduced by imputation.

Both mean and KNN imputation are single-imputation methods: they produce one complete dataset and treat imputed values as if they were observed. This understates the uncertainty due to missingness and produces overconfident inferences (Rubin, 1987).

### 2.2.3 Multiple Imputation by Chained Equations (MICE)

Multiple imputation generates M > 1 complete datasets by drawing imputed values from the predictive distribution P(X_mis | X_obs), combines the analysis results across datasets using Rubin's rules, and produces inferences that properly account for missingness uncertainty.

MICE (van Buuren and Groothuis-Oudshoorn, 2011), also known as Fully Conditional Specification (FCS), is the dominant multiple imputation method. For a dataset with p partially observed variables, MICE iterates a cycle of p conditional imputation models: for each variable j, it fits a regression model of X_j on all other variables X_{-j} using the currently completed data, then draws imputations from the posterior predictive distribution of this model. The cycle repeats until convergence, producing one complete dataset per chain.

The key strength of MICE is its flexibility: each variable can use a different model family (linear regression for continuous variables, logistic regression for binary variables, polytomous regression for categorical variables). MICE is theoretically valid under MAR and produces imputations with the correct conditional distribution of each variable given the others.

However, MICE's optimality criterion is minimum expected squared error, which corresponds to imputing conditional means. Van Buuren (2018, §2.6) explicitly states:

> "The minimum mean squared error is achieved by regression imputation, which suppresses variance and produces biased statistical inference."

This is not a flaw but a structural property: by imputing E[X_j | X_{-j}], MICE systematically underestimates the marginal variance Var(X_j). By the law of total variance, Var(X_j) = E[Var(X_j | X_{-j})] + Var(E[X_j | X_{-j}]), and MICE discards the first term. For analyses that require the correct marginal distribution — distributional hypothesis tests, bootstrap inference, synthetic data generation — MICE produces biased results. This is the fundamental motivation for MIGA.

---

## 2.3 Genetic Algorithms

### 2.3.1 Foundations

A Genetic Algorithm (GA) is a population-based metaheuristic search method inspired by Darwinian natural selection (Holland, 1975). A GA maintains a population of candidate solutions, evaluates each candidate using a fitness function, and iteratively produces better solutions by applying selection, crossover (recombination), and mutation operators.

The canonical GA operates as follows (Goldberg, 1989):

1. **Initialisation:** Generate an initial population of l individuals, typically at random.
2. **Evaluation:** Compute the fitness f(x) for each individual x.
3. **Selection:** Select individuals for reproduction with probability proportional to fitness.
4. **Crossover:** With probability p_c, combine pairs of selected individuals to produce offspring.
5. **Mutation:** With probability p_m, randomly alter components of offspring.
6. **Replacement:** Form the new population from survivors and offspring.
7. **Repeat** steps 2–6 for G generations or until a convergence criterion is met.

GAs are particularly suited to problems where the fitness landscape is discontinuous, multimodal, or not differentiable — properties that make gradient-based methods inapplicable. The exploration-exploitation tradeoff is controlled by the balance between diversity (random initialisation, mutation, diversity injection) and selection pressure.

### 2.3.2 MIGA's Genetic Algorithm

MIGA (Figueroa-García et al., 2023) uses a custom GA that departs from the canonical scheme in several respects. Most notably, MIGA uses an aggressive diversity injection: at each generation, the majority of the population (approximately 90% under the paper's default parameters) is replaced with randomly generated individuals drawn from per-variable generators R_j. This effectively restarts the search from random each generation, preserving only the best c individuals. The consequence — documented in this thesis (§3.3) — is that the diversity injection, not the mutation rate, is the primary driver of MIGA's exploration behaviour.

The GA optimises the fitness function F_r over the space of possible imputed values for the missing cells. Each individual in the population represents one complete candidate imputation: a vector of values for all missing cells in the dataset. The fitness function F_r measures distributional distance between the available rows X_A and the completed rows X_C, penalising divergence in means, covariance structure, skewness, and (in the extended Fr+ variant proposed in this thesis) kurtosis.

### 2.3.3 Exploration-Exploitation in MIGA

The paper parameterises the operator outputs as:
- **c:** elite individuals carried forward unchanged
- **c1 × c3:** mutated copies of elite individuals (c1 elites, each producing c3 variants)
- **c2:** crossover offspring from pairs of elites
- **l − c − c1·c3 − 2·(c2−1):** random individuals from generators (diversity injection)

Under the paper's default settings (l=200, c=3, c1=3, c2=2, c3=5), the diversity injection fills 180/200 = 90% of the population. The mutation offspring (c1·c3 = 15) constitute only 7.5%. This unusual balance means that MIGA is closer to a repeated random restart with elite preservation than a conventional GA. Chapter 3 discusses the implications for the adaptive mutation schedule (C3).

---

## 2.4 The MIGA Paper

Figueroa-García, Neruda, and Hernandez-Pérez (2023) proposed MIGA in "A genetic algorithm for multivariate missing data imputation," published in *Information Sciences* (619, 947–967, doi:10.1016/j.ins.2022.11.037).

### 2.4.1 Problem Formulation

Given a dataset X with n rows and p features, MIGA partitions it into X_A (the n_A rows with no missing values) and X_C (the n_C = n − n_A rows with at least one missing value). The goal is to find imputed values for the missing cells of X_C such that the statistical properties of the completed X_C match those of X_A.

### 2.4.2 The Fitness Function F_r

The fitness function F_r quantifies the distributional distance between X_A and X_C using three terms (Definitions 1–5, Equations 5–10):

$$F_r = D_r(\tilde{x}_A, \tilde{x}_C) + D_r(\tilde{S}, I) + D_r(b_A, b_C)$$

where $D_r$ denotes the Minkowski distance of order r (with r = ∞ for most benchmark datasets, reducing $D_r$ to the Chebyshev / L∞ norm), and:

- $\tilde{x} = \bar{x} / S_p$ is the standardised mean vector, with $S_p$ the pooled standard deviation
- $\tilde{S} = S_A^{-1/2} S_C S_A^{-1/2}$ is the relative covariance matrix, equal to the identity when $S_C = S_A$
- $b$ is the bias-corrected skewness vector

The three terms are dimensionless and additive. $F_r = 0$ if and only if X_C has the same mean, covariance, and skewness as X_A.

### 2.4.3 Per-Variable Generators

Each variable j has a random generator $R_j$ used to sample candidate values. The paper specifies that generators are "obtained from samples" of the observed values of variable j, which this thesis interprets as bootstrap resampling from the empirical distribution (§3.2.5). For integer-valued or discrete variables, this ensures that imputed values are always valid integers; for continuous variables, it approximates the true marginal distribution.

### 2.4.4 Algorithm

MIGA runs Q independent GA runs (to reduce the risk of premature convergence) and returns the best individual across all Q runs. Each run operates on the same fixed missing pattern. The paper reports results on seven benchmark datasets (Iris, Wine, Glass, Haberman, Wholesale, Cardiotocography, Adult) under MAR at 30–60% missing rates, comparing against CMIM and ANNI as baselines.

### 2.4.5 Gaps in the Original Paper

Three gaps motivate this thesis:

1. **No public implementation.** The paper contains no source code, and implementation details are underspecified at several points (generator construction, eigenvalue handling, stopping criterion for the convergence check).

2. **Weak baseline comparison.** CMIM (2005) and ANNI are significantly weaker than modern methods. No comparison against KNN or MICE is provided.

3. **No MNAR evaluation.** All experiments assume MAR. The behaviour of MIGA's fitness function under non-ignorable missingness is unknown.

---

## 2.5 Related Work

### 2.5.1 Distributional Imputation

The observation that RMSE-optimal imputation suppresses variance has motivated several lines of work. Siddique and Belin (2008) proposed predictive mean matching as a way to preserve distributional properties by imputing observed values rather than model predictions. Van Buuren (2018, §3.4) discusses "proper imputation" — drawing from the full posterior predictive distribution rather than its mean — as a way to avoid variance suppression. MIGA can be understood as a non-parametric alternative to proper imputation: rather than drawing from a parametric posterior, it uses a GA to find imputations that match distributional moments directly.

Muzellec et al. (2020) proposed optimal transport-based imputation, which finds imputed values that minimise the Sinkhorn divergence between the completed distribution and the observed distribution — directly analogous to MIGA's F_r objective, but using a differentiable surrogate. They note that distributional methods will generically achieve higher RMSE than regression-based methods, which is consistent with the findings of this thesis.

### 2.5.2 Neural Network Imputation

More recent work has applied deep generative models to imputation. GAIN (Yoon et al., 2018) uses a Generative Adversarial Network to generate imputations, with a discriminator that learns to distinguish imputed from observed values — conceptually similar to MIGA's distributional objective but in a learned feature space. GRAPE (You et al., 2020) formulates imputation as a graph completion problem, using message passing over a bipartite graph of observations and features. These methods achieve strong RMSE performance on large datasets but require significant compute and are not evaluated in the original MIGA paper. A comparison against GAIN or GRAPE is identified as future work (§5.4).

### 2.5.3 Evaluation of Imputation Methods

Oberman and Vink (2024) survey evaluation practices for imputation and find no universal standard for RMSE normalisation, making cross-paper comparisons unreliable. This thesis uses range-normalised RMSE (NRMSE_j = RMSE_j / (max_j − min_j)) to match the paper's reported baseline values. Ipsen et al. (2022) show that evaluating imputation quality under MNAR using RMSE on the missing cells is itself biased, because the missing cells are a non-representative sample of the marginal distribution — a point directly relevant to Chapter 4's MNAR analysis.

---

*[Chapter 2 draft complete. Citations to be formatted to institution style during final writing.]*
