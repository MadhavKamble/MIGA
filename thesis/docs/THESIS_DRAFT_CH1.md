# Chapter 1 — Introduction (Draft)

---

## 1.1 Background and Motivation

Missing data is a pervasive challenge in real-world machine learning and statistical analysis. Medical records omit test results for patients who did not present with relevant symptoms; financial surveys suffer from non-response bias; sensor networks produce gaps due to transmission failures or device faults. The proportion of missing values in real-world datasets commonly ranges from 5% to 40% [CITE: van Buuren 2018], and naive strategies for handling missingness can severely compromise the integrity of downstream analyses.

Three broad strategies are typically employed. **Complete case analysis** discards any observation with at least one missing value, sacrificing statistical power and potentially introducing selection bias. **Single imputation** (e.g., replacing missing values with column means) fills the gaps with a single plausible value, but distorts the variance structure of the data and treats imputed values as though they were observed. **Multiple imputation** generates several plausible complete datasets and combines inferences across them, appropriately accounting for the uncertainty introduced by missingness.

Among multiple imputation methods, MICE (Multiple Imputation by Chained Equations; van Buuren 2011) has become the standard. MICE iteratively regresses each incomplete variable on all other variables, producing imputations that are optimal in the sense of minimising expected squared error. However, this optimality comes at a cost: MICE imputes conditional means, which suppresses marginal variance. Van Buuren (2018, §2.6) explicitly notes that "the minimum mean squared error is achieved by regression imputation, which suppresses variance and produces biased statistical inference." For any analysis that requires the correct marginal distribution of the imputed variables — distributional testing, bootstrap inference, synthetic data generation — MICE produces systematically distorted results.

This motivates a distributional approach to imputation: rather than minimising pointwise squared error, find imputed values whose *distribution* matches the observed distribution of complete cases. This is precisely the objective of MIGA.

## 1.2 The MIGA Algorithm

Figueroa-García, Neruda, and Hernandez-Pérez (2023) proposed MIGA (Missing data Imputation via Genetic Algorithm), published in *Information Sciences* (619, 947–967). MIGA frames missing data imputation as an optimisation problem: find the set of imputed values that minimises the statistical distance between the set of complete rows (X_A) and the set of rows with at least one missing value after imputation (X_C).

The fitness function (Fr) is a sum of three Minkowski distances measuring differences in mean vectors, covariance structure, and skewness between X_A and X_C:

$$F_r = D_r(\tilde{x}_A, \tilde{x}_C) + D_r(\tilde{S}, I) + D_r(b_A, b_C)$$

where $\tilde{x}$ denotes standardised means, $\tilde{S} = S_A^{-\frac{1}{2}} S_C S_A^{-\frac{1}{2}}$ is the relative covariance, and $b$ denotes skewness vectors. A genetic algorithm with $l$ individuals over $G$ generations across $Q$ independent runs is used to minimise Fr.

MIGA is notable for several reasons. It handles mixed variable types (continuous, discrete, binary) natively through variable-specific random generators. It imposes no parametric assumptions on the data distribution. And it explicitly targets distributional fidelity rather than pointwise accuracy, making it the correct tool when the downstream analysis requires correct distributional properties.

However, the paper provides no public implementation, and several implementation details are underspecified. No systematic comparison against modern baselines (KNN, MICE) is reported. The algorithm has not been evaluated under Missing Not At Random (MNAR) conditions. This thesis addresses all three gaps.

## 1.3 Research Objectives

This thesis pursues four primary objectives:

1. **Reproduce MIGA** from scratch, providing the first open-source implementation, and verify replication of the paper's reported results on the seven benchmark datasets.

2. **Extend MIGA** with principled improvements: Ledoit-Wolf shrinkage covariance (addressing rank-deficiency at high missingness rates) and an adaptive mutation schedule (improving convergence under compute-limited settings).

3. **Evaluate MIGA under MNAR**, providing the first systematic assessment of MIGA's distributional objective when the missing mechanism is not at random.

4. **Characterise the Fr vs RMSE trade-off**, quantifying MIGA's distributional advantage over MICE and identifying the conditions under which each method is preferable.

## 1.4 Contributions

The specific contributions of this thesis are:

**C1 — Open-source reimplementation.** A complete Python reimplementation of MIGA from the paper specification, with 14 Jupyter notebooks reproducing all paper experiments across seven benchmark datasets (Iris, Wine, Glass, Haberman, Wholesale, Cardiotocography, Adult).

**C2 — Ledoit-Wolf shrinkage covariance.** We identify that MIGA's fitness function requires a positive-definite covariance estimate for X_A. At high missingness rates, the sample covariance of X_A becomes rank-deficient (n_A < m). We replace the sample covariance with the Ledoit-Wolf shrinkage estimator, restoring mathematical consistency (d_cov(X_A, X_A) = 0) even when n_A < m.

**C3 — Adaptive mutation schedule.** We propose a linear decay schedule for the mutation offspring count c3, reducing from a high initial value (broad exploration) to a low final value (fine exploitation). Experiments show faster early convergence under compute-limited budgets (G ≤ 100 generations).

**C4 — First MNAR evaluation.** We extend MIGA's evaluation to three MNAR mechanisms (top-quantile, bottom-quantile, tails censoring) across three datasets. A striking finding: under top-quantile MNAR, MIGA achieves its lowest-ever Fr (0.81 — better than MAR) while simultaneously achieving its highest-ever RMSE (0.38). This is the clearest possible empirical proof that Fr minimisation and RMSE minimisation are structurally orthogonal objectives.

**C5 — Systematic Fr vs RMSE comparison with significance testing.** We compute both Fr and RMSE for all four methods (Mean, KNN, MICE, MIGA) on all benchmark datasets, validated across 10 independent seeds with Wilcoxon tests. MIGA achieves 2–1.97× lower Fr than MICE on Iris and Haberman (Wilcoxon p < 0.001); MICE achieves 1.33–2.08× lower RMSE than MIGA. A dimension-dependent effect emerges: on Glass (p=10 features), MICE achieves lower Fr than MIGA — iterative conditional regression incidentally captures the joint distribution when p is large, negating MIGA's advantage. MIGA preserves marginal variance (variance ratio ≈ 1.0 on multi-feature datasets); MICE suppresses it (variance ratio 0.5–0.9 across features).

## 1.5 Thesis Organisation

The remainder of this thesis is organised as follows. Chapter 2 reviews the background on missing data mechanisms, classical and modern imputation methods, and genetic algorithms. Chapter 3 describes the MIGA algorithm in detail, documents every implementation decision, and presents the novel extensions. Chapter 4 reports experimental results on replication, MNAR evaluation, and the baseline comparison. Chapter 5 discusses the central finding (Fr vs RMSE decoupling), limitations of MIGA under MNAR, and directions for future work.

---

*[Draft — citations to be completed during final writing. All experimental results from notebooks 02–14 in the project repository.]*
