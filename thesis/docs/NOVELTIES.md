# MIGA — Novelty Extensions Research

Comprehensive catalogue of extensions beyond the base paper (Figueroa-García et al., Information Sciences 619, 2023).
Each entry includes what changes in MIGA, motivation, and key references.

---

## Category 1 — Covariance Estimation Improvements

### N1.1 Linear Ledoit-Wolf Shrinkage ⭐ PLANNED
**What changes in MIGA:** Replace the raw sample covariance S_A (which becomes singular when n_A < m) with the Ledoit-Wolf convex combination S_LW = (1−α)S + αμI, where α is computed analytically. The d_cov fitness term is structurally unchanged but always numerically valid.

**Why:** Directly fixes the Wine rank-deficiency limitation. Adds no hyperparameters; α is data-driven. Available in `sklearn.covariance.LedoitWolf`.

**References:**
- Ledoit & Wolf, "A well-conditioned estimator for large-dimensional covariance matrices," *Journal of Multivariate Analysis* 88(2), 365–411, 2004. DOI: 10.1016/S0047-259X(03)00096-4

---

### N1.2 Oracle Approximating Shrinkage (OAS)
**What changes:** Swap LW for OAS, which minimises a tighter MSE bound using Rao-Blackwellisation.

**Why:** Consistently outperforms LW in the n ≪ m regime (exactly what Wine/Cardio encounter at high missing rates). Available as `sklearn.covariance.OAS`.

**References:**
- Chen, Wiesel, Eldar & Hero, "Shrinkage algorithms for MMSE covariance estimation," *IEEE Trans. Signal Processing* 58(10), 5016–5029, 2010. DOI: 10.1109/TSP.2010.2053029

---

### N1.3 Analytical Nonlinear Shrinkage
**What changes:** Apply Ledoit-Wolf 2020 nonlinear shrinkage — eigenvalue-specific shrinkage rather than a single global α. Provides the tightest known covariance estimator under random matrix theory asymptotics.

**Why:** Provably optimal in the large-p, large-n regime; ~1000× faster than earlier numerical methods and handles m up to 10,000. More principled than a fixed eigenvalue floor.

**References:**
- Ledoit & Wolf, "Analytical nonlinear shrinkage of large-dimensional covariance matrices," *Annals of Statistics* 48(5), 2020. DOI: 10.1214/19-AOS1921

---

### N1.4 Graphical Lasso / Sparse Precision Matrix
**What changes:** Estimate a sparse precision matrix (Θ = S^{-1}) via GLASSO and use it directly in d_cov instead of computing S_A^{-½} S_C S_A^{-½}. This models conditional independence structure among features.

**Why:** Prevents overfitting the covariance term to noisy sample correlations in high-dimensional settings. `sklearn.covariance.GraphicalLassoCV` handles the regularisation path automatically.

**References:**
- Friedman, Hastie & Tibshirani, "Sparse inverse covariance estimation with the graphical lasso," *Biostatistics* 9(3), 432–441, 2008. DOI: 10.1093/biostatistics/kxm045
- Städler & Bühlmann, "Missing values: sparse inverse covariance estimation and an extension to sparse regression," *Statistics and Computing* 22(1), 219–235, 2012.

---

### N1.5 Latent Factor Model Covariance
**What changes:** Estimate S = ΛΛᵀ + Ψ (low-rank + diagonal) from incomplete data. Use the factor structure inside d_cov instead of the raw covariance.

**Why:** For datasets with correlated features, factor models give parsimony and naturally handle rank-deficiency by design.

**References:**
- Xiong & Pelger, "Large dimensional latent factor modeling with missing observations," *Journal of Econometrics* 233(1), 113–131, 2023. DOI: 10.1016/j.jeconom.2022.01.005

---

## Category 2 — Alternative Fitness Functions

### N2.1 Wasserstein / Optimal Transport Fitness
**What changes:** Replace the three-component F_r with a Sinkhorn divergence (regularised Wasserstein distance) between the empirical distributions of X_A and X_C. Captures full distributional geometry.

**Why:** OT fitness is differentiable, handles multimodal/heavy-tailed distributions, and does not assume independence of features. Demonstrated for imputation in Muzellec et al. (2020).

**References:**
- Muzellec, Josse, Boyer & Cuturi, "Missing data imputation using optimal transport," *ICML 2020*, PMLR 119, 7130–7140. arXiv: 2002.03860

---

### N2.2 Maximum Mean Discrepancy (MMD) Fitness
**What changes:** Replace F_r with kernel MMD² between X_A and X_C in a chosen RKHS. The GA evaluates candidate completions by MMD score.

**Why:** Distribution-free, kernelisable to any data type, robust to outliers, and has proven finite-sample bounds. Recent work shows MMD estimators are robust even under MNAR.

**References:**
- Gretton et al., "A kernel two-sample test," *JMLR* 13, 723–773, 2012.
- Szabó & Sriperumbudur, "Characteristics and universal consistency of MMD-based statistics," *arXiv 2405.14051*, 2024.

---

### N2.3 Gaussian Copula Likelihood Fitness
**What changes:** Fit a Gaussian copula to X_A. Fitness = negative copula log-likelihood of X_C under the fitted copula. Replaces d_cov and d_skew with a single principled term.

**Why:** Copulas decouple marginal from joint structure; naturally handles mixed continuous/ordinal data without label encoding. Extends MIGA to categorical features.

**References:**
- Zhao & Udell, "Missing value imputation for mixed data via Gaussian copula," *KDD 2020*. arXiv: 1910.12845

---

### N2.4 Robust Fitness with Median / IQR
**What changes:** Replace standardised means with medians, covariance with median absolute deviations, skewness with quantile-based skewness. All Minkowski distances apply to these robust statistics.

**Why:** Directly addresses the undocumented MAD formula (Limitation §7.3) with a clearly defined robust alternative. Makes fitness insensitive to outliers in X_A.

**References:**
- Rousseeuw & Croux, "Alternatives to the median absolute deviation," *JASA* 88(424), 1273–1283, 1993.
- Brys, Hubert & Struyf, "A robust measure of skewness," *Journal of Computational & Graphical Statistics* 13(4), 996–1017, 2004.

---

### N2.5 Information-Theoretic (KL / Jensen-Shannon) Fitness
**What changes:** Fit KDEs to X_A and X_C distributions and minimise their KL or Jensen-Shannon divergence.

**Why:** Gives the fitness a maximum-likelihood interpretation. KL divergence directly measures how likely X_C is to be a draw from X_A's distribution.

**References:**
- Pérez-Cruz, "Kullback-Leibler divergence estimation of continuous distributions," *IEEE ISIT 2008*.

---

## Category 3 — Generator Improvements

### N3.1 Kernel Density Estimation (KDE) Generators
**What changes:** Fit a KDE per feature on available data; sample from the KDE instead of bootstrap resampling. Smooths the generator distribution and allows values between observed data points.

**Why:** Bootstrap resampling can only produce observed values verbatim. KDE generators produce novel values on the data manifold, which is important when n is small.

**References:**
- Silverman, *Density Estimation for Statistics and Data Analysis*, Chapman & Hall, 1986.
- `sklearn.neighbors.KernelDensity` — bandwidth via Scott's rule or cross-validation.

---

### N3.2 Gaussian Copula Generators (Joint Sampling)
**What changes:** Fit a Gaussian copula to the available data; generate candidates by sampling from the copula's joint distribution. Replaces independent marginal generators with a multivariate generator.

**Why:** Directly fixes Limitation 5 (feature independence assumption). Candidates are plausible multivariate completions from the start, accelerating GA convergence.

**References:**
- Zhao & Udell, KDD 2020 (cited above).
- Li, Xue & Ding, "A general approach for copula-based imputation," *Communications in Statistics — Simulation and Computation*, 2022. DOI: 10.1080/03610918.2022.2025839

---

### N3.3 Normalising Flow Generators
**What changes:** Train a normalising flow (e.g., RealNVP, Glow) on the complete rows X_A. Sample latent codes and push through the flow to generate candidates that lie on the data manifold.

**Why:** Flow generators provide exact density evaluation; the GA can search in the compact latent space rather than the high-dimensional feature space.

**References:**
- Richardson, Wu, Lin, Xu & Bernal, "MCFlow: Monte Carlo flow models for data imputation," *CVPR 2020*. arXiv: 2003.12628
- "EMFlow: Data imputation in latent space via EM and deep flow models," *OpenReview 2022*.

---

### N3.4 VAE / MIWAE Generators
**What changes:** Train a VAE (MIWAE variant for incomplete data) on X; use the decoder as generator. The GA searches in the latent space; crossover and mutation occur in latent coordinates.

**Why:** Latent space is lower-dimensional and semantically structured. Crossover in latent space produces coherent offspring. Combines MIGA's fitness with deep generative structure.

**References:**
- Mattei & Frellsen, "MIWAE: Deep generative modelling and imputation of incomplete data sets," *ICML 2019*, PMLR 97, 4413–4423. arXiv: 1812.02633
- Ma, Shi & Jiao, "MIVAE: Multiple imputation based on variational autoencoder," *Engineering Applications of AI*, 2023. DOI: 10.1016/j.engappai.2023.106689

---

### N3.5 Gaussian Process Generators
**What changes:** Per feature, fit a GP regression with observed features as inputs. Sample from the GP posterior predictive for each missing feature, conditioning on observed row values.

**Why:** GP generators provide feature-conditional (not marginal) samples and natural uncertainty estimates that can guide GA exploration.

**References:**
- Agrawal, Daumé & Basu, "Gaussian processes for missing value imputation," *Knowledge-Based Systems* 270, 110568, 2023. arXiv: 2204.04648

---

## Category 4 — Missing Mechanism Extensions

### N4.1 MNAR via Selection Model ⭐ PLANNED
**What changes:** Augment MIGA with a model for the missingness mechanism R = f(X_obs, X_miss, θ). Weight the distributional distances in F_r by estimated inverse probability of observation (IPW), correcting for non-ignorable missingness.

**Why:** Directly addresses Limitation 3. Under MNAR, ignoring the missing mechanism biases all statistical distances in the fitness function.

**References:**
- Yoon, Zame & van der Schaar, "Identifiable generative models for missing not at random data imputation," *NeurIPS 2021*. arXiv: 2110.14708
- Ipsen, Mattei & Frellsen, "not-MIWAE: Deep generative modelling with MNAR data," *ICLR 2021*. arXiv: 2006.12871
- Chen et al., "Deep generative imputation model for MNAR data," *CIKM 2023*. DOI: 10.1145/3583780.3614835

---

### N4.2 MAR + MNAR Sensitivity Analysis
**What changes:** Run MIGA under MAR, then perturb the assumption with sensitivity parameter δ (correlation between missingness and unobserved value). Report imputation uncertainty as a function of δ.

**Why:** Provides honest uncertainty bounds when the MAR assumption may not hold. Converts MIGA into a sensitivity analysis tool for practitioners.

**References:**
- Resseguier, Giorgi & Paoletti, "A multiple imputation-based sensitivity analysis for data subject to MNAR," *Statistics in Medicine* 2011.
- Cro et al., "Sensitivity analysis for clinical trials with missing continuous outcome data using controlled multiple imputation," *Statistics in Medicine* 2020. DOI: 10.1002/sim.8569

---

### N4.3 Structured / Block / Monotone Missingness
**What changes:** Specialise chromosome encoding and generators for block missingness (entire columns missing for subgroups) and monotone missingness (common in longitudinal data) using sequential conditional generation.

**Why:** Many real-world datasets have structured missingness, not arbitrary cell-level MAR. Exploiting structure reduces chromosome dimensionality and speeds convergence.

**References:**
- Xie et al., "Blockwise PCA for monotone missing data imputation," arXiv: 2305.06042, 2023.
- Bhattacharya, Nandram & Bhattacharya, "Monotone missing data: a blessing and a curse," arXiv: 2411.03848, 2024.
- Zhan et al., "Multiple imputation via GAN for high-dimensional blockwise missing value problems," *PMC* 2022.

---

## Category 5 — GA / Optimisation Improvements

### N5.1 CMA-ES as Drop-in Optimiser ⭐ PLANNED
**What changes:** Replace the standard GA (selection, crossover, mutation) with CMA-ES, which learns the full covariance structure of the search space. MIGA's F_r is used unchanged as the black-box objective.

**Why:** CMA-ES is state-of-the-art for continuous black-box optimisation; adapts step size and search direction automatically. Available via the `cma` Python package.

**References:**
- Hansen, "The CMA evolution strategy: A tutorial," arXiv: 1604.00772, 2016.
- Hansen & Ostermeier, "Completely derandomised self-adaptation in evolution strategies," *Evolutionary Computation* 9(2), 159–195, 2001. DOI: 10.1162/106365601750190398

---

### N5.2 Differential Evolution (DE)
**What changes:** Replace GA operators with DE/rand/1/bin scheme with self-adaptive F and CR parameters (jDE variant). Chromosome and fitness function unchanged.

**Why:** DE consistently outperforms canonical GA on continuous real-parameter problems. Self-adaptive parameters eliminate manual tuning burden.

**References:**
- Brest, Greiner, Boskovic et al., "Self-adapting control parameters in differential evolution," *IEEE Trans. Evolutionary Computation* 10(6), 646–657, 2006.

---

### N5.3 Island Model GA (Parallel Populations) ⭐ PLANNED
**What changes:** Split population into K islands evolving independently with periodic migration. Islands run in parallel on multi-core machines, fixing Limitation 4 (sequential Q-runs).

**Why:** Maintains diversity, avoids premature convergence, and gives near-K× speedup. A simple version uses Python `multiprocessing.Pool`.

**References:**
- Cantú-Paz, "A survey of parallel genetic algorithms," *Calculateurs Parallèles* 10(2), 141–171, 1998.
- Gong et al., "On the behavior of parallel island models in genetic algorithms," *Applied Soft Computing* 147, 2023. DOI: 10.1016/j.asoc.2023.110817

---

### N5.4 Estimation of Distribution Algorithm (EDA)
**What changes:** Replace crossover/mutation with a probabilistic model (Gaussian mixture or copula) fitted to the best candidates each generation. New candidates are sampled from this model.

**Why:** EDA explicitly captures inter-feature dependencies, conceptually tightly aligned with MIGA's distributional-matching objective. Avoids premature convergence via model-based search.

**References:**
- Larrañaga & Lozano (Eds.), *Estimation of Distribution Algorithms*, Kluwer, 2002.
- Pan, Peng & Shi, "An efficient mixture sampling model for Gaussian EDA," *Information Sciences* 609, 1–18, 2022. DOI: 10.1016/j.ins.2022.07.059

---

### N5.5 Surrogate-Assisted Fitness Evaluation
**What changes:** Maintain a Gaussian Process surrogate that approximates F_r. Evaluate the true fitness only for the most promising candidates; use the surrogate for the rest.

**Why:** Directly addresses Limitation 6 (large dataset slowness). Reduces full covariance computations per generation by 80–90%.

**References:**
- Jin, "Surrogate-assisted evolutionary computation: Recent advances and future challenges," *Swarm and Evolutionary Computation* 1(2), 61–70, 2011.
- Wang, Jin & Doherty, "Committee-based active learning for surrogate-assisted particle swarm optimisation," *IEEE Trans. Cybernetics* 2017.

---

## Category 6 — Multiple Imputation Framework

### N6.1 Rubin-Pooled Multiple Imputation ⭐ PLANNED
**What changes:** Treat each of the Q MIGA runs' best chromosomes as one imputed dataset. Apply Rubin's (1987) combining rules to pool point estimates and compute between-imputation variance for any downstream statistic.

**Why:** Converts MIGA from single- to proper multiple imputation, providing calibrated confidence intervals at near-zero additional computational cost. Statistically major upgrade.

**References:**
- Rubin, *Multiple Imputation for Nonresponse in Surveys*, Wiley, 1987.
- Schafer & Graham, "Missing data: Our view of the state of the art," *Psychological Methods* 7(2), 147–177, 2002. DOI: 10.1037/1082-989X.7.2.147

---

### N6.2 Bootstrap Rubin Variance
**What changes:** Bootstrap the entire MIGA optimisation B times (resample rows, re-run) to estimate imputation uncertainty, consistent even when the imputation model is misspecified.

**Why:** Bootstrap variance is valid under uncongenial imputation models — the typical case for MIGA since F_r is not a proper likelihood.

**References:**
- Schomaker & Heumann, "Bootstrap inference when using multiple imputation," *Statistics in Medicine* 37(14), 2018. DOI: 10.1002/sim.7654

---

## Category 7 — Scalability

### N7.1 Mini-Batch Fitness Evaluation ⭐ PLANNED
**What changes:** Instead of computing distances over all n_A complete rows, randomly subsample a mini-batch of size B << n_A each generation to estimate F_r. Estimator is unbiased.

**Why:** Directly addresses Limitation 6. With B=500 instead of n=48K, each fitness evaluation is ~96× cheaper, making Adult-scale data practical.

**References:**
- Bottou, Curtis & Nocedal, "Optimization methods for large-scale machine learning," *SIAM Review* 60(2), 223–311, 2018.

---

### N7.2 Distributed Spark-Based Evaluation
**What changes:** Distribute fitness evaluation of each chromosome across a Spark cluster. Population is broadcast; distance computations are mapped across workers.

**Why:** Parallelises across the population axis (chromosomes evaluated simultaneously), complementing island model parallelism.

**References:**
- Fernández-Vara & Luque, "Parallel and distributed GA on Apache Spark," *Applied Soft Computing* 101, 2021. DOI: 10.1016/j.asoc.2020.107091

---

### N7.3 Sketched / Approximate Covariance
**What changes:** Use random projection sketching to maintain an O(m·k) sketch (k << n) of the sample covariance, updated incrementally as new candidates are evaluated.

**Why:** Reduces memory and time for recomputing S_C per candidate from O(n·m²) to O(n·m·k), critical for large m.

**References:**
- Mahoney, "Randomised algorithms for matrices and data," *Foundations and Trends in Machine Learning* 3(2), 123–224, 2011.

---

## Category 8 — Categorical / Mixed Data

### N8.1 Gower Distance Fitness for Mixed Data
**What changes:** Replace Minkowski d_means with Gower distance, handling continuous, ordinal, and nominal variables in a single normalised metric.

**Why:** Allows MIGA to handle datasets with categorical columns without external preprocessing. Distance is bounded and interpretable.

**References:**
- Gower, "A general coefficient of similarity and some of its properties," *Biometrics* 27(4), 857–871, 1971.
- Mix & Alvarez-Esteban, "Distances with mixed-type variables, some modified Gower's coefficients," arXiv: 2101.02481, 2021.

---

### N8.2 Probabilistic Mixed-Type Fitness (Copula)
**What changes:** Model complete data as a Gaussian copula with mixed marginals (Gaussian / ordinal probit / multinomial). Fitness = negative copula log-likelihood of the candidate completion.

**Why:** Unified, principled fitness for any feature type mix. Replaces three ad-hoc distance terms with a single well-understood log-likelihood.

**References:**
- Zhao & Udell, KDD 2020 (cited above).
- Christoffersen & Veredas, "Probabilistic imputation for mixed categorical and ordered data," *NeurIPS 2022*.

---

## Category 9 — Ensemble Imputation

### N9.1 MIGA + SuperMICE Stacked Ensemble
**What changes:** Use MIGA imputation as a warm start for MICE, or stack MIGA and MICE imputations via a meta-learner that selects per-feature or per-cell the best imputation.

**Why:** Ensemble methods reduce variance. Individual imputers fail in complementary ways; stacking provides robust overall performance.

**References:**
- Laqueur, Shev & Kagawa, "SuperMICE: An ensemble ML approach to MICE," *American Journal of Epidemiology* 191(3), 516–525, 2022. DOI: 10.1093/aje/kwab271
- Carpenito & Manjourides, "MISL: Multiple imputation by super learning," *Statistical Methods in Medical Research* 31(8), 2022.

---

### N9.2 Multiple Imputation Ensembles (MIE)
**What changes:** Generate Q MIGA + Q MICE + Q MissForest imputations. Combine 3Q imputed datasets via bagging or stacking.

**Why:** Demonstrated empirically to reduce imputation MSE relative to any single method; MIGA's distributional matching provides complementary strengths to regression-based methods.

**References:**
- Gaffert, Meinfelder & Ulber, "Multiple imputation ensembles (MIE)," *SN Computer Science* 1, 2020. DOI: 10.1007/s42979-020-00131-0
- Stekhoven & Bühlmann, "MissForest — non-parametric missing value imputation," *Bioinformatics* 28(1), 112–118, 2012. DOI: 10.1093/bioinformatics/btr597

---

## Category 10 — Theoretical Contributions

### N10.1 EM Algorithm Interpretation of MIGA
**What changes:** Show analytically that MIGA's loop is a stochastic EM: E-step = chromosome generation (filling missing values); M-step = selection / elitism (maximising fitness). Derive convergence rate.

**Why:** Provides a formal convergence guarantee (currently absent from the paper). Motivates principled stopping criteria based on likelihood increase rather than fixed generation counts.

**References:**
- Dempster, Laird & Rubin, "Maximum likelihood from incomplete data via the EM algorithm," *JRSS-B* 39(1), 1–38, 1977.
- Liang et al., "An imputation-regularised optimisation algorithm for high-dimensional missing data," *JRSS-B* 80(5), 899–926, 2018. DOI: 10.1111/rssb.12279

---

### N10.2 Sample Complexity Bounds for the Fitness Estimator
**What changes:** Derive concentration inequalities: how large must n_A be for F_r to approximate the population fitness within ε with probability 1−δ?

**Why:** Explains the Wine failure quantitatively. Guides practitioners on the minimum dataset size needed for MIGA to work reliably.

**References:**
- Vershynin, *High-Dimensional Probability*, Cambridge University Press, 2018.
- Ledoit & Wolf, *Annals of Statistics* 2020 (spectral concentration results).
- Szabo & Sriperumbudur, "Characteristics and universal consistency of MMD-based statistics," arXiv: 2405.14051, 2024.

---

### N10.3 Schema Theorem Applied to Imputation Chromosomes
**What changes:** Formalise MIGA's gene structure and apply Holland's schema theorem to predict which sub-structures (patterns of imputed values) are building blocks exploited early in the GA.

**Why:** Provides theoretical basis for choosing crossover operators (uniform vs. one-point) specifically for imputation chromosomes. Explains why MIGA converges fast on some features and stagnates on others.

**References:**
- Holland, *Adaptation in Natural and Artificial Systems*, MIT Press, 1975.
- Goldberg, *Genetic Algorithms in Search, Optimisation and Machine Learning*, Addison-Wesley, 1989.

---

### N10.4 Identifiability Analysis Under Different Missing Mechanisms
**What changes:** Characterise conditions under which MIGA's fitness minimum uniquely identifies the true complete-data distribution under MCAR, MAR, and MNAR.

**Why:** Establishes when MIGA can and cannot be trusted. A rigorous theoretical contribution that complements the empirical benchmarks.

**References:**
- Yoon et al., NeurIPS 2021 (cited above).
- Miao, Tchetgen & Tchetgen, "Identification, doubly robust estimation under MNAR," *Annals of Statistics* 44(4), 2016.

---

---

# Novelty Rankings

Scored on five dimensions (each 1–5):
- **Ease**: How easy to implement (5 = very easy, ≤ 1 week)
- **Impact**: Expected RMSE improvement / thesis contribution (5 = high)
- **Novelty**: How novel vs. existing literature (5 = highly original)
- **Complexity**: Implementation complexity (5 = very complex; inverse of ease in terms of effort)
- **Thesis Value**: Contribution to thesis storyline (5 = essential)

| ID | Novelty | Ease | Impact | Novelty | Complexity | Thesis Value | Priority |
|----|---------|------|--------|---------|------------|--------------|----------|
| N1.1 | Ledoit-Wolf Shrinkage | 5 | 5 | 3 | 1 | 5 | **1 — Start here** |
| N6.1 | Rubin Multiple Imputation | 5 | 4 | 3 | 1 | 5 | **2** |
| N7.1 | Mini-Batch Fitness | 5 | 4 | 2 | 1 | 4 | **3** |
| N5.3 | Island Model / Parallel Runs | 4 | 3 | 2 | 2 | 3 | **4** |
| N3.1 | KDE Generators | 4 | 3 | 2 | 2 | 3 | **5** |
| N2.4 | Robust Fitness (Median/IQR) | 4 | 3 | 3 | 2 | 4 | **6** |
| N1.2 | OAS Covariance | 5 | 4 | 3 | 1 | 3 | **7** |
| N5.1 | CMA-ES Optimiser | 3 | 4 | 3 | 2 | 4 | **8** |
| N5.2 | Differential Evolution | 3 | 3 | 3 | 2 | 3 | 9 |
| N3.2 | Copula Generators | 3 | 4 | 4 | 3 | 4 | 10 |
| N4.2 | MAR+MNAR Sensitivity | 3 | 3 | 4 | 3 | 4 | 11 |
| N9.2 | MissForest Ensemble (MIE) | 3 | 3 | 2 | 2 | 3 | 12 |
| N2.3 | Copula Likelihood Fitness | 2 | 4 | 4 | 3 | 4 | 13 |
| N1.4 | Graphical Lasso (GLASSO) | 3 | 3 | 4 | 3 | 3 | 14 |
| N10.1 | EM Interpretation (Theory) | 3 | 3 | 4 | 3 | 5 | 15 |
| N10.2 | Sample Complexity Bounds | 2 | 3 | 4 | 4 | 5 | 16 |
| N5.4 | EDA | 2 | 3 | 4 | 3 | 3 | 17 |
| N4.1 | MNAR via Selection Model | 1 | 5 | 5 | 5 | 5 | 18 |
| N2.1 | Wasserstein Fitness | 2 | 4 | 5 | 4 | 4 | 19 |
| N2.2 | MMD Fitness | 2 | 4 | 5 | 4 | 4 | 20 |
| N3.4 | VAE/MIWAE Generators | 1 | 5 | 5 | 5 | 4 | 21 |
| N3.3 | Normalising Flow Generators | 1 | 4 | 5 | 5 | 3 | 22 |
| N8.2 | Copula Mixed-Type Fitness | 1 | 4 | 4 | 4 | 3 | 23 |
| N5.5 | Surrogate-Assisted Fitness | 2 | 3 | 3 | 4 | 3 | 24 |
| N4.3 | Block / Monotone Missingness | 2 | 2 | 3 | 4 | 2 | 25 |

---

## Recommended Thesis Implementation Roadmap

### Phase 1 — Quick Wins (1–2 weeks total)
1. **N1.1 Ledoit-Wolf** — fix Wine, theory grounded, 1 day
2. **N6.1 Rubin MI Framework** — proper uncertainty, zero extra compute, 1 day
3. **N7.1 Mini-Batch Fitness** — makes large datasets tractable, 2 days
4. **N5.3 Island Model / Parallel Runs** — speedup, 2 days

### Phase 2 — Algorithmic Novelties (2–3 weeks)
5. **N3.1 KDE Generators** — better than bootstrap, sklearn, 2 days
6. **N2.4 Robust Fitness** — fixes MAD discrepancy, documented justification, 3 days
7. **N5.1 CMA-ES** — state-of-the-art optimiser, pycma package, 3 days

### Phase 3 — Major Contributions (4–8 weeks)
8. **N3.2 Copula Generators** — fixes independence assumption, 1 week
9. **N4.2 MNAR Sensitivity** — practical contribution, 1 week
10. **N10.1 EM Theory** — convergence guarantee, math-only, 2 weeks

### Future Work (thesis discussion only, not implemented)
- N4.1 MNAR selection model (NeurIPS-level contribution)
- N2.1 Wasserstein fitness (computationally demanding)
- N3.4 VAE/MIWAE generators (requires deep learning infrastructure)
- N10.4 Identifiability analysis
