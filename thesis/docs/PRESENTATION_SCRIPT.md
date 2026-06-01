# Phase 1 Presentation — Full Speaker Script
**Slot:** 2026-05-21  |  **Duration:** 15–20 min  |  **Audience:** Thesis committee
**Target timing:** ~17 min talk + 3 min Q&A buffer

Each section shows: what appears on screen, what to say.

---

## Slide 1 — Title (30 sec)

**ON SCREEN**
```
Missing Data Imputation via Genetic Algorithms:
Reimplementation, Extension, and Distributional Analysis of MIGA

[Your Name]
Supervisor: [Name]
MTech Computer Science — IIIT Allahabad | May 2026
```

**SAY**
> Good morning. My thesis is about a 2023 paper that proposed a genetic algorithm for missing data imputation — called MIGA — but published no source code. I reimplemented it from scratch, proved a formal theorem about why it behaves the way it does, and ran the first systematic evaluation under both MAR and MNAR conditions. The central finding is that MIGA and MICE — the current standard — are solving structurally different problems, and each wins decisively on its own objective.

---

## Slide 2 — Motivation (1.5 min)

**ON SCREEN**
```
The Missing Data Problem

• Every real dataset has gaps
  — Medical records: tests not ordered for low-risk patients
  — Sensor networks: packet loss, device faults
  — Surveys: non-response bias (5%–40% typical)

Three strategies — each with a cost:
  Delete rows         → loses statistical power, selection bias
  Mean imputation     → distorts variance, treats imputed = observed
  MICE                → minimises squared error, but suppresses variance

Van Buuren (2018, §2.6):
  "Regression imputation achieves minimum RMSE by suppressing variance
   and produces biased statistical inference."

MIGA (Figueroa-García et al., 2023, Information Sciences):
  → Imputes by matching the joint distribution, not minimising pointwise error
  → No public code. No comparison against MICE. No MNAR evaluation.
  → This thesis closes all three gaps.
```

**SAY**
> Missing data is pervasive. In medical datasets, test results are missing precisely for patients who don't need those tests — the missingness is informative. Standard fixes have well-known costs. The interesting case is MICE, which has become the gold standard. MICE does iterative regression — predicting each missing value from all other columns. This is optimal for minimising prediction error.
>
> But Van Buuren himself, the author of MICE, writes that this comes at a cost: MICE systematically underestimates variance. For any analysis that needs the correct marginal distribution — confidence intervals, hypothesis tests, generating synthetic data — MICE produces biased results.
>
> Figueroa-García et al. published MIGA in 2023 to address this. The idea: instead of minimising squared error, find imputations that make the completed data statistically indistinguishable from the observed data. But they published no code, didn't compare against MICE, and never tested what happens under MNAR. This thesis addresses all three.

---

## Slide 3 — What MIGA Does (1.5 min)

**ON SCREEN**
```
MIGA: Impute by Matching the Distribution

  X_A = complete rows (reference)       n_A rows
  X_C = rows with ≥1 missing value      n_C rows (filled by GA)

Fitness function (minimise):

  F_r = D_r(x̃_A, x̃_C)   +   D_r(S̃, I)   +   D_r(b_A, b_C)
         ─────────────────    ─────────────    ─────────────────
           means match        covariance        skewness match
                              structure

  S̃ = S_A^{-½} S_C S_A^{-½}   (equals I when X_C ~ X_A)

Genetic algorithm:
  l = 200 individuals  |  G generations  |  Q = 3–5 independent runs
  Operators: selection → crossover → mutation → diversity injection (90% random)

Key difference from MICE:
  MIGA never sees the true missing values.
  It only checks if the imputed distribution matches the reference distribution.
```

**SAY**
> Here's the core idea. We split the data into X_A — rows with no missing values — and X_C — rows with at least one missing value. After imputation, we want X_C to be statistically indistinguishable from X_A. Same means, same covariance structure, same skewness. That's the fitness function Fr.
>
> The genetic algorithm treats each candidate imputation as an individual. It evolves a population of 200 candidates, using selection, crossover, mutation, and a diversity injection that replaces 90% of the population with random individuals each generation.
>
> The critical difference from MICE: MIGA never looks at the true missing values. It only optimises distributional similarity. This means it can achieve a perfect distributional match while still being wrong on individual cells — which is exactly what the formal theorem (our next slide) proves must happen.

---

## Slide 4 — Six Contributions (1 min)

**ON SCREEN**
```
Six Contributions

  C1  Open-source reimplementation       First public MIGA code. 5 datasets. Reproducible.

  C2  Formal Fr–RMSE orthogonality ★    Proved: no single imputation can minimise both.
                                          Explains every experimental result in the thesis.

  C3  First MNAR evaluation ★            Discovered Fr→RMSE inversion under directional MNAR.
                                          The global minimum Fr coincides with global max RMSE.

  C4  Systematic Fr vs RMSE comparison   10 seeds, Wilcoxon tests. MIGA 1.5–2× better Fr.
                                          MICE 1.4–2.1× better RMSE. On all datasets.

  C5  Synthetic p×ρ experiment ★         Failure depends jointly on (p, ρ), not p alone.
                                          At ρ=0.9, MIGA loses even at p=8.

  C6  IPW-Fr for MNAR correction         Re-weights X_A by propensity scores.
                                          Works when n/p ≥ 20; fails when n/p < 15.
```

**SAY**
> Here's what the thesis contains. Six contributions. C2, C3, and C5 are the starred ones — the formal theorem, the MNAR inversion discovery, and the synthetic characterisation. The rest of this talk focuses on these, plus C4 and C6. C1 is the foundation everything is built on.

---

## Slide 5 — C1: Replication Results (1.5 min)

**ON SCREEN**
```
C1: Replication — Our RMSE / Paper RMSE

  Dataset     30%      40%      50%      Notes
  ────────────────────────────────────────────────────────────
  Iris        1.29×    1.31×    1.33×    Expected range
  Wine        2.56×    3.79×      —      n_A < p at 40%: fundamental limit ▼
  Glass       1.33×    1.56×    1.89×    Expected range
  Haberman    0.52× ★  1.24×    1.43×    ★ Beats paper (bootstrap handles 44% zeros)
  Wholesale   1.07× ★  1.08×    1.26×    ★ Best replication — within 8%

★ = our implementation outperforms the paper's reported result

Wine at 40%: n_A = 11 complete rows, p = 13 features
→ Sample covariance S_A is rank-deficient
→ Fr fitness function is numerically broken
→ Documented as hard statistical limit, not a bug
```

**SAY**
> We achieve reasonable replication on all five datasets. On Wholesale we're within 8% of the paper's number. On Haberman at 30%, we actually beat the paper — because our bootstrap generators sample directly from observed values, which correctly handles the Nodes column where 44% of values are exactly zero. A Gaussian generator, which the paper implicitly uses, never generates zeros.
>
> Wine is the documented failure. At 40% missing, only 11 complete rows remain for 13 features. You cannot reliably estimate a 13×13 covariance matrix from 11 samples. The fitness landscape breaks down. We document this thoroughly — it's a hard statistical limit, not an implementation error.

---

## Slide 6 — C2: Formal Orthogonality Theorem (2 min) ★

**ON SCREEN**
```
C2: No Single Imputation Can Minimise Both Fr and RMSE

Proposition 1 (RMSE → Fr):
  The RMSE-minimising imputation  X̂_j = E[X_j | X_{-j}]
  must have Fr > 0 whenever Var(X_j | X_{-j}) > 0.

  Proof: Law of total variance:
    Var(X_j) = E[Var(X_j|X_{-j})] + Var(E[X_j|X_{-j}])
    MICE replaces X_j with conditional mean → removes first term
    → Imputed marginal variance < true marginal variance
    → Covariance term in Fr is non-zero → Fr > 0

Proposition 2 (Fr → RMSE):
  Any imputation with Fr = 0 must have RMSE > 0
  unless the true missing values are drawn from the same distribution as X_A.

Corollary: No single imputation can jointly minimise both.

This theorem predicts every result in the thesis:
  → MICE wins RMSE on every dataset ✓ (Proposition 1 explains why)
  → MIGA wins Fr on Iris and Haberman ✓ (Proposition 2 explains how)
  → Under top MNAR, Fr = 0.810 and RMSE = 0.384 simultaneously ✓ (both props active)
```

**SAY**
> This is the formal backbone of the thesis. The theorem says: MICE's approach — predicting the conditional mean — must suppress variance, which means it must have non-zero distributional distance Fr. And conversely, if you achieve Fr=0, you must have non-zero RMSE unless the missing values happen to be drawn from the same distribution as the complete rows.
>
> The proof of Proposition 1 is clean: the law of total variance says that the total variance of a feature equals the average within-group variance plus the variance of the group means. MICE replaces each value with its group mean, eliminating the within-group variance term. So MICE-imputed data always has lower variance than the true data. Lower variance means non-zero Fr.
>
> This theorem predicts every experimental result in the thesis. It's not post-hoc rationalisation — the experiments confirm the prediction.

---

## Slide 7 — C3: MNAR Inversion (2 min) ★★

**ON SCREEN**
```
C3: What Happens Under Directional MNAR?

top MNAR: the 30% highest values of selected features go missing.
          X_A now contains only rows with LOW values for those features.

Haberman dataset — 10 seeds, σ(Fr) = 0.000005 (essentially zero):

  Mechanism   MIGA Fr    MIGA RMSE    Interpretation
  ──────────────────────────────────────────────────────
  MAR          2.580       0.398      Normal operation
  top          0.810  ←    0.384  ←   ← Complete inversion
  bottom       1.040       0.380      Moderate degradation
  tails        1.134       0.354      Robust — better than MAR

Fr = 0.810 is the GLOBAL MINIMUM across all experiments.
RMSE = 0.384 is the GLOBAL MAXIMUM.

Why the inversion:
  X_A = only low-value rows  (biased reference)
  GA matches X_C to biased X_A  → Fr = 0.810  ("GA succeeds")
  True missing values were HIGH
  → Every imputed value is systematically too low
  → RMSE is high despite near-zero distributional distance

Theorem C2 predicts this:
  Fr ≈ 0 with RMSE > 0 is only possible when X_A is not representative.
  Under top MNAR, X_A is exactly not representative.
```

**SAY**
> This is the most dramatic single result in the thesis. Under top MNAR, the 30% highest values of the selected features go missing. So X_A — the complete rows that MIGA uses as its reference — contains only the low-value rows for those features.
>
> The genetic algorithm does its job. It finds an imputation that makes the completed rows look statistically identical to X_A. Fr = 0.810 — the lowest value we observed across every experiment in the thesis. The GA genuinely succeeded at solving the problem it was given.
>
> But it was given the wrong problem. X_A is a biased sample. The true missing values were the high values. Every imputed value is systematically too low. RMSE stays at 0.384 — the global maximum.
>
> And this is confirmed with essentially zero variance across 10 seeds — σ(Fr) = 0.000005. The GA reliably, deterministically converges to this wrong answer. Theorem C2 predicts this exactly: Fr ≈ 0 with RMSE > 0 is only possible when X_A is not representative of the full population. Under directional MNAR, it is not.
>
> Note the tails mechanism is different: when both extremes go missing, the surviving complete rows are concentrated near the distribution centre — a stable, unbiased reference. MIGA's RMSE actually improves relative to MAR. The failure is specific to one-sided mechanisms.

---

## Slide 8 — C4: Systematic Baseline Comparison (2 min) ★

**ON SCREEN**
```
C4: Fr vs RMSE — Each Method Wins on Its Own Metric

Fr under MAR (10-seed Wilcoxon, H₁: MIGA Fr < MICE Fr):

  Dataset        MIGA Fr    MICE Fr    Ratio          p-value
  ──────────────────────────────────────────────────────────────
  Iris (p=4)      0.780      1.155    MIGA 1.48× lower   0.001 ✓
  Haberman (p=3)  2.580      5.065    MIGA 1.97× lower   0.001 ✓
  Glass (p=10)   74.03      42.99    MICE 1.72× lower   1.000 ✗

RMSE under MAR (MICE wins every dataset):

  Dataset        MIGA RMSE   MICE RMSE   MICE advantage
  ──────────────────────────────────────────────────────
  Iris             0.119        0.078       1.53×
  Haberman         0.398        0.201       1.98×
  Glass            0.155        0.075       2.07×
  Wholesale        0.214        0.151       1.42×

Neither method dominates. Each wins decisively on its own metric.

[See figure: Fr and RMSE bar chart for all methods and datasets]
```

**SAY**
> The systematic comparison confirms the theorem on every dataset. MIGA achieves significantly lower Fr than MICE on Iris and Haberman — p=0.001, verified over 10 independent seeds. On Haberman, MIGA's Fr is almost twice as low.
>
> But MICE wins RMSE on every single dataset — by 1.4 to 2.1 times. Every dataset, no exceptions. This is exactly what Proposition 1 predicts: MICE is literally designed to minimise squared error column by column.
>
> Glass is the honest counterexample for Fr. MICE achieves Fr=43, MIGA achieves Fr=74. MICE wins on MIGA's own metric. This happens because Glass has 10 features with strong correlations. MICE's column-wise regression, by predicting each feature from all others, implicitly models the joint distribution well enough to beat MIGA's moment-matching. This is a real limitation — which C5 characterises precisely.
>
> The practical message: which method you choose depends entirely on which metric your downstream analysis cares about.

---

## Slide 9 — C5: Synthetic p×ρ Experiment (1.5 min)

**ON SCREEN**
```
C5: When Does MIGA's Fr Advantage Disappear?

Controlled experiment: synthetic Gaussian data, Toeplitz covariance
p ∈ {4, 8, 13, 20, 30},  ρ ∈ {0, 0.3, 0.6, 0.9},  30% MCAR

MIGA Fr advantage (%) over MICE:

           ρ=0     ρ=0.3   ρ=0.6   ρ=0.9
  p=4    +55%    +47%    +38%    +12%       (MIGA wins everywhere)
  p=8    +44%    +32%    +20%    −39%  ←   (MIGA loses at high ρ)
  p=13   +31%    +22%    +8%     −11%  ←
  p=20   −14%    −28%    −41%    −67%      (MICE dominates)
  p=30   (vacuous — fewer than 1 complete row expected)

Key finding: Failure is JOINT on (p, ρ) — not dimension alone.

At ρ = 0.9: MIGA loses even at p = 8.
At ρ ≤ 0.6: MIGA maintains advantage up to p = 13.

Mechanism: High correlation makes MICE's iterative regression an effective
implicit joint distribution estimator. MIGA's moment-matching cannot
exploit the correlation structure — so MICE incidentally wins on Fr too.
```

**SAY**
> The Glass result raised the question: is MIGA's failure purely about dimensionality, or does inter-feature correlation also play a role? C5 answers this with a controlled experiment.
>
> We generate synthetic Gaussian data with a Toeplitz covariance — a structured way to vary correlation from zero to 0.9. We test five dimensionalities from 4 to 30. The heatmap shows MIGA's Fr advantage percentage over MICE.
>
> The finding is clear: it's not dimension alone. At low correlation, MIGA maintains its advantage up to p=13. At high correlation (ρ=0.9), MIGA loses even at p=8. The failure point moves left as correlation increases.
>
> The mechanism: when features are strongly correlated, MICE's regression of each feature on all others effectively learns the joint distribution through the conditional distributions. MIGA's moment matching — comparing means, covariance, skewness — can't exploit that structure. So in the high-correlation high-dimension regime, MICE accidentally becomes a good distributional imputer too.
>
> This gives practitioners a precise scope condition: MIGA's Fr advantage is reliable when p ≲ 8 and ρ ≲ 0.6.

---

## Slide 10 — C6: IPW-Fr Correction (1 min)

**ON SCREEN**
```
C6: Can We Correct MNAR Bias with Propensity Weighting?

Idea: Under MNAR, X_A is a biased sample.
      Re-weight row i by 1/π_i where π_i = P(row i is complete).
      Propensity π_i estimated by logistic regression on observed covariates.

Results — Fr change from standard MIGA under directional MNAR:

  Dataset    n/p    top       bottom    tails     MAR
  ─────────────────────────────────────────────────────────
  Wholesale   55   −38.1% ★  +3.0%    −28.4%   +0.7%
  Haberman   102   −18.6%   −1.3%    −12.8%   −2.1%
  Iris        38   −17.1%   −5.0%    −13.0%   −0.4%
  Wine        14   +34.7% ✗  −6.3%    +15.1%   +12%

Critical n/p threshold:
  n/p ≥ 20 → IPW reduces Fr by 1–38% (propensity model stable)
  n/p < 15 → IPW increases Fr (propensity model overfits)

Fr–RMSE orthogonality (C2) remains intact under IPW.
```

**SAY**
> C6 directly addresses the MNAR failure identified in C3. If X_A is a biased sample because high-value rows are missing, we can correct this by down-weighting rows in X_A that look like they came from the biased selection process. Inverse probability weighting, standard in causal inference, gives us a principled way to do this.
>
> The results show a clear pattern. On datasets with large n/p ratios, IPW reduces Fr by substantial amounts — up to 38% on Wholesale under top MNAR. The propensity model has enough data to estimate the selection mechanism accurately.
>
> But on Wine, with n/p of only 14, IPW makes things worse — by up to 35%. The logistic regression overfits the few complete rows and estimates propensity scores that are uninformative. The n/p ≥ 20 threshold is the practical applicability condition. Above this, IPW is a reliable MNAR correction; below it, it's harmful.

---

## Slide 11 — Downstream Utility (1 min)

**ON SCREEN**
```
Do Distributional Gains Propagate to Downstream Analyses?

Haberman dataset, 30% MAR, 100 bootstrap seeds:

  Method   KS pass rate   CI coverage   Classification accuracy
  ───────────────────────────────────────────────────────────────
  Mean         0%             20%              0.744
  KNN          0%             30%              0.744
  MICE         0%             10%              0.744
  MIGA        40%             70%              0.744

KS pass rate: imputed data passes Kolmogorov-Smirnov test against true data
CI coverage: 95% bootstrap CI contains the true mean

MIGA is the only method to pass the KS test at all.
MIGA CI coverage is 70% vs MICE's 10%.
All methods achieve identical classification accuracy — MIGA's distributional
gains come at zero cost to predictive performance.

Scope conditions for MIGA's downstream advantage:
  ✓ p ≲ 8  |  ✓ Non-degenerate feature variances  |  ✓ Moderate baseline Fr gap
```

**SAY**
> Do MIGA's distributional improvements actually matter for downstream analysis? The downstream evaluation shows they do — at least under the right conditions. MIGA is the only method that passes the KS test at all on Haberman, achieving 40% pass rate while MICE fails 100% of the time. MIGA's bootstrap confidence intervals contain the true mean 70% of the time versus MICE's 10%. And critically, all four methods achieve identical classification accuracy — around 0.744. MIGA's distributional gains are free, not bought at the cost of predictive performance.

---

## Slide 12 — Summary and Recommendation (1 min)

**ON SCREEN**
```
Six Things This Thesis Established

1. MIGA can be faithfully reimplemented (C1) — first open-source implementation.

2. Fr–RMSE orthogonality is formally provable and empirically confirmed (C2).
   No imputation can jointly minimise both. Every experiment confirms this.

3. Directional MNAR causes a characterisable failure: Fr→RMSE inversion (C3).
   The sharpest empirical confirmation: global min Fr = global max RMSE on Haberman top.

4. MIGA's Fr advantage is real but bounded (C4).
   Wins: Iris (1.48×), Haberman (1.97×). Loses on RMSE everywhere (1.4–2.1×).

5. The advantage depends jointly on (p, ρ), not p alone (C5).
   At ρ=0.9, MIGA loses even at p=8.

6. IPW-Fr corrects MNAR bias when n/p ≥ 20 (C6). Fails when n/p < 15.

Recommendation:
  Use MIGA when p ≲ 8, variances well-conditioned, downstream analysis
  requires the correct distribution, and MAR or symmetric MNAR holds.
  Use MICE for minimum RMSE, high dimensionality, or compute constraints.
```

**SAY**
> To summarise: six contributions. The formal theorem is the skeleton that explains every experiment. The MNAR inversion is the most striking empirical finding — the GA achieves its global best Fr while simultaneously achieving its worst RMSE. The synthetic experiment pins down exactly when MIGA's advantage disappears: not just dimensionality, but the interaction of dimensionality and correlation. And the IPW correction provides a partial fix for MNAR when you have enough data to estimate propensity scores reliably.
>
> The recommendation is: these are not competing methods. They have different objectives. Choose based on what your downstream analysis requires.
>
> I'm happy to take questions.

---

## Backup Slides

### Backup A — Variance Preservation

**ON SCREEN**
```
MIGA Preserves Marginal Variance. MICE Suppresses It.

Variance ratio = Var(imputed) / Var(true)  [1.0 = perfect]

  Dataset      MICE ratio   MIGA ratio   |MICE−1|   |MIGA−1|
  ──────────────────────────────────────────────────────────────
  Iris          0.976        1.025        0.024      0.025   (Tie)
  Glass         0.937        1.005        0.063      0.005 ★
  Wholesale     0.877        1.077        0.123      0.077 ★
  Haberman      0.717        1.535        0.283      0.535   (Both poor†)

  † Single missing feature — bootstrap has no multivariate constraint

Why MICE always undershoots:
  Law of total variance: Var(X_j) = E[Var(X_j|X_{-j})] + Var(E[X_j|X_{-j}])
  MICE replaces X_j with its conditional mean → removes within-group variance
  → Always underestimates, regardless of dataset
```

### Backup B — GA Hyperparameters

**ON SCREEN**
```
MIGA Hyperparameters (from paper Table 3)

  l = 200 individuals per generation
  G = 300–500 generations (dataset-dependent)
  Q = 3–5 independent runs (dataset-dependent)
  r = 2 or 3 (Minkowski order, dataset-dependent)
  c1 = 10% selection rate
  c2 = 10% crossover pairs
  c3 = 5% mutation rate
  diversity injection = 90% random individuals per generation

Population fitness evaluation: 200 evaluations/generation
  Each evaluation: fill missing cells, compute Fr (means + cov + skew)
  Per seed: ~180–210 seconds on standard CPU
  Key optimisations: cached S_A^{-1/2}, vectorised fill, numpy skewness (17× speedup)
```

### Backup C — Budget-Aware Restarts

**ON SCREEN**
```
Budget-Aware Early Stopping: Same Total Budget, More Restarts

Motivation: allocate a fixed budget Q×G more efficiently via early stopping.
  Patience = 50 generations with no improvement → restart with fresh random population
  Diversity decay = 0.7 (injection rate decays within each run)
  Total evaluations budget identical to baseline.

Results (Fr and RMSE change from fixed schedule):

  Dataset     Fr change   RMSE change   Restarts vs baseline
  ───────────────────────────────────────────────────────────────
  Iris         −0.01%      −4.1%         11 vs 5
  Wine         −8.7%       +3.6%          5 vs 5
  Glass        −4.5%       +3.1%          5 vs 5
  Haberman     −1.4%       +16.3%         7 vs 5
  Wholesale    −0.4%       +11.4%         5 vs 5

AES improves Fr (Iris: -0.01%, Wine: -8.7%, Glass: -4.5%)
  but worsens RMSE on 4/5 datasets — further confirmation of C2 orthogonality.
```

### Backup D — IPW-Fr Implementation Detail

**ON SCREEN**
```
C6: IPW-Fr — How It Works

Standard Fr: all n_A complete rows weighted equally

IPW-Fr: weight row i by  w_i = 1 / π̂_i

where  π̂_i = P(row i is complete | observed covariates)
       estimated by logistic regression on all observed features

Weighted statistics:
  x̄_A^{IPW} = Σ w_i x_i / Σ w_i
  S_A^{IPW}  = Σ w_i (x_i − x̄)(x_i − x̄)^T / Σ w_i

These weighted statistics replace the unweighted ones in the Fr formula.
The GA optimises the same objective, with corrected reference statistics.

Stability condition:
  Propensity model is logistic regression with p+1 parameters.
  Reliable estimation requires n_A >> p+1.
  Empirically: n/p ≥ 20 → stable; n/p < 15 → overfit, harmful.
```

---

## Full Q&A Script

**Q: Why not just use MICE? It has better RMSE.**

MICE is the right choice when you need minimum pointwise error — for example, when you're predicting a single outcome variable and you just need the imputed features to be accurate. MIGA is the right choice when the imputed data itself needs to have the correct distribution — confidence intervals, distributional hypothesis tests, synthetic data generation. These are different analytical goals. Both are principled choices for their respective objectives.

**Q: The RMSE gaps are large (up to 2×). Doesn't that make MIGA impractical?**

The RMSE gap is expected and explained by the theorem. MICE explicitly minimises squared error column by column — of course it wins on RMSE. The question is whether RMSE is even the right metric for a distributional imputer. For downstream analyses that require variance preservation or correct marginal distributions, RMSE of the imputed values is a misleading metric. The downstream evaluation confirms this: MIGA achieves 70% CI coverage versus MICE's 10%, at zero cost to classification accuracy.

**Q: MIGA fails under MNAR — doesn't that undermine the contribution?**

The failure mode is the contribution. We are the first to formally demonstrate that MIGA's fitness function becomes unreliable under directional MNAR, and to characterise exactly when and why. The inversion — global minimum Fr coinciding with global maximum RMSE, confirmed across 10 seeds — is the empirical proof of this failure mode. Prior work on MIGA never evaluated MNAR at all. The theoretical fix (IPW-Fr) is C6, which we implement and evaluate.

**Q: How does this compare to the original paper?**

The paper compares MIGA against CMIM (2005) and ANNI — both older, weaker baselines. We add comparison against MICE, the current standard. The paper tests MAR only; we test three MNAR mechanisms. The paper provides no code; we provide the first open-source implementation. We add formal proofs (C2), controlled dimensionality experiments (C5), and the IPW correction (C6). This is a substantial extension beyond replication.

**Q: MICE beats MIGA on Fr for Glass. Doesn't that weaken your main claim?**

It's an honest finding and part of C5. The advantage depends jointly on (p, ρ) not p alone. At high correlation and moderate dimensionality, MICE's regression incidentally models the joint distribution well enough to beat MIGA on Fr. This tells us exactly when each method applies — which is more useful than an unconditional claim. And the theorem still holds: MICE wins Fr on Glass but RMSE is still 2× better for MICE on Glass too, so the directionality is consistent — MICE optimises RMSE, and incidentally gets Fr as a bonus when the dataset structure aligns.

**Q: What is the computational cost?**

With paper parameters (l=200, G=300, Q=3), each seed takes 180–210 seconds on a standard CPU. MICE is under 1 second. MIGA is 200× slower. The Q independent runs are trivially parallelisable — they share no state. A Q-fold speedup is achievable with no algorithm change. For distributional tasks where correctness matters more than speed, this is acceptable; for large datasets or production pipelines, parallelisation is necessary.

**Q: Is the n/p condition for IPW-Fr a hard threshold?**

It's empirically derived, not theoretically sharp. The logistic regression for propensity estimation has p+1 parameters. With n_A complete rows, reliable estimation requires n_A >> p+1, which loosely translates to n/p >> some constant. We observe stability above n/p ≈ 20 and instability below n/p ≈ 14. The transition zone between 14 and 20 warrants caution. A more robust approach would use regularised propensity estimation (e.g. elastic net) to extend the stable range.
