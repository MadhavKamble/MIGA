# PPT Generation Prompt for Claude Code

Paste the following prompt into a new Claude Code session. It is fully self-contained.

---

## Prompt to give Claude Code

```
Create a professional PowerPoint presentation using python-pptx for an MTech thesis defence.
The presentation is titled:
  "Missing Data Imputation via Genetic Algorithms:
   Reimplementation, Extension, and Distributional Analysis of MIGA"

Save the output as: miga_presentation.pptx

## Design Specifications

Colour palette (IIIT Allahabad theme):
  - Title background: #1A1A5E (dark navy blue)
  - Title text: #FFFFFF (white)
  - Section header slides: #1A1A5E background, white text
  - Body slides: #FFFFFF background
  - Accent / highlight colour: #C8102E (IIIT-A red)
  - Table header fill: #1A1A5E, text white
  - Table row alternating: #F0F0F0 and #FFFFFF
  - Bullet points: #2C2C2C (near black)
  - Code/formula blocks: #F5F5F5 background, #1A1A5E text, monospace font

Fonts:
  - Headings: Calibri Bold, 28–32pt
  - Body text: Calibri, 18–20pt
  - Table text: Calibri, 14–16pt
  - Code/formula blocks: Courier New, 14pt

Slide dimensions: 13.33 × 7.5 inches (widescreen 16:9)

Logo: place IIITA_Logo.jpg in top-right corner of every non-title slide (height 0.5 inch)
Header image: place IIITA_Header.png as a thin banner (height 0.6 inch) on the title slide only

## Image Files Available
(all images are in the same folder as the script)
  - fig1_fr_rmse.png       — grouped bar chart: Fr and RMSE for Mean/KNN/MICE/MIGA across datasets
  - fig2_variance.png      — variance ratio chart: Var(imputed)/Var(true) for all methods
  - fig3_dimension.png     — Fr advantage vs dimensionality (line chart)
  - fig4_downstream.png    — KS pass rate and CI coverage bar chart
  - fig5_synthetic_dim.png — p × ρ heatmap of MIGA Fr advantage over MICE
  - IIITA_Logo.jpg         — IIIT Allahabad logo

## Slide Structure (12 main slides + 4 backup slides)

---

### Slide 1 — Title Slide

Layout: full dark navy background
Top: IIITA_Header.png as banner
Centre-top (bold white, 36pt):
  Missing Data Imputation via Genetic Algorithms:
  Reimplementation, Extension, and Distributional Analysis of MIGA
Centre-bottom (white, 22pt):
  [Your Name]
  Supervisor: [Name]
  MTech Computer Science — IIIT Allahabad
  May 2026

---

### Slide 2 — Motivation

Heading: "The Missing Data Problem"
Bullet points (18pt):
• Every real dataset has gaps
  – Medical records: tests not ordered for low-risk patients
  – Sensor networks: packet loss, device faults
  – Surveys: non-response bias (5–40% typical)
• Three strategies — each with a cost:
  – Delete rows → loses statistical power, selection bias
  – Mean imputation → distorts variance, treats imputed = observed
  – MICE → minimises squared error, but suppresses variance

Quote box (indented, italic, #C8102E border on left, 16pt):
  "Regression imputation achieves minimum RMSE by suppressing variance
   and produces biased statistical inference."
  — Van Buuren (2018, §2.6)

Bottom text (bold, 18pt):
  MIGA (Figueroa-García et al., 2023): imputes by matching the joint distribution.
  No public code. No comparison against MICE. No MNAR evaluation.
  → This thesis closes all three gaps.

---

### Slide 3 — What MIGA Does

Heading: "MIGA: Impute by Matching the Distribution"

Left column (60% width):
  Notation box (light grey background):
    X_A = complete rows (reference)     n_A rows
    X_C = rows with ≥1 missing value    n_C rows

  Formula box (code-style):
    F_r = D_r(x̃_A, x̃_C)  +  D_r(S̃, I)  +  D_r(b_A, b_C)
               means             covariance       skewness

  Explanation bullets:
    • D_r = Minkowski distance of order r
    • S̃ = S_A^{-½} S_C S_A^{-½}  (equals I when X_C ~ X_A)
    • GA: l=200 individuals, G generations, Q=3–5 independent runs

Right column (40% width):
  Highlighted callout box (navy border):
    Key difference from MICE:
    MIGA never sees the true missing values.
    It only checks if the imputed distribution
    matches the reference distribution.

---

### Slide 4 — Contributions Overview

Heading: "Six Contributions"

Table with 3 columns: #, Contribution, Key Result
Rows:
  C1 | Open-source reimplementation | First public MIGA code; 5 datasets
  C2 | Formal Fr–RMSE orthogonality theorem ★ | Proved: no single imputation can minimise both
  C3 | First MNAR evaluation ★ | Fr→RMSE inversion discovered
  C4 | Systematic Fr vs RMSE comparison | MIGA 1.5–2× better Fr; MICE 1.4–2.1× better RMSE
  C5 | Synthetic p×ρ experiment ★ | Failure depends jointly on (p, ρ)
  C6 | IPW-Fr for MNAR correction | Works when n/p ≥ 20

Mark starred rows (C2, C3, C5) with #C8102E background on the # cell.
Footer note: "10 seeds, Wilcoxon significance tests throughout"

---

### Slide 5 — C1: Replication Results

Heading: "C1: Replication — Our RMSE vs Paper RMSE"

Main table (5 datasets × 4 columns: Dataset, 30%, 40%, Notes):
  Iris      | 1.29× | 1.31× | Expected range
  Wine      | 2.56× | 3.79× | n_A < p: fundamental limit ▼
  Glass     | 1.33× | 1.56× | Expected range
  Haberman  | 0.52× ★ | 1.24× | ★ Beats paper (bootstrap handles 44% zeros)
  Wholesale | 1.07× ★ | 1.08× | ★ Best replication

Colour Haberman 30% and Wholesale 30% cells green (ratio ≤ 1.0 = beats paper).
Colour Wine cells orange (failure case).

Callout box at bottom (navy border, 16pt):
  Wine at 40% missing: n_A = 11 rows, p = 13 features
  → S_A is rank-deficient → Fr fitness function numerically broken
  → Hard statistical limit — fully documented in thesis

---

### Slide 6 — C2: Orthogonality Theorem

Heading: "C2: No Single Imputation Can Minimise Both Fr and RMSE"

Two proposition boxes side by side:

Left box (navy border, 40% width):
  Proposition 1 (RMSE → Fr)
  The RMSE-minimising imputation X̂_j = E[X_j|X_{-j}]
  must have Fr > 0 when Var(X_j|X_{-j}) > 0.

  Proof: Law of total variance:
  Var(X_j) = E[Var(X_j|X_{-j})] + Var(E[X_j|X_{-j}])
  MICE removes first term → imputed variance < true
  → Covariance term in Fr ≠ 0 → Fr > 0

Right box (navy border, 40% width):
  Proposition 2 (Fr → RMSE)
  Any Fr = 0 imputation must have RMSE > 0
  unless missing values are drawn from
  the same distribution as X_A.

Corollary box (red background, full width, white text):
  Corollary: No single imputation can jointly minimise both Fr and RMSE.

Bottom bullet:
  • This theorem predicts every experimental result in the thesis.
  • MICE wins RMSE on all 5 datasets (Prop 1). MIGA wins Fr on Iris and Haberman (Prop 2).
  • Under top MNAR, Fr = 0.810 and RMSE = 0.384 simultaneously — both propositions active.

---

### Slide 7 — C3: MNAR Inversion

Heading: "C3: The Fr→RMSE Inversion Under Directional MNAR"

Top explanation (16pt):
  top MNAR: 30% highest values go missing → X_A contains only LOW-value rows (biased reference)

Main table — Haberman dataset, 10 seeds:
  Mechanism | MIGA Fr | MIGA RMSE | Interpretation
  MAR       |   2.580  |   0.398   | Normal operation
  top       |   0.810  |   0.384   | ← COMPLETE INVERSION
  bottom    |   1.040  |   0.380   | Moderate degradation
  tails     |   1.134  |   0.354   | Robust — better than MAR

Highlight the "top" row with #C8102E background, white text.
Add annotation arrow on top row: "Global minimum Fr = Global maximum RMSE"

Explanation bullets:
  • σ(Fr) across 10 seeds = 0.000005 — GA reliably converges to this wrong answer
  • Fr ≈ 0 does NOT guarantee correct imputation when X_A is a biased reference
  • Tails MNAR is benign: removing both extremes leaves a stable, unbiased reference

Image: insert fig1_fr_rmse.png on the right side (30% width) as supporting evidence

---

### Slide 8 — C4: Systematic Comparison

Heading: "C4: Each Method Wins on Its Own Metric"

Left table (Fr under MAR — Wilcoxon 10-seed test):
  Dataset      | MIGA Fr | MICE Fr | Ratio      | p-value
  Iris (p=4)   |  0.780  |  1.155  | 1.48× ↓    | 0.001 ✓
  Haberman(p=3)|  2.580  |  5.065  | 1.97× ↓    | 0.001 ✓
  Glass (p=10) | 74.03   | 42.99   | MICE 1.72× | 1.000 ✗

Colour p-value 0.001 cells green; 1.000 cell orange.

Right table (RMSE under MAR):
  Dataset   | MIGA RMSE | MICE RMSE | MICE adv.
  Iris      |   0.119   |   0.078   |   1.53×
  Haberman  |   0.398   |   0.201   |   1.98×
  Glass     |   0.155   |   0.075   |   2.07×
  Wholesale |   0.214   |   0.151   |   1.42×

Colour all MICE RMSE cells light green (MICE always wins).

Bottom callout box (navy):
  Neither method dominates — each wins decisively on its own metric.
  MIGA wins Fr on Iris and Haberman (p < 0.001).
  MICE wins RMSE on every dataset (1.4–2.1×).

Place fig1_fr_rmse.png below the tables (full width, 35% slide height).

---

### Slide 9 — C5: Synthetic p×ρ Experiment

Heading: "C5: Failure Depends Jointly on (p, ρ) — Not Dimension Alone"

Left side (50% width):
  Explanation text:
    Controlled experiment: synthetic Gaussian, Toeplitz covariance
    p ∈ {4, 8, 13, 20, 30}
    ρ ∈ {0, 0.3, 0.6, 0.9}
    30% MCAR, 10 seeds

  Key findings (bullets):
    • At ρ = 0.9: MIGA loses even at p = 8 (−39% advantage)
    • At ρ ≤ 0.6: MIGA wins up to p = 13
    • p = 30: vacuous (fewer than 1 complete row expected)
    • Mechanism: high ρ → MICE regression models joint distribution

Right side (50% width):
  Image: fig5_synthetic_dim.png (p×ρ heatmap)

---

### Slide 10 — C6: IPW-Fr Correction

Heading: "C6: Correcting MNAR Bias with Propensity Weighting"

Left column (50% width):
  Concept box:
    IPW-Fr: weight row i by w_i = 1/π̂_i
    where π̂_i = P(row complete | covariates)
    estimated by logistic regression

  Results table:
    Dataset   | n/p | top Fr Δ | tails Fr Δ
    Wholesale |  55 | −38.1% ★ | −28.4%
    Haberman  | 102 | −18.6%   | −12.8%
    Iris      |  38 | −17.1%   | −13.0%
    Wine      |  14 | +34.7% ✗ | +15.1%

  Colour Wholesale and Haberman top cells green. Wine top cell red.

Right column (50% width):
  Critical threshold box (navy border):
    n/p ≥ 20 → IPW reduces Fr by 1–38%
               (propensity model stable)

    n/p < 15 → IPW INCREASES Fr
               (propensity model overfits)

    Wine: n/p = 14 → harmful (+34.7%)
    Wholesale: n/p = 55 → effective (−38.1%)

  Note at bottom:
    Fr–RMSE orthogonality (C2) remains intact under IPW.

---

### Slide 11 — Downstream Utility

Heading: "Do Distributional Gains Propagate to Downstream Analyses?"

Left side (50% width):
  Table — Haberman, 30% MAR, 100 bootstrap seeds:
    Method | KS pass rate | CI coverage | Accuracy
    Mean   |     0%       |    20%      |  0.744
    KNN    |     0%       |    30%      |  0.744
    MICE   |     0%       |    10%      |  0.744
    MIGA   |    40%       |    70%      |  0.744

  Highlight MIGA row green.

  Callout box:
    MIGA is the ONLY method to pass the KS test.
    70% CI coverage vs MICE's 10%.
    Zero cost to classification accuracy.

Right side (50% width):
  Image: fig4_downstream.png

Bottom bullets:
  Scope conditions for MIGA's advantage to propagate:
  ✓ p ≲ 8  |  ✓ Non-degenerate variances  |  ✓ Moderate baseline Fr gap

---

### Slide 12 — Summary and Recommendation

Heading: "The Central Finding"

Top callout box (red background, white text, large):
  MIGA and MICE are structurally orthogonal objectives.
  Neither dominates — each wins decisively on its own metric.

Two side-by-side boxes:

Left box (navy background, white text):
  Use MIGA when:
  ✓ Downstream analysis requires the correct distribution
  ✓ p ≲ 8, non-degenerate feature variances
  ✓ MAR or symmetric (tails) MNAR
  ✓ Features are discrete or heavy-tailed
  Examples: confidence intervals, KS tests, synthetic data

Right box (dark grey background, white text):
  Use MICE when:
  ✓ Minimum pointwise prediction error required
  ✓ p ≥ 10 with strong inter-feature correlations
  ✓ Compute budget is limited
  ✓ Data is high-dimensional
  Examples: feature imputation for ML models

Footer: "Thank you — Questions?"

---

### Backup Slide A — Variance Preservation

Heading: "MIGA Preserves Variance. MICE Suppresses It."

Table:
  Dataset   | MICE ratio | MIGA ratio | |MICE−1| | |MIGA−1|
  Iris      |   0.976    |   1.025    |  0.024   |  0.025 (tie)
  Glass     |   0.937    |   1.005    |  0.063   |  0.005 ★
  Wholesale |   0.877    |   1.077    |  0.123   |  0.077 ★
  Haberman  |   0.717    |   1.535    |  0.283   |  0.535 (both poor†)

Note: Ratio = Var(imputed)/Var(true). 1.0 = perfect.

Image: fig2_variance.png

Explanation box:
  Why MICE always undershoots: law of total variance
  MICE removes within-group variance component
  → Always underestimates, regardless of dataset

---

### Backup Slide B — GA Parameters and Runtime

Heading: "MIGA Hyperparameters and Runtime"

Two-column layout:
  Left: parameter table
    l = 200 individuals per generation
    G = 300–500 generations
    Q = 3–5 independent runs
    c1 = 10% selection, c2 = 10% crossover, c3 = 5% mutation
    diversity injection = 90% random individuals/generation

  Right: runtime and optimisation
    Per seed: ~180–210 seconds (standard CPU)
    MICE: <1 second
    MIGA overhead: ~200×

    Key optimisations implemented:
    • Cached S_A^{-1/2} (computed once at init)
    • Vectorised fill with numpy advanced indexing
    • numpy skewness replacing scipy (17× speedup)
    • Q runs are trivially parallelisable → Q× wall-clock speedup

---

### Backup Slide C — IPW Implementation Detail

Heading: "C6: IPW-Fr — Implementation Details"

Code-style box:
  # Propensity estimation
  completeness = (observed_rows).astype(int)   # 1 if row is complete
  model = LogisticRegression(C=1.0, max_iter=1000)
  model.fit(X_observed, completeness)
  pi = model.predict_proba(X_A)[:, 1]          # P(complete | covariates)
  weights = 1.0 / np.clip(pi, 0.05, 0.95)      # clipped for stability

  # Weighted Fr statistics
  x_bar_A = np.average(X_A, weights=weights, axis=0)
  cov_A   = np.cov(X_A.T, aweights=weights)

Applicability conditions:
  n/p ≥ 20   → stable propensity estimation → IPW reduces Fr
  n/p < 15   → logistic regression overfits → IPW harmful
  Transition: 15 ≤ n/p < 20 → caution; use regularised propensity (elastic net)

---

### Backup Slide D — Budget-Aware Restarts

Heading: "Budget-Aware Early Stopping"

Concept:
  Fixed total budget Q×G. Early stopping at patience=50 → more restarts.

Results table:
  Dataset   | Fr change | RMSE change | Restarts
  Iris      |  −0.01%   |   −4.1%     |  11 vs 5
  Wine      |  −8.7%    |   +3.6%     |   5 vs 5
  Glass     |  −4.5%    |   +3.1%     |   5 vs 5
  Haberman  |  −1.4%    |  +16.3%     |   7 vs 5
  Wholesale |  −0.4%    |  +11.4%     |   5 vs 5

Note: AES improves Fr but worsens RMSE on 4/5 datasets — confirms C2 orthogonality.

---

## Python Code Requirements

1. Use python-pptx library
2. Implement helper functions:
   - add_title_slide(prs, title, subtitle, logo_path, header_path)
   - add_content_slide(prs, title, content, logo_path)
   - add_table(slide, data, headers, left, top, width, height, header_color, alt_color)
   - add_image(slide, image_path, left, top, width, height)
   - add_callout_box(slide, text, left, top, width, height, bg_color, text_color, border_color)
   - add_two_column_layout(slide, left_content, right_content, split=0.5)

3. All measurements in inches using Inches() from pptx.util
4. All font sizes in points using Pt() from pptx.util
5. All colours as RGBColor(r, g, b)

6. Key colours:
   NAVY   = RGBColor(26,  26,  94)   # #1A1A5E
   RED    = RGBColor(200, 16,  46)   # #C8102E
   WHITE  = RGBColor(255, 255, 255)
   LGREY  = RGBColor(240, 240, 240)  # light grey for alternating rows
   GREEN  = RGBColor(0,   153, 76)   # for positive results
   ORANGE = RGBColor(255, 140, 0)    # for failure cases / warnings

7. Handle missing image files gracefully (skip if file not found, add placeholder text)

8. Save as: miga_presentation.pptx

Generate the complete Python script to create this presentation.
```

---

## Images You Need to Provide

Copy these files into the same folder where you run the script:

| File | What it shows | Used in slide |
|---|---|---|
| **fig1_fr_rmse.png** | Grouped bar chart: Fr and RMSE for Mean/KNN/MICE/MIGA across all datasets | Slides 7, 8 |
| **fig2_variance.png** | Variance ratio (Var(imputed)/Var(true)) for all methods | Backup A |
| **fig4_downstream.png** | KS pass rate and CI coverage comparison | Slide 11 |
| **fig5_synthetic_dim.png** | p×ρ heatmap of MIGA Fr advantage over MICE | Slide 9 |
| **IIITA_Logo.jpg** | IIIT Allahabad logo | Every slide (top-right) |
| **IIITA_Header.png** | IIIT Allahabad header banner | Title slide only |

All of these already exist in:
`thesis_template/thesis_template_0.1/Figures/`

Copy them to the folder where you run the python-pptx script.

## Optional: Additional Charts to Create (if you want cleaner PPT visuals)

These are currently embedded as tables in the slides. You can ask Claude Code to generate matplotlib charts for them instead:

1. **MNAR inversion chart** (Slide 7): 2-panel bar chart showing Fr and RMSE by mechanism for Haberman (MAR/top/bottom/tails). Key visual: top bar has low Fr but high RMSE — opposite of what you'd expect.

2. **IPW results chart** (Slide 10): grouped bar chart showing Fr change (%) under IPW for each dataset and MNAR mechanism. Colour positive green, negative red. n/p values annotated on x-axis.

3. **Replication ratio chart** (Slide 5): bar chart with RMSE ratio (our/paper) for each dataset × missingness %, with dashed line at 1.0. Bars below 1.0 in green (we beat paper).

To generate these, ask Claude Code:
> "Using matplotlib and the data in results/ directory (JSON files), generate publication-quality PNG charts: (1) MNAR inversion bar chart for Haberman from results/06_mnar_haberman_30.json, (2) IPW results chart from results/12_ipw_*.json, (3) replication ratio chart from results/01_replication_iris_30.json etc."
