# PPT Revision Instructions — MIGA Thesis Defence

I have an existing 16-slide PowerPoint presentation (MIGA-Thesis-Defence.pptx) for an MTech thesis defence. I need specific changes made to individual slides. Apply only the changes described below. Do not alter slides not mentioned.

---

## Slide 2 — Motivation

**Problem:** Large white gap between the Van Buuren quote block and the MIGA Opportunity box at the bottom-right. Also, the terms MAR/MCAR/MNAR are used throughout but never defined.

**Change:** Add a new callout box in the white gap area (right column, between the quote and the MIGA Opportunity box). Use a light grey background with a thin navy left border. Content:

```
MISSINGNESS MECHANISMS (Rubin 1976)

MCAR — Missing Completely At Random
        Probability of missing is independent of the data

MAR  — Missing At Random
        Depends on observed values only (ignorable)

MNAR — Missing Not At Random
        Depends on the missing value itself ← hardest case
        MIGA is evaluated under all three
```

Font: monospace, 13pt, navy text. Label "MISSINGNESS MECHANISMS (Rubin 1976)" in small caps red.

---

## Slide 3 — MIGA: imputation by distribution matching

**Problem:** Large white space below the GA configuration line on the left, and nothing below the "Key Difference from MICE" box on the right.

**Change 1:** Below the "GA configuration — ℓ=200 individuals, G generations, Q=3–5 runs" line, add a code-style block (same grey background as the notation block):

```
GA OPERATORS

  Selection       → top c₁ = 10% of population retained each generation
  Crossover       → c₂ = 10% pairs recombined (uniform crossover)
  Mutation        → c₃ = 5% values resampled from feature marginal
  Diversity inj.  → 90% of population replaced with random individuals
                    (dominant source of exploration — not mutation)

  Output: Q independent imputed datasets
          → combined via Rubin's rules (1987) for inference
```

**Change 2:** Below the "Key Difference from MICE" navy box on the right, add a second smaller box (dark grey background, white text, 13pt):

```
MULTIPLE IMPUTATION OUTPUT

Q runs → Q complete datasets
Each analysed separately → results pooled
Standard errors from Rubin's combining rules:
  T = Ū + (1 + 1/Q) · B
  Ū = within-imputation variance
  B = between-imputation variance
```

---

## Slide 4 — Six contributions

**Problem:** Approximately 40% of the slide is blank below the contributions table. This looks unfinished.

**Change 1:** Directly below the table, add a dataset summary block (light grey background, full width of the table):

```
BENCHMARK DATASETS (UCI ML Repository)

  Dataset     n      p    Domain              Missing at 30%
  ─────────────────────────────────────────────────────────
  Iris        150    4    Flower morphology   45 cells
  Wine        178   13    Chemical analysis   521 cells  ← rank-deficient at 40%
  Glass       214   10    Forensic science    642 cells
  Haberman    306    3    Survival data       92 cells
  Wholesale   440    8    Retail sales        1056 cells

  NRMSE_j = RMSE_j / (max_j − min_j), averaged over missing features
  Missingness: MCAR at cell level, floor(p/2) features selected per run
```

Font: monospace, 13pt. Header row in navy bold.

**Change 2:** The existing footnote "★ = NOVEL CONTRIBUTION BEYOND THE ORIGINAL PAPER · 10 SEEDS · WILCOXON SIGNIFICANCE TESTS THROUGHOUT" — keep it, but move it to just below the dataset table so it anchors the bottom of the content, not floating at the bottom of a half-empty slide.

---

## Slide 7 — C3: the Fr → RMSE inversion

**Problem:** White space below the three bullet points on the left side (below "Tails-MNAR is benign..."). Missing explicit link back to the C2 theorem.

**Change:** Add a highlighted box at the bottom-left, below the bullet points. Navy background, white text, 14pt:

```
CONNECTION TO C2 (THEOREM)

This is Proposition 2 instantiated in the data.

Fr ≈ 0 with RMSE > 0 is provably possible only when
X_A is not a representative sample of the full population.

Under top-MNAR, X_A is exactly not representative.
The GA found the correct answer to the wrong problem.
```

---

## Slide 9 — C5: failure depends jointly on (p, ρ)

**Problem:** The left column has white space below the four bullet points. The results in the heatmap are not translated into an actionable guideline.

**Change:** Below the last bullet point ("Mechanism: high ρ lets MICE regression recover the joint distribution implicitly."), add a callout box (red left border, light background):

```
PRACTICAL SCOPE CONDITIONS

  Use MIGA when:   p ≤ 8   AND   ρ ≤ 0.6
  MICE dominates:  p ≥ 13   OR   ρ ≥ 0.9

  Vacuous threshold: at p = 30, 30% MCAR
    E[complete rows] = n · (1 − 0.3)^30 ≈ 0
    X_A is empty → MIGA cannot operate

  This chart is the empirical derivation of the scope conditions
  stated in the thesis conclusion.
```

---

## Slide 10 — C6: correcting MNAR bias with propensity weighting

**Problem 1:** "IPW" is never expanded in the slide title. Problem 2: White space at the bottom of the left column below the table. Problem 3: Theoretical grounding (Horvitz-Thompson) not mentioned.

**Change 1:** Update slide title to:
`C6 — IPW-Fr: Inverse Probability Weighted fitness for MNAR correction`

**Change 2:** Below the table on the left, add a small box (grey background):

```
THEORETICAL BASIS

Horvitz-Thompson estimator (1952) — standard in survey
sampling for correcting non-response selection bias.

IPW-Fr adapts this to the distributional fitness objective:
  weighted reference X_A corrects biased complete-case set
  without changing the GA or the Fr formula structure.
```

**Change 3:** In the Critical Threshold box (right side), add one line at the bottom:

```
Transition zone 15 ≤ n/p < 20: use regularised propensity
estimation (elastic net) for more stable weight estimates.
```

---

## Slide 11 — Do distributional gains propagate downstream?

**Problem 1:** White space on the left below the "WHAT THIS MEANS" navy box. Problem 2:** KS test is referenced but never defined.

**Change 1:** Above the "WHAT THIS MEANS" box (or between the table and the box), add a definitions line in small monospace text:

```
KS test = Kolmogorov-Smirnov test (α = 0.05): tests whether
          imputed data and true data come from the same distribution.
CI coverage = fraction of 100 bootstrap seeds where the 95% CI
              contains the true feature mean (nominal target: 95%).
```

**Change 2:** Below the "WHAT THIS MEANS" box, add a small explanation box (light grey, navy border):

```
WHY MICE FAILS THE KS TEST

MICE imputes E[X_j | X_{-j}] → compresses the marginal distribution
→ imputed variance < true variance (Proposition 1, C2 theorem)
→ KS test detects a statistically significant distributional shift
→ MICE fails KS 100% of the time — not dataset-specific, provably structural
```

---

## Slide 12 — Summary and recommendation

**Problem:** The "Use MIGA when" and "Use MICE when" boxes have 4 bullet points each but are sized for 8 lines — large white space inside both boxes.

**Change 1:** Inside the "Use MIGA when" (navy) box, add after the 4 bullets:

```
─────────────────────────────────
Empirical evidence:
  Iris: KS pass 100%, CI coverage 95%
  Haberman: KS pass 40% vs MICE 0%
            CI coverage 70% vs MICE 10%
  Zero cost to classification accuracy (Δacc = 0.000)
```

**Change 2:** Inside the "Use MICE when" (dark grey) box, add after the 4 bullets:

```
─────────────────────────────────
Empirical evidence:
  RMSE 1.4–2.1× lower on all 5 datasets (no exceptions)
  Fr wins on Glass (p=10, high inter-feature correlation)
  Runtime < 1s vs MIGA's 180–210s per seed
```

---

## Backup Slide C — IPW-Fr implementation details

**Problem:** Top-right quadrant of the slide is entirely empty (the code block is left, the Applicability box is bottom-right).

**Change:** In the top-right area, add a box titled "WEIGHTED Fr FORMULA" (grey background, navy border):

```
WEIGHTED Fr STATISTICS

  x̄_A^IPW = Σ wᵢxᵢ / Σ wᵢ           (weighted mean)

  S_A^IPW = Σ wᵢ(xᵢ − x̄)(xᵢ − x̄)ᵀ / Σ wᵢ   (weighted covariance)

  b_A^IPW = weighted skewness (bias-corrected)

  These replace the unweighted statistics in:
    Fr = Dr(x̃_A, x̃_C) + Dr(S̃, I) + Dr(b_A, b_C)

  The GA and fitness structure are unchanged.
  Only the reference distribution X_A is re-weighted.
```

Font: monospace 13pt.

---

## Global Changes (apply to all slides)

1. **Missingness mechanism colour coding** (optional but impactful): Wherever "MAR", "MNAR", "MCAR" appear in tables or bullet points, apply consistent colour:
   - MAR → standard black text
   - MNAR → red (#C8102E) text to signal danger/difficulty
   - MCAR → grey text (mildest case)

2. **No other design changes.** Keep navy/red/white palette, IIIT-A logo position, slide numbering format, and footer style unchanged.

---

## What NOT to change

- Slide 1 (Title) — complete as-is
- Slide 5 (Replication) — white space on right is acceptable; Wine explainer anchors the layout
- Slide 6 (C2 Theorem) — complete and dense, do not alter
- Slide 8 (C4 Comparison) — complete and dense, do not alter
- Backup A (Variance) — complete, do not alter
- Backup B (Hyperparameters) — complete, do not alter
- Backup D (Restarts) — complete, do not alter
