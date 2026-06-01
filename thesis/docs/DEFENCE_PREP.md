# MIGA Thesis Defence — Full Preparation Guide
**Presentation:** 20 slides, ~17 min talk + Q&A
**How to use:** Read WHAT TO SAY out loud (not silently). Drill NOVICE Q&A first — if you can explain it to a stranger, you own it. Then drill EXPERT Q&A. The PLAIN ENGLISH section before each slide is your cheat-sheet for on-the-spot explanations.

---

## MASTER GLOSSARY IN PLAIN ENGLISH

Read this once before anything else. These are every jargon word in the thesis explained as if you are talking to your neighbour who has never studied statistics.

---

**Missing Data**
Some cells in a spreadsheet are blank. You collected data but some measurements were not taken. The question is: what do you do with those blanks?

**Imputation**
Filling in the blank cells with educated guesses, using the information in the other columns. Like a detective using available clues to reconstruct missing evidence.

**RMSE (Root Mean Squared Error)**
How wrong was your guess, on average? You compare each imputed value to the real value (which you know in simulation), square the errors, average them, take the square root. Lower = better guesses. Like the average distance between your darts and the bullseye.

**Fr (Fitness Function / Distributional Distance)**
A score asking: "Does the imputed data *look like* the complete reference data — same spread, same shape, same asymmetry?" It doesn't care whether individual guesses are right; it cares whether the statistical fingerprint matches. Lower Fr = better distributional match.

**Distribution**
The pattern of how values are spread out. For example, human heights in India — most people are between 155–175 cm, with fewer taller or shorter. A distribution tells you how common each value range is.

**Variance**
How spread out the data is. Small variance = values are clustered close together (like heights of people aged 25–30). Large variance = values are spread far apart (like heights of people aged 5–70). Variance = average of squared distances from the mean.

**Mean**
The average. Add all values, divide by count.

**Skewness**
Asymmetry of a distribution. A symmetric distribution (like a bell curve) has skewness = 0. A distribution with a long tail to the right (like income, where most people earn moderate amounts but a few earn enormously) has positive skewness.

**Kurtosis**
How heavy/fat the tails are. A distribution with many extreme values (earthquakes, stock crashes) has high kurtosis. A normal bell curve has kurtosis = 3.

**Covariance**
How much two features move together. If height increases when weight increases, they have positive covariance. Covariance matrix = all pairwise covariances between all features.

**Correlation (ρ)**
Like covariance, but scaled to be between –1 and +1. ρ = 1 means perfect positive relationship. ρ = 0 means no relationship. ρ = 0.9 means very strong relationship.

**Conditional Mean**
The average value of one feature given what the other features are. E.g., the average height of a 30-year-old male who weighs 80kg. MICE imputes this as its best guess for a missing height.

**Law of Total Variance**
A mathematical law saying: total spread = (average spread within groups) + (spread between group averages). Example: total height variance = (average of variance-within-males + variance-within-females) + (variance due to gender difference). When MICE imputes the group average for every row, it throws away the first term (within-group spread), making the imputed data less spread out than the real data.

**Joint Distribution**
How all features relate to each other simultaneously. Not just: "what's the distribution of height?" and "what's the distribution of weight?" separately — but: "what's the combined distribution of height AND weight AND age together?" Knowing the joint distribution tells you things like "tall AND heavy people are common; tall AND very light people are rare."

**Chebyshev Distance (r = ∞)**
A way of measuring how far apart two sets of numbers are, by looking only at the single worst mismatch. If you're comparing exam scores across 5 subjects, Chebyshev distance = score of your worst subject. It penalises the worst offender, ignoring how well the others did.

**Genetic Algorithm (GA)**
An optimisation method inspired by biological evolution. You start with 200 random guesses. Score each guess (fitness). Keep the best 20. Create 180 new guesses by mixing and mutating the good ones (plus fresh random ones). Repeat 500 times. The population "evolves" toward better guesses. Used when you can't just take a derivative and do calculus because the problem is too complex.

**Chromosome (in this GA)**
One candidate solution = one complete set of imputed values for all the blank cells in the dataset. If there are 50 blank cells, a chromosome is 50 numbers representing the imputed values for all of them simultaneously.

**Gene (in this GA)**
One individual imputed value — one blank cell's candidate value. A chromosome is a collection of all genes.

**Population**
The set of all candidate solutions in one generation. MIGA uses 200 chromosomes per generation.

**Elitism**
Keeping the best solutions from the current generation unchanged in the next generation, so progress is never lost.

**Crossover**
Combining two parent chromosomes to make a child. Like taking the first half of one parent's answer sheet and the second half of the other parent's.

**Mutation**
Randomly changing a few genes (individual values) by resampling from the observed data distribution. Keeps the search from getting stuck.

**Diversity Injection**
The most important operator in MIGA: 90% of the population in every generation is replaced with completely fresh random chromosomes. This keeps the search exploratory — like constantly adding new players to the team instead of only cloning the current best player.

**Generation**
One cycle of: evaluate fitness → keep elite → create new population. MIGA runs 300–500 generations.

**Q Independent Runs**
Running the entire GA process 3–5 separate times from scratch. Each run gives a different imputed dataset. This captures uncertainty — like asking 5 different doctors for an opinion.

**MCAR, MAR, MNAR**
Three types of "why is data missing?"
- MCAR: Missing Completely At Random — like a random server crash deleting some rows. No pattern.
- MAR: Missing At Random — missingness depends on observed columns. Older patients skip certain tests, but age is recorded.
- MNAR: Missing Not At Random — missingness depends on the value itself. Rich people skip salary surveys because of their high income, so the data on high incomes is absent.

**X_A (Available complete rows)**
The rows in your dataset that have no missing values. These are the "reference" — MIGA tries to make the imputed rows look like X_A.

**X_C (Rows Containing missing values)**
The rows that have at least one blank cell. These are the rows MIGA needs to impute.

**Relative Covariance (S̃)**
S̃ = S_A^{–1/2} S_C S_A^{–1/2}. Think of it as: "After we rescale everything to the reference group's units, how different is the imputed group's spread?" When S̃ equals the identity matrix (all 1s on the diagonal, 0s everywhere else), the two groups have perfectly matching covariance structure. The distance from the identity = the covariance mismatch term in Fr.

**Rank Deficiency**
A covariance matrix becomes rank-deficient when you have more features (columns) than complete rows. Like trying to solve 13 equations with only 11 variables — the system is underdetermined. The matrix can't be inverted. Fr breaks down. This is the Wine at 40% problem.

**Eigenvalue**
A property of a matrix. For a covariance matrix, eigenvalues tell you how much variance exists in each "direction" in the data. Near-zero eigenvalues = near-rank-deficiency. MIGA floors near-zero eigenvalues to avoid numerical collapse.

**Propensity Score (π_i)**
The probability that row i happens to be a complete row (no missing values), given its observed feature values. Estimated by logistic regression. Used in IPW to correct for bias.

**Inverse Probability Weighting (IPW)**
Re-weighting rows by 1/π_i. Rows that are "surprisingly" complete (low π_i) get high weight because they represent an underrepresented part of the population. Rows that are "expected" to be complete (high π_i) get low weight. Corrects the biased sample problem.

**Logistic Regression**
A model for predicting a 0/1 outcome (here: is this row complete = 1, or does it have missing values = 0?) from numeric features. Outputs a probability between 0 and 1.

**KS Test (Kolmogorov-Smirnov)**
A statistical test asking: "Do these two datasets come from the same underlying distribution?" It finds the maximum gap between the two cumulative distribution curves. Small gap = likely same distribution (pass). Large gap = likely different (fail). Here: we test whether imputed data matches the true data distributionally.

**Confidence Interval (CI)**
A range of values that should contain the true mean 95% of the time if you repeated your experiment many times. A 95% CI should have 95% coverage. If variance is suppressed (by MICE), the CI is too narrow and captures the true mean far less than 95% of the time.

**CI Coverage**
What fraction of experiments have a CI that actually contains the true value. Should be 95% for a 95% CI. MICE achieves only 10% on Haberman — its CI is so narrow that it misses the truth 90% of the time.

**Wilcoxon Signed-Rank Test**
A non-parametric statistical test (no assumption that data is bell-shaped). Used here to test whether MIGA's Fr is systematically lower than MICE's Fr across 10 runs. Non-parametric = appropriate when sample size is small (10 runs).

**p-value**
The probability of seeing a result this extreme by chance alone, if there were actually no difference between methods. p = 0.001 means 1 in 1000 chance — very strong evidence the difference is real.

**Toeplitz Covariance Matrix**
A synthetic covariance matrix where every pair of features has exactly the same correlation ρ. Used in the controlled synthetic experiment (C5) to dial ρ precisely from 0 (completely independent) to 0.9 (very strongly correlated). Real datasets have messy, unequal correlations; Toeplitz gives a clean controlled experiment.

**Bootstrap Resampling**
Generating new samples by randomly drawing (with replacement) from the existing observed values. If a feature has values [1, 2, 0, 0, 0, 5, 3], a bootstrap sample might be [0, 0, 3, 1, 0, 5, 2]. Preserves the original distribution including discrete values and zeros. Used in MIGA to initialise chromosome genes.

**Multiple Imputation (Rubin's Combining Rules)**
Instead of imputing once, impute Q times. Get Q slightly different complete datasets. Analyse each. Combine results using the formula T = Ū + (1 + 1/Q)·B, where Ū = average within-run uncertainty, B = disagreement between runs. The combined estimate is more honest about total uncertainty than a single imputation.

**EM Algorithm**
Expectation-Maximisation. An older method for missing data that assumes data is normally distributed. Alternates between: (E) estimate missing values given current parameters, and (M) re-estimate parameters given filled-in data. Not used in this thesis directly, but contextual background.

**IPW-Fr**
MIGA's fitness function Fr, but with X_A statistics re-weighted by 1/π_i. Corrects for directional MNAR bias when n/p ≥ 20.

---

## BEFORE YOU START — Numbers to Know Cold

| Fact | Value |
|---|---|
| GA population size | ℓ = 200 individuals |
| GA generations | G = 300–500 |
| GA independent runs | Q = 3–5 |
| Elite kept per generation | top 10% = 20 individuals |
| Diversity injection rate | 90% of population replaced fresh each generation |
| Distance metric | r = ∞ (Chebyshev — the worst single mismatch wins) |
| Haberman top-MNAR Fr | 0.810 — global minimum across ALL experiments |
| Haberman top-MNAR RMSE | 0.384 — global maximum across ALL experiments |
| Seed std on that result | σ = 0.000005 — GA always finds this same wrong answer |
| MICE RMSE advantage | 1.4–2.1× better on every single dataset |
| MIGA Fr advantage | 1.48–1.97× better on Iris, Haberman, Wholesale |
| IPW works when | n/p ≥ 20 |
| IPW backfires when | n/p = 14 (Wine: Fr gets 34.7% worse) |
| KS test: MIGA on Haberman | 40% pass |
| KS test: MICE on Haberman | 0% pass |
| CI coverage: MIGA | 70% |
| CI coverage: MICE | 10% |
| Classification accuracy | 0.744 for ALL four methods — no difference |

---

## SLIDE 1 — TITLE

### WHAT TO SAY

> Good morning. My name is Madhav Kamble, roll number MDE2025008, and my supervisor is Dr. Gaurav Srivastava. My thesis is about a 2023 paper published in Information Sciences — one of Elsevier's highest-ranked journals — that proposed a new approach to missing data imputation using genetic algorithms. They called it MIGA. The paper introduced a genuinely interesting idea, but it published no source code, no comparison against the standard method MICE, and no evaluation under MNAR — the hardest class of missing data.

> My thesis closes all three gaps. More than that, I discovered something the original paper did not anticipate: MIGA and the standard method MICE are not competing for the same objective. They are solving structurally different problems. I proved this formally, confirmed it empirically across five datasets and ten seeds, and characterised exactly when each method is better. That is the central contribution.

> The talk will take about 17 minutes. I have backup slides for technical details.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Information Sciences journal:** A top-tier scientific journal published by Elsevier. Getting accepted here means the work passed rigorous peer review by international experts.

**MIGA:** An acronym for the method — Missing data Imputation via Genetic Algorithm. A 2023 paper proposed this idea.

**MICE:** Multivariate Imputation by Chained Equations — the current industry standard method, used everywhere from medical research to government surveys.

**MNAR:** The hardest type of missing data where the probability that a value is missing depends on what that value would have been. Example: people with very high blood pressure are less likely to show up for their check-up, so those extreme measurements go missing.

**Formal proof:** A mathematical argument that is logically airtight — not just "we observed this in experiments" but "this MUST be true, here is the proof."

---

### NOVICE QUESTIONS

**Q: What is missing data? Why is it a problem?**
> Imagine a hospital keeps records of 1000 patients. For some patients, the blood sugar test was never done — so that cell in the spreadsheet is blank. Now if a doctor wants to study the relationship between blood sugar and heart disease, what do they do with those blank cells?
>
> Option 1: Delete those rows. You lose information and introduce bias — the deleted patients might systematically differ from the kept ones.
>
> Option 2: Fill every blank with the column average. Quick, but wrong — every imputed value is the same number, which makes the data look artificially concentrated. The spread (variance) is destroyed.
>
> Option 3: Use a smarter method like MICE or MIGA — use the other columns to make an educated guess for each blank. This thesis studies when each smart method is appropriate.

**Q: What is imputation?**
> It is the process of filling in blank cells with informed guesses. Think of it like a detective: you don't have the missing value directly, but you have clues from the other columns. If a patient's height is missing but their weight and age are known, imputation uses the statistical relationship between height, weight, and age (learned from patients whose height IS known) to make a reasonable estimate.

**Q: What is MICE?**
> MICE stands for Multivariate Imputation by Chained Equations. The idea in simple terms: for each column that has missing values, build a prediction model using all the other columns. Use that model to predict the missing values. Repeat this for every column, going around multiple cycles until the predictions stabilise. It has been the standard in scientific research since 2011 — the official R package is called "mice" and is used in medical and social science studies worldwide.

**Q: What is a genetic algorithm?**
> Think of 200 students all attempting to solve the same puzzle simultaneously. Each student's solution is scored. The bottom 180 students are replaced — but the replacements are partly inspired by the top 20's approaches (crossover), partly random variations of them (mutation), and partly completely new fresh attempts (diversity injection). After 500 rounds of this, the pool of 200 solutions has "evolved" toward the best possible answer. This is a genetic algorithm — evolution applied to finding solutions.

---

### EXPERT QUESTIONS

**Q: Why a genetic algorithm specifically for imputation? What's the theoretical motivation?**
> Standard imputation methods — MICE, KNN, EM — all implicitly minimise reconstruction loss, i.e., they output the conditional expectation E[X_j | X_{-j}]. The law of total variance tells us this removes within-group variance, so the imputed marginal variance is always lower than the true variance. This is a structural bias that cannot be fixed by tuning. MIGA sidesteps this entirely by not minimising reconstruction loss — instead it searches for imputed values that match the joint distribution of the complete rows. A GA is suited to this because the Fr objective is non-differentiable (involves matrix operations, skewness, Chebyshev distance) and doesn't admit gradient-based search.

---

## SLIDE 2 — MOTIVATION

### WHAT TO SAY

> Missing data is universal. It appears in medical records, sensor networks, financial surveys — anywhere you collect data in the real world. The three classical responses each have costs. Deletion loses statistical power and introduces selection bias. Mean imputation is the worst — it replaces every missing value in a column with the same number, which destroys the variance structure entirely. MICE is far better than both, but it has a fundamental limitation that van Buuren — the creator of the mice R package — documents explicitly in his textbook: regression imputation suppresses variance by construction, which produces biased statistical inference. This is not a bug in MICE — it is the mathematical consequence of how it works.

> On the right side you can see the three missingness mechanisms formalised by Rubin in 1976. MCAR means the probability that a value is missing is completely independent of the data — like a random sensor dropout. MAR means the probability of missingness depends on other observed columns, but not on the missing value itself — manageable with most methods. MNAR is the hardest: the value is missing precisely because of what it would have been. My thesis is the first to evaluate MIGA under all three mechanisms.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Variance suppression:** When you replace missing values with a prediction (like a regression model's output), you're giving every row a "smooth, average" value. You never impute extreme values — the tails. So the imputed column has a narrower spread than the real column. Like replacing every student's guessed score with the class average — the resulting list looks artificially uniform.

**Biased statistical inference:** If you run a statistical test (like "is the mean significantly different from some value?") on MICE-imputed data, the result will be wrong because the variance is smaller than it should be. Confidence intervals will be too narrow, p-values will be too small. The test gives you a false sense of certainty.

**Selection bias:** When the rows you deleted (because they had missing values) are systematically different from the rows you kept. Like a hospital study that deletes patients who didn't come back — those patients may have died or moved, making the remaining sample healthier on average.

**MCAR / MAR / MNAR:** Explained in the master glossary above. Think of them on a scale of difficulty:
- MCAR = completely random = easy to handle
- MAR = depends on observed things = manageable
- MNAR = depends on the missing value itself = hard, biases everything

---

### NOVICE QUESTIONS

**Q: Can you give a simple real-life example of MCAR, MAR, and MNAR?**
> Sure. Let's use a student exam database.
>
> **MCAR:** The school's computer crashed randomly and lost 10% of exam scores with no pattern — some from toppers, some from average students, some from weak students. Completely random. If you delete these, your remaining sample is still representative.
>
> **MAR:** Students who failed the previous exam are less likely to show up for the next one. But you have the previous exam score (it's observed). Given that you know someone failed before, whether they're absent is predictable. The missingness depends on observed data, not on the absent score itself.
>
> **MNAR:** Students who scored very well in this exam are embarrassed to report it (seems unlikely, but bear with me). Or, more realistically: people with very high blood pressure don't come in for check-ups because they feel fine — so the highest blood pressure readings are missing. The missing value is missing BECAUSE of how high it was.

**Q: What does "variance suppression" mean in plain words?**
> Imagine all the missing heights in a dataset are replaced with 165 cm (the column mean). Now the height column has: real heights that vary from 150 to 185 cm, AND a bunch of identical values all at 165 cm. The column now looks "narrower" — it has less spread — because all those imputed values cluster at the centre. MICE does something smarter than the mean — it uses regression — but the result is the same problem: all predicted values cluster around a smooth fitted line, never imputing extreme values.

**Q: What does "regression imputation" mean and why does it suppress variance?**
> Regression imputation means: fit a straight line (or curve) through the data, then use that line to predict the missing values. A fitted line always gives the central tendency — it never strays to the extremes. Imagine predicting a child's height from their weight. The fitted line gives: "for a child weighing 30kg, the predicted height is 120cm." But in reality, children at 30kg can range from 110cm to 130cm. The regression only gives 120cm — never 110cm or 130cm. Those extreme values are never imputed, so the distribution narrows.

**Q: Why does van Buuren — the MICE creator himself — warn about this?**
> Because he's honest about the trade-off. MICE was designed to minimise prediction error — to get individual values as close to the truth as possible. It does this very well. But in doing so, it sacrifices distributional fidelity. He explicitly warns in his 2018 textbook that MICE-imputed data should not be used for distributional tests or variance estimation without correction. The motivation for MIGA is: what if we design an imputer specifically to preserve the distribution, accepting that individual values won't be as accurate?

---

### EXPERT QUESTIONS

**Q: Under what formal conditions is MICE unbiased?**
> MICE is consistent under MAR when the imputation model is correctly specified and the missing fraction is not so large that the complete case set becomes unrepresentative. Under MNAR, MICE is generally biased even with correct model specification, because the missingness mechanism is informative about the values being imputed.

**Q: What is the formal statement of the variance suppression problem?**
> By the law of total variance: Var(X_j) = E[Var(X_j | X_{-j})] + Var(E[X_j | X_{-j}]). MICE imputes E[X_j | X_{-j}] — the conditional mean — removing the within-group variance term. So the imputed marginal variance equals only Var(E[X_j | X_{-j}]), which is strictly less than Var(X_j) whenever the conditional variance is non-zero. This is what Proposition 1 of the C2 theorem formalises.

---

## SLIDE 3 — LITERATURE

### WHAT TO SAY

> This slide places MIGA in the literature landscape. On the left are the foundational works. Dempster, Laird and Rubin (1977) gave us the EM algorithm — the first principled maximum-likelihood approach to missing data. Rubin (1976) and Little and Rubin (1987) formalised the MCAR/MAR/MNAR taxonomy and developed combining rules that make multiple imputation valid. Van Buuren (2018) is the MICE gold standard.

> On the right are the evolutionary and ML methods. Abdella and Marwala (2005) were the first to use a GA for imputation — but they used it to train neural network weights, not to directly search in the imputation space. Sefidian and Daneshpour (2020) published CMIM, a strong regression-chain baseline.

> The key point is the red box: Figueroa-García et al. (2023) introduced the core idea — searching directly in imputation space using a distributional fitness function — but left three gaps. My thesis closes all of them.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**EM Algorithm:** A two-step math procedure for handling missing data. Step E: guess the missing values. Step M: re-estimate the model parameters using those guesses. Repeat until stable. It works well but assumes the data is bell-shaped (Gaussian), which is often not true in real data.

**Maximum-likelihood:** A way of fitting a statistical model — find the parameters that make the observed data most probable. Like asking: "Given what we see, what is the most likely population mean and variance?"

**Combining rules (Rubin 1987):** When you impute Q times and get Q estimates, how do you merge them into one final answer? Rubin's formula: final variance = average within-run variance + (1 + 1/Q) × between-run disagreement. It properly accounts for two sources of uncertainty: within each imputed dataset AND between imputed datasets.

**CMIM:** An acronym for a competing method (Cluster-based Multiple Imputation with Mutual Information features). Not important to explain in detail — just know it's a strong but RMSE-focused method from 2020.

---

### NOVICE QUESTIONS

**Q: What is the EM algorithm in plain words?**
> Think of it like this: You're trying to find the average height of all students in a school, but some heights are missing. Step 1 (E): guess the missing heights using your current best estimate of the average and spread. Step 2 (M): re-compute the average and spread including your guesses. Repeat until the guesses stabilise. The key limitation: it assumes heights follow a bell curve, which is not always true.

**Q: What is the difference between MICE and EM?**
> EM models the whole dataset jointly and assumes a specific shape (Gaussian). It's elegant but rigid. MICE models each column separately using whatever model fits best — a logistic regression for yes/no columns, a linear regression for continuous columns. MICE is more flexible and works on messy real-world data. Both handle MAR data; neither handles MNAR well.

**Q: Why is the original MIGA paper's comparison weak?**
> They compared MIGA only against older, less powerful methods from 2005 and 2020. MICE is the gold standard used in thousands of published medical and social science studies. Not comparing against MICE is like testing a new car's speed against a bicycle but not against other cars. My thesis makes the proper comparison.

---

### EXPERT QUESTIONS

**Q: What is the key algorithmic difference between CMIM and MIGA?**
> CMIM uses grey-based fuzzy c-means clustering combined with mutual information feature selection and regression chains — essentially a reconstruction-based objective similar to MICE in spirit. MIGA is fundamentally different: it makes no parametric assumption about the conditional distribution and optimises a distributional objective. The solution space is all possible complete matrices, not the conditional mean function.

**Q: Is there prior work on distribution-matching for imputation?**
> GAIN (Yoon et al. 2018) and GRAPE (You et al. 2020) use neural approaches but are trained under reconstruction loss, making them conditional-mean estimators by Proposition 1. Muzellec et al. (2020) propose Sinkhorn-divergence imputation and report the same RMSE trade-off. The Fr objective — moment matching via Chebyshev distance — is specific to MIGA. No prior work uses a GA directly in imputation space with a distributional fitness function.

---

## SLIDE 4 — MIGA: METHOD

### WHAT TO SAY

> MIGA's core idea is captured in this fitness function. Fr is a sum of three distances: one measuring how well the means match, one measuring how well the covariance structure matches, and one measuring how well the skewness matches. All three compare the completed rows (X_C) against the available complete rows (X_A).

> The covariance term is the most sophisticated. S̃ is the relative covariance — when S̃ equals the identity matrix, the two groups have identical covariance structure. Fr = 0 means perfect distributional match on all three moments. The GA searches for imputed values that drive Fr as close to zero as possible.

> The critical insight: MIGA never sees the true missing values. It only asks: does the filled-in dataset look statistically identical to the complete rows? This is fundamentally different from minimising prediction error.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Fr = D_r(x̃_A, x̃_C) + D_r(S̃, I) + D_r(b_A, b_C)**
Break it down:
- First term: are the *averages* of each feature the same between complete rows and imputed rows?
- Second term: is the *spread and correlation structure* the same?
- Third term: are the *shapes* (asymmetries) the same?
Lower Fr = better. Fr = 0 = perfect match on all three aspects.

**D_r with r = ∞ (Chebyshev Distance):**
Imagine comparing two report cards with 5 subjects each. Chebyshev distance = your worst subject score difference. Even if you matched 4 subjects perfectly, one bad subject gives a large distance. In MIGA this means: if even one feature has badly mismatched distribution, Fr is large regardless of other features.

**Relative Covariance S̃ = S_A^{–1/2} S_C S_A^{–1/2}:**
A "normalised" comparison of how the features co-vary in the imputed rows versus the complete rows. Think of it as asking: "After converting to a common scale, are the inter-feature relationships the same?" If S̃ = I (identity matrix — diagonal of 1s, zeros elsewhere), perfect match. Distance from I = how mismatched they are.

**Identity Matrix I:**
A square matrix with 1s on the diagonal and 0s everywhere else. It represents "no transformation" — the baseline of perfect covariance match.

**Skewness (b):**
A measure of the asymmetry of a distribution. A perfectly symmetric bell curve has skewness = 0. Income distributions (many poor, few very rich) have positive skewness. MIGA matches skewness between X_C and X_A.

---

### NOVICE QUESTIONS

**Q: What exactly is a chromosome in this GA? Can you draw it for me?**
> Imagine a spreadsheet where some cells are blank. A chromosome is one complete set of guesses for all the blank cells simultaneously.
>
> For example, suppose rows 3, 7, and 12 have missing values:
> - Row 3 is missing: Height
> - Row 7 is missing: Weight and Age
> - Row 12 is missing: Height
>
> One chromosome = one guess for all four blanks: [Height_row3=165, Weight_row7=62, Age_row7=28, Height_row12=170].
>
> The population of 200 chromosomes = 200 different complete sets of guesses. The GA evolves all 200 simultaneously.

**Q: What is a gene in this context?**
> A gene is one individual imputed value — one blank cell's candidate guess. In the example above, 165 is a gene (the guessed height for row 3). The full chromosome has four genes. The GA mutates, crosses over, and evolves these gene values.

**Q: Why does MIGA use the "worst mismatch" (Chebyshev) instead of average mismatch?**
> Using average would let you get away with imputing one feature very badly as long as you do well on others. Chebyshev says: "Your weakest link defines your score." This forces the GA to balance all features — it can't sacrifice one feature to boost another. For distributional fidelity, this is sensible: you want ALL features to look right, not a trade-off.

**Q: What are X_A and X_C?**
> Split the dataset into two groups:
> - X_A = rows with ALL values present. These are your reference — the "clean" data.
> - X_C = rows with at least ONE value missing. These need imputation.
>
> MIGA says: after we impute X_C, it should look statistically indistinguishable from X_A. Same average. Same spread. Same shape.

**Q: Why does MIGA not just look at the actual missing values to compute the score?**
> Because in real life you don't HAVE the actual missing values — that's the whole point. You can't compute RMSE in practice because you don't know the truth. Fr is computable purely from the observed data (X_A) and whatever values you're trying (X_C with imputed values). This is why Fr is a practical fitness function — it doesn't require knowledge of the true values.

**Q: What is Rubin's combining rules? What is T = Ū + (1+1/Q)·B?**
> After Q independent runs, you have Q imputed datasets and Q sets of statistical analyses. How do you merge them?
>
> Ū = on average, how much uncertainty was in each individual analysis.
> B = how much do the Q analyses disagree with each other?
> T = total uncertainty = average within-run uncertainty + (slightly inflated) between-run disagreement.
>
> The (1 + 1/Q) factor is a correction for using a finite number of runs — if you only did Q=3 runs, you're not perfectly capturing the uncertainty, so you inflate B slightly. This formula makes the final result statistically honest about both types of uncertainty.

---

### EXPERT QUESTIONS

**Q: Why Chebyshev (r=∞) specifically? Have you tried other values of r?**
> The original paper used r=∞ for replication fidelity. Chebyshev is a natural minimax criterion. The Fr-RMSE orthogonality theorem holds for any r ≥ 1, so the choice of r doesn't affect the theoretical result.

**Q: Why moments (mean, covariance, skewness) specifically? Why not Wasserstein or MMD?**
> Moments are O(n·p) to compute (mean), O(n·p²) (covariance) — fast. For a GA calling the fitness function ~100,000 times per run, computational cost matters. Wasserstein requires solving an optimal transport problem (O(n² log n)) — entirely intractable here. This is also identified as future work: replacing the three-moment Fr with MMD or Wasserstein for richer distributional comparison.

**Q: The S̃ term requires S_A to be invertible. What happens when S_A is rank-deficient?**
> When n_A < p (fewer complete rows than features), S_A is singular. This is the Wine dataset failure at 40% missing: only 11 complete rows for 13 features. S_A^{-1/2} is undefined, Fr is degenerate, the GA cannot function. Documented as a hard statistical limit, not an implementation bug. Ledoit-Wolf shrinkage could regularise S_A but changes the Fr formula — identified as future work.

---

## SLIDE 5 — MIGA ALGORITHM (FULL PIPELINE)

### WHAT TO SAY

> The flowchart on the left is directly from the original paper. Let me walk you through the five steps on the right.

> Step 1: Initialise. Create 200 candidate imputations. For each missing cell, sample a value from that feature's distribution (estimated from the complete rows X_A). This gives 200 random but plausible-looking complete datasets.

> Step 2: Evaluate fitness. Compute Fr for each of the 200 candidates. Lower is better. This is where the three-term formula runs — means, covariance, skewness.

> Step 3: Evolve. Keep the top 10% (20 individuals) unchanged. Then replenish: 10% from crossover, 5% from mutation, and critically — 90% brand-new random individuals. This 90% diversity injection is the dominant exploration operator.

> Step 4: Repeat for 300–500 generations.

> Step 5: Run Q independent times from different random starts. Select the best overall imputation.

> The Chebyshev distance r = ∞ is used throughout.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Initialisation:** At the start, you have no good guess. So you just randomly fill every blank cell with a value sampled from that column's observed distribution. All 200 chromosomes start as random noise — but plausible noise (values within the observed range).

**Fitness Evaluation:** Score all 200 guesses using Fr. The lower the score, the more the imputed data looks like the reference.

**Elitism (top 10% survive unchanged):** The 20 best chromosomes from this generation automatically move to the next generation without any modification. This ensures the best solution found so far is never accidentally destroyed.

**Crossover (10%):** Take two elite parents. For each gene, randomly pick which parent to take it from. The child inherits a mix. Like a child inheriting some features from Mum and some from Dad.

**Mutation (5%):** Take some chromosomes and randomly change a few genes by sampling fresh values from the column distribution. Introduces small random changes to explore nearby solutions.

**90% fresh injection:** The most important step. 90% of the population is completely replaced with brand-new random chromosomes. Why? Because the imputation space is huge — with 50 blank cells, there are infinite possible solutions. Without fresh random attempts every generation, the GA gets stuck exploring only a tiny corner of that space.

**Generations:** One cycle of evaluate → keep elite → create new population = one generation. Running 500 generations is running 500 such cycles.

---

### NOVICE QUESTIONS

**Q: What is crossover? Can you explain with an example?**
> Parent 1's imputed values: [165, 62, 28, 170] (for 4 missing cells)
> Parent 2's imputed values: [160, 70, 31, 175]
>
> For each cell, flip a coin: heads = take from Parent 1, tails = take from Parent 2.
> Result child: [165, 70, 28, 175] (mix of both)
>
> The idea: if Parent 1 found a good value for cell 1 and Parent 2 found a good value for cell 2, the child combines both good values.

**Q: What is mutation? How is it different from crossover?**
> Crossover combines two existing solutions. Mutation randomly changes one gene to a completely new random value sampled from that feature's distribution. If gene 3 currently says "28 years old," mutation might change it to "41 years old" (a random sample from the observed ages in X_A). Only 5% of the population is mutated, and only a few genes per chromosome. It's a minor tweak, not a complete rebuild.

**Q: Why is 90% diversity injection so high? Doesn't that throw away all the learning?**
> It sounds wasteful, but it's necessary. Here's the intuition: the search space is enormous. If you have 50 blank cells, and each can take any value in a continuous range, the number of possible solutions is effectively infinite. Without constantly adding fresh random attempts, the GA would only explore solutions that are variations of the original 200 random starts — a tiny island in an infinite ocean. The 90% injection keeps searching the whole ocean while the 10% elite preserves the best island found so far.

**Q: What does "Q independent runs" mean? Why not just run once?**
> Each GA run starts from a different random point and may converge to a different solution. Running Q=3 or 5 times gives you 3-5 different complete datasets that represent different plausible imputations. This captures uncertainty — just like getting a second (or third or fourth) doctor's opinion. The Q imputed datasets are combined using Rubin's rules to give a statistically honest final answer that accounts for imputation uncertainty.

**Q: What is elitism, and why is it important?**
> Elitism = protecting the best solutions from being accidentally destroyed. Imagine you spend 200 generations evolving a really good solution, then a bad crossover or mutation happens to destroy it. Elitism prevents this: the top 20 solutions are copied unchanged to the next generation. The fitness can only improve or stay the same over time — it can never get worse by accident.

---

### EXPERT QUESTIONS

**Q: What is the convergence criterion? How many generations are needed?**
> G = 300–500 fixed generations. No adaptive stopping in the original implementation. Empirically, a plateau is observed around 300 generations on Haberman and Wholesale — more generations give diminishing returns. An early stopping criterion based on Fr stagnation (patience = 50 generations) is evaluated in Appendix D — it reduces Fr slightly but increases RMSE on 4/5 datasets, consistent with C2.

**Q: Is the 90% diversity injection theoretically principled?**
> Heuristic in its specific value, principled in its motivation. The imputation space is a Cartesian product of n_C continuous marginal distributions. Without fresh sampling, crossover and mutation cannot escape early elite basins. The 90% figure comes from the original paper's tuning. C5 shows reducing it below ~70% causes premature convergence at high dimensions.

---

## SLIDE 6 — SIX CONTRIBUTIONS

### WHAT TO SAY

> The six contributions span three types of work: implementation (C1), formal theory (C2), and empirical investigation (C3–C6). The three marked with red stars are genuinely novel beyond the original paper.

> C1: First public Python implementation of MIGA. Replicated paper results on five UCI datasets.

> C2: Formal proof that Fr and RMSE cannot be jointly minimised. This theorem predicts every experimental result.

> C3: Under top-quantile MNAR, MIGA achieves its lowest Fr and simultaneously its highest RMSE. The GA found the right answer to the wrong problem.

> C4: Systematic head-to-head. MIGA wins Fr on 3/4 datasets; MICE wins RMSE on all 5 — without exception.

> C5: The failure boundary depends jointly on dimensionality p and correlation ρ, not p alone.

> C6: IPW-Fr — re-weighting the reference X_A to correct MNAR bias. Works when n/p ≥ 20.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Open-source reimplementation:** Writing the code from scratch based on the paper's description and publishing it publicly so anyone can run, verify, and build on it. This is necessary for scientific reproducibility.

**Formal proof:** Not just "we found this in experiments." A mathematical proof that shows this MUST be true under stated assumptions. Like proving that the sum of angles in a triangle is always 180° — not because we measured many triangles but because it follows logically from the axioms.

**Empirically confirmed:** The formal proof says it should happen; the experiments show it actually does happen, with specific numbers, across multiple datasets, with statistical tests.

**Statistical significance:** A result is statistically significant if the probability that it happened by chance alone is below a threshold (usually 5%). p = 0.001 means 1-in-1000 probability of chance — very strong evidence.

---

### NOVICE QUESTIONS

**Q: What is the difference between Fr and RMSE? Why do we need two metrics?**
> RMSE = how close was your individual guess? Like the average error in a number-guessing game. Did you guess 42 when the answer was 45? Error = 3.
>
> Fr = does the overall pattern of your guesses look right? Even if every individual guess is slightly off, do the guesses collectively have the right average, spread, and shape?
>
> They measure different things. A method can have good Fr but bad RMSE (it preserved the shape but individual values are wrong). A method can have good RMSE but bad Fr (it predicted individual values well but the distribution looks wrong). My thesis proves they cannot both be minimised at once.

**Q: What does "open-source reimplementation" mean as a contribution?**
> The original paper described the algorithm in words and equations but never published the code. I wrote the full Python code from scratch, based only on the paper description, and made it freely available. This matters because:
> 1. Science should be reproducible — anyone can now run and verify the results.
> 2. I had to make decisions the paper left vague (how to handle edge cases, what to do when the covariance matrix breaks, etc.).
> 3. All my other contributions (C2–C6) are only possible because I built this foundation first.

**Q: What are the UCI datasets?**
> UCI Machine Learning Repository is a free public library of datasets maintained by the University of California Irvine, used as benchmark tests in research since the 1980s. The five datasets are: Iris (measurements of flower petals and sepals, 150 rows, 4 features), Wine (chemical measurements of wines, 178 rows, 13 features), Glass (chemical composition of glass samples, 214 rows, 10 features), Haberman (breast cancer survival data, 306 rows, 3 features), Wholesale (wholesale food purchase data, 440 rows, 8 features).

---

### EXPERT QUESTIONS

**Q: Why these five datasets? Why not the ones from the original paper?**
> These ARE the same five datasets as the original paper (they also tested Cardio and Adult — large n > 1000 datasets I excluded due to runtime). Iris (p=4), Wine (p=13), Glass (p=10), Haberman (p=3), Wholesale (p=8) span the relevant range of p and n/p.

**Q: Why Wilcoxon tests rather than t-tests?**
> Wilcoxon signed-rank is non-parametric — no distributional assumption. With 10 seeds, the CLT hasn't kicked in and assuming normality of differences would be unjustified. Wilcoxon tests the symmetry of differences around zero. H₁: MIGA < baseline (one-tailed).

---

## SLIDE 7 — C1: REPLICATION

### WHAT TO SAY

> The replication table shows the ratio of my RMSE to the paper's RMSE. A ratio of 1.0 means exact match. Above 1 means I'm worse; below 1 means I beat the paper.

> For Iris and Glass — ratios of 1.29× and 1.33× — this is within the expected range of variation from different random seeds. The paper used different seeds and a different implementation.

> For Wine at 40%, the ratio is 3.79× — I'm much worse. Not a code error. At 40% missing, only 11 complete rows remain for 13 features. The covariance matrix S_A is rank-deficient — it can't be inverted. The Fr fitness function breaks down. Hard statistical limit.

> The two stars: Haberman at 30% where I achieve 0.52× — I beat the paper. Their implementation likely couldn't handle Haberman's 44% zero values in the survival column; bootstrap sampling handles this correctly. Wholesale at 30% is nearly exact at 1.07×.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Replication ratio = my RMSE / paper's RMSE:** A simple comparison. If I get ratio 1.29, my method produces 29% higher error than the original paper did. If I get 0.52, I actually do better. Note: GAs are random, so even rerunning the same code gives slightly different results each time.

**Rank-deficient covariance matrix:** Imagine you want to describe how 13 features relate to each other (a 13×13 covariance table), but you only have 11 complete rows of data. You have 13 unknowns but only 11 equations — mathematically impossible to solve uniquely. The matrix becomes "singular" (can't be inverted). Wine at 40% hits this wall.

**Bootstrap generator vs Gaussian generator:** For Haberman's survival time column, 44% of values are exactly 0. If you assume a bell curve (Gaussian), you'd never sample a 0 — because the bell curve extends smoothly and assigns very low probability to exactly 0. Bootstrap sampling from the actual observed values correctly preserves these zeros. This is why my implementation outperforms the original paper on Haberman.

---

### NOVICE QUESTIONS

**Q: What does 1.29× mean exactly?**
> If the original paper's RMSE was 0.100 (each guess is off by 10%), my RMSE is 1.29 × 0.100 = 0.129 (off by 12.9%). Since RMSE lower is better, a ratio above 1 means I'm slightly worse. A ratio below 1 (0.52× on Haberman) means I'm better. Ratios up to 1.4× are generally considered "successfully replicated" given that GAs are random and the paper gave no code or seeds.

**Q: Why is it hard to exactly match the paper's results?**
> Genetic algorithms are like weather — inherently random. The random seed (the starting point for random number generation) determines which solutions are explored. The original paper did not publish their code, their seeds, or even all their implementation details. I rebuilt the method from the description. Small differences in: how I initialise values, how I handle boundary conditions, which random number generator I use — all affect the final RMSE. Getting within 1.3× with no code to compare against is a successful replication.

**Q: What is rank deficiency? Can you explain without math?**
> Imagine you're trying to figure out the price of 13 different items in a grocery bag. Each "complete row" in your data is like one receipt that lists all 13 items. If you only have 11 receipts but you need to figure out 13 unknowns, you don't have enough information — the system is underdetermined. Mathematically, the covariance matrix (which captures relationships between all 13 features) requires at least 13 complete data points to be solvable. With only 11, it collapses — you can't invert it.

---

### EXPERT QUESTIONS

**Q: Have you verified the replication numerically against the paper?**
> Yes. The paper reports RMSE for each dataset at 30/40/50% missing. My table shows the ratios. For Iris, Glass, Wholesale: 1.07–1.33× at 30% — within expected variation. Haberman 0.52× is a genuine improvement: I confirmed the bootstrap vs. Gaussian difference empirically — Gaussian initialisation on Haberman gives ratio ~1.2×; bootstrap gives 0.52×.

**Q: Could Ledoit-Wolf shrinkage fix the Wine failure?**
> Yes — it provides a regularised, always-full-rank covariance estimator. But it changes the Fr formula: S̃ would now use a shrunk S_A, changing what the fitness function measures. Faithful replication requires the original Fr formula; Ledoit-Wolf is identified as a future extension.

---

## SLIDE 8 — C2: THE THEOREM

### WHAT TO SAY

> This is the theoretical centrepiece. I'll state both propositions and the corollary.

> Proposition 1: The RMSE-minimising imputation is the conditional expectation — the model's predicted value. By the law of total variance, this removes the within-group variance. So the imputed variance is strictly less than the true variance. Therefore the covariance term in Fr is non-zero. Fr > 0 — every time, for every RMSE-minimising method.

> Proposition 2: Suppose you impute with Fr = 0 — perfect distributional match. Does that mean you got the right individual values? No. Fr = 0 means the completed rows look like the available rows, not that they ARE the true missing values. Fr = 0 is compatible with very wrong individual guesses.

> Corollary: No single imputation can jointly minimise both.

> This theorem predicts everything. MICE always wins RMSE — confirmed on all 5 datasets. MIGA wins Fr on low-dimensional datasets. And the top-MNAR result — Fr = 0.810 with RMSE = 0.384 — is both propositions activating simultaneously.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Law of Total Variance — plain explanation:**
Imagine measuring student heights across all years of school (grades 1–12).
- Total variance = how much heights vary across ALL students.
- Within-group variance = how much heights vary within a single grade (e.g., within grade 7 students).
- Between-group variance = how much the average height differs between grades (grade 1 average vs grade 12 average).
- Total = Within + Between.
- When MICE imputes: every student in a "group" gets the same imputed height — the group mean. Within-group variance drops to zero. The imputed distribution only has between-group spread. It's narrower than reality.

**Proposition 1 in one sentence:** MICE predicts the group average for every row, destroying within-group spread, so its Fr score must be positive.

**Proposition 2 in one sentence:** MIGA can match the distribution of complete rows without matching individual true values — so low Fr doesn't mean low RMSE.

**Corollary:** These two propositions together mean no single method can achieve both Fr = 0 and RMSE = 0 at the same time. They pull in opposite directions.

**Impossibility theorem:** A mathematical result that says something CANNOT be done, not just "we haven't done it yet." Like proving no one can trisect an arbitrary angle using only a compass and straightedge. My corollary is an imputation impossibility theorem.

---

### NOVICE QUESTIONS

**Q: Can you explain the law of total variance with a real-world example?**
> Sure. Consider the weight of all people in a city.
>
> Total variance of weight = how much people's weights vary overall.
>
> Split people into groups by height (short, medium, tall). Within each height group, people's weights still vary (because weight depends on more than height). That's within-group variance.
>
> The average weight of short people, medium people, and tall people are different from each other. That difference contributes between-group variance.
>
> Total weight variance = average within-group weight variance + between-group weight variance.
>
> Now imagine MICE imputing weight. For a person with known height, MICE predicts: "given this height, the weight is 70kg" (the group average for that height). It assigns 70kg to every person with that height, ignoring the fact that people at that height actually weigh anywhere from 55kg to 90kg. The within-group variation is wiped out.

**Q: Can you give a simple, concrete example of Proposition 2?**
> Imagine all the complete rows in your dataset are short women (heights 150–165 cm). Most of the rows with missing heights are actually tall men (175–190 cm), but you don't know this because the heights are missing.
>
> MIGA looks at X_A (all short women) and tries to make X_C match that reference. It succeeds — Fr = 0. It imputes heights of 155, 162, 158 cm.
>
> But the true missing heights are 178, 185, 183 cm. RMSE is huge.
>
> Fr = 0 means "imputed distribution matches the reference." It says nothing about whether the reference was appropriate. This is exactly the top-MNAR scenario.

**Q: Is this theorem new? Has nobody proved this before?**
> Not in this exact form for the Fr objective. Rubin (1987) observed informally that imputation for distributional analysis and imputation for prediction are different problems. Van Buuren (2018) discusses variance suppression descriptively. But a formal mathematical proof with the exact structure I present — two propositions, a corollary, covering the Fr objective specifically — is original to this thesis.

---

### EXPERT QUESTIONS

**Q: Is the theorem tight? When does equality hold in Proposition 1?**
> Equality (Fr = 0) could hold only if Var(X_j | X_{-j}) = 0 for all j — features are perfectly linearly related with no noise. In that degenerate case, the conditional mean IS the true value. Not a real-world scenario — it requires perfect multicollinearity.

**Q: Does the theorem apply only to MICE, or to all conditional-mean estimators?**
> It applies to any method outputting the conditional mean under MSE loss: linear regression imputation, MICE, GAIN, GRAPE, missForest. This is the basis for the C2-generalisation future work — if confirmed for neural methods, C2 becomes a universal impossibility theorem.

**Q: What if you use predictive mean matching in MICE?**
> PMM adds noise by sampling from observed values closest to the conditional mean, partially restoring variance. It is no longer a pure conditional-mean estimator. It could achieve lower Fr than pure regression imputation. However, it still undersamples the tails. My theorem applies strictly to methods outputting E[X_j | X_{-j}]; PMM is a grey area requiring empirical evaluation.

**Q: Can you construct the Pareto frontier between Fr and RMSE?**
> Conceptually yes — it exists. MICE is one endpoint (RMSE-optimal), MIGA is the other (Fr-optimal). Any convex combination traces a Pareto frontier. Parameterising and visualising this is future work. The practical value: choosing an operating point based on the downstream task — classifier → MICE end; distributional test → MIGA end.

---

## SLIDE 9 — C3: THE MNAR INVERSION

### WHAT TO SAY

> Under top-quantile MNAR, the 30% highest values in each feature are removed. X_A — the remaining complete rows — contains only the rows with low values. The GA matches X_C to X_A. It imputes low values. The true missing values were high. Fr = 0.810 (near-zero — looks like a success). RMSE = 0.384 (global maximum — actually the worst). σ = 0.000005 across 10 seeds — the GA reliably converges to this wrong answer.

> The phrase: the GA found the correct answer to the wrong problem.

> Proposition 2 of C2 predicts this exactly: Fr ≈ 0 implies correct imputation only when X_A is a representative sample. Under top-MNAR, it is not.

> Note: tails-MNAR (removing both extremes) is actually benign — the central reference is stable and RMSE improves vs MAR.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Top-MNAR — plain explanation:**
Sort each column from smallest to largest. The top 30% (the highest values) are removed. Think of it like a salary survey where everyone earning above ₹10 lakh refuses to report their salary. The remaining "complete" respondents (X_A) all earn below ₹10 lakh. MIGA tries to make the imputed respondents look like these below-₹10-lakh people — and it succeeds! But the missing respondents actually earn above ₹10 lakh.

**Fr → RMSE inversion:**
Normally you'd expect: if Fr is low (distribution matched well), RMSE would also be moderate. The inversion is when Fr is at its absolute minimum and RMSE is at its absolute maximum simultaneously. This is the worst case of the C2 theorem made concrete.

**σ = 0.000005:**
Standard deviation across 10 independent runs = 0.000005. This means the GA converges to essentially the same Fr value in every run. It's not a lucky fluke — it's systematic. The GA is 100% confident it found the best answer. But the best answer is wrong.

**Tails-MNAR:**
Remove the top 15% AND bottom 15% — both extremes. The remaining X_A is the "core" of the distribution — the middle 70%. This is actually a more stable reference than random (MAR) rows. Imputing from a stable core produces better results.

---

### NOVICE QUESTIONS

**Q: What is top-MNAR? Explain it like I'm a student.**
> Imagine 100 students took an exam. We remove the scores of the top 30 students (ranks 71–100). Their scores are "missing." The remaining 70 students (X_A) are the ones who scored below average.
>
> MIGA sees X_A and says: "OK, the reference group has a certain average and spread. Let me impute the missing values to match that." So it imputes relatively low scores.
>
> But the missing students actually got high scores. MIGA's imputed values are systematically too low.
>
> Result: Fr is excellent (the imputed data matches X_A well — because it imputed low values that look like X_A). RMSE is terrible (the imputed values are far from the true high values).

**Q: Why does σ = 0.000005 matter? What does it mean?**
> You ran the same experiment 10 times with different random starting points. If the result were random, the 10 outcomes would vary a lot. But σ = 0.000005 means all 10 outcomes are essentially identical.
>
> This tells you: the fitness landscape under top-MNAR has a single very clear "best valley." Every random starting point leads to the same valley. The GA is deterministically converging to this one answer.
>
> The disturbing part: the answer it's converging to is wrong (high RMSE). The GA is VERY confident and VERY wrong.

**Q: What is tails-MNAR and why is it benign?**
> Remove the bottom 15% and top 15% — the extreme values on both ends. What remains as X_A is the central 70% — the "typical" cases. Because you removed extremes from both sides equally, X_A is a balanced, central representation.
>
> Now MIGA tries to impute X_C to look like X_A. It imputes "central" values. The missing values were extremes — not too far from the centre. So the imputed values are close to the truth.
>
> RMSE actually improves versus MAR. Symmetric removal = benign. One-sided removal (top or bottom only) = dangerous.

---

### EXPERT QUESTIONS

**Q: Can you characterise the conditions under which the Fr→RMSE inversion occurs?**
> Two conditions: (1) directional MNAR creates a biased X_A — complete rows are not representative; (2) the GA successfully optimises Fr to near-zero, which means n_A is large enough for stable Fr computation (n_A ≥ p). Severity scales with the directional strength and missingness rate.

**Q: σ = 0.000005 — is this a convergence artifact?**
> Not spurious. Under top-MNAR, X_A has a consistent low-value signature. The fitness landscape has a single deep basin. Multiple runs find the same basin because it's the only deep one. The GA is working correctly — correctly optimising the wrong objective.

**Q: IPW only corrects Fr by 1–38%. Why can't it eliminate the inversion?**
> IPW re-weights available rows. It cannot recover information about high-value rows that are entirely absent from the complete-case set. Under severe top-MNAR (30%), there are very few high-value complete rows to up-weight. The propensity model estimates relative probabilities — it cannot create information that was never observed.

---

## SLIDE 10 — C4: SYSTEMATIC COMPARISON

### WHAT TO SAY

> Head-to-head verdict on Fr and RMSE under standard MAR conditions.

> Fr: MIGA wins on Iris (1.48× better, p=0.001), Haberman (1.97× better, p=0.001), and Wholesale (0.985×, p=0.031). MIGA loses on Glass (p=10) where MICE achieves 1.72× lower Fr.

> RMSE: MICE wins every dataset without exception. Advantage 1.41× to 2.07×. Exactly predicted by Proposition 1.

> Neither method dominates. Each wins decisively on its own metric.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Head-to-head:** Running both methods on the same data, under the same conditions, 10 times each, and comparing the numbers directly.

**MAR (again):** The baseline, fairest condition for imputation. Data is missing in a way that depends on other observed columns, not on the missing value itself. Both MICE and MIGA are designed for this setting.

**Ratio (MICE/MIGA):** Fr ratio of 1.48× means MICE's Fr is 1.48 times MIGA's Fr. Since lower Fr is better, this means MIGA is 1.48 times better at distributional matching. RMSE ratio of 1.41× means MIGA's RMSE is 1.41 times MICE's. Since lower RMSE is better, MICE is 1.41× better at pointwise accuracy.

**p-value in context:** p = 0.001 means: if there was actually no difference between MIGA and MICE, we'd see this result only 1 in 1000 times. We can confidently say the difference is real, not random noise.

**Wholesale margin (0.985×):** MIGA's Fr is 98.5% of MICE's Fr — very close to a tie. A 1.5% improvement. Yet the Wilcoxon test says this is statistically significant (p = 0.031). At Wholesale's precision (many decimal places), 1.5% is consistently reproducible across 10 seeds.

---

### NOVICE QUESTIONS

**Q: What does p = 0.001 mean? Can you explain without using statistics jargon?**
> Imagine I claim a coin is biased toward heads. I flip it 10 times and get 9 heads. You might say: "That could happen by luck." The p-value answers: how likely is it to get 9 or more heads by pure luck if the coin is actually fair? The answer for 9/10 heads is about 1.1% — unlikely but possible.
>
> In our case: if MIGA and MICE were truly equal on Fr, how likely is it that MIGA would look 1.48× better in 10 trials just by chance? p = 0.001 = 0.1% chance. Almost impossible to be a fluke. The difference is real.

**Q: Why does MICE win RMSE on every single dataset? Is that suspicious?**
> Not at all — it's expected. MICE is specifically designed to minimise the squared error between its predictions and the true values. It's doing what it was designed to do. It would be deeply suspicious if MICE did NOT win RMSE. My Proposition 1 mathematically proves that MICE must win RMSE. So MICE winning RMSE on all 5 datasets is a confirmation that the theorem is correct, not a suspicious outcome.

**Q: Why does MICE win Fr on Glass (p=10) even though MIGA is the "distributional" method?**
> Glass has strong correlations between its features. When features are strongly correlated, MICE's regression models become very accurate — knowing most features pins down the missing one very precisely. When MICE predicts very accurately, the imputed values end up close to the true values, which also means the distribution of imputed values matches the true distribution reasonably well.
>
> It's an accidental win: MICE got the distribution right as a by-product of excellent individual predictions.
>
> MIGA, on the other hand, struggles with the complex correlation structure in Glass — its fitness landscape becomes difficult to optimise.

---

### EXPERT QUESTIONS

**Q: Wholesale's Fr advantage (0.985×) is barely above 1.0. Why do you claim MIGA wins?**
> p = 0.031 < 0.05 establishes that the difference is not due to chance. But you're right — 0.985× is nearly parity. The claim is worded carefully: "MIGA wins with statistical significance" — not "wins decisively." The key inference: Wholesale (p=8) sits at the boundary, consistent with C5's finding that the crossover is between p=8 and p=10.

---

## SLIDE 11 — C5: SYNTHETIC p×ρ EXPERIMENT

### WHAT TO SAY

> C5 is a controlled experiment to isolate the joint effect of dimensionality and correlation on MIGA's Fr advantage. I generated synthetic Gaussian data with a Toeplitz covariance matrix and swept p from 4 to 30 and ρ from 0 to 0.9.

> Key finding: at ρ = 0.9, MIGA loses even at p = 8 (−39% disadvantage). At ρ ≤ 0.6, MIGA maintains a positive advantage up to p = 13. At p = 30, essentially no complete rows exist — MIGA cannot run.

> Mechanism: high correlation makes MICE's regression extremely effective, incidentally preserving the distribution. MIGA's explicit moment matching can't exploit this structure.

> Scope condition: use MIGA when p ≤ 8 and ρ ≤ 0.6. MICE dominates when p ≥ 13 or ρ ≥ 0.9.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Toeplitz covariance — plain explanation:**
Imagine 8 friends in a group. In a Toeplitz setup, every pair of friends likes each other with exactly the same strength ρ. ρ = 0 means they're strangers (no relationship). ρ = 0.9 means they're very close friends (strongly related). This gives you a clean controlled experiment where you can dial "how much are features related" from 0 to 1 precisely.

In real datasets like Iris or Haberman, different pairs of features have different correlations — messy and uncontrolled. The synthetic Toeplitz experiment removes this messiness so you can see purely the effect of ρ.

**Dimensionality p:**
How many features (columns) the dataset has. Iris has 4, Wholesale has 8, Glass has 10, Wine has 13. The synthetic experiment sweeps from p = 4 to p = 30 to see where MIGA's advantage disappears.

**ρ (correlation):**
How strongly features move together. ρ = 0 = independent. ρ = 0.9 = very strong relationship between every pair.

**Heatmap:**
A table where each cell's value is represented by a colour. Green = MIGA wins (positive advantage). Red = MICE wins (negative advantage). Lets you read the p × ρ grid at a glance.

**The "hard wall" at p = 30:**
With 30% MCAR and 30 features, each row independently has a (0.7)^30 = 0.0002 probability of being completely intact. In 200 rows, you expect 0.04 complete rows. Zero. MIGA needs complete rows to function. This isn't a soft limitation — it's mathematically impossible.

---

### NOVICE QUESTIONS

**Q: What is Toeplitz covariance? I've never heard this word.**
> "Toeplitz" is the name of a German mathematician. A Toeplitz matrix is one where every diagonal has the same value. For our covariance matrix, this means: every pair of features has EXACTLY the same pairwise correlation ρ.
>
> In real life, Feature A might correlate with Feature B at 0.8, with Feature C at 0.3, and with Feature D at 0.1. In a Toeplitz setup, all three pairs have exactly the same correlation ρ. It's an artificial, controlled setting for a clean experiment.
>
> Why use it? Because we want to isolate "what happens when we change only ρ, holding everything else fixed." Toeplitz gives us that control.

**Q: What is 30% MCAR at cell level? How is that different from 30% missing rows?**
> Cell-level MCAR: each individual cell (one value in one row) independently has a 30% probability of being deleted. A single row might lose 0, 1, 2, or all of its values.
>
> Row-level 30% missing: 30% of ENTIRE ROWS are deleted. That's not what we use.
>
> Cell-level MCAR is harder for imputation methods because even "mostly complete" rows might have one missing value, making them part of X_C.
>
> At p = 30 features and 30% cell-level MCAR, the probability that ALL 30 cells in a row survive = (0.7)^30 ≈ 0.0002. Essentially zero rows remain complete.

**Q: What does "MIGA loses at ρ = 0.9 even at p = 8" mean in practical terms?**
> It means: if you have an 8-feature dataset where every pair of features has a 0.9 correlation (very strongly related), MICE will outperform MIGA on Fr (not just RMSE). The scope condition "use MIGA when p ≤ 8" is not sufficient — you also need ρ ≤ 0.6. At ρ = 0.9, MICE's regression becomes so accurate that it accidentally matches the distribution better than MIGA does.

**Q: Why does high correlation help MICE but not MIGA?**
> When features are strongly correlated (ρ = 0.9), knowing the other 7 features almost completely determines the 8th. MICE's regression prediction is nearly perfect. When you predict nearly the true value for every missing cell, the distribution of imputed values is also nearly the true distribution — MICE accidentally wins Fr.
>
> MIGA can't exploit correlation directly. Its fitness function measures the distributional gap but doesn't use the correlation structure to guide imputation. In a high-correlation space, MIGA is navigating a more complex fitness landscape without the shortcut that MICE gets from precise conditional prediction.

---

### EXPERT QUESTIONS

**Q: Can you explain theoretically why high ρ helps MICE but not MIGA?**
> Under high correlation, P(X_j | X_{-j}) is tightly concentrated — regression prediction is accurate, residual variance small. Low RMSE → imputed values close to truth → distributional moments approximately correct. MIGA explicitly matches moments without using correlation structure. In high-ρ spaces, the fitness landscape becomes harder — many configurations yield similar Fr values — reducing GA convergence.

**Q: Why is n=200 sufficient for this experiment?**
> n=200 matches typical UCI dataset scales and is enough to estimate X_A reliably at low p. At high p (p=20, 30), the vacuous-threshold result is analytically derived — it doesn't depend on n. The experiment demonstrates the structural p×ρ pattern; the scope conditions are conservative empirical guidelines, not analytical bounds.

---

## SLIDE 12 — C6: IPW-Fr

### WHAT TO SAY

> C6 proposes a correction for the MNAR bias in C3. The idea: under MNAR, X_A is a biased sample. We estimate the probability that each row is complete using logistic regression — this is the propensity score. We re-weight each complete row by 1/propensity. Rows that are "surprisingly" complete get high weight; expected-to-be-complete rows get low weight. This re-centres the reference distribution.

> Works when n/p ≥ 20. Wholesale (55): 38.1% reduction in Fr under top-MNAR. Haberman (102) and Iris (38) also improve. Wine (14): Fr gets 34.7% worse — overfitting.

> The Fr-RMSE orthogonality remains intact under IPW. Correcting X_A's distribution does not make MIGA competitive on RMSE.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Propensity Score — plain explanation:**
"Propensity" = tendency. The propensity score for a row = the probability that this row would be complete (no missing values), given what we know about it.

Example: In a top-MNAR salary survey, low-salary people are more likely to answer (they have no reason to hide). High-salary people are less likely to answer. A person earning ₹3 lakhs might have a propensity score of 0.9 (90% chance of being complete). A person earning ₹15 lakhs might have propensity score 0.1 (10% chance).

**Inverse Probability Weighting — plain explanation:**
Once you have propensity scores, give each complete row a weight = 1 / propensity.
- Low-salary person (propensity = 0.9): weight = 1/0.9 = 1.1 (almost normal weight)
- High-salary-looking complete person (propensity = 0.1): weight = 1/0.1 = 10 (very high weight)

The high-weight rows drag the reference distribution toward the underrepresented high-salary region. The re-weighted X_A looks more like the true population.

**n/p ratio:**
Number of rows divided by number of features. This tells you how much data you have per feature. Low n/p = not enough data to estimate reliable relationships. High n/p = plenty of data.

**Logistic regression overfitting (Wine n/p=14):**
With only 14 rows per feature, the logistic regression model that estimates propensity scores has too little data. It memorises the training rows instead of learning a general pattern. Some rows get propensity score ≈ 0.001 (extreme), giving weight 1000 — which destabilises everything.

**Horvitz-Thompson estimator (1952):**
The classical survey sampling tool that this adapts. In surveys, if certain types of people are less likely to respond, you up-weight their responses to make the sample representative. Same idea, applied to imputation.

---

### NOVICE QUESTIONS

**Q: What is a propensity score? Explain it like I'm ten.**
> Imagine you have 100 kids and you ask them to bring their lunch boxes to show the class. Some kids always bring their lunch box (high propensity to show up = high probability of being "complete"). Other kids often forget (low propensity).
>
> The propensity score for each kid = how likely they are to show up with their lunch box. We estimate this using what we know about each kid (their habits, their distance from school, etc.).
>
> Kids who rarely show up but did today are "surprisingly present." We give their lunch boxes extra attention (high weight) because they represent the usually-absent kids.

**Q: What is inverse probability weighting? Why "inverse"?**
> If a row has a 20% probability of being complete (propensity = 0.20), we give it weight 1/0.20 = 5. Because only 1 in 5 such rows typically shows up as complete, each one that does show up represents 5 similar rows in the true population. The inverse weight = "how many real-world units does this one observed row represent?"
>
> A very common row (propensity = 0.90, 90% chance of being complete) gets weight 1/0.90 ≈ 1.1 — it's well-represented already.
>
> A rare row (propensity = 0.10, rarely complete) gets weight 10 — it's underrepresented, so we amplify it.

**Q: Why does n/p ≥ 20 matter for this to work?**
> We're using logistic regression to estimate propensity scores. Logistic regression has p parameters to estimate (one per feature). To do this reliably, you need many more observations than parameters — as a rule of thumb, at least 10–20 observations per parameter.
>
> Wholesale: n/p = 440/8 = 55 — excellent. 55 rows per parameter. Reliable scores.
> Wine: n/p = 178/13 = 14 — too few. The logistic model memorises the 178 training rows instead of learning a general pattern. Some rows get assigned extreme probabilities (like 0.001), giving weight 1000 — completely unreliable.

---

### EXPERT QUESTIONS

**Q: Horvitz-Thompson requires true propensity scores. You estimate them — what's the consequence of misspecification?**
> With estimated propensity scores, the IPW estimator is consistent only if the model is correctly specified. Misspecification (e.g., true mechanism is non-linear) means the reweighted reference may not converge to the true population distribution. Doubly-robust estimators would be more robust but more complex.

**Q: You clip propensity scores at [0.05, 0.95]. Why?**
> Without clipping, near-zero propensities create extreme weights (1/0.001 = 1000), giving single rows excessive leverage. Clipping at 0.05 bounds weights at 20. Trade-off: introduces bias (not using true estimated weights) but reduces variance (extreme weights bounded). Standard practice in causal inference.

---

## SLIDE 13 — DOWNSTREAM UTILITY

### WHAT TO SAY

> Do MIGA's distributional gains translate into better downstream statistical analyses?

> 100 bootstrap seeds on Haberman at 30% MAR. Two evaluations: KS test pass rate (does imputed data pass a distributional test against the true data?) and 95% CI coverage (does the confidence interval for the mean actually contain the true mean 95% of the time?).

> MIGA: 40% KS pass, 70% CI coverage. MICE: 0% KS pass, 10% CI coverage. All four methods get identical 0.744 classification accuracy.

> Why MICE always fails the KS test: Proposition 1 guarantees it. MICE's variance suppression is systematic and consistent. The KS test detects this systematic narrowing — not by luck, but structurally.

> The full panel shows: on Iris (p=4) and Haberman (p=3), MIGA leads. On Glass (p=10) and Wholesale (p=8), MICE leads — consistent with scope conditions from C5.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**KS Test — plain explanation:**
The Kolmogorov-Smirnov test compares two groups of numbers and asks: "Could these two groups have come from the same underlying population?"
Think of it like two histograms. The KS test finds the biggest gap between the two histograms at any point. If the gap is large, the distributions are different (fail). If the gap is small, they could be the same (pass).
In our case: we compare MIGA-imputed data versus true data. "Pass" means the imputed data looks statistically identical to the truth.

**Confidence Interval (CI) — plain explanation:**
A CI is a range within which the true value probably lies. A "95% CI" means: if you repeated this experiment 100 times, in 95 of them the CI would contain the true value.
If MICE suppresses variance, the CI is too narrow. Like using a ruler that's too small — you report the answer is "between 3.0 and 3.5 cm" but the true answer is 4.2 cm. The CI missed.

**95% CI Coverage:**
What fraction of times (across 100 experiments) does the 95% CI actually contain the true value? It SHOULD be 95%. MICE achieves only 10% on Haberman — its CI misses the truth 90% of the time. MIGA achieves 70% — not perfect, but dramatically better.

**Classification accuracy:**
The fraction of times a standard classifier (like a decision tree or logistic regression) correctly labels a test example. All four methods get 0.744 = 74.4% correct. Imputation method doesn't matter for this task on this dataset.

**Bootstrap seeds (100):**
Instead of running the experiment once, run it 100 times with different random splits of the data. This gives a reliable estimate of how often each method succeeds. 40% KS pass means: across 100 independent experiments, MIGA passed the KS test in 40 of them.

---

### NOVICE QUESTIONS

**Q: What is the Kolmogorov-Smirnov test? Can you explain without math?**
> Imagine you have two dice. One is the "true data" distribution, one is the "imputed data" distribution. You roll each die 100 times and plot the results as a histogram.
>
> The KS test asks: "Are these two histograms consistent with being from the same die?" It finds the biggest gap between the two cumulative curves — if that gap is too large, the dice are different (fail). If the gap is small enough, they might be the same die (pass).
>
> α = 0.05 means: we allow a 5% chance of falsely declaring them different when they're actually the same.
>
> "40% KS pass rate" means: in 40 out of 100 experiments, the imputed data's histogram was close enough to the true data's histogram that we couldn't distinguish them.

**Q: What does 95% CI coverage mean in plain English?**
> Imagine you're trying to estimate the average income in a city. You take a sample, do your calculation, and report a range: "I'm 95% confident the true average is between ₹45,000 and ₹55,000."
>
> If you repeated this with 100 different samples, a good method would have the true average fall inside the CI in 95 of the 100 tries. That's 95% coverage.
>
> MICE achieves only 10% coverage on Haberman. That means in 90 out of 100 experiments, MICE's CI does NOT contain the true mean. Its CI is so narrow (because variance is suppressed) that it constantly misses.
>
> MIGA achieves 70% — still not perfect, but 7 times better than MICE for this task.

**Q: Why does classification accuracy not differ between methods? Shouldn't a better method produce better downstream results?**
> Not necessarily. Classification depends on whether the model can separate classes — it depends on the mean structure and decision boundaries between groups. All four imputation methods preserve the mean structure reasonably well. The distributional differences (variance, shape) don't affect the classifier's decision boundary much.
>
> The distributional gains from MIGA matter for tasks that USE the distribution: confidence intervals, hypothesis tests, generating synthetic data. Classification doesn't use the distribution — it just needs "which class is more likely?" and all methods agree on that.

---

### EXPERT QUESTIONS

**Q: MIGA achieves 40% KS pass on Haberman. Why not higher?**
> Two factors. First, MIGA doesn't drive Fr to zero (Fr = 2.580 under MAR) — some distributional mismatch remains. Second, the KS test at Haberman's n=306 with n_C ≈ 92 has reasonable power to detect remaining mismatches. 40% is a genuine result — not perfect, but statistically and practically significantly better than 0%.

**Q: Why does MICE achieve exactly 0% KS pass?**
> Proposition 1 shows MICE systematically under-imputes variance. The KS test detects systematic distributional shifts. At Haberman's n=306, the test has sufficient power to detect MICE's variance suppression consistently across all 100 seeds. In the limit, as n grows, power approaches 1 against any fixed distributional shift — hence consistent 0%.

---

## SLIDE 14 — SUMMARY AND RECOMMENDATION

### WHAT TO SAY

> Central finding: MIGA and MICE are structurally orthogonal — each wins decisively on its own metric and loses on the other's.

> Use MIGA when: downstream analysis requires the correct distribution — confidence intervals, distributional tests, synthetic data generation. Also when p ≤ 8, ρ ≤ 0.6, MAR or tails-MNAR.

> Use MICE when: minimum prediction error required — feature imputation for a supervised ML pipeline. Also when p ≥ 10, strongly correlated data, or compute budget is limited. MIGA: ~200 seconds per seed. MICE: under 1 second.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**"Structurally orthogonal":**
Orthogonal means perpendicular — at right angles. In maths, two vectors are orthogonal if they have nothing in common in direction. Here it means: Fr and RMSE are measuring completely different, unrelated things. Improving one doesn't improve the other. They pull in opposite directions.

**Downstream analysis:**
What you're going to DO with the data after imputation. "Downstream" = the next step in your pipeline. Common downstream tasks: (1) train a classifier, (2) run a hypothesis test, (3) compute confidence intervals, (4) generate synthetic data. The choice of imputation method should match the downstream task.

**200 seconds per seed:**
Running MIGA once on a modern laptop takes about 3.5 minutes. With Q = 5 seeds, that's about 17 minutes total. MICE runs in under 1 second for the same data. This is why MIGA is a research tool, not a production deployment tool.

---

### NOVICE QUESTIONS

**Q: If MIGA is better at preserving distributions, why would anyone ever use MICE?**
> Most real-world imputation is done to feed data into a machine learning model — a classifier or predictor. That model only cares about getting the right answer for each row. For that, RMSE is the right metric, and MICE is better.
>
> MIGA's advantage only matters when the downstream task uses the SHAPE of the distribution — like computing how spread out something is, or testing whether two groups are statistically different, or generating synthetic data. These are important in research and clinical settings, but less common in everyday ML applications.

**Q: What is 200 seconds per seed? Is that a dealbreaker?**
> 200 seconds = ~3.5 minutes per seed. With Q = 5 seeds, that's about 17 minutes for a full analysis on one dataset.
> For research: acceptable — you run it once, publish results.
> For production (imputing new incoming data every few seconds): not viable.
>
> The future work parallelisation (running all Q seeds simultaneously on Q CPU cores) reduces wall-clock time to ~40 seconds. That makes it more practical, but still far slower than MICE's 1 second.

---

### EXPERT QUESTIONS

**Q: Is there any theoretical reason the scope conditions cannot be extended?**
> The conditions are empirically derived, not analytically bounded. The fundamental limit is n_A ≥ p for S_A invertibility. Beyond this, GA convergence degrades with dimensionality because search space grows exponentially while population is fixed. Theoretically, larger ℓ or G could extend scope at the cost of runtime. The ρ threshold is driven by MICE's improvement at high correlation — not a MIGA failure mode but a MICE success mode.

**Q: The kurtosis improvement (4.2× better than MICE) is mentioned. Why isn't kurtosis in Fr?**
> Fr includes moments 1–3: mean (moment 1), covariance (moment 2), skewness (moment 3). Kurtosis is moment 4. The finding that MIGA achieves 4.2× better kurtosis without a kurtosis term is a free bonus — matching moments 1–3 implicitly improves moment 4. Adding a kurtosis term is future work.

---

## SLIDE 15 — FUTURE WORK

### WHAT TO SAY

> AD-MIGA: Adaptive Diversity Injection. The C5 finding shows diversity injection is the dominant operator — 90% of the population is replaced randomly each generation, fixed for all G generations. Early: 90% is good for exploration. Late: it wastes budget. Adaptive decay — α(g) = α₀ · exp(−λ · g/G) — reduces injection as generations progress, shifting from exploration to exploitation. Novel because no prior GA-imputation paper identified diversity injection as the primary driver.

> Generalising C2 to neural methods: if GAIN and GRAPE are also conditional-mean estimators under MSE loss, Proposition 1 applies to them too — and C2 becomes a universal impossibility theorem: no imputation method optimising reconstruction loss can achieve distributional fidelity. This is publishable.

---

### PLAIN ENGLISH: TERMS ON THIS SLIDE

**Adaptive decay — plain explanation:**
Think of a search party looking for a missing person in a forest. Early in the search: explore widely — send people all over the forest (high injection rate). Late in the search: you've found some promising areas — focus there (low injection rate). The current MIGA uses the same "explore widely" rate throughout. AD-MIGA proposes starting wide and narrowing as you find good solutions.

**Exploration vs. exploitation:**
Exploration = trying new random things. Exploitation = refining the best things found so far. A good search balances both. Too much exploration = random search, no improvement. Too much exploitation = stuck in one local best. The exponential decay schedule is a principled way to shift the balance over time.

**GAIN and GRAPE:**
Neural network-based imputers. GAIN uses GANs (two competing networks). GRAPE uses graph neural networks. Both are trained primarily under reconstruction loss (MSE) — which, if Proposition 1 applies, means they also suppress variance and would fail the KS test. Testing this would confirm or refute whether C2 is a universal law.

**Universal impossibility theorem:**
A result that would say: ANY method trained under MSE/reconstruction loss CANNOT achieve distributional fidelity. Not just MICE — every neural method, every regression-based method. This would be a fundamental theoretical contribution to the field.

---

### NOVICE QUESTIONS

**Q: What is adaptive decay? Why not just set a lower injection rate from the start?**
> A lower fixed rate (say 50%) would be better late in the run but WORSE early — the population converges too fast to a local solution before exploring the space properly.
>
> Adaptive decay starts HIGH (aggressive exploration, like the early stage of a job search where you apply broadly) and ends LOW (focused exploitation, like intensively preparing for the few best companies). You get the best of both phases.
>
> The exponential function α(g) = α₀ · exp(−λ · g/G) starts at α₀ = 0.9 and smoothly decreases. This is similar to "simulated annealing" in other optimisation methods.

**Q: What are GAIN and GRAPE?**
> GAIN = Generative Adversarial Imputation Networks. Uses two neural networks: a generator (tries to produce fake complete data) and a discriminator (tries to distinguish real from fake). They compete until the generator is very good at producing realistic completions. Published in 2018.
>
> GRAPE = GRAph-based imputation by Propagation. Treats the dataset as a graph — rows are one set of nodes, features are another set. Missing values are like broken edges. A graph neural network propagates information through the graph to fill in the gaps. Published in 2020.
>
> Both are state-of-the-art neural methods. The question for my future work: do they ALSO suppress variance (and thus fail the KS test)?

---

### EXPERT QUESTIONS

**Q: For the C2 generalisation — GAIN uses adversarial loss, not MSE. Does Proposition 1 still apply?**
> GAIN uses a hint-based adversarial objective combined with an MSE reconstruction term. If the MSE term dominates the gradient signal, the generator effectively outputs conditional means, and Proposition 1 applies. The adversarial term could, in principle, encourage full conditional distribution sampling rather than the mean — partially restoring variance. The empirical test (KS test pass rate on all 5 datasets) would disambiguate. The hypothesis: MSE dominates, GAIN fails the KS test.

**Q: Is AD-MIGA truly novel in the EA literature?**
> Adaptive mutation rates and population sizing are studied in evolutionary computation. However, the specific combination of: (1) identifying diversity injection (not mutation) as the dominant operator, (2) applying exponential injection decay to imputation quality, and (3) the empirical motivation from C5 — is novel in the missing data imputation literature.

---

## SLIDES 17–20 — APPENDICES

### WHEN TO USE THEM

These are backup slides for Q&A. Do NOT present them during the main talk. Know where each one is so you can navigate instantly.

- **Appendix A (Variance):** If asked "does MIGA actually preserve variance?" Show this. MICE ratio < 1 on all datasets (undershoot). MIGA ratio > 1 on most (slight overshoot). Both imperfect, but MIGA is closer to 1.0 on Glass and Wholesale.

- **Appendix B (Hyperparameters):** If asked about runtime, GA settings, or implementation optimisations. numpy skewness computation = 17× faster than scipy — important for running 100,000+ fitness evaluations per run.

- **Appendix C (IPW Details):** If asked for the IPW formula or code structure. Shows weighted mean, covariance, skewness. Weight clipping at [0.05, 0.95].

- **Appendix D (Restarts/AES):** If asked about adaptive early stopping. AES improves Fr slightly but worsens RMSE on 4/5 datasets — empirical confirmation of C2.

---

## GENERAL Q&A STRATEGY

### If you don't know the exact answer
> "That's a great question — I don't have the exact number in front of me, but based on the C2 theorem, what we'd expect is..." — always anchor to your theorem when uncertain.

### If a question challenges your results
> "That's a fair concern. The finding is consistent across 10 seeds with Wilcoxon significance (p = 0.001 on the key results), so it's unlikely to be random variation. The theoretical mechanism — as Proposition 1 proves — is..."

### If asked about something you didn't test
> "That's identified as future work. The prediction, from the C2 theorem, is... Testing it would confirm or refute whether the theorem extends to that setting."

### If a complete novice asks about a term
> Start with the simplest analogy. Use numbers from everyday life (income, height, exam scores). Confirm they're following before going into formulas.

### If an expert challenges your proof
> "The proof relies on the law of total variance, which holds whenever the conditional variance is non-zero. The edge case where it doesn't hold is perfect multicollinearity — features are linearly determined by others, with zero noise. In that degenerate case, RMSE = 0 and Fr = 0 simultaneously, but this never occurs in real data."

---

## STRESS-TEST QUESTIONS

**Q: MNAR is untestable in practice — how can you evaluate it if you can't detect it?**
> In practice, MNAR cannot be distinguished from MAR using only observed data — you're right. However, this is a SIMULATION study: I imposed MNAR by construction. I know the mechanism because I created it. The contribution is not "detect MNAR in practice" but "characterise what happens to MIGA when MNAR is present." The Fr→RMSE inversion is a WARNING to practitioners: if your complete-case sample is unrepresentative (diagnosable indirectly via covariate balance tests), IPW-Fr is warranted.

**Q: Your Fr formula weights three terms equally. Is there a principled reason?**
> No — the equal weighting is a design choice in the original paper. A one-unit mismatch in mean, covariance, or skewness is treated as equally important. Learning the weights from data (or the downstream task objective) could improve performance. This is not explored in this thesis but is related to the moment-expansion future work.

**Q: Can Fr and RMSE be jointly minimised as n → ∞?**
> In the asymptotic limit under MAR, MICE's conditional mean prediction approaches the true conditional mean, which reconstructs the true distribution — variance suppression shrinks as the number of groups approaches the sample size. So yes, the tension softens asymptotically. However, this thesis operates at finite n (n = 150 for Iris, 306 for Haberman) — the regime where the theorem is empirically confirmed and practically relevant. Most real-world datasets are finite.

**Q: What if the committee asks about feature interaction in the Fr covariance term?**
> The covariance term S̃ = S_A^{-1/2} S_C S_A^{-1/2} captures all pairwise feature interactions simultaneously. It's a symmetric p×p matrix comparison. The distance D_∞(S̃, I) = max_{i,j} |S̃_{ij} - I_{ij}| finds the single worst element mismatch across all pairs. This ensures even one badly modelled feature interaction contributes strongly to the fitness penalty.

**Q: What if someone asks why you didn't run MIGA on the large Adult or Cardio datasets?**
> Adult (n=48,842) and Cardio (n=70,000): MIGA's O(n × p × ℓ × G) runtime scales with n. On a standard CPU at these sizes, a single MIGA run would take many hours. The original paper tested them and found similar results to the five I studied. I focused on the five datasets where I could run the full 10-seed Wilcoxon analysis within a reasonable compute budget (~200 seconds × 10 seeds × 5 datasets ≈ 3 hours).

---

*End of preparation guide.*
*Estimated talk time: 17 minutes.*
*Estimated preparation time: Read this once (2 hours). Drill Q&A out loud (2–3 hours). Especially: practice explaining each PLAIN ENGLISH section without reading it.*
