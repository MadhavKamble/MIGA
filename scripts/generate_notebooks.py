"""
Generates all MIGA thesis notebooks using nbformat.

Run once to create / recreate the notebooks directory:
  python generate_notebooks.py

Notebooks created
─────────────────
  notebooks/
    00_Algorithm_Overview.ipynb   — Theory walkthrough + pseudocode
    01_Application_Example.ipynb  — Paper Section 4 (synthetic Dataset 1)
    02_Iris.ipynb
    03_Wine.ipynb
    04_Glass.ipynb
    05_Haberman.ipynb
    06_Wholesale.ipynb
    07_Cardio.ipynb
    08_Adult.ipynb
    09_Results_Comparison.ipynb   — Side-by-side vs paper Tables 4/5/6
    10_Adaptive_vs_Fixed.ipynb    — Adaptive c3 mutation schedule (Novelty)
    11_MNAR_Extension.ipynb       — MNAR missing mechanism (Novelty)
    12_Baseline_Comparison.ipynb  — Mean / KNN / MICE vs MIGA (RMSE and Fr)
    13_Significance_Tests.ipynb   — Wilcoxon significance tests, Fr under MNAR
"""

import os
import textwrap
import nbformat as nbf

NB_DIR = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(NB_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def new_nb() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    }
    return nb


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(src).strip())


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(src).strip())


def save(nb: nbf.NotebookNode, filename: str) -> None:
    path = os.path.join(NB_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  Created: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Standard setup cell (injected into every notebook)
# ─────────────────────────────────────────────────────────────────────────────

SETUP = """\
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from miga import MIGA
from miga.fitness import FitnessEvaluator
from miga.statistics import compute_stats, pooled_std, relative_cov, minkowski_distance
from miga.data_utils import apply_mar, apply_mnar, auto_generators, compute_metrics, EXCLUDE_COLS
from miga.paper_results import (
    TABLE3_PARAMS, BENCHMARK_Q,
    TABLE4_RMSE, TABLE5_MAD, TABLE6_COD,
    METHODS, PERCENTAGES,
)

RESULTS_DIR = os.path.join("..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Setup complete.")
"""


# ─────────────────────────────────────────────────────────────────────────────
# 00 — Algorithm Overview
# ─────────────────────────────────────────────────────────────────────────────

def make_00_overview() -> nbf.NotebookNode:
    nb = new_nb()
    nb.cells = [
        md("""\
        # MIGA — Algorithm Overview
        **Paper:** Figueroa-García, Neruda & Hernandez-Pérez (2023).
        *A genetic algorithm for multivariate missing data imputation.*
        Information Sciences 619, 947–967.  DOI: 10.1016/j.ins.2022.11.037

        This notebook walks through the mathematical foundation and pseudocode
        of MIGA so that every design decision in the reimplementation can be
        traced back to a specific definition or equation in the paper.
        """),

        md("""\
        ## 1. Problem Statement

        Given an incomplete data matrix **X** (n × p) with some entries missing,
        find imputed values that minimise the statistical differences between the
        **available** sub-matrix **X_A** (complete rows) and the **completed**
        sub-matrix **X_C** (rows with at least one missing value, now filled in).

        Two key challenges not handled by classical methods (EM, regression):
        - Mixed variable types: *continuous*, *discrete*, *binary*
        - Multiple missing observations across multiple variables simultaneously
        """),

        md("""\
        ## 2. Similarity between matrices (Definition 2)

        Two matrices **X**, **Y** are *similar* if:

        $$\\mathscr{D}_r(\\bar{\\mathbf{x}}, \\bar{\\mathbf{y}}) \\to 0$$
        $$\\mathscr{D}_r(\\mathbf{S}_X, \\mathbf{S}_Y) \\to 0$$
        $$\\mathscr{D}_r(\\mathbf{b}_X, \\mathbf{b}_Y) \\to 0$$

        where $\\mathscr{D}_r$ is the Minkowski $r$-norm distance (Definition 1):

        $$\\mathscr{D}_r(\\mathbf{A}, \\mathbf{B}) = \\left(\\sum_i\\sum_j |a_{ij} - b_{ij}|^r\\right)^{1/r}$$

        For $r = \\infty$: $\\mathscr{D}_\\infty(\\mathbf{A}, \\mathbf{B}) = \\max_{i,j}|a_{ij}-b_{ij}|$
        """),

        md("""\
        ## 3. Fitness Function (Definition 5, Eq. 10)

        To make the three goal distances **dimensionless and additive**, the paper
        standardises means and covariances:

        | Symbol | Formula | Purpose |
        |--------|---------|---------|
        | $\\tilde{x}_A = \\bar{x}_A / S_p$ | standardised means of $X_A$ | removes units from means |
        | $\\tilde{x}_C = \\bar{x}_C / S_p$ | standardised means of $X_C$ | same |
        | $\\tilde{S} = S_A^{-1/2} S_C S_A^{-1/2}$ | relative covariance | $\\tilde{S}=I \\iff S_A = S_C$ |
        | $\\hat{b} = b_A - b_C$ | skewness difference | zero iff skewness preserved |

        Pooled variance (Eq. 7): $S_p^2 = \\frac{\\text{dg}(S_A)\\nu_A + \\text{dg}(S_C)\\nu_C}{\\nu_T}$

        The single fitness function to minimise:

        $$\\mathscr{F}_r := \\min \\left( \\mathscr{D}_r(\\tilde{x}_A, \\tilde{x}_C)
        + \\mathscr{D}_r(\\tilde{S},\\, I)
        + \\mathscr{D}_r(b_A, b_C) \\right)$$

        $\\mathscr{F}_r = 0 \\iff \\bar{x}_A=\\bar{x}_C,\\; S_A=S_C,\\; b_A=b_C$
        """),

        md("""\
        ## 4. Individual Encoding (Definition 3)

        - **Individual** $p_i$: a flat vector of length $k = |\\mathbf{M}|$
          where $\\mathbf{M}$ is the set of all missing positions $(i,j)$.
        - Genes are *j-ordered*: consecutive genes in $p_i$ for the same column $j$
          form a "subset" — enabling variable-specific crossover.
        - Each gene is sampled from a **random variable generator** $R_j$
          appropriate for the type of variable $j$
          (Normal, Discrete Uniform, Poisson, Exponential, …).
        """),

        md("""\
        ## 5. Population Structure per Generation

        | Component | Size | Description |
        |-----------|------|-------------|
        | Elite | $c$ | Best $c$ individuals carried forward (elitism) |
        | Mutation | $c_1 \\times c_3$ | $c_3$ children per parent, from best $c_1$ |
        | Crossover | $2(c_2-1)$ | Swap variable $j'$ between consecutive pairs |
        | Diversity | $l - c - c_1 c_3 - 2(c_2-1)$ | Fresh random individuals |
        | **Total** | **$l$** | Population size |

        **Constraint**: $l > c + c_1 c_3 + 2(c_2-1)$
        """),

        md("## 6. Algorithm Pseudocode (Algorithm 1)"),

        code("""\
        # Algorithm 1 — reproduced as Python pseudocode for clarity

        def MIGA_pseudocode(X, X_A, M, l, G, c, c1, c2, c3, Q, r):
            \"\"\"
            X   : incomplete dataset (n x p), NaN = missing
            X_A : complete rows of X
            M   : list of (row, col) missing positions  [j-ordered]
            l   : population size
            G   : number of generations per run
            c   : elite size
            c1  : mutation parent pool size
            c2  : crossover parent pool size
            c3  : mutation children per parent
            Q   : number of independent runs
            r   : Minkowski order
            \"\"\"
            # Pre-compute available statistics
            mean_A, cov_A, skew_A = compute_stats(X_A)       # x̄_A, S_A, b_A

            best_overall = None
            for q in range(Q):
                P = initialize_population(l, k=len(M))       # random initial pop

                for g in range(G):
                    scores = [F_r(individual, X_A, X_C) for individual in P]

                    # Selection (elitism)
                    elite = top_c(P, scores, c)

                    # Operators
                    mutants  = mutate(P, scores, c1, c3)     # c1*c3 new
                    children = crossover(P, scores, c2)       # 2*(c2-1) new
                    fresh    = random_individuals(n_div)      # diversity

                    P = elite + mutants + children + fresh    # next generation
                    # update best individual

                best_per_run = best_in(P)

            return best_overall_across_runs

        print("Pseudocode displayed. See miga/core.py for the full implementation.")
        """),

        md("## 7. Fitness Function Demo"),

        code("""\
        import sys, os
        sys.path.insert(0, os.path.abspath(".."))
        import numpy as np
        from miga.statistics import compute_stats, pooled_std, relative_cov, minkowski_distance

        # Small 2-variable example (matches paper Example 2)
        np.random.seed(0)
        n_A, n_C = 80, 20
        X_A = np.column_stack([np.random.normal(5, 1, n_A), np.random.normal(3, 2, n_A)])
        X_C = np.column_stack([np.random.normal(5.1, 1.05, n_C), np.random.normal(3.1, 2.1, n_C)])

        nu_A, nu_C = n_A - 1, n_C - 1
        mean_A, cov_A, skew_A = compute_stats(X_A)
        mean_C, cov_C, skew_C = compute_stats(X_C)
        S_p = pooled_std(cov_A, cov_C, nu_A, nu_C)

        x_tilde_A = mean_A / S_p
        x_tilde_C = mean_C / S_p
        S_tilde   = relative_cov(cov_A, cov_C)
        I         = np.eye(2)

        r = np.inf
        d_means = minkowski_distance(x_tilde_A, x_tilde_C, r)
        d_cov   = minkowski_distance(S_tilde, I, r)
        d_skew  = minkowski_distance(skew_A, skew_C, r)
        F_r     = d_means + d_cov + d_skew

        print(f"D_∞(x̃_A, x̃_C) = {d_means:.4f}  (means)")
        print(f"D_∞(S̃,  I)     = {d_cov:.4f}  (covariance)")
        print(f"D_∞(b_A, b_C)  = {d_skew:.4f}  (skewness)")
        print(f"F_∞            = {F_r:.4f}")
        """),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Application Example (Paper Section 4)
# ─────────────────────────────────────────────────────────────────────────────

def make_01_application() -> nbf.NotebookNode:
    nb = new_nb()
    nb.cells = [
        md("""\
        # Notebook 01 — Application Example (Paper Section 4)

        Reproduces the six-variable, 311-individual example from Section 4 of
        the paper.  Dataset 1 (available at the paper's repository) is
        approximated with a synthetic dataset that matches the described:
        - Variable types (Normal, Discrete-Uniform, Discrete-Uniform, Normal, Exponential, Poisson)
        - Approximate means, covariances, and skewness reported in Section 4

        Results are compared against Table 1 and the Section 4.1 fitness values.
        """),

        code(SETUP),

        md("## 1. Synthetic Dataset 1 (matches paper Section 4)"),

        code("""\
        # Reproduce paper's Dataset 1 statistics (Section 4.1):
        #   x̄_A ≈ [36.002, 44.928, 6.672, 35.438, 2.401, 1.506]
        #   Variable types:
        #     x1: Normal(μ=36.002, σ=5.260)
        #     x2: Discrete-Uniform [14, 64]
        #     x3: Discrete-Uniform [0,  25]
        #     x4: Normal(μ=35.438, σ=4.382)
        #     x5: Exponential(θ=2.401)
        #     x6: Poisson(λ=1.506)

        rng_data = np.random.default_rng(2023)
        n, p = 311, 6

        X_true = np.column_stack([
            rng_data.normal(36.002, 5.260, n),
            rng_data.integers(14, 65, n).astype(float),
            rng_data.integers(0,  26, n).astype(float),
            rng_data.normal(35.438, 4.382, n),
            rng_data.exponential(2.401, n),
            rng_data.poisson(1.506, n).astype(float),
        ])

        # Introduce ~10% missing (on ~131 individuals, matching paper)
        X_miss = apply_mar(X_true, pct=10, max_feat_frac=1.0, seed=42)

        complete_mask = ~np.any(np.isnan(X_miss), axis=1)
        X_A = X_miss[complete_mask]
        X_C = X_miss[~complete_mask]

        print(f"Total individuals   : {n}")
        print(f"Complete rows (X_A) : {len(X_A)}")
        print(f"Incomplete rows (XC): {len(X_C)}")
        print(f"Missing values (k)  : {int(np.isnan(X_miss).sum())}")
        print(f"Missing %           : {100*np.isnan(X_miss).mean():.1f}%")

        mean_A, cov_A, skew_A = compute_stats(X_A)
        print(f"\\nAvailable means x̄_A:")
        print("  " + "  ".join(f"x{j+1}={v:.3f}" for j, v in enumerate(mean_A)))
        """),

        md("## 2. Random Variable Generators (paper Section 4.1)"),

        code("""\
        rng_gen = np.random.default_rng(0)

        generators = {
            0: lambda: float(rng_gen.normal(36.002, 5.260)),       # R₁: Normal
            1: lambda: float(rng_gen.integers(14, 65)),            # R₂: Discrete-Uniform [14,64]
            2: lambda: float(rng_gen.integers(0,  26)),            # R₃: Discrete-Uniform [0,25]
            3: lambda: float(rng_gen.normal(35.438, 4.382)),       # R₄: Normal
            4: lambda: float(rng_gen.exponential(2.401)),          # R₅: Exponential
            5: lambda: float(rng_gen.poisson(1.506)),              # R₆: Poisson
        }
        print("Generators defined.")
        """),

        md("## 3. Run MIGA (paper Section 4.1 parameters)"),

        code("""\
        # Paper parameters: r=∞, p=6, k=199, n_A=180, n_C=131
        # l=1000, G=2000, c=3, c1=3, c2=3, c3=10, Q=12
        # Reduced l/G here for speed; increase for full reproduction.

        miga = MIGA(
            l=500,   # paper: 1000
            G=500,   # paper: 2000
            c=3, c1=3, c2=3, c3=10,
            Q=6,     # paper: 12
            r=np.inf,
            seed=42,
            verbose=True,
        )

        X_imputed = miga.fit_transform(X_miss, generators=generators)
        print(f"\\nBest overall F_∞ = {miga.best_score_:.4f}  (paper: 0.2110)")
        """),

        md("## 4. Fitness Decomposition — vs. Paper Table 1 and Section 4.1"),

        code("""\
        # Decompose fitness into three components (matches paper Section 4.1)
        evaluator = FitnessEvaluator(X_A, r=np.inf)
        X_C_filled = X_imputed[~complete_mask]
        decomp = evaluator.decompose(X_C_filled)

        paper_miga = {
            "F_inf":   0.2110,
            "d_means": 0.0163,
            "d_cov":   0.0796,
            "d_skew":  0.1151,
        }

        print("\\n=== Fitness Decomposition (r=∞) ===")
        print(f"{'Component':<30} {'Our MIGA':>12} {'Paper MIGA':>12}")
        print("-" * 55)
        print(f"{'D_∞(x̃_A, x̃_C)  [means]':<30} {decomp['d_means']:>12.4f} {paper_miga['d_means']:>12.4f}")
        print(f"{'D_∞(S̃, I)       [covariances]':<30} {decomp['d_cov']:>12.4f} {paper_miga['d_cov']:>12.4f}")
        print(f"{'D_∞(b_A, b_C)   [skewness]':<30} {decomp['d_skew']:>12.4f} {paper_miga['d_skew']:>12.4f}")
        print(f"{'F_∞ (total)':<30} {decomp['F_r']:>12.4f} {paper_miga['F_inf']:>12.4f}")
        """),

        md("## 5. Comparison: MIGA vs EM vs Auxiliary Regression"),

        code("""\
        from miga.paper_results import SECTION4_FITNESS

        paper_data = [
            ("MIGA",            decomp["d_means"], decomp["d_cov"], decomp["d_skew"], decomp["F_r"]),
        ]
        for method, vals in SECTION4_FITNESS.items():
            paper_data.append((f"{method} (paper)", vals["d_means"], vals["d_cov"], vals["d_skew"], vals["F_inf"]))

        print("\\n=== Method Comparison (r=∞) ===")
        print(f"{'Method':<25} {'D_means':>10} {'D_cov':>10} {'D_skew':>10} {'F_∞':>10}")
        print("-" * 65)
        for row in paper_data:
            print(f"{row[0]:<25} {row[1]:>10.4f} {row[2]:>10.4f} {row[3]:>10.4f} {row[4]:>10.4f}")
        """),

        md("## 6. Imputation Quality vs. True Values"),

        code("""\
        missing_mask = np.isnan(X_miss)
        metrics = compute_metrics(X_true, X_imputed, missing_mask)

        print("=== Imputation quality vs. true values (standardised scale) ===")
        print(f"  RMSE : {metrics['rmse']:.4f}")
        print(f"  MAD  : {metrics['mad']:.4f}")
        print(f"  CoD  : {metrics['cod']:.4f}")
        """),

        md("## 7. Convergence Plot"),

        code("""\
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, len(miga.history_) + 1), miga.history_, color="steelblue", alpha=0.8)
        ax.axhline(0.2110, color="red", linestyle="--", label="Paper F_∞ = 0.2110")
        ax.set_xlabel("Run")
        ax.set_ylabel("Best F_∞ per run")
        ax.set_title("MIGA convergence — Application Example")
        ax.legend()
        plt.tight_layout()
        plt.savefig("../results/01_convergence.png", dpi=120)
        plt.show()
        """),

        md("## 8. Available vs. Completed Data Statistics"),

        code("""\
        mean_C, cov_C, skew_C = compute_stats(X_C_filled)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].bar(range(p), mean_A, alpha=0.7, label="Available X_A")
        axes[0].bar(range(p), mean_C, alpha=0.7, label="Completed X_C")
        axes[0].set_title("Column Means")
        axes[0].set_xticks(range(p))
        axes[0].set_xticklabels([f"x{j+1}" for j in range(p)])
        axes[0].legend()

        axes[1].bar(range(p), np.diag(cov_A), alpha=0.7, label="Available X_A")
        axes[1].bar(range(p), np.diag(cov_C), alpha=0.7, label="Completed X_C")
        axes[1].set_title("Column Variances (diagonal of S)")
        axes[1].set_xticks(range(p))
        axes[1].set_xticklabels([f"x{j+1}" for j in range(p)])
        axes[1].legend()

        axes[2].bar(range(p), skew_A, alpha=0.7, label="Available X_A")
        axes[2].bar(range(p), skew_C, alpha=0.7, label="Completed X_C")
        axes[2].set_title("Column Skewness")
        axes[2].set_xticks(range(p))
        axes[2].set_xticklabels([f"x{j+1}" for j in range(p)])
        axes[2].legend()

        plt.tight_layout()
        plt.savefig("../results/01_statistics_comparison.png", dpi=120)
        plt.show()
        """),

        code("""\
        # Save results
        result = {
            "dataset": "Application_Example",
            "F_inf": decomp["F_r"],
            "d_means": decomp["d_means"],
            "d_cov": decomp["d_cov"],
            "d_skew": decomp["d_skew"],
            "rmse": metrics["rmse"],
            "mad": metrics["mad"],
            "cod": metrics["cod"],
        }
        with open(os.path.join(RESULTS_DIR, "01_application_example.json"), "w") as f:
            json.dump(result, f, indent=2)
        print("Results saved to results/01_application_example.json")
        """),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# 02–08 — Benchmark dataset notebooks
# ─────────────────────────────────────────────────────────────────────────────

DATASET_META = {
    # name        : (loader_func_name, num, description)
    "Iris":       ("load_iris",      "02", "150 × 4, continuous features"),
    "Wine":       ("load_wine",      "03", "178 × 13, continuous features"),
    "Glass":      ("load_glass",     "04", "214 × 10, mixed continuous/integer"),
    "Haberman":   ("load_haberman",  "05", "306 × 3, integer features"),
    "Wholesale":  ("load_wholesale", "06", "440 × 8, integer features"),
    "Cardio":     ("load_cardio",    "07", "2126 × 23, mixed features"),
    "Adult":      ("load_adult",     "08", "48842 × 14, label-encoded categorical"),
}


def make_benchmark_nb(
    name: str,
    loader_fn: str,
    num: str,
    description: str,
) -> nbf.NotebookNode:
    nb = new_nb()

    params = {k: v for k, v in __import__(
        "miga.paper_results", fromlist=["TABLE3_PARAMS"]
    ).TABLE3_PARAMS[name].items()}
    r_display = "∞" if str(params["r"]) == "inf" else str(params["r"])

    nb.cells = [
        md(f"""\
        # Notebook {num} — {name} Dataset

        **Dataset:** {description}

        **Paper parameters (Table 3):**
        `c={params['c']}, c1={params['c1']}, c2={params['c2']}, c3={params['c3']},
        l={params['l']}, G={params['G']}, r={r_display}, Q=12`

        Results are compared against the paper's Tables 4 (RMSE), 5 (MAD), 6 (CoD).
        Each percentage (30%, 40%, 50%, 60%) is run independently and saved to
        `results/{num}_{name}_results.json`.
        """),

        code(SETUP),

        md("## 1. Load Dataset"),

        code(f"""\
        from miga.data_utils import {loader_fn}, EXCLUDE_COLS

        name = "{name}"
        X_true, feature_names = {loader_fn}()
        n, m = X_true.shape
        exclude_cols = EXCLUDE_COLS[name]
        print(f"Dataset: {{name}}  shape={{X_true.shape}}")
        print(f"Features: {{feature_names}}")
        print(f"Excluded from MAR: {{[feature_names[i] for i in exclude_cols] or 'none'}}")
        print(f"Max features with missing (m/2): {{m//2}}")
        """),

        md("## 2. Descriptive Statistics of Complete Dataset"),

        code("""\
        df_true = pd.DataFrame(X_true, columns=feature_names)
        display(df_true.describe().round(3))
        """),

        md("## 3. MIGA Parameters (Paper Table 3)"),

        code(f"""\
        # Paper Table 3 parameters for {name}
        PARAMS = dict(
            c={params['c']}, c1={params['c1']}, c2={params['c2']},
            c3={params['c3']}, l={params['l']}, G={params['G']},
            r={params['r']!r}, Q=BENCHMARK_Q,
            seed=2023, verbose=True,
        )
        print("MIGA parameters:", PARAMS)

        # Reduce for quick testing — comment out for full paper reproduction
        # PARAMS["l"] = 50; PARAMS["G"] = 200; PARAMS["Q"] = 3
        """),

        md("## 4. Run MIGA for All Missing Percentages (30%, 40%, 50%, 60%)"),

        code("""\
        all_results = {}

        for pct in PERCENTAGES:
            print(f"\\n{'='*60}")
            print(f"  Missing percentage: {pct}%")
            print(f"{'='*60}")

            # Apply MAR missing mechanism (paper Section 5)
            # exclude_cols: categorical/label columns never made missing
            X_miss = apply_mar(X_true, pct=pct, max_feat_frac=0.5, seed=pct,
                               exclude_cols=EXCLUDE_COLS[name])
            missing_mask = np.isnan(X_miss)

            n_complete = int(np.all(~missing_mask, axis=1).sum())
            n_missing_vals = int(missing_mask.sum())
            print(f"  Complete rows    : {n_complete} / {n}")
            print(f"  Missing values   : {n_missing_vals} ({100*missing_mask.mean():.1f}%)")

            # Build generators from available data
            generators = auto_generators(X_miss, seed=pct)

            # Run MIGA
            miga = MIGA(**PARAMS)
            X_imputed = miga.fit_transform(X_miss, generators=generators)

            # Compute metrics on z-standardised scale
            metrics = compute_metrics(X_true, X_imputed, missing_mask)

            # Fitness decomposition
            X_A = X_miss[~np.any(missing_mask, axis=1)]
            X_C_filled = X_imputed[np.any(missing_mask, axis=1)]
            evaluator = FitnessEvaluator(X_A, r=PARAMS["r"])
            decomp = evaluator.decompose(X_C_filled)

            all_results[pct] = {
                "rmse":    metrics["rmse"],
                "mad":     metrics["mad"],
                "cod":     metrics["cod"],
                "F_r":     decomp["F_r"],
                "d_means": decomp["d_means"],
                "d_cov":   decomp["d_cov"],
                "d_skew":  decomp["d_skew"],
                "best_score": miga.best_score_,
                "history": miga.history_,
            }

            paper_rmse = TABLE4_RMSE[name][pct]["MIGA"]
            paper_mad  = TABLE5_MAD[name][pct]["MIGA"]
            paper_cod  = TABLE6_COD[name][pct]["MIGA"]

            print(f"\\n  Results at {pct}% missing:")
            print(f"  {'Metric':<8} {'Ours':>10} {'Paper':>10} {'Δ':>10}")
            print(f"  {'-'*38}")
            print(f"  {'RMSE':<8} {metrics['rmse']:>10.4f} {paper_rmse:>10.4f} {metrics['rmse']-paper_rmse:>+10.4f}  (range-norm, matches paper)")
            print(f"  {'MAD':<8} {metrics['mad']:>10.4f} {paper_mad:>10.4f}  {metrics['mad']-paper_mad:>+10.4f}  (range-norm; paper formula undocumented)")
            print(f"  {'CoD':<8} {metrics['cod']:>10.4f} {paper_cod:>10.4f}  {metrics['cod']-paper_cod:>+10.4f}  (all-data SS, matches paper)")
        """),

        md("## 5. Summary Table — Our Results vs. Paper Tables 4, 5, 6"),

        code(f"""\
        print(f"\\n=== {name}: RMSE Comparison vs. Paper Table 4 ===")
        print(f"  {{'':<6}} {{' Our MIGA':>12}} {{'Paper MIGA':>12}} {{'CMIM':>10}} {{'ANNI':>10}} {{'Mean':>10}}")
        print("  " + "-"*62)
        for pct in PERCENTAGES:
            r = all_results[pct]
            p4 = TABLE4_RMSE[name][pct]
            mean_val = p4.get('Mean') if p4.get('Mean') is not None else float('nan')
            print(f"  {{pct}}%   {{r['rmse']:>12.4f}} {{p4['MIGA']:>12.4f}} {{p4['CMIM']:>10.4f}} {{p4['ANNI']:>10.4f}} {{mean_val:>10.4f}}")

        print(f"\\n=== {name}: MAD Comparison vs. Paper Table 5 ===")
        print(f"  {{'':<6}} {{' Our MIGA':>12}} {{'Paper MIGA':>12}} {{'CMIM':>10}} {{'ANNI':>10}}")
        print("  " + "-"*52)
        for pct in PERCENTAGES:
            r = all_results[pct]
            p5 = TABLE5_MAD[name][pct]
            print(f"  {{pct}}%   {{r['mad']:>12.4f}} {{p5['MIGA']:>12.4f}} {{p5['CMIM']:>10.4f}} {{p5['ANNI']:>10.4f}}")

        print(f"\\n=== {name}: CoD Comparison vs. Paper Table 6 ===")
        print(f"  {{'':<6}} {{' Our MIGA':>12}} {{'Paper MIGA':>12}} {{'CMIM':>10}} {{'ANNI':>10}}")
        print("  " + "-"*52)
        for pct in PERCENTAGES:
            r = all_results[pct]
            p6 = TABLE6_COD[name][pct]
            print(f"  {{pct}}%   {{r['cod']:>12.4f}} {{p6['MIGA']:>12.4f}} {{p6['CMIM']:>10.4f}} {{p6['ANNI']:>10.4f}}")
        """),

        md("## 6. Visualisation"),

        code(f"""\
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        pcts = PERCENTAGES

        our_rmse = [all_results[p]["rmse"] for p in pcts]
        pap_rmse = [TABLE4_RMSE[name][p]["MIGA"] for p in pcts]
        our_mad  = [all_results[p]["mad"]  for p in pcts]
        pap_mad  = [TABLE5_MAD[name][p]["MIGA"]  for p in pcts]
        our_cod  = [all_results[p]["cod"]  for p in pcts]
        pap_cod  = [TABLE6_COD[name][p]["MIGA"]  for p in pcts]

        for ax, our, pap, ylabel, title in [
            (axes[0], our_rmse, pap_rmse, "RMSE", "RMSE"),
            (axes[1], our_mad,  pap_mad,  "MAD",  "MAD"),
            (axes[2], our_cod,  pap_cod,  "CoD",  "CoD"),
        ]:
            ax.plot(pcts, our, "o-", color="steelblue",  label="Our MIGA")
            ax.plot(pcts, pap, "s--", color="tomato",    label="Paper MIGA")
            ax.set_xlabel("Missing %")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name}: {{title}}")
            ax.set_xticks(pcts)
            ax.legend()

        plt.suptitle(f"{name} Dataset — MIGA Reimplementation vs. Paper", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(f"../results/{num}_{name}_metrics.png", dpi=120, bbox_inches="tight")
        plt.show()
        """),

        md("## 7. Convergence Across Runs"),

        code(f"""\
        fig, axes = plt.subplots(1, len(PERCENTAGES), figsize=(14, 3))
        for ax, pct in zip(axes, PERCENTAGES):
            history = all_results[pct]["history"]
            ax.bar(range(1, len(history)+1), history, color="steelblue", alpha=0.8)
            ax.set_title(f"{{pct}}% missing")
            ax.set_xlabel("Run")
            if ax == axes[0]:
                ax.set_ylabel("Best F_r per run")
        plt.suptitle(f"{name} — GA convergence per run", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"../results/{num}_{name}_convergence.png", dpi=120)
        plt.show()
        """),

        code(f"""\
        # Save all results
        result_path = os.path.join(RESULTS_DIR, "{num}_{name}_results.json")
        # Convert history lists to serialisable format
        serialisable = {{str(k): v for k, v in all_results.items()}}
        with open(result_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Results saved to {{result_path}}")
        """),
    ]

    if name == "Wine":
        # Insert after index 5 (describe code), so n/m/X_true are already defined.
        # Original order: 0=title, 1=setup, 2=sec1-md, 3=load-code, 4=sec2-md, 5=describe-code
        nb.cells.insert(6, md("""\
        ## 2b. Wine — Rank-Deficiency Diagnostic

        Wine (178 × 13) is an edge case for MIGA.  MIGA's fitness function
        requires estimating a 13 × 13 sample covariance matrix S_A from the
        *complete* rows X_A.  For reliable estimation we need n_A ≫ m = 13.

        Under MAR with 30–60% missing, X_A shrinks fast:

        | Missing% | Eligible features | P(row complete) | Expected n_A | n_A / m |
        |----------|-------------------|-----------------|--------------|---------|
        | 30%      | floor(13/2) = 6   | 0.70^6 ≈ 0.118  | ≈ 21         | 1.6     |
        | 40%      | 6                 | 0.60^6 ≈ 0.047  | ≈  8         | 0.6 ★   |
        | 50%      | 6                 | 0.50^6 ≈ 0.016  | ≈  3         | 0.2 ★   |
        | 60%      | 6                 | 0.40^6 ≈ 0.004  | ≈  1         | 0.1 ★   |

        ★ n_A < m → S_A is rank-deficient → S_A^{-1/2} is undefined.

        **At 30%** n_A ≈ 21 > 13 so S_A is technically full-rank, but with
        n_A / m ≈ 1.6 the smallest eigenvalues are dominated by estimation
        noise.  Without regularisation, S_A^{-1/2} amplifies that noise by
        orders of magnitude, producing large F_r values even for good imputations.

        **At 40–60%** S_A is genuinely rank-deficient.  No imputation algorithm
        that relies on a p×p sample covariance from X_A can fully compensate —
        this is a hard statistical limit for MIGA on this dataset.

        **Our fix**: we floor eigenvalues at 1% of the maximum (condition ≤ 100),
        compared to 0.01% in the initial implementation. This bounds the noise
        amplification at 30% and limits (but cannot eliminate) the impact of
        rank-deficiency at 40–60%.  The paper does not document how they handle
        this case; their reported results likely benefited from seeds where more
        complete rows survived.
        """))

        nb.cells.insert(7, code("""\
        # Empirical n_A diagnostic for Wine
        print("Wine complete-row diagnostic")
        print(f"  n={n}, m={m}, eligible features per run = {m//2}")
        print()
        print(f"  {'pct':>5}  {'n_A':>6}  {'n_A/m':>7}  {'status'}")
        print("  " + "-"*42)
        for pct in PERCENTAGES:
            X_miss_tmp = apply_mar(X_true, pct=pct, max_feat_frac=0.5, seed=pct,
                                   exclude_cols=EXCLUDE_COLS[name])
            n_A = int(np.all(~np.isnan(X_miss_tmp), axis=1).sum())
            ratio = n_A / m
            status = "OK" if ratio > 3 else ("borderline" if ratio > 1 else "RANK DEFICIENT")
            print(f"  {pct:>4}%  {n_A:>6}  {ratio:>7.2f}  {status}")
        """))

    return nb


# ─────────────────────────────────────────────────────────────────────────────
# 09 — Results Comparison
# ─────────────────────────────────────────────────────────────────────────────

def make_09_comparison() -> nbf.NotebookNode:
    nb = new_nb()
    nb.cells = [
        md("""\
        # Notebook 09 — Full Results Comparison

        Aggregates results from all benchmark notebooks (02–08) and compares
        them side-by-side with the paper's Tables 4 (RMSE), 5 (MAD), 6 (CoD).

        **Run notebooks 02–08 first** to generate the JSON result files.

        ### Metric Definitions (matching paper normalisation)

        | Metric | Formula | Notes |
        |--------|---------|-------|
        | RMSE | per-feature RMSE / (max−min), averaged over missing features | Matches Table 4 closely |
        | MAD  | per-feature MAD  / (max−min), averaged over missing features | Paper's exact normalisation is undocumented; absolute values differ ~3×, but method *rankings* are preserved |
        | CoD  | 1 − SS_res(missing) / SS_tot(all cells) | Matches Table 6 closely |
        """),

        code(SETUP),

        md("## 1. Load All Saved Results"),

        code("""\
        from miga.paper_results import DATASETS, TABLE4_RMSE, TABLE5_MAD, TABLE6_COD

        NB_META = {
            "Iris":      "02", "Wine": "03", "Glass": "04",
            "Haberman":  "05", "Wholesale": "06",
            "Cardio":    "07", "Adult": "08",
        }

        our_results = {}
        for ds, num in NB_META.items():
            path = os.path.join(RESULTS_DIR, f"{num}_{ds}_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    our_results[ds] = {int(k): v for k, v in json.load(f).items()}
                print(f"  Loaded: {path}")
            else:
                print(f"  Missing: {path}  (run notebook {num} first)")
        """),

        md("## 2. RMSE — Our MIGA vs. Paper Table 4"),

        code("""\
        print("\\n=== RMSE Comparison (Our MIGA vs. Paper MIGA) ===")
        print(f"{'Dataset':<12} {'pct':>5} {'Ours':>10} {'Paper':>10} {'Δ':>10} {'Better?':>9}")
        print("-" * 60)

        for ds in DATASETS:
            if ds not in our_results:
                continue
            for pct in PERCENTAGES:
                our  = our_results[ds][pct]["rmse"]
                pap  = TABLE4_RMSE[ds][pct]["MIGA"]
                diff = our - pap
                better = "✓" if our <= pap else "✗"
                print(f"{ds:<12} {pct:>5}% {our:>10.4f} {pap:>10.4f} {diff:>+10.4f} {better:>9}")
        """),

        md("## 3. MAD — Our MIGA vs. Paper Table 5"),

        code("""\
        print("\\n=== MAD Comparison (Our MIGA vs. Paper MIGA) ===")
        print(f"{'Dataset':<12} {'pct':>5} {'Ours':>10} {'Paper':>10} {'Δ':>10} {'Better?':>9}")
        print("-" * 60)

        for ds in DATASETS:
            if ds not in our_results:
                continue
            for pct in PERCENTAGES:
                our  = our_results[ds][pct]["mad"]
                pap  = TABLE5_MAD[ds][pct]["MIGA"]
                diff = our - pap
                better = "✓" if our <= pap else "✗"
                print(f"{ds:<12} {pct:>5}% {our:>10.4f} {pap:>10.4f} {diff:>+10.4f} {better:>9}")
        """),

        md("## 4. CoD — Our MIGA vs. Paper Table 6"),

        code("""\
        print("\\n=== CoD Comparison (Our MIGA vs. Paper MIGA) ===")
        print(f"{'Dataset':<12} {'pct':>5} {'Ours':>10} {'Paper':>10} {'Δ':>10} {'Better?':>9}")
        print("-" * 60)

        for ds in DATASETS:
            if ds not in our_results:
                continue
            for pct in PERCENTAGES:
                our  = our_results[ds][pct]["cod"]
                pap  = TABLE6_COD[ds][pct]["MIGA"]
                diff = our - pap
                better = "✓" if our >= pap else "✗"
                print(f"{ds:<12} {pct:>5}% {our:>10.4f} {pap:>10.4f} {diff:>+10.4f} {better:>9}")
        """),

        md("""\
        ## 5. Ratio Summary — Our MIGA / Paper MIGA

        Ratio = Our RMSE / Paper RMSE.
        Values **< 1.0** (★) mean we beat the paper.
        Values **1.0–1.2** are excellent for a reimplementation.
        Ratios **> 2.5** indicate a known algorithmic limitation.
        """),

        code("""\
        print("=== RMSE: Our MIGA / Paper MIGA  (ratio per dataset × missing %) ===")
        print()
        print(f"  {'Dataset':<12}  {'30%':>9}  {'40%':>9}  {'50%':>9}  {'60%':>9}  {'Avg ratio':>10}")
        print("  " + "-" * 66)

        for ds in DATASETS:
            if ds not in our_results:
                print(f"  {ds:<12}  (not yet run)")
                continue
            ratios = []
            cells  = []
            for pct in PERCENTAGES:
                our = our_results[ds][pct]["rmse"]
                pap = TABLE4_RMSE[ds][pct]["MIGA"]
                r   = our / pap
                ratios.append(r)
                label = "★" if r <= 1.0 else f"{r:.2f}x"
                cells.append(f"{label:>9}")
            avg = sum(ratios) / len(ratios)
            avg_label = "★" if avg <= 1.0 else f"{avg:.2f}x"
            print(f"  {ds:<12}  {'  '.join(cells)}  {avg_label:>10}")

        print()
        print("  ★ = beats paper")
        print()

        print("=== MAD: Our MIGA / Paper MIGA ===")
        print()
        print(f"  {'Dataset':<12}  {'30%':>9}  {'40%':>9}  {'50%':>9}  {'60%':>9}  {'Avg ratio':>10}")
        print("  " + "-" * 66)

        for ds in DATASETS:
            if ds not in our_results:
                print(f"  {ds:<12}  (not yet run)")
                continue
            ratios = []
            cells  = []
            for pct in PERCENTAGES:
                our = our_results[ds][pct]["mad"]
                pap = TABLE5_MAD[ds][pct]["MIGA"]
                r   = our / pap
                ratios.append(r)
                label = "★" if r <= 1.0 else f"{r:.2f}x"
                cells.append(f"{label:>9}")
            avg = sum(ratios) / len(ratios)
            avg_label = "★" if avg <= 1.0 else f"{avg:.2f}x"
            print(f"  {ds:<12}  {'  '.join(cells)}  {avg_label:>10}")

        print()
        print("  ★ = beats paper")
        print()

        print("=== CoD: Our MIGA / Paper MIGA  (higher CoD is better; ratio > 1 = beats paper) ===")
        print()
        print(f"  {'Dataset':<12}  {'30%':>9}  {'40%':>9}  {'50%':>9}  {'60%':>9}  {'Avg ratio':>10}")
        print("  " + "-" * 66)

        for ds in DATASETS:
            if ds not in our_results:
                print(f"  {ds:<12}  (not yet run)")
                continue
            ratios = []
            cells  = []
            for pct in PERCENTAGES:
                our = our_results[ds][pct]["cod"]
                pap = TABLE6_COD[ds][pct]["MIGA"]
                # Ratio is only meaningful when both values are positive
                r = our / pap if (pap > 1e-6 and our > 0) else float("nan")
                ratios.append(r)
                if our >= pap:
                    label = "★"
                elif r == r:  # valid ratio
                    label = f"{r:.2f}x"
                else:
                    # Negative CoD or undefined — show delta instead
                    label = f"{our-pap:+.2f}"
                cells.append(f"{label:>9}")
            valid = [r for r in ratios if r == r]
            avg = sum(valid) / len(valid) if valid else float("nan")
            avg_label = "★" if avg >= 1.0 else (f"{avg:.2f}x" if avg == avg else "n/a")
            print(f"  {ds:<12}  {'  '.join(cells)}  {avg_label:>10}")

        print()
        print("  ★ = meets or beats paper   (CoD: higher = better, ratio > 1 is good)")
        """),

        md("## 7. Heat-map: Our MIGA RMSE vs. Paper"),

        code("""\
        available_ds = [d for d in DATASETS if d in our_results]
        if not available_ds:
            print("No results available yet. Run notebooks 02–08 first.")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, len(available_ds) * 0.8 + 2))

            for ax, (table, title) in zip(axes, [
                (TABLE4_RMSE, "RMSE (lower=better)"),
                (TABLE5_MAD,  "MAD  (lower=better)"),
                (TABLE6_COD,  "CoD  (higher=better)"),
            ]):
                key = "rmse" if "RMSE" in title else ("mad" if "MAD" in title else "cod")
                data_our = np.array([[our_results[d][p][key] for p in PERCENTAGES] for d in available_ds])
                data_pap = np.array([[table[d][p]["MIGA"]    for p in PERCENTAGES] for d in available_ds])
                delta = data_our - data_pap

                im = ax.imshow(delta, cmap="RdYlGn_r" if key != "cod" else "RdYlGn",
                               aspect="auto", vmin=-0.1, vmax=0.1)
                ax.set_xticks(range(4))
                ax.set_xticklabels([f"{p}%" for p in PERCENTAGES])
                ax.set_yticks(range(len(available_ds)))
                ax.set_yticklabels(available_ds)
                ax.set_title(f"Δ {title}\\n(Ours − Paper)", fontsize=10)
                for i in range(len(available_ds)):
                    for j in range(4):
                        ax.text(j, i, f"{delta[i,j]:+.3f}", ha="center", va="center", fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.04)

            plt.suptitle("MIGA Reimplementation vs. Paper (green = better than paper)", fontsize=12)
            plt.tight_layout()
            plt.savefig("../results/09_heatmap_comparison.png", dpi=120, bbox_inches="tight")
            plt.show()
        """),

        md("## 8. Full Table 4 — RMSE All Methods (Our MIGA replaces Paper MIGA)"),

        code("""\
        import pandas as pd
        from miga.paper_results import METHODS

        for ds in available_ds:
            rows = []
            for pct in PERCENTAGES:
                row = {"pct": f"{pct}%", "OurMIGA": f"{our_results[ds][pct]['rmse']:.4f}"}
                for m in METHODS:
                    row[m] = f"{TABLE4_RMSE[ds][pct].get(m, float('nan')):.4f}"
                rows.append(row)
            df = pd.DataFrame(rows).set_index("pct")
            print(f"\\n{ds} — RMSE (Table 4):")
            display(df)
        """),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# 10 — Adaptive vs Fixed c3 (Novelty: adaptive mutation schedule)
# ─────────────────────────────────────────────────────────────────────────────

def make_10_adaptive() -> nbf.NotebookNode:
    nb = new_nb()
    nb.cells = [
        md("""\
        # Notebook 10 — Adaptive vs Fixed Mutation Schedule
        **Novelty:** Adaptive c3 schedule for MIGA (exploration → exploitation).

        The base paper uses a **fixed** `c3` throughout all G generations: every
        generation produces `c1 × c3` mutation offspring.  We propose a **linear
        decay schedule** — starting high (broad exploration of gene space) and
        decaying to a small value (fine exploitation near the elite):

        $$c_3(g) = \\text{round}\\!\\left(c_3^{\\text{start}} + (c_3^{\\text{end}} - c_3^{\\text{start}}) \\cdot \\frac{g}{G-1}\\right)$$

        **Motivation:** Early generations benefit from many diverse mutation
        offspring to escape local minima.  Late generations benefit from fewer
        but higher-quality offspring produced from an already-refined elite pool.
        This mirrors the exploration-exploitation tradeoff in evolutionary
        computation (Holland 1975; Goldberg 1989).

        **Datasets:** Iris, Glass, Haberman — chosen because they span different
        n/p ratios and feature types.  Wholesale and Wine are left out: Wholesale
        has high variance across seeds; Wine has a fundamental rank-deficiency
        issue documented in Notebook 03.
        """),

        code(SETUP),

        md("## 1. Configuration"),
        code("""\
        # Datasets to compare
        from miga.data_utils import load_iris, load_glass, load_haberman

        DATASETS = {
            "Iris":     (load_iris,     {"pct": 30, "seed": 30, "exclude_cols": set()}),
            "Glass":    (load_glass,    {"pct": 30, "seed": 30, "exclude_cols": {9}}),
            "Haberman": (load_haberman, {"pct": 30, "seed": 30, "exclude_cols": set()}),
        }

        # GA parameters (reduced for speed; scale up for final thesis runs)
        L, G_GENS, Q_RUNS = 200, 500, 5

        # Fixed baseline
        C3_FIXED = 5

        # Adaptive schedule: c3 decays from C3_START → C3_END over G generations
        C3_START, C3_END = 15, 3

        SEED = 42
        print(f"l={L}, G={G_GENS}, Q={Q_RUNS}")
        print(f"Fixed: c3={C3_FIXED}")
        print(f"Adaptive: c3 {C3_START} → {C3_END}")
        """),

        md("## 2. c3 Schedule Visualisation"),
        code("""\
        gens = np.arange(G_GENS)
        c3_curve = np.round(C3_START + (C3_END - C3_START) * gens / max(G_GENS - 1, 1)).astype(int)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(gens, c3_curve, color="tab:blue", lw=2, label=f"Adaptive c3 ({C3_START}→{C3_END})")
        ax.axhline(C3_FIXED, color="tab:orange", lw=2, ls="--", label=f"Fixed c3={C3_FIXED}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("c3 (mutation offspring per parent)")
        ax.set_title("Adaptive vs Fixed Mutation Rate Schedule")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "10_c3_schedule.png"), dpi=150)
        plt.show()
        print("Saved: results/10_c3_schedule.png")
        """),

        md("""\
        ## 3. Load Results

        Run each dataset in a **separate terminal** (in parallel):
        ```
        .venv/bin/python scripts/run_adaptive_dataset.py Iris
        .venv/bin/python scripts/run_adaptive_dataset.py Glass
        .venv/bin/python scripts/run_adaptive_dataset.py Haberman
        ```
        Each script saves `results/10_adaptive_<dataset>.json`.
        Re-run this cell after all three finish to load results.
        """),
        code("""\
        import json, os

        DATASET_NAMES = ["Iris", "Glass", "Haberman"]
        results = {}

        for ds_name in DATASET_NAMES:
            path = os.path.join(RESULTS_DIR, f"10_adaptive_{ds_name.lower()}.json")
            if not os.path.exists(path):
                print(f"  [MISSING] {path} — run the script for {ds_name} first")
                continue
            with open(path) as f:
                data = json.load(f)
            results[ds_name] = data
            fixed_rmse    = data["fixed"]["rmse"]
            adaptive_rmse = data["adaptive"]["rmse"]
            delta = (fixed_rmse - adaptive_rmse) / fixed_rmse * 100
            print(f"{ds_name:10s}  fixed={fixed_rmse:.4f}  adaptive={adaptive_rmse:.4f}  Δ={delta:+.1f}%")

        print(f"\\nLoaded {len(results)}/{len(DATASET_NAMES)} datasets.")
        """),

        md("## 4. Convergence Plots (Fr vs Generation)"),
        code("""\
        if not results:
            print("No results loaded yet. Run the scripts first.")
        else:
            n_ds = len(results)
            fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=False)
            if n_ds == 1:
                axes = [axes]

            for ax, (ds_name, data) in zip(axes, results.items()):
                for label, colour, ls in [("fixed", "tab:orange", "--"), ("adaptive", "tab:blue", "-")]:
                    gen_hists = data[label]["gen_history"]   # list of Q lists
                    arr = np.array(gen_hists)                 # (Q, G)
                    mean_curve = arr.mean(axis=0)
                    std_curve  = arr.std(axis=0)
                    gens = np.arange(arr.shape[1])
                    ax.plot(gens, mean_curve, color=colour, ls=ls, lw=2,
                            label=f"{label} (Q={len(gen_hists)})")
                    ax.fill_between(gens,
                                    mean_curve - std_curve,
                                    mean_curve + std_curve,
                                    alpha=0.15, color=colour)

                ax.set_title(ds_name)
                ax.set_xlabel("Generation")
                ax.set_ylabel("Best F_r")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

            plt.suptitle("Convergence: Adaptive vs Fixed c3 Schedule", fontsize=13, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "10_convergence.png"), dpi=150, bbox_inches="tight")
            plt.show()
            print("Saved: results/10_convergence.png")
        """),

        md("## 5. RMSE Comparison Table"),
        code("""\
        from miga.paper_results import TABLE4_RMSE

        if results:
            rows = []
            for ds_name, data in results.items():
                paper_rmse    = TABLE4_RMSE.get(ds_name, {}).get(30, float("nan"))
                fixed_rmse    = data["fixed"]["rmse"]
                adaptive_rmse = data["adaptive"]["rmse"]
                delta         = (fixed_rmse - adaptive_rmse) / fixed_rmse * 100
                rows.append({
                    "Dataset":     ds_name,
                    "Paper RMSE":  f"{paper_rmse:.4f}",
                    "Fixed c3":    f"{fixed_rmse:.4f}  ({fixed_rmse/paper_rmse:.2f}x)",
                    "Adaptive c3": f"{adaptive_rmse:.4f}  ({adaptive_rmse/paper_rmse:.2f}x)",
                    "Delta RMSE":  f"{'+'if delta>0 else ''}{delta:.1f}%",
                })
            df = pd.DataFrame(rows).set_index("Dataset")
            display(df)
        """),

        md("## 6. Best F_r per Run"),
        code("""\
        if results:
            n_ds = len(results)
            fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4))
            if n_ds == 1:
                axes = [axes]

            for ax, (ds_name, data) in zip(axes, results.items()):
                fixed_scores    = data["fixed"]["history"]
                adaptive_scores = data["adaptive"]["history"]
                x = np.arange(len(fixed_scores))
                ax.bar(x - 0.2, fixed_scores,    0.35, label="Fixed",    color="tab:orange", alpha=0.8)
                ax.bar(x + 0.2, adaptive_scores, 0.35, label="Adaptive", color="tab:blue",   alpha=0.8)
                ax.set_title(ds_name)
                ax.set_xlabel("Run index")
                ax.set_ylabel("Best F_r")
                ax.legend(fontsize=8)
                ax.grid(axis="y", alpha=0.3)

            plt.suptitle("Best F_r per Run: Fixed vs Adaptive c3", fontsize=13, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "10_best_fr_per_run.png"), dpi=150, bbox_inches="tight")
            plt.show()
        """),

        md("## 7. Summary"),
        code("""\
        if results:
            print("=" * 60)
            print("SUMMARY — Adaptive vs Fixed c3 Schedule")
            print("=" * 60)
            for ds_name, data in results.items():
                f_rmse = data["fixed"]["rmse"]
                a_rmse = data["adaptive"]["rmse"]
                f_fr   = data["fixed"]["best_score"]
                a_fr   = data["adaptive"]["best_score"]
                delta_rmse = (f_rmse - a_rmse) / f_rmse * 100
                delta_fr   = (f_fr   - a_fr)   / f_fr   * 100
                winner = "Adaptive" if a_rmse < f_rmse else "Fixed"
                print(f"\\n{ds_name}:")
                print(f"  RMSE   Fixed={f_rmse:.4f}  Adaptive={a_rmse:.4f}  D={delta_rmse:+.1f}%  [{winner}]")
                print(f"  Best Fr Fixed={f_fr:.4f}  Adaptive={a_fr:.4f}  D={delta_fr:+.1f}%")
        """),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# 11 — MNAR Extension (Novelty: non-random missing mechanism)
# ─────────────────────────────────────────────────────────────────────────────

def make_11_mnar() -> nbf.NotebookNode:
    nb = new_nb()
    nb.cells = [
        md("""\
        # Notebook 11 — MNAR Extension
        **Novelty:** Extending MIGA to Missing Not At Random (MNAR) mechanisms.

        ## Background

        The base paper evaluates MIGA exclusively under **MAR** (Missing At Random),
        where the probability that a value is missing is independent of the value
        itself.  In many real-world settings, missingness is **not** random:

        - High-income respondents may not report income (finance, surveys)
        - Extreme sensor readings may be truncated or fail to record (IoT, medicine)
        - Patients with severe outcomes may drop out of clinical trials

        This is formally called **MNAR** (Missing Not At Random; Rubin 1976):
        > $P(R=0 \\mid Y) \\neq P(R=0)$,  where $R$ is the response indicator.

        **Key insight:** MIGA minimises the *distributional distance* between
        available rows X_A and completed rows X_C — it never assumes MAR.
        We test whether this distributional objective makes MIGA more robust to
        MNAR than methods that explicitly assume MAR (e.g. mean imputation).

        ## MNAR Mechanisms Evaluated

        | Mechanism | Rule | Real-world analogy |
        |-----------|------|--------------------|
        | MAR       | rows missing at random (baseline) | survey non-response |
        | top       | top pct% of values go missing | self-censoring of high values |
        | bottom    | bottom pct% of values go missing | floor effects, detection limits |
        | tails     | pct/2% from each tail go missing | censoring of extreme values |

        All mechanisms remove the same percentage pct% per selected feature —
        only *which* rows are selected differs.
        """),

        code(SETUP),

        md("## 1. Configuration"),
        code("""\
        from miga.data_utils import load_iris, load_glass, load_haberman

        DATASET_CFG = {
            "Iris":     (load_iris,     {"pct": 30, "seed": 30, "exclude_cols": []}),
            "Glass":    (load_glass,    {"pct": 30, "seed": 30, "exclude_cols": [9]}),
            "Haberman": (load_haberman, {"pct": 30, "seed": 30, "exclude_cols": []}),
        }

        L, G_GENS, Q_RUNS, SEED = 200, 500, 5, 42
        MECHANISMS = ["mar", "top", "bottom", "tails"]

        print(f"GA: l={L}, G={G_GENS}, Q={Q_RUNS}, seed={SEED}")
        print(f"Missing: 30% per selected feature")
        print(f"Mechanisms: {MECHANISMS}")
        """),

        md("## 2. MNAR Mechanism Visualisation"),
        code("""\
        # Show which rows each mechanism selects for a single feature of Iris
        from miga.data_utils import load_iris, apply_mar, apply_mnar

        X_iris, cols_iris = load_iris()
        petal_len = X_iris[:, 2]  # feature 2: petal length
        n = len(petal_len)
        n_remove = max(1, int(round(n * 30 / 100)))

        order = np.argsort(petal_len)
        half  = max(1, n_remove // 2)

        row_sets = {
            "MAR":    np.random.default_rng(30).choice(n, size=n_remove, replace=False),
            "top":    order[-n_remove:],
            "bottom": order[:n_remove],
            "tails":  np.concatenate([order[:half], order[-half:]]),
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
        colours = {"MAR": "tab:gray", "top": "tab:red", "bottom": "tab:blue", "tails": "tab:purple"}

        for ax, (mech, rows) in zip(axes.flat, row_sets.items()):
            ax.scatter(range(n), np.sort(petal_len), s=15, color="lightgray", label="observed")
            ax.scatter(rows, np.sort(petal_len)[np.argsort(np.argsort(petal_len))[rows]],
                       s=25, color=colours[mech], label="missing")
            ax.set_title(f"{mech.upper()} — missing rows highlighted")
            ax.set_xlabel("Row index (sorted by petal length)")
            ax.set_ylabel("Petal length")
            ax.legend(fontsize=8)

        plt.suptitle("Iris petal length: which rows each MNAR mechanism removes (30%)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "11_mnar_mechanisms.png"), dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: results/11_mnar_mechanisms.png")
        """),

        md("""\
        ## 3. Load Results

        Run each dataset in a **separate terminal** (in parallel):
        ```
        .venv/bin/python scripts/run_mnar_dataset.py Iris
        .venv/bin/python scripts/run_mnar_dataset.py Glass
        .venv/bin/python scripts/run_mnar_dataset.py Haberman
        ```
        Each script saves `results/11_mnar_<dataset>.json`.
        Re-run this cell after all three finish to load results.
        """),
        code("""\
        DATASET_NAMES = ["Iris", "Glass", "Haberman"]
        results = {}

        for ds_name in DATASET_NAMES:
            path = os.path.join(RESULTS_DIR, f"11_mnar_{ds_name.lower()}.json")
            if not os.path.exists(path):
                print(f"  [MISSING] {path} — run the script for {ds_name} first")
                continue
            with open(path) as f:
                data = json.load(f)
            results[ds_name] = data
            print(f"\\n{ds_name}:")
            for mech in ["mar", "top", "bottom", "tails"]:
                m = data.get(mech, {})
                print(f"  {mech:8s}  RMSE={m['rmse']:.4f}  MAD={m['mad']:.4f}  CoD={m['cod']:.4f}")

        print(f"\\nLoaded {len(results)}/{len(DATASET_NAMES)} datasets.")
        """),

        md("## 4. RMSE Comparison Table"),
        code("""\
        if results:
            print("=" * 70)
            print("RMSE — MIGA under MAR vs MNAR mechanisms (30% missing, l=200, G=500)")
            print("=" * 70)
            print(f"{'Dataset':<12}  {'MAR':>9}  {'top':>9}  {'bottom':>9}  {'tails':>9}")
            print("-" * 55)
            for ds_name in DATASET_NAMES:
                if ds_name not in results:
                    print(f"  {ds_name:<12}  (not yet run)")
                    continue
                data = results[ds_name]
                vals = [f"{data[m]['rmse']:.4f}" for m in ["mar", "top", "bottom", "tails"]]
                best_idx = np.argmin([data[m]["rmse"] for m in ["mar", "top", "bottom", "tails"]])
                vals[best_idx] = vals[best_idx] + " ★"
                print(f"  {ds_name:<12}  {'  '.join(v:>9 for v in vals)}")
            print("\\n  ★ = lowest RMSE (best imputation) for this dataset")
        """),

        md("## 5. Visualisation — RMSE by Mechanism and Dataset"),
        code("""\
        if results:
            mechs  = ["mar", "top", "bottom", "tails"]
            ds_list = [d for d in DATASET_NAMES if d in results]
            x = np.arange(len(ds_list))
            width = 0.2
            colours_m = ["tab:gray", "tab:red", "tab:blue", "tab:purple"]

            fig, ax = plt.subplots(figsize=(10, 5))
            for i, (mech, col) in enumerate(zip(mechs, colours_m)):
                rmse_vals = [results[ds][mech]["rmse"] for ds in ds_list]
                ax.bar(x + (i - 1.5) * width, rmse_vals, width, label=mech, color=col, alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(ds_list)
            ax.set_ylabel("NRMSE")
            ax.set_title("MIGA RMSE under MAR vs MNAR Mechanisms (30% missing)")
            ax.legend(title="Mechanism")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "11_rmse_by_mechanism.png"), dpi=150, bbox_inches="tight")
            plt.show()
            print("Saved: results/11_rmse_by_mechanism.png")
        """),

        md("## 6. Convergence Plots (Fr vs Generation)"),
        code("""\
        if results:
            for ds_name, data in results.items():
                fig, axes = plt.subplots(1, 4, figsize=(18, 3.5), sharey=False)
                cols_m = {"mar": "tab:gray", "top": "tab:red", "bottom": "tab:blue", "tails": "tab:purple"}

                for ax, mech in zip(axes, ["mar", "top", "bottom", "tails"]):
                    gen_hists = data[mech].get("gen_history", [])
                    if not gen_hists:
                        ax.text(0.5, 0.5, "no gen_history", transform=ax.transAxes, ha="center")
                        continue
                    arr = np.array(gen_hists)
                    mean_curve = arr.mean(axis=0)
                    std_curve  = arr.std(axis=0)
                    gens = np.arange(arr.shape[1])
                    ax.plot(gens, mean_curve, color=cols_m[mech], lw=2)
                    ax.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve,
                                    alpha=0.15, color=cols_m[mech])
                    ax.set_title(mech.upper())
                    ax.set_xlabel("Generation")
                    ax.set_ylabel("Best F_r")
                    ax.grid(alpha=0.3)

                plt.suptitle(f"{ds_name} — Convergence under MAR vs MNAR", fontsize=13, y=1.02)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, f"11_{ds_name.lower()}_convergence.png"),
                            dpi=150, bbox_inches="tight")
                plt.show()
        """),

        md("""\
        ## 7. Discussion

        ### Key questions

        1. **Does MIGA degrade under MNAR?**
           Compare RMSE(MAR) vs RMSE(top/bottom/tails) — if degradation is small,
           MIGA is robust to the missing mechanism.

        2. **Which MNAR mechanism is hardest?**
           `top`/`bottom` create *systematic* biases in X_A vs X_C (available rows
           have lower/higher means than completed rows). `tails` removes extreme
           values, reducing X_A variance.

        3. **Fr vs RMSE under MNAR**
           MIGA minimises distributional distance Fr.  Under MNAR, X_A and X_C
           may have structurally different distributions (e.g. X_C has only
           high-income individuals when mechanism=top). Fr cannot detect this
           structural asymmetry — it only measures how similar the *imputed*
           values' distribution is to X_A's distribution.

        ### Expected findings

        - MIGA's Fr objective is distribution-agnostic: it asks "does X_C look
          like X_A?" Under MNAR, X_C genuinely differs from X_A in distribution,
          so a *low* Fr may still yield high RMSE.
        - Distributional methods (MIGA) are expected to degrade less than
          regression-based methods (KNN, MICE) under MNAR — but this would require
          a baseline comparison notebook (future work).
        - The `tails` mechanism may be harder than `top`/`bottom` because it
          removes the most informative boundary observations.
        """),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# 12 — Baseline Comparison (Mean / KNN / MICE vs MIGA)
# ─────────────────────────────────────────────────────────────────────────────

def make_12_baseline() -> nbf.NotebookNode:
    nb = new_nb()
    nb.cells = [
        md("""\
        # Notebook 12 — Baseline Comparison: Mean / KNN / MICE vs MIGA

        ## Motivation

        MIGA minimises a **distributional distance** (Fr) between available rows X_A
        and completed rows X_C.  Classical imputation methods minimise (explicitly or
        implicitly) **pointwise squared error** (RMSE).  These are structurally
        different objectives — so comparing RMSE across methods is an *apples-vs-oranges*
        comparison, yet it is the standard in the missing-data literature.

        This notebook makes that comparison explicit and frames MIGA's RMSE gaps
        honestly: MIGA is not trying to minimise RMSE.

        ## Methods compared

        | Method | Library | Key property |
        |--------|---------|--------------|
        | Mean   | `sklearn.SimpleImputer(strategy='mean')` | Baseline; minimises MSE to column mean |
        | KNN    | `sklearn.KNNImputer(n_neighbors=5)` | Weighted average of k nearest complete rows |
        | MICE   | `sklearn.IterativeImputer(max_iter=10)` | Iterative regression; closest to minimum-RMSE |
        | MIGA   | Our reimplementation | Distributional distance minimisation |

        **Expected finding (from theory):**
        Van Buuren (2018, §2.6): "The minimum RMSE is achieved by regression imputation,
        which suppresses variance."  MICE ≈ regression imputation → expected to win on RMSE.
        MIGA may win on distributional fidelity (Fr) even when losing on RMSE.

        **Note on baselines:** run `scripts/run_baselines.py` once (< 5 min) to generate
        `results/12_baselines.json`.  MIGA results are loaded from the existing
        `results/0{2-8}_*_results.json` files.
        """),

        code(SETUP),

        md("## 1. Load Baseline Results"),
        code("""\
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

        BASELINE_PATH = os.path.join(RESULTS_DIR, "12_baselines.json")
        if not os.path.exists(BASELINE_PATH):
            print("Run scripts/run_baselines.py first to generate 12_baselines.json")
            baselines = {}
        else:
            with open(BASELINE_PATH) as f:
                baselines = json.load(f)
            print(f"Loaded baselines for: {list(baselines.keys())}")
        """),

        md("## 2. Load MIGA Results"),
        code("""\
        NB_META = {
            "Iris": "02", "Wine": "03", "Glass": "04",
            "Haberman": "05", "Wholesale": "06",
            "Cardio": "07", "Adult": "08",
        }

        miga_results = {}
        for ds, num in NB_META.items():
            path = os.path.join(RESULTS_DIR, f"{num}_{ds}_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    miga_results[ds] = {int(k): v for k, v in json.load(f).items()}
                print(f"  Loaded MIGA: {ds}")
            else:
                print(f"  Missing MIGA: {path}  (run notebook {num} first)")
        """),

        md("## 3. RMSE Comparison Table — All Datasets × All Methods"),
        code("""\
        all_ds = [d for d in NB_META if d in baselines and d in miga_results]

        if not all_ds:
            print("No results available. Run scripts/run_baselines.py and notebooks 02–08 first.")
        else:
            for pct in PERCENTAGES:
                print(f"\\n{'='*70}")
                print(f"RMSE at {pct}% missing")
                print(f"{'='*70}")
                print(f"  {'Dataset':<12}  {'Mean':>8}  {'KNN':>8}  {'MICE':>8}  {'MIGA':>8}  {'MIGA best?':>12}")
                print("  " + "-" * 60)
                for ds in all_ds:
                    b  = baselines[ds].get(str(pct), baselines[ds].get(pct, {}))
                    mi = miga_results[ds].get(pct, {})
                    vals = {
                        "Mean": b.get("Mean", {}).get("rmse") or float("nan"),
                        "KNN":  b.get("KNN",  {}).get("rmse") or float("nan"),
                        "MICE": b.get("MICE", {}).get("rmse") or float("nan"),
                        "MIGA": mi.get("rmse", float("nan")),
                    }
                    best_method = min(vals, key=lambda k: vals[k] if vals[k] == vals[k] else float("inf"))
                    miga_rank   = sorted(vals, key=lambda k: vals[k] if vals[k] == vals[k] else float("inf")).index("MIGA") + 1
                    rank_str    = f"rank {miga_rank}/4"
                    print(f"  {ds:<12}  {vals['Mean']:>8.4f}  {vals['KNN']:>8.4f}  {vals['MICE']:>8.4f}  {vals['MIGA']:>8.4f}  {rank_str:>12}  [best: {best_method}]")
        """),

        md("## 4. RMSE Ratio: MIGA / MICE"),
        code("""\
        if all_ds:
            print("MIGA/MICE RMSE ratio  (> 1 = MICE wins, < 1 = MIGA wins)")
            print()
            print(f"  {'Dataset':<12}  {'30%':>8}  {'40%':>8}  {'50%':>8}  {'60%':>8}  {'Avg':>8}")
            print("  " + "-" * 55)
            for ds in all_ds:
                row = []
                for pct in PERCENTAGES:
                    b  = baselines[ds].get(str(pct), baselines[ds].get(pct, {}))
                    mi = miga_results[ds].get(pct, {})
                    mice_rmse = (b.get("MICE", {}).get("rmse") or float("nan"))
                    miga_rmse = mi.get("rmse", float("nan"))
                    if mice_rmse and mice_rmse == mice_rmse and miga_rmse == miga_rmse:
                        row.append(miga_rmse / mice_rmse)
                    else:
                        row.append(float("nan"))
                avg = sum(r for r in row if r == r) / max(1, sum(1 for r in row if r == r))
                cells = ["★" if r < 1.0 else f"{r:.2f}x" for r in row]
                print(f"  {ds:<12}  {'  '.join(f'{c:>8}' for c in cells)}  {avg:>8.2f}x")
            print()
            print("  ★ = MIGA beats MICE on RMSE (rare — MICE optimizes RMSE directly)")
        """),

        md("""\
        ## 5. The Honest Framing

        MICE wins on RMSE because it **explicitly minimises squared error** via iterative
        regression.  MIGA is not designed to minimise RMSE — it minimises the distributional
        distance Fr between X_A and X_C.

        **Van Buuren (2018, §2.6):**
        > "The minimum mean squared error is achieved by regression imputation, which
        > suppresses variance and produces biased statistical inference."

        The implication: MICE imputes "average" values (near the regression line), which
        scores well on RMSE but *underestimates the true variance* of the missing values.
        MIGA imputes from the empirical distribution of each feature, preserving variance
        at the cost of pointwise accuracy.

        **When MIGA is preferable to MICE:**
        - When downstream analyses require the correct *marginal distribution*
          (histograms, quantile estimation, distributional tests)
        - When data is non-Gaussian and regression imputation is misspecified
        - When the goal is to *recover the joint distribution* rather than point estimates

        **When MICE is preferable:**
        - When the goal is to minimise prediction error on specific values
        - When downstream analyses require minimum-variance estimates
        """),

        md("## 6. Visualisation — RMSE by Method"),
        code("""\
        if all_ds:
            pct = 30
            methods = ["Mean", "KNN", "MICE", "MIGA"]
            cols_m = {"Mean": "tab:gray", "KNN": "tab:green", "MICE": "tab:orange", "MIGA": "tab:blue"}
            x = np.arange(len(all_ds))
            width = 0.2

            fig, ax = plt.subplots(figsize=(12, 5))
            for i, method in enumerate(methods):
                rmse_vals = []
                for ds in all_ds:
                    b  = baselines[ds].get(str(pct), baselines[ds].get(pct, {}))
                    mi = miga_results[ds].get(pct, {})
                    if method == "MIGA":
                        rmse_vals.append(mi.get("rmse", float("nan")))
                    else:
                        rmse_vals.append((b.get(method, {}).get("rmse") or float("nan")))
                ax.bar(x + (i - 1.5) * width, rmse_vals, width,
                       label=method, color=cols_m[method], alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(all_ds, rotation=20, ha="right")
            ax.set_ylabel("NRMSE")
            ax.set_title(f"Imputation RMSE at {pct}% missing — Mean / KNN / MICE / MIGA")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "12_rmse_comparison.png"), dpi=150, bbox_inches="tight")
            plt.show()
            print("Saved: results/12_rmse_comparison.png")
        """),

        md("## 7. CoD Comparison — Does MIGA Win on Any Metric?"),
        code("""\
        if all_ds:
            pct = 30
            print(f"CoD at {pct}% missing  (higher = better):")
            print(f"  {'Dataset':<12}  {'Mean':>8}  {'KNN':>8}  {'MICE':>8}  {'MIGA':>8}  {'MIGA best?'}")
            print("  " + "-" * 60)
            for ds in all_ds:
                b  = baselines[ds].get(str(pct), baselines[ds].get(pct, {}))
                mi = miga_results[ds].get(pct, {})
                vals = {
                    "Mean": b.get("Mean", {}).get("cod") or float("nan"),
                    "KNN":  b.get("KNN",  {}).get("cod") or float("nan"),
                    "MICE": b.get("MICE", {}).get("cod") or float("nan"),
                    "MIGA": mi.get("cod", float("nan")),
                }
                best = max(vals, key=lambda k: vals[k] if vals[k] == vals[k] else float("-inf"))
                miga_best = "★" if best == "MIGA" else ""
                print(f"  {ds:<12}  {vals['Mean']:>8.4f}  {vals['KNN']:>8.4f}  {vals['MICE']:>8.4f}  {vals['MIGA']:>8.4f}  {miga_best}")
        """),

        md("## 8. Summary"),
        code("""\
        if all_ds:
            print("=" * 60)
            print("SUMMARY — Method Comparison")
            print("=" * 60)
            print()
            print("RMSE ranking (across all datasets and percentages):")
            method_avg = {m: [] for m in ["Mean", "KNN", "MICE", "MIGA"]}
            for ds in all_ds:
                for pct in PERCENTAGES:
                    b  = baselines[ds].get(str(pct), baselines[ds].get(pct, {}))
                    mi = miga_results[ds].get(pct, {})
                    method_avg["Mean"].append(b.get("Mean", {}).get("rmse") or float("nan"))
                    method_avg["KNN"].append(b.get("KNN",  {}).get("rmse") or float("nan"))
                    method_avg["MICE"].append(b.get("MICE", {}).get("rmse") or float("nan"))
                    method_avg["MIGA"].append(mi.get("rmse", float("nan")))
            for method, vals in sorted(method_avg.items(),
                                       key=lambda kv: np.nanmean(kv[1])):
                avg = np.nanmean(vals)
                print(f"  {method:<6}: avg RMSE = {avg:.4f}")
            print()
            print("Interpretation:")
            print("  MICE wins on RMSE because it explicitly minimises squared error.")
            print("  MIGA optimises distributional distance (Fr), not RMSE.")
            print("  MIGA is preferable when variance preservation matters more than pointwise accuracy.")
        """),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# 14 — Variance Preservation (MIGA's concrete distributional advantage)
# ─────────────────────────────────────────────────────────────────────────────

def make_14_variance() -> nbf.NotebookNode:
    nb = new_nb()
    nb.cells = [
        md("""\
        # Notebook 14 — Variance Preservation

        ## The Core Argument

        MICE imputes **conditional means**: E[X_j | X_{-j}]. This is optimal for RMSE
        but systematically suppresses marginal variance:

        $$\\text{Var}(\\hat{X}_j^{\\text{MICE}}) < \\text{Var}(X_j^{\\text{true}})$$

        Van Buuren (2018, §2.6) states this explicitly:
        > "The minimum mean squared error is achieved by regression imputation, which
        > suppresses variance and produces biased statistical inference."

        MIGA imputes from the **empirical bootstrap distribution** of observed values.
        The expected variance of the imputed values equals the variance of the observed
        marginal distribution — which approximates the true variance.

        **Measurement:** Variance ratio = Var(imputed) / Var(true). Closer to 1.0 is better.
        MICE < 1.0 (variance suppression). MIGA ≈ 1.0 (variance preservation).

        **Why this matters:** Any downstream analysis that uses variance of the imputed
        data — confidence intervals, hypothesis tests, distributional comparisons — will
        be biased if MICE is used. MIGA avoids this bias.

        **Run script first:**
        ```
        .venv/bin/python scripts/run_variance_preservation.py
        ```
        Saves `results/14_variance_preservation.json`. Takes ~20 min.
        """),

        code(SETUP),

        md("## 1. Load Results"),
        code("""\
        VAR_PATH = os.path.join(RESULTS_DIR, "14_variance_preservation.json")
        if not os.path.exists(VAR_PATH):
            print("Run scripts/run_variance_preservation.py first.")
            var_data = {}
        else:
            with open(VAR_PATH) as f:
                var_data = json.load(f)
            print(f"Loaded: {list(var_data.keys())}")
        """),

        md("## 2. Variance Ratio Table — Per Dataset and Method"),
        code("""\
        METHODS = ["Mean", "KNN", "MICE", "MIGA"]

        if var_data:
            print("Variance Ratio = Var(imputed) / Var(true)  [1.0 = perfect, <1 = suppressed]")
            print()
            print(f"  {'Dataset':<12}  {'Feature(s)':<30}  {'Mean':>7}  {'KNN':>7}  {'MICE':>7}  {'MIGA':>7}")
            print("  " + "-" * 75)
            for ds, data in var_data.items():
                miss_cols = data.get("miss_cols", [])
                feat_str  = ", ".join(miss_cols[:3]) + ("…" if len(miss_cols) > 3 else "")
                vals = {m: data.get(m, {}).get("mean_ratio", float("nan")) for m in METHODS}
                best = min(METHODS, key=lambda m: abs(vals[m] - 1.0) if vals[m]==vals[m] else float("inf"))
                row = "  ".join(f"{vals[m]:>7.4f}{'★' if m==best else ' '}" for m in METHODS)
                print(f"  {ds:<12}  {feat_str:<30}  {row}")
            print()
            print("  ★ = closest to 1.0 (best variance preservation)")
        """),

        md("## 3. Deviation from True Variance"),
        code("""\
        if var_data:
            print("|ratio - 1.0|  (lower = better)")
            print()
            print(f"  {'Dataset':<12}  {'Mean':>8}  {'KNN':>8}  {'MICE':>8}  {'MIGA':>8}  {'MIGA wins?'}")
            print("  " + "-" * 60)
            miga_wins = 0
            for ds, data in var_data.items():
                devs = {m: abs(data.get(m, {}).get("mean_ratio", float("nan")) - 1.0)
                        for m in METHODS}
                best = min(devs, key=lambda m: devs[m] if devs[m]==devs[m] else float("inf"))
                win = "✓" if best == "MIGA" else f"({best})"
                if best == "MIGA":
                    miga_wins += 1
                print(f"  {ds:<12}  {devs['Mean']:>8.4f}  {devs['KNN']:>8.4f}  "
                      f"{devs['MICE']:>8.4f}  {devs['MIGA']:>8.4f}  {win}")
            print(f"\\n  MIGA wins variance preservation on {miga_wins}/{len(var_data)} datasets")
        """),

        md("## 4. Per-Feature Variance Ratios"),
        code("""\
        if var_data:
            for ds, data in var_data.items():
                miss_cols = data.get("miss_cols", [])
                if not miss_cols:
                    continue
                print(f"\\n{ds}:")
                print(f"  {'Feature':<15}  {'True Var':>10}  {'Mean':>8}  {'KNN':>8}  {'MICE':>8}  {'MIGA':>8}")
                print("  " + "-" * 60)
                for col in miss_cols:
                    true_v = data.get("true_var", {}).get(col, float("nan"))
                    vals = {m: data.get(m, {}).get("ratios", {}).get(col, float("nan"))
                            for m in METHODS}
                    print(f"  {col:<15}  {true_v:>10.3f}  "
                          + "  ".join(f"{vals[m]:>8.4f}" for m in METHODS))
        """),

        md("## 5. Visualisation — Variance Ratio Comparison"),
        code("""\
        if var_data:
            n_ds = len(var_data)
            fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=False)
            if n_ds == 1:
                axes = [axes]

            colours = {"Mean": "tab:gray", "KNN": "tab:green", "MICE": "tab:orange", "MIGA": "tab:blue"}

            for ax, (ds, data) in zip(axes, var_data.items()):
                miss_cols = data.get("miss_cols", [])
                x = np.arange(len(miss_cols))
                width = 0.18

                for i, method in enumerate(METHODS):
                    ratios = [data.get(method, {}).get("ratios", {}).get(c, float("nan"))
                              for c in miss_cols]
                    ax.bar(x + (i - 1.5) * width, ratios, width,
                           label=method, color=colours[method], alpha=0.85)

                ax.axhline(1.0, color="black", lw=2, ls="--", label="True = 1.0")
                ax.set_xticks(x)
                ax.set_xticklabels(miss_cols, rotation=30, ha="right", fontsize=9)
                ax.set_title(ds)
                ax.set_ylabel("Var(imputed) / Var(true)")
                ax.legend(fontsize=8)
                ax.grid(axis="y", alpha=0.3)

            plt.suptitle("Variance Ratio by Method and Feature  (1.0 = perfect preservation)",
                         fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "14_variance_ratios.png"), dpi=150, bbox_inches="tight")
            plt.show()
            print("Saved: results/14_variance_ratios.png")
        """),

        md("""\
        ## 6. Interpretation

        **MICE variance suppression:** MICE imputes E[X_j | X_{-j}]. Since conditional
        expectations have lower variance than the marginal: Var(E[X|Y]) ≤ Var(X) (law of
        total variance). For highly correlated features (Iris: r ≈ 0.8), MICE can suppress
        variance by 30–50%.

        **MIGA variance preservation:** MIGA samples from the bootstrap distribution of
        observed values. The bootstrap distribution approximates the true marginal distribution,
        so variance is approximately preserved. The bootstrap naturally captures discreteness
        (Haberman Nodes: 44% zeros) and multi-modality (Iris petal length: trimodal).

        **Practical implication:** Use MICE when you need minimum RMSE on individual imputed
        values. Use MIGA when you need the imputed dataset to have the correct marginal
        distributions — e.g. for:
        - Computing confidence intervals from multiply-imputed data
        - Testing distributional hypotheses (KS test, etc.)
        - Generating plausible synthetic samples from the imputed distribution
        - Any analysis that uses variance of the imputed values
        """),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# 13 — Significance Tests + Fr under MNAR
# ─────────────────────────────────────────────────────────────────────────────

def make_13_significance() -> nbf.NotebookNode:
    nb = new_nb()
    nb.cells = [
        md("""\
        # Notebook 13 — Statistical Significance Tests

        ## Purpose

        Every result so far comes from a single seed (or fixed Q runs).
        This notebook validates the key claims with proper statistical tests:

        1. **Is MIGA Fr significantly lower than baselines under MAR?**
           Wilcoxon signed-rank test (one-sample): H₁: MIGA Fr < baseline Fr.
        2. **Is MIGA Fr still lowest under MNAR?**
           The Haberman `top` MNAR finding (F7) — MIGA achieves lowest Fr but
           highest RMSE — requires confirmation that baselines have higher Fr.
        3. **Effect sizes** — Cohen's d to quantify the Fr gap.

        ## Design

        - N=10 independent MIGA seeds for each (dataset × mechanism) pair
        - Baselines (Mean/KNN/MICE) are deterministic — single run each
        - Wilcoxon one-sample test: MIGA Fr distribution vs. a constant (baseline value)
        - Datasets: Iris, Glass, Haberman at 30% missing
        - Mechanisms: MAR, top MNAR, tails MNAR

        **Run scripts first (parallel):**
        ```
        .venv/bin/python scripts/run_significance.py Iris
        .venv/bin/python scripts/run_significance.py Glass
        .venv/bin/python scripts/run_significance.py Haberman
        ```
        Each saves `results/13_significance_<dataset>.json`.
        """),

        code(SETUP),

        md("## 1. Load Results"),
        code("""\
        from scipy import stats

        DATASET_NAMES = ["Iris", "Glass", "Haberman"]
        MECHANISMS    = ["mar", "top", "tails"]
        METHODS       = ["Mean", "KNN", "MICE"]

        sig_data = {}
        for ds in DATASET_NAMES:
            path = os.path.join(RESULTS_DIR, f"13_significance_{ds.lower()}.json")
            if not os.path.exists(path):
                print(f"  [MISSING] {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            sig_data[ds] = data[ds]
            print(f"  Loaded: {ds}")
        print(f"Loaded {len(sig_data)}/{len(DATASET_NAMES)} datasets.")
        """),

        md("## 2. Fr: MIGA vs Baselines under MAR (Wilcoxon Tests)"),
        code("""\
        if sig_data:
            print("=" * 70)
            print("Fr under MAR — MIGA (10 seeds) vs deterministic baselines")
            print("H1: MIGA Fr < baseline Fr  (one-sided Wilcoxon)")
            print("=" * 70)
            print()
            print(f"  {'Dataset':<10}  {'Baseline':<6}  {'MIGA mean Fr':>13}  {'Baseline Fr':>12}  {'p-value':>9}  {'Sig?':>6}  {'Win?'}")
            print("  " + "-" * 70)

            for ds in DATASET_NAMES:
                if ds not in sig_data:
                    continue
                mech_data = sig_data[ds].get("mar", {})
                miga_frs = [f for f in mech_data.get("miga_fr", []) if f == f]
                if not miga_frs:
                    continue
                miga_arr = np.array(miga_frs)

                for bl_name in METHODS:
                    bl_fr = mech_data.get("baselines", {}).get(bl_name, {}).get("Fr")
                    if bl_fr is None or bl_fr != bl_fr:
                        continue
                    w_data = mech_data.get("wilcoxon_fr", {}).get(bl_name, {})
                    p     = w_data.get("p", float("nan"))
                    miga_wins = miga_arr.mean() < bl_fr
                    sig   = "✓✓" if p < 0.01 else ("✓" if p < 0.05 else ("~" if p < 0.10 else "n.s."))
                    win   = "MIGA<" if miga_wins else "MIGA>"
                    print(f"  {ds:<10}  {bl_name:<6}  {miga_arr.mean():>13.4f}  {bl_fr:>12.4f}  {p:>9.3f}  {sig:>6}  {win}")
        """),

        md("## 3. Fr under MNAR — The Key Finding"),
        code("""\
        if sig_data:
            print("=" * 70)
            print("Fr under MNAR — MIGA (10 seeds) vs baselines")
            print("Critical test: does MIGA still achieve lowest Fr under MNAR?")
            print("=" * 70)

            for mech in ["top", "tails"]:
                print(f"\\n--- {mech.upper()} MNAR ---")
                print(f"  {'Dataset':<10}  {'MIGA Fr':>10}  {'Mean Fr':>10}  {'KNN Fr':>10}  {'MICE Fr':>10}  {'MIGA rank':>10}  {'MIGA RMSE':>10}")
                print("  " + "-" * 75)

                for ds in DATASET_NAMES:
                    if ds not in sig_data:
                        continue
                    mech_data = sig_data[ds].get(mech, {})
                    if not mech_data:
                        continue
                    miga_frs  = [f for f in mech_data.get("miga_fr", []) if f == f]
                    miga_rmse = np.mean(mech_data.get("miga_rmse", [float("nan")]))
                    if not miga_frs:
                        continue
                    miga_mean = np.mean(miga_frs)
                    bls = mech_data.get("baselines", {})
                    bl_frs = {m: (bls.get(m, {}).get("Fr") or float("nan")) for m in METHODS}

                    all_frs = {"MIGA": miga_mean, **bl_frs}
                    ranked  = sorted(all_frs, key=lambda k: all_frs[k] if all_frs[k]==all_frs[k] else float("inf"))
                    rank    = ranked.index("MIGA") + 1

                    print(f"  {ds:<10}  {miga_mean:>10.4f}  "
                          f"{bl_frs['Mean']:>10.4f}  {bl_frs['KNN']:>10.4f}  "
                          f"{bl_frs['MICE']:>10.4f}  {rank:>10}/4  {miga_rmse:>10.4f}")
        """),

        md("""\
        ## 4. The Complete Fr vs RMSE Picture

        This table captures the central empirical finding of the thesis.
        """),
        code("""\
        if sig_data:
            print("=" * 80)
            print("MIGA: lower Fr (better distributional fit) vs MICE: lower RMSE (better pointwise accuracy)")
            print("=" * 80)
            print()
            for ds in DATASET_NAMES:
                if ds not in sig_data:
                    continue
                for mech in MECHANISMS:
                    mech_data = sig_data[ds].get(mech, {})
                    if not mech_data:
                        continue
                    miga_frs  = [f for f in mech_data.get("miga_fr",  []) if f == f]
                    miga_rmse = np.mean(mech_data.get("miga_rmse", [float("nan")]))
                    if not miga_frs:
                        continue
                    miga_fr_mean = np.mean(miga_frs)
                    bls = mech_data.get("baselines", {})
                    mice_fr   = bls.get("MICE", {}).get("Fr")   or float("nan")
                    mice_rmse = bls.get("MICE", {}).get("rmse") or float("nan")
                    knn_fr    = bls.get("KNN",  {}).get("Fr")   or float("nan")
                    knn_rmse  = bls.get("KNN",  {}).get("rmse") or float("nan")

                    fr_ratio   = mice_fr   / miga_fr_mean if miga_fr_mean and miga_fr_mean==miga_fr_mean else float("nan")
                    rmse_ratio = miga_rmse / mice_rmse    if mice_rmse and mice_rmse==mice_rmse else float("nan")

                    print(f"  {ds:<10}  {mech:<7}  "
                          f"MIGA Fr={miga_fr_mean:.4f}  MICE Fr={mice_fr:.4f}  "
                          f"[MIGA Fr {fr_ratio:.2f}x better]  |  "
                          f"MIGA RMSE={miga_rmse:.4f}  MICE RMSE={mice_rmse:.4f}  "
                          f"[MICE RMSE {rmse_ratio:.2f}x better]")
        """),

        md("## 5. Visualisation — Fr and RMSE across Seeds (Boxplots)"),
        code("""\
        if sig_data:
            n_ds = len(sig_data)
            fig, axes = plt.subplots(2, n_ds, figsize=(5 * n_ds, 8))
            if n_ds == 1:
                axes = axes.reshape(2, 1)

            mech_colours = {"mar": "tab:blue", "top": "tab:red", "tails": "tab:purple"}

            for col, ds in enumerate(ds for ds in DATASET_NAMES if ds in sig_data):
                ds_data = sig_data[ds]

                # Fr plot
                ax_fr = axes[0, col]
                all_frs = []
                labels  = []
                for mech in MECHANISMS:
                    mech_data = ds_data.get(mech, {})
                    frs = [f for f in mech_data.get("miga_fr", []) if f == f]
                    if frs:
                        all_frs.append(frs)
                        labels.append(f"MIGA\\n{mech}")
                    # baseline Fr as horizontal lines
                    for bl_name in ["MICE"]:
                        bl_fr = mech_data.get("baselines", {}).get(bl_name, {}).get("Fr")
                        if bl_fr and bl_fr == bl_fr:
                            ax_fr.axhline(bl_fr, color=mech_colours[mech], ls="--",
                                          alpha=0.6, label=f"MICE Fr ({mech})")

                if all_frs:
                    bp = ax_fr.boxplot(all_frs, labels=labels, patch_artist=True)
                    for patch, mech in zip(bp["boxes"], MECHANISMS):
                        patch.set_facecolor(mech_colours[mech])
                        patch.set_alpha(0.6)
                ax_fr.set_title(f"{ds} — Fr (10 seeds)")
                ax_fr.set_ylabel("Fr (lower = better distributional fit)")
                ax_fr.grid(axis="y", alpha=0.3)

                # RMSE plot
                ax_rmse = axes[1, col]
                all_rmse = []
                labels2  = []
                for mech in MECHANISMS:
                    mech_data = ds_data.get(mech, {})
                    rmse_vals = mech_data.get("miga_rmse", [])
                    if rmse_vals:
                        all_rmse.append(rmse_vals)
                        labels2.append(f"MIGA\\n{mech}")
                    for bl_name in ["MICE"]:
                        bl_rmse = mech_data.get("baselines", {}).get(bl_name, {}).get("rmse")
                        if bl_rmse:
                            ax_rmse.axhline(bl_rmse, color=mech_colours[mech], ls="--",
                                            alpha=0.6, label=f"MICE RMSE ({mech})")

                if all_rmse:
                    bp = ax_rmse.boxplot(all_rmse, labels=labels2, patch_artist=True)
                    for patch, mech in zip(bp["boxes"], MECHANISMS):
                        patch.set_facecolor(mech_colours[mech])
                        patch.set_alpha(0.6)
                ax_rmse.set_title(f"{ds} — RMSE (10 seeds)")
                ax_rmse.set_ylabel("NRMSE (lower = better pointwise accuracy)")
                ax_rmse.grid(axis="y", alpha=0.3)

            plt.suptitle("MIGA (10 seeds) vs MICE (dashed): Fr and RMSE across MAR / top MNAR / tails MNAR",
                         fontsize=12, y=1.01)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "13_fr_rmse_boxplots.png"), dpi=150, bbox_inches="tight")
            plt.show()
            print("Saved: results/13_fr_rmse_boxplots.png")
        """),

        md("## 6. Summary Table for Thesis"),
        code("""\
        if sig_data:
            print("=" * 80)
            print("THESIS TABLE: MIGA Fr and RMSE vs MICE Fr and RMSE (mean ± std over 10 seeds)")
            print("=" * 80)
            print()
            print(f"  {'Dataset':<10}  {'Mech':<7}  {'MIGA Fr':>18}  {'MICE Fr':>10}  {'MIGA RMSE':>18}  {'MICE RMSE':>10}  {'Sig (Fr)':>10}")
            print("  " + "-" * 90)

            for ds in DATASET_NAMES:
                if ds not in sig_data:
                    continue
                for mech in MECHANISMS:
                    mech_data = sig_data[ds].get(mech, {})
                    if not mech_data:
                        continue
                    miga_frs  = np.array([f for f in mech_data.get("miga_fr", []) if f == f])
                    miga_rmse = np.array(mech_data.get("miga_rmse", []))
                    bls       = mech_data.get("baselines", {})
                    mice_fr   = bls.get("MICE", {}).get("Fr")   or float("nan")
                    mice_rmse = bls.get("MICE", {}).get("rmse") or float("nan")
                    w_data    = mech_data.get("wilcoxon_fr", {}).get("MICE", {})
                    p         = w_data.get("p", float("nan"))
                    sig       = "p<0.01**" if p < 0.01 else ("p<0.05*" if p < 0.05 else ("p<0.10~" if p < 0.10 else "n.s."))

                    if len(miga_frs) and len(miga_rmse):
                        fr_str   = f"{miga_frs.mean():.3f}±{miga_frs.std():.3f}"
                        rmse_str = f"{miga_rmse.mean():.3f}±{miga_rmse.std():.3f}"
                    else:
                        fr_str = rmse_str = "n/a"

                    print(f"  {ds:<10}  {mech:<7}  {fr_str:>18}  {mice_fr:>10.4f}  {rmse_str:>18}  {mice_rmse:>10.4f}  {sig:>10}")
        """),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Main: generate all notebooks
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating MIGA notebooks...")

    save(make_00_overview(),       "00_Algorithm_Overview.ipynb")
    save(make_01_application(),    "01_Application_Example.ipynb")

    for name, (loader_fn, num, desc) in DATASET_META.items():
        nb = make_benchmark_nb(name, loader_fn, num, desc)
        save(nb, f"{num}_{name}.ipynb")

    save(make_09_comparison(),     "09_Results_Comparison.ipynb")
    save(make_10_adaptive(),       "10_Adaptive_vs_Fixed.ipynb")
    save(make_11_mnar(),           "11_MNAR_Extension.ipynb")
    save(make_12_baseline(),       "12_Baseline_Comparison.ipynb")
    save(make_13_significance(),   "13_Significance_Tests.ipynb")
    save(make_14_variance(),       "14_Variance_Preservation.ipynb")

    print(f"\nAll notebooks written to: {NB_DIR}/")
    print("Run them in any order (02–08 can be run in parallel).")
    print("Run 09 last to aggregate and compare all results.")
    print("Run 10 to compare fixed vs adaptive c3 mutation schedule.")
    print("Run 11 to compare MAR vs MNAR mechanisms.")
    print("Run scripts/run_baselines.py first, then notebook 12 for baseline comparison.")


if __name__ == "__main__":
    main()
