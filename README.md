# MIGA — Multiple Imputation Genetic Algorithm

Implementation of the MIGA algorithm for multivariate missing data imputation, based on:

> Figueroa-García, Neruda & Hernandez-Pérez. *A genetic algorithm for multivariate missing data imputation.* Information Sciences 619 (2023), 947–967.

This repository accompanies a thesis that extends the original algorithm with:
- **Kurtosis-extended fitness (Fr+)** — adds 4th-moment matching for heavy-tailed variables
- **Ledoit-Wolf shrinkage covariance** — more stable covariance estimation on high-dimensional data
- **Multiple Imputation ensemble** — runs MIGA M times with different seeds and pools predictions (Rubin's Rules)
- **AD-MIGA** — adaptive diversity control during evolution
- **MNAR extension** — handling Missing Not At Random patterns via IPW weighting

---

## Repository structure

```
MIGA/
├── miga/           core package
├── data/           datasets (Iris, Wine, Glass, Haberman, Wholesale, CTG)
├── notebooks/      experiment notebooks (00–14)
├── scripts/        analysis and plotting scripts
├── results/        JSON + PNG outputs from experiments
├── examples/       usage example
├── paper/          reference paper (PDF)
├── thesis/
│   ├── docs/       chapter drafts and notes
│   ├── latex/      LaTeX source
│   └── defence/    defence presentation slides
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

No separate install needed — import directly from the repo root:

```python
import sys
sys.path.insert(0, '/path/to/MIGA')
from miga import MIGA
```

---

## Quick start

```python
import numpy as np
from miga import MIGA

# Dataset with ~10% missing values
X = np.array([...])   # shape (n_samples, n_features), NaN where missing

miga = MIGA(
    l=150,   # population size
    G=300,   # generations per run
    Q=4,     # independent runs
    c=3, c1=3, c2=2, c3=6,
    cov_estimator='ledoit_wolf',
    use_kurtosis=True,
    n_jobs=4,
    seed=42,
)

X_imputed = miga.fit_transform(X)
```

See [examples/run_example.py](examples/run_example.py) for a full reproduction of the paper's Section 4 example.

---

## Notebooks

| Notebook | Description |
|---|---|
| [00_Algorithm_Overview](notebooks/00_Algorithm_Overview.ipynb) | Algorithm walkthrough |
| [01_Application_Example](notebooks/01_Application_Example.ipynb) | Paper Section 4 reproduction |
| [02–08](notebooks/) | Dataset experiments (Iris, Wine, Glass, Haberman, Wholesale, Cardio, Adult) |
| [09_Results_Comparison](notebooks/09_Results_Comparison.ipynb) | Cross-dataset comparison |
| [10_Adaptive_vs_Fixed](notebooks/10_Adaptive_vs_Fixed.ipynb) | AD-MIGA vs fixed diversity |
| [11_MNAR_Extension](notebooks/11_MNAR_Extension.ipynb) | MNAR handling |
| [12_Baseline_Comparison](notebooks/12_Baseline_Comparison.ipynb) | vs. mean/kNN/MICE/GAIN |
| [13_Significance_Tests](notebooks/13_Significance_Tests.ipynb) | Statistical significance |
| [14_Variance_Preservation](notebooks/14_Variance_Preservation.ipynb) | Variance preservation analysis |

---

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `l` | Population size | 150 |
| `G` | Generations per run | 300 |
| `Q` | Independent runs | 4 |
| `c` | Elite individuals | 3 |
| `c1`, `c2`, `c3` | Mutation/crossover/diversity counts | 3, 2, 6 |
| `cov_estimator` | `'sample'` or `'ledoit_wolf'` | `'sample'` |
| `use_kurtosis` | Include 4th-moment term in fitness | `False` |
| `n_jobs` | Parallel runs | 1 |
| `seed` | Random seed | None |

---

## Reference

```bibtex
@article{figueroa2023miga,
  title   = {A genetic algorithm for multivariate missing data imputation},
  author  = {Figueroa-García, Juan Carlos and Neruda, Roman and Hernandez-Pérez, Germán},
  journal = {Information Sciences},
  volume  = {619},
  pages   = {947--967},
  year    = {2023}
}
```
