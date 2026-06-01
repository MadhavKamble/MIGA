# References

All sources consulted during reimplementation and thesis writing.

---

## Primary Paper

**[MIGA]** Figueroa-García, J. C., Neruda, R., & Hernandez-Pérez, G. (2023).
*A genetic algorithm for multivariate missing data imputation.*
**Information Sciences**, 619, 947–967.
https://doi.org/10.1016/j.ins.2022.11.037

> Source of Algorithm 1, Definitions 1–5, Tables 2–6, and all paper benchmark results used in comparison.

---

## Datasets

**[Iris]** Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.*
Annals of Eugenics, 7(2), 179–188.
Accessed via: `sklearn.datasets.load_iris`

**[Wine]** Aeberhard, S., Coomans, D., & de Vel, O. (1992). *Comparison of classifiers in high dimensional settings.*
UCI Machine Learning Repository.
Accessed via: `sklearn.datasets.load_wine`

**[Glass]** German, B. (1987). *Glass Identification Database.*
UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/glass+identification

**[Haberman]** Haberman, S. J. (1976). *Generalized Residuals for Log-Linear Models.*
Proceedings of the 9th International Biometrics Conference.
UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/haberman%27s+survival

**[Wholesale]** Abreu, N. (2011). *Analise do perfil do cliente Recheio e desenvolvimento de um sistema promocional.*
UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/wholesale+customers

**[Cardio]** Ayres-de-Campos, D., Bernardes, J., Garrido, A., Marques-de-Sá, J., & Pereira-Leite, L. (2000).
*SisPorto 2.0: A program for automated analysis of cardiotocograms.*
Journal of Maternal-Fetal Medicine, 9(5), 311–318.
UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/cardiotocography
Local file: `data/CTG.xls`

**[Adult]** Kohavi, R. (1996). *Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid.*
Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining.
UCI Machine Learning Repository / OpenML.
Accessed via: `sklearn.datasets.fetch_openml("adult", version=2)`

---

## Algorithms & Methods Referenced

**[GA-survey]** Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems.*
University of Michigan Press.
> Background on genetic algorithm operators (selection, crossover, mutation).

**[Ledoit-Wolf]** Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices.*
Journal of Multivariate Analysis, 88(2), 365–411.
https://doi.org/10.1016/S0047-259X(03)00096-4
> Cited as principled alternative to ad hoc eigenvalue flooring for rank-deficient covariance. Referenced in METHODS.md §7.1 and PLAN.md §B1.

**[MAR-def]** Rubin, D. B. (1976). *Inference and missing data.*
Biometrika, 63(3), 581–592.
https://doi.org/10.1093/biomet/63.3.581
> Formal definition of Missing At Random (MAR) mechanism used in paper §5.

**[MICE]** van Buuren, S., & Groothuis-Oudshoorn, K. (2011). *mice: Multivariate Imputation by Chained Equations in R.*
Journal of Statistical Software, 45(3), 1–67.
https://doi.org/10.18637/jss.v045.i03
> Referenced as a comparison baseline approach (CMIM in paper Table 4).

**[kNNI]** Troyanskaya, O., et al. (2001). *Missing value estimation methods for DNA microarrays.*
Bioinformatics, 17(6), 520–525.
https://doi.org/10.1093/bioinformatics/17.6.520
> k-NN imputation baseline referenced in paper Table 4 (kNNI column).

---

## Software & Libraries

**[NumPy]** Harris, C. R., et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362.
https://doi.org/10.1038/s41586-020-2649-2

**[SciPy]** Virtanen, P., et al. (2020). *SciPy 1.0: Fundamental algorithms for scientific computing in Python.*
Nature Methods, 17, 261–272.
https://doi.org/10.1038/s41592-019-0686-2
> Used for `scipy.stats.skew` (bias-corrected skewness, Definition 3).

**[scikit-learn]** Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.*
Journal of Machine Learning Research, 12, 2825–2830.
http://jmlr.org/papers/v12/pedregosa11a.html
> Used for dataset loaders (`load_iris`, `load_wine`, `fetch_openml`) and `LabelEncoder`.

**[pandas]** McKinney, W. (2010). *Data Structures for Statistical Computing in Python.*
Proceedings of the 9th Python in Science Conference, 56–61.
> Used for dataset loading (CSV/Excel) and result formatting.

**[nbformat]** Project Jupyter. *nbformat — Jupyter Notebook format.*
https://nbformat.readthedocs.io
> Used in `generate_notebooks.py` to programmatically generate all experiment notebooks.

---

## Notes

- All UCI dataset URLs accessed 2025–2026.
- Paper benchmark results (Tables 4, 5, 6) transcribed to `miga/paper_results.py` for automated comparison.
- Paper PDF stored locally at `A_genetic_algorithm_for_multivariate_missing_data_imputation.pdf`.
