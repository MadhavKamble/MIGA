"""
MIGA — Multiple Imputation Genetic Algorithm
Figueroa-García, Neruda & Hernandez-Pérez, Information Sciences 619 (2023).

Implements Algorithm 1 (pseudocode, p. 953):

  Require: X, X_A, M, l, G, c, c1, c2, c3, q, r
  COMPUTE x̄_A, S_A, b_A, x̃_A from X_A
  for q = 1 → Q:
    for g = 1 → G:
      INITIALISE population P^g
      EVALUATE  F_r  for every individual l
      SELECT    c best individuals (elitism)
      COMPUTE   mutation, crossover, diversity
      COMPUTE   F_r  for new individuals
      PRESERVE  best individual p*_g
    RETURN best per run  p*_q
  RETURN best overall  p*

Population structure per generation (sizes must sum to l):
  c          — elite individuals carried forward
  c1 × c3    — mutation offspring
  2(c2 - 1)  — crossover offspring
  remainder  — fresh diversity individuals
"""

from __future__ import annotations

import concurrent.futures
import numpy as np
from typing import Callable

from .fitness import FitnessEvaluator
from .operators import (
    build_var_groups,
    initialize_population,
    mutate,
    crossover,
    diversity,
)


# ── Module-level worker (must be top-level for ProcessPoolExecutor pickling) ──

def _worker(args: tuple):
    """Run one independent GA run inside a subprocess.

    Generators are recreated from (col_means, col_stds) inside the worker so
    that only picklable data crosses the process boundary.
    Returns (best_ind, best_score, gen_history, run_idx) for ordered collection.
    """
    (run_idx, X_C_base, missing_index, var_groups, k, evaluator,
     col_means, col_stds, l, G, c, c1, c2, c3, c3_schedule,
     decay_mode, decay_lambda, early_stopping_patience,
     verbose, verbose_gen, Q, seed) = args

    rng = np.random.default_rng(seed)
    generators = {
        j: (lambda mu=float(col_means[j]), sig=float(col_stds[j]):
            float(rng.normal(mu, sig)))
        for j in range(len(col_means))
    }
    best_ind, best_score, gen_history = MIGA._single_run_impl(
        X_C_base, missing_index, var_groups, k, evaluator, generators,
        run_idx, l, G, c, c1, c2, c3, c3_schedule,
        decay_mode, decay_lambda, early_stopping_patience,
        verbose, verbose_gen, Q, rng,
    )
    return best_ind, best_score, gen_history, run_idx


class MIGA:
    """
    Parameters
    ----------
    l       : population size  (paper recommends 100 ≤ l ≤ 1000, G ≥ 2l)
    G       : generations per run
    c       : elite size (elitism)
    c1      : mutation — individuals selected from elite pool
    c2      : crossover — individuals selected from elite pool
    c3      : mutation — children produced per parent
    Q       : independent runs  (paper recommends Q ≥ 6)
    r       : Minkowski order  (np.inf recommended for most cases)
    seed    : integer seed for reproducibility
    verbose : whether to print per-run progress
    verbose_gen : whether to print per-generation progress
    """

    def __init__(
        self,
        l: int = 200,
        G: int = 500,
        c: int = 3,
        c1: int = 3,
        c2: int = 2,
        c3: int = 5,
        Q: int = 6,
        r=np.inf,
        cov_estimator: str = 'sample',
        use_kurtosis: bool = False,
        use_ipw: bool = False,
        c3_schedule: tuple[int, int] | None = None,
        diversity_decay: float = 0.0,
        decay_mode: str = 'none',
        decay_lambda: float = 2.0,
        early_stopping_patience: int | None = None,
        n_jobs: int = 1,
        seed: int | None = None,
        verbose: bool = True,
        verbose_gen: bool = False,
    ):
        self.l = l
        self.G = G
        self.c = c
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.Q = Q
        self.r = r
        self.cov_estimator = cov_estimator
        self.use_kurtosis = use_kurtosis
        self.use_ipw = use_ipw
        self.c3_schedule = c3_schedule
        self.diversity_decay = diversity_decay
        self.decay_mode = decay_mode        # 'none' | 'linear' | 'exponential'
        self.decay_lambda = decay_lambda    # exponential rate λ (used when decay_mode='exponential')
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.verbose_gen = verbose_gen
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        if c3_schedule is not None:
            c3_max = c3_schedule[0]
            n_ops_max = c + c1 * c3_max + 2 * max(c2 - 1, 0)
            if n_ops_max >= l:
                raise ValueError(
                    f"c3_schedule start={c3_max} is too large: "
                    f"c + c1*c3_start + 2*(c2-1) = {n_ops_max} >= l={l}."
                )

        # Results populated by fit_transform
        self.best_score_: float | None = None
        self.best_individual_: np.ndarray | None = None
        self.X_imputed_: np.ndarray | None = None
        self.history_: list[float] = []           # best score per run
        self.generation_history_: list[list[float]] = []  # best Fr per gen, per run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: np.ndarray,
        generators: dict[int, Callable[[], float]] | None = None,
    ) -> np.ndarray:
        """
        Impute missing values (NaN) in X.

        Parameters
        ----------
        X          : (n × p) float array with NaN marking missing entries.
        generators : dict mapping column index j → zero-arg callable that
                     draws a sample from R_j (the variable-specific random
                     generator). If None, defaults to N(μ_j, σ_j) estimated
                     from available data in each column.

        Returns
        -------
        X_imputed : (n × p) float array with no NaN values.
        """
        X = np.array(X, dtype=float)
        n, p = X.shape

        complete_mask = ~np.any(np.isnan(X), axis=1)
        X_A = X[complete_mask]
        X_C_base = X[~complete_mask].copy()

        if len(X_A) == 0:
            raise ValueError(
                "No complete rows found. MIGA requires at least one "
                "row with no missing values to form X_A."
            )
        if len(X_C_base) == 0:
            return X.copy()

        # Build missing index  M = [(local_row, col), ...]  (j-ordered)
        missing_index = self._build_missing_index(X_C_base, p)
        k = len(missing_index)

        if k == 0:
            return X.copy()

        var_groups = build_var_groups(missing_index)

        if generators is None:
            generators = self._default_generators(X, p, self.rng)

        if self.use_ipw:
            from .statistics import estimate_ipw_weights
            ipw_weights = estimate_ipw_weights(X)
            if self.verbose:
                print(f"  IPW weights: min={ipw_weights.min():.3f}  "
                      f"max={ipw_weights.max():.3f}  "
                      f"mean={ipw_weights.mean():.3f}")
        else:
            ipw_weights = None

        evaluator = FitnessEvaluator(X_A, r=self.r, cov_estimator=self.cov_estimator,
                                     use_kurtosis=self.use_kurtosis,
                                     ipw_weights=ipw_weights)

        # Sanity check: population can hold all operator outputs
        n_ops = self.c + self.c1 * self.c3 + 2 * max(self.c2 - 1, 0)
        if n_ops >= self.l:
            raise ValueError(
                f"Population size l={self.l} is too small for the given "
                f"c={self.c}, c1={self.c1}, c2={self.c2}, c3={self.c3}. "
                f"Need l > c + c1*c3 + 2*(c2-1) = {n_ops}."
            )

        best_overall_ind: np.ndarray | None = None
        best_overall_score = np.inf
        self.history_ = []
        self.generation_history_ = []

        # Deterministic per-run seeds — same regardless of n_jobs value.
        master_seed = self._seed if self._seed is not None else int(
            self.rng.integers(0, 2**31))
        ss = np.random.SeedSequence(master_seed)

        if self.early_stopping_patience is not None:
            # Budget-aware restart mode: fixed total budget Q*G generations.
            # Runs that stagnate early free up budget for additional restarts.
            # (Parallelisation is disabled in this mode — run count is dynamic.)
            child_seeds = ss.spawn(self.Q * 10)  # generous upper bound
            total_budget = self.Q * self.G
            gens_used = 0
            run_idx = 0
            while gens_used < total_budget:
                G_avail = min(self.G, total_budget - gens_used)
                child_rng = np.random.default_rng(child_seeds[run_idx])
                run_ind, run_score, run_gen_hist = self._single_run(
                    X_C_base, missing_index, var_groups, k, evaluator,
                    generators, run_idx, G_max=G_avail, rng=child_rng,
                )
                gens_used += len(run_gen_hist)
                self.history_.append(run_score)
                self.generation_history_.append(run_gen_hist)
                if run_score < best_overall_score:
                    best_overall_score = run_score
                    best_overall_ind = run_ind.copy()
                if self.verbose:
                    print(f"  Run {run_idx+1}  gens={len(run_gen_hist)}"
                          f"  budget_used={gens_used}/{total_budget}"
                          f"  best_F_r={best_overall_score:.6f}")
                run_idx += 1
        elif self.n_jobs == 1 or self.Q == 1:
            # Standard serial mode.
            child_seeds = ss.spawn(self.Q)
            for q in range(self.Q):
                child_rng = np.random.default_rng(child_seeds[q])
                run_ind, run_score, run_gen_hist = self._single_run(
                    X_C_base, missing_index, var_groups, k, evaluator,
                    generators, q, rng=child_rng,
                )
                self.history_.append(run_score)
                self.generation_history_.append(run_gen_hist)
                if run_score < best_overall_score:
                    best_overall_score = run_score
                    best_overall_ind = run_ind.copy()
        else:
            # Parallel mode: each run in its own subprocess.
            n_workers = self.Q if self.n_jobs == -1 else min(self.n_jobs, self.Q)
            child_seeds_seq = ss.spawn(self.Q)
            child_seeds_int = [
                int(s.generate_state(1)[0]) for s in child_seeds_seq
            ]
            col_means, col_stds = self._col_stats(X, p)
            worker_args = [
                (q, X_C_base, missing_index, var_groups, k, evaluator,
                 col_means, col_stds,
                 self.l, self.G, self.c, self.c1, self.c2, self.c3,
                 self.c3_schedule, self.decay_mode, self.decay_lambda,
                 self.early_stopping_patience, self.verbose, self.verbose_gen,
                 self.Q, child_seeds_int[q])
                for q in range(self.Q)
            ]
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_workers) as pool:
                run_results = list(pool.map(_worker, worker_args))
            # Collect results in order
            run_results_ordered = [None] * self.Q
            for result in run_results:
                run_ind, run_score, run_gen_hist, q_idx = result
                run_results_ordered[q_idx] = (run_ind, run_score, run_gen_hist)
            for run_ind, run_score, run_gen_hist in run_results_ordered:
                self.history_.append(run_score)
                self.generation_history_.append(run_gen_hist)
                if run_score < best_overall_score:
                    best_overall_score = run_score
                    best_overall_ind = run_ind.copy()

        # Reconstruct imputed dataset
        X_imputed = X.copy()
        X_C_filled = X_C_base.copy()
        if best_overall_ind is None:
            # All runs produced nan fitness (e.g. rank-deficient X_A).
            # Fall back to column-mean imputation for the missing entries.
            col_means = np.nanmean(X, axis=0)
            for pos, (local_row, j) in enumerate(missing_index):
                X_C_filled[local_row, j] = col_means[j]
        else:
            for pos, (local_row, j) in enumerate(missing_index):
                X_C_filled[local_row, j] = best_overall_ind[pos]
        X_imputed[~complete_mask] = X_C_filled

        self.best_score_ = best_overall_score
        self.best_individual_ = best_overall_ind
        self.X_imputed_ = X_imputed

        return X_imputed

    def fitness_decomposition(self, X_A: np.ndarray, X_C: np.ndarray) -> dict:
        """Return the three component distances for a completed X_C."""
        evaluator = FitnessEvaluator(X_A, r=self.r, cov_estimator=self.cov_estimator,
                                     use_kurtosis=self.use_kurtosis)
        return evaluator.decompose(X_C)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_run(
        self,
        X_C_base: np.ndarray,
        missing_index: list[tuple[int, int]],
        var_groups: dict[int, list[int]],
        k: int,
        evaluator: FitnessEvaluator,
        generators: dict[int, Callable],
        run_idx: int,
        G_max: int | None = None,
        rng=None,
    ) -> tuple[np.ndarray, float, list[float]]:
        if rng is None:
            rng = self.rng
        return self._single_run_impl(
            X_C_base, missing_index, var_groups, k, evaluator, generators,
            run_idx, self.l, self.G, self.c, self.c1, self.c2, self.c3,
            self.c3_schedule, self.decay_mode, self.decay_lambda,
            self.early_stopping_patience, self.verbose, self.verbose_gen,
            self.Q, rng, G_max=G_max,
        )

    @staticmethod
    def _single_run_impl(
        X_C_base: np.ndarray,
        missing_index: list[tuple[int, int]],
        var_groups: dict[int, list[int]],
        k: int,
        evaluator: FitnessEvaluator,
        generators: dict[int, Callable],
        run_idx: int,
        l: int, G: int, c: int, c1: int, c2: int, c3: int,
        c3_schedule, decay_mode: str, decay_lambda: float,
        early_stopping_patience, verbose: bool, verbose_gen: bool,
        Q: int, rng, G_max: int | None = None,
    ) -> tuple[np.ndarray, float, list[float]]:
        """One full Q-run: G generations of evolution.

        Returns (best_individual, best_score, generation_history) where
        generation_history[g] is the best F_r after generation g.
        Static so it can be called from the module-level worker.
        """
        _rows = np.array([r for r, j in missing_index], dtype=np.intp)
        _cols = np.array([j for r, j in missing_index], dtype=np.intp)

        population = initialize_population(l, k, generators, missing_index, rng)
        scores = MIGA._eval_population(population, X_C_base, missing_index,
                                       evaluator, _rows, _cols)

        valid = np.where(np.isfinite(scores))[0]
        if len(valid) == 0:
            best_ind = population[0].copy()
            best_score = float("nan")
        else:
            best_ind = population[valid[np.argmin(scores[valid])]].copy()
            best_score = float(scores[valid[np.argmin(scores[valid])]])

        gen_history: list[float] = []
        stagnation_count = 0
        G_effective = G_max if G_max is not None else G

        for g in range(G_effective):
            # ── Adaptive or fixed c3 ──────────────────────────────────
            if c3_schedule is not None:
                c3_start, c3_end = c3_schedule
                c3_now = max(1, round(
                    c3_start + (c3_end - c3_start) * g / max(G - 1, 1)
                ))
            else:
                c3_now = c3

            # ── Elitism ──────────────────────────────────────────────
            order = np.argsort(scores)
            elite_inds   = population[order[:c]].copy()
            elite_scores = scores[order[:c]].copy()

            # ── Mutation ─────────────────────────────────────────────
            mut_inds = mutate(population, scores, c1, c3_now,
                              generators, missing_index, rng)

            # ── Crossover ────────────────────────────────────────────
            cross_inds = crossover(population, scores, c2, var_groups, rng)

            # ── Diversity ────────────────────────────────────────────
            # Three modes (controlled by decay_mode):
            #   'none'        — full injection every generation (original MIGA)
            #   'linear'      — frac shrinks from 1.0 to (1 - diversity_decay)
            #   'exponential' — frac = exp(-decay_lambda * g / G)  (AD-MIGA)
            max_n_div = l - c - len(mut_inds) - len(cross_inds)
            if decay_mode == 'exponential' and G > 1:
                frac = float(np.exp(-decay_lambda * g / G))
                n_div = max(0, int(max_n_div * frac))
            elif decay_mode == 'linear' and G > 1:
                frac = max(0.0, 1.0 - decay_lambda * g / (G - 1))
                n_div = max(0, int(max_n_div * frac))
            else:
                n_div = max_n_div
            div_inds = diversity(n_div, k, generators, missing_index, rng)

            # ── Assemble new population ───────────────────────────────
            parts = [elite_inds, mut_inds]
            if len(cross_inds):
                parts.append(cross_inds)
            if len(div_inds):
                parts.append(div_inds)
            assembled = np.vstack(parts)
            n_short = l - len(assembled)
            if n_short > 0:
                extra = mutate(population, scores, c1, n_short,
                               generators, missing_index, rng)[:n_short]
                assembled = np.vstack([assembled, extra])
            population = assembled[:l]

            # Reuse elite scores; evaluate only new slots
            new_scores = MIGA._eval_population(
                population[c:], X_C_base, missing_index, evaluator, _rows, _cols
            )
            scores = np.concatenate([elite_scores, new_scores])

            valid_g = np.where(np.isfinite(scores))[0]
            if len(valid_g) > 0:
                gen_best_idx = int(valid_g[np.argmin(scores[valid_g])])
                if scores[gen_best_idx] < best_score - 1e-8:
                    best_score = float(scores[gen_best_idx])
                    best_ind = population[gen_best_idx].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                stagnation_count += 1

            gen_history.append(best_score)

            if (early_stopping_patience is not None
                    and stagnation_count >= early_stopping_patience):
                break

            if verbose_gen and (g + 1) % max(G // 10, 1) == 0:
                c3_str = f"  c3={c3_now}" if c3_schedule else ""
                print(f"    g={g+1:4d}/{G}  F_r={best_score:.6f}{c3_str}")

        if verbose:
            print(f"  Run {run_idx + 1}/{Q}  best F_r = {best_score:.6f}")

        return best_ind, best_score, gen_history

    # ------------------------------------------------------------------

    @staticmethod
    def _build_missing_index(
        X_C: np.ndarray, p: int
    ) -> list[tuple[int, int]]:
        """
        Build j-ordered missing index  M = [(row, col), ...]
        iterating columns first so genes for the same variable are contiguous.
        """
        index = []
        for j in range(p):
            for i in range(len(X_C)):
                if np.isnan(X_C[i, j]):
                    index.append((i, j))
        return index

    @staticmethod
    def _eval_population(
        population: np.ndarray,
        X_C_base: np.ndarray,
        missing_index: list[tuple[int, int]],
        evaluator: FitnessEvaluator,
        _rows: np.ndarray | None = None,
        _cols: np.ndarray | None = None,
    ) -> np.ndarray:
        # Pre-extract row/col arrays if not already done (amortised across gens)
        if _rows is None:
            _rows = np.array([r for r, j in missing_index], dtype=np.intp)
            _cols = np.array([j for r, j in missing_index], dtype=np.intp)
        scores = np.empty(len(population))
        for idx, individual in enumerate(population):
            X_C = X_C_base.copy()
            X_C[_rows, _cols] = individual   # vectorised fill — replaces Python loop
            scores[idx] = evaluator.evaluate(X_C)
        return scores

    @staticmethod
    def _col_stats(X: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-column (mean, std) from available data — picklable arrays."""
        means = np.zeros(p)
        stds  = np.ones(p)
        for j in range(p):
            col = X[:, j][~np.isnan(X[:, j])]
            means[j] = float(np.mean(col))
            stds[j]  = max(float(np.std(col, ddof=1)) if len(col) > 1 else 1.0, 1e-6)
        return means, stds

    @staticmethod
    def _default_generators(
        X: np.ndarray, p: int, rng=None
    ) -> dict[int, Callable]:
        """Fall-back: N(μ_j, σ_j) estimated from available data per column."""
        if rng is None:
            rng = np.random.default_rng()
        gens = {}
        for j in range(p):
            col = X[:, j][~np.isnan(X[:, j])]
            mu = float(np.mean(col))
            sig = float(np.std(col, ddof=1)) if len(col) > 1 else 1.0
            sig = max(sig, 1e-6)
            gens[j] = (lambda mu=mu, sig=sig: float(rng.normal(mu, sig)))
        return gens
