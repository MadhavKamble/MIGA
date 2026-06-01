"""
GA operators: population initialisation, mutation, crossover, diversity.

Individual encoding (Definition 3 in the paper):
  An individual  p_i  is a flat vector of length  k = |M|,
  ordered by variable j  (j-ordered).  Position  pos  in p_i
  corresponds to missing entry  (i, j) = missing_index[pos].

  var_groups : dict  j → [pos0, pos1, ...]  — positions in p_i for column j.

Population sizes per generation:
  elite         :  c
  mutation      :  c1 × c3   (c1 parents, c3 children each with 1 mutated gene)
  crossover     :  2 × (c2 - 1)   (consecutive pairs, 1 feature j' swapped)
  diversity     :  l - c - c1·c3 - 2(c2-1)   (fresh random individuals)
  ─────────────────────────────────────────
  total         :  l
"""

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_var_groups(missing_index: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Map each column j to the positions it occupies in an individual vector."""
    groups: dict[int, list[int]] = {}
    for pos, (_, j) in enumerate(missing_index):
        groups.setdefault(j, []).append(pos)
    return groups


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def initialize_population(
    l: int,
    k: int,
    generators: dict[int, Callable],
    missing_index: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create an initial random population of l individuals (l × k).

    Each gene p_i[pos] is sampled from R_{j(pos)} — the random variable
    generator for the variable at that position.
    """
    pop = np.empty((l, k), dtype=float)
    for pos, (_, j) in enumerate(missing_index):
        for ind in range(l):
            pop[ind, pos] = generators[j]()
    return pop


# ---------------------------------------------------------------------------
# Mutation  (Section 3.4, produces c1 × c3 new individuals)
# ---------------------------------------------------------------------------

def mutate(
    population: np.ndarray,
    scores: np.ndarray,
    c1: int,
    c3: int,
    generators: dict[int, Callable],
    missing_index: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mutation operator.

    Strategy (paper Section 3.4):
      For each of c1 × c3 steps:
        1. Select a random individual from the best c1 (elitist pool).
        2. Copy it, then replace ONE randomly chosen gene with a fresh sample
           from the appropriate generator R_j.

    This produces exactly c1 × c3 new individuals — consistent with the
    diversity formula:  l - c - c1·c3 - 2(c2-1).
    """
    order = np.argsort(scores)
    elite_pool = population[order[:c1]]
    k = population.shape[1]
    new_inds = np.empty((c1 * c3, k), dtype=float)

    idx = 0
    for _ in range(c1 * c3):
        parent_idx = int(rng.integers(0, c1))
        child = elite_pool[parent_idx].copy()
        pos = int(rng.integers(0, len(missing_index)))
        _, j = missing_index[pos]
        child[pos] = generators[j]()
        new_inds[idx] = child
        idx += 1

    return new_inds


# ---------------------------------------------------------------------------
# Crossover  (Section 3.4, produces 2(c2-1) new individuals)
# ---------------------------------------------------------------------------

def crossover(
    population: np.ndarray,
    scores: np.ndarray,
    c2: int,
    var_groups: dict[int, list[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Crossover operator (Fig. 1 in the paper).

    Strategy:
      1. Take the best c2 individuals (ordered by ascending fitness).
      2. Choose a random feature j' from the variables that have missing values.
      3. For each consecutive pair (i, i+1) in the elite pool:
           swap the genes corresponding to variable j' between the two individuals.
         This produces 2 new children per pair → 2(c2-1) total.
    """
    order = np.argsort(scores)
    elite_pool = population[order[:c2]].copy()

    col_keys = list(var_groups.keys())
    j_prime = int(rng.choice(col_keys))
    positions = var_groups[j_prime]

    n_children = 2 * (c2 - 1)
    if n_children <= 0:
        return np.empty((0, population.shape[1]), dtype=float)

    children = np.empty((n_children, population.shape[1]), dtype=float)
    out = 0
    for i in range(c2 - 1):
        p1 = elite_pool[i].copy()
        p2 = elite_pool[i + 1].copy()
        # Swap genes for variable j'
        tmp = p1[positions].copy()
        p1[positions] = p2[positions]
        p2[positions] = tmp
        children[out] = p1
        children[out + 1] = p2
        out += 2

    return children


# ---------------------------------------------------------------------------
# Diversity injection
# ---------------------------------------------------------------------------

def diversity(
    n_new: int,
    k: int,
    generators: dict[int, Callable],
    missing_index: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate n_new entirely random individuals to maintain diversity.
    Replaces the worst (l - c - c1·c3 - 2(c2-1)) individuals each generation.
    """
    if n_new <= 0:
        return np.empty((0, k), dtype=float)
    pop = np.empty((n_new, k), dtype=float)
    for pos, (_, j) in enumerate(missing_index):
        for ind in range(n_new):
            pop[ind, pos] = generators[j]()
    return pop
