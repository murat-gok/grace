"""Metaheuristic escape operators for the GRACE-QAOA loop.

When the RL refiner stalls (improvement < eps for T steps), the controller
triggers one of these to perturb (gamma, beta) out of a local optimum.

Two operators are provided so the ablation (c) CPO vs DE is ready out of the box.
Both are deliberately *short-budget* local escapes, not full global searches:
the RL refiner does the fine work, the metaheuristic just relocates the search.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def differential_evolution_escape(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: tuple[float, float] = (0.0, np.pi),
    pop_size: int = 12,
    iters: int = 15,
    F: float = 0.6,
    CR: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Classic DE/rand/1/bin seeded around x0. Returns best params found."""
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    # Seed population near x0 plus some spread for diversity.
    pop = np.clip(x0[None, :] + rng.normal(0, 0.5, size=(pop_size, dim)), lo, hi)
    pop[0] = x0.copy()
    fitness = np.array([cost_fn(ind) for ind in pop])
    for _ in range(iters):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = pop[rng.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lo, hi)
            cross = rng.random(dim) < CR
            if not cross.any():
                cross[rng.integers(dim)] = True
            trial = np.where(cross, mutant, pop[i])
            f_trial = cost_fn(trial)
            if f_trial < fitness[i]:
                pop[i], fitness[i] = trial, f_trial
    return pop[int(np.argmin(fitness))]


def cpo_escape(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: tuple[float, float] = (0.0, np.pi),
    pop_size: int = 12,
    iters: int = 15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crested Porcupine Optimizer style escape with cyclic population reduction.

    Simplified CPO: four defensive behaviours map onto exploration (sight/sound)
    and exploitation (odour/physical attack), with a cyclic population reduction
    that shrinks then restores the population to balance diversity and convergence.
    This is a compact local-escape variant tuned for the small (2*p)-dim QAOA
    parameter space, not a full benchmark CPO implementation.
    """
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    pop = np.clip(x0[None, :] + rng.normal(0, 0.5, size=(pop_size, dim)), lo, hi)
    pop[0] = x0.copy()
    fitness = np.array([cost_fn(ind) for ind in pop])
    best = pop[int(np.argmin(fitness))].copy()
    best_f = fitness.min()

    n_cycles = 2
    for cycle in range(n_cycles):
        for t in range(iters):
            # Cyclic population reduction: shrink the active pool over the cycle.
            frac = 1.0 - (t / max(iters, 1)) * 0.7
            active = max(4, int(pop_size * frac))
            for i in range(active):
                r = rng.random()
                if r < 0.25:        # sight: move away from a random other (explore)
                    other = pop[rng.integers(pop_size)]
                    cand = pop[i] + rng.normal(0, 0.4, dim) * (pop[i] - other)
                elif r < 0.5:       # sound: random long jump (explore)
                    cand = pop[i] + rng.normal(0, 0.6, dim)
                elif r < 0.75:      # odour: drift toward best (exploit)
                    cand = pop[i] + rng.uniform(0, 1, dim) * (best - pop[i])
                else:               # physical attack: tight local refine (exploit)
                    cand = best + rng.normal(0, 0.1, dim)
                cand = np.clip(cand, lo, hi)
                f_cand = cost_fn(cand)
                if f_cand < fitness[i]:
                    pop[i], fitness[i] = cand, f_cand
                    if f_cand < best_f:
                        best, best_f = cand.copy(), f_cand
    return best


def random_restart_escape(
    cost_fn,
    x0,
    bounds: tuple[float, float] = (0.0, np.pi),
    pop_size: int = 12,
    iters: int = 15,
    rng=None,
):
    """Naive control: sample `pop_size*iters` random points, keep the best.

    This is the "is the metaheuristic actually doing anything?" baseline. If CPO
    does not beat random restart with the same evaluation budget, the
    sophistication is not justified -- a reviewer will demand exactly this check.
    Budget is matched to CPO/DE (pop_size * iters cost function calls).
    """
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    best, best_f = x0.copy(), cost_fn(x0)
    for _ in range(pop_size * iters):
        cand = rng.uniform(lo, hi, dim)
        f = cost_fn(cand)
        if f < best_f:
            best, best_f = cand, f
    return best


ESCAPE_OPERATORS = {
    "de": differential_evolution_escape,
    "cpo": cpo_escape,
    "random": random_restart_escape,
}

# Register the extra metaheuristics (GA/PSO/ACO/WOA/GWO/HHO) for the
# "why CPO?" ablation. Imported here to keep a single registry.
try:
    from grace_qaoa.metaheuristic.escape_extra import EXTRA_ESCAPE_OPERATORS
    ESCAPE_OPERATORS.update(EXTRA_ESCAPE_OPERATORS)
except ImportError:
    pass


class _BudgetReached(Exception):
    pass


def run_escape_fair(name, cost_fn, x0, rng, max_evals, bounds=(0.0, np.pi)):
    """Run any registered escape operator under an IDENTICAL evaluation budget.

    Wraps cost_fn with a hard cap so every operator gets exactly `max_evals`
    cost-function calls, regardless of its internal loop structure. This makes
    the "why CPO?" ablation fair: any performance difference reflects the search
    mechanism, not a larger budget. Returns the best x seen within the budget.
    """
    op = ESCAPE_OPERATORS[name]
    state = {"n": 0, "best_x": np.asarray(x0, dtype=float).copy(),
             "best_f": np.inf}

    def capped(x):
        if state["n"] >= max_evals:
            raise _BudgetReached()
        state["n"] += 1
        f = cost_fn(x)
        if f < state["best_f"]:
            state["best_f"] = f
            state["best_x"] = np.asarray(x, dtype=float).copy()
        return f

    try:
        out = op(capped, np.asarray(x0, dtype=float).copy(), bounds=bounds,
                 rng=rng)
        # operator finished under budget; prefer its returned point if better
        try:
            fo = cost_fn(out)
            if fo < state["best_f"]:
                return np.asarray(out, dtype=float)
        except Exception:
            pass
        return state["best_x"]
    except _BudgetReached:
        return state["best_x"]
