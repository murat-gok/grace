"""Additional metaheuristic escape operators for the "why CPO?" ablation.

All operators share the SAME interface and the SAME evaluation budget
(pop_size * iters cost-function calls, plus a small fixed overhead) as the CPO
and DE escapes, so the comparison is fair. Each is a local escape seeded around
x0 in the small (2*p)-dimensional QAOA parameter space, not a full global search.

Operators: GA, PSO, ACO (continuous adaptation, ACO_R), WOA, GWO, HHO.

These are compact, standard implementations adapted for continuous bounded
optimization. They are deliberately matched in budget and seeding strategy to
CPO so that any performance difference reflects the SEARCH MECHANISM, not extra
evaluations.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _seed_population(x0, pop_size, dim, lo, hi, rng, spread=0.5):
    """Population seeded near x0 (so it's a local escape) with x0 included."""
    pop = np.clip(x0[None, :] + rng.normal(0, spread, size=(pop_size, dim)), lo, hi)
    pop[0] = x0.copy()
    return pop


# --------------------------------------------------------------------------- #
# Genetic Algorithm (real-coded, tournament + blend crossover + Gaussian mut)  #
# --------------------------------------------------------------------------- #
def ga_escape(cost_fn, x0, bounds=(0.0, np.pi), pop_size=12, iters=15,
              rng=None, mut_rate=0.3, mut_scale=0.2):
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    pop = _seed_population(x0, pop_size, dim, lo, hi, rng)
    fit = np.array([cost_fn(ind) for ind in pop])
    for _ in range(iters):
        new = []
        for _ in range(pop_size):
            # tournament selection (size 2)
            i, j = rng.integers(pop_size, size=2)
            p1 = pop[i] if fit[i] < fit[j] else pop[j]
            k, l = rng.integers(pop_size, size=2)
            p2 = pop[k] if fit[k] < fit[l] else pop[l]
            # blend crossover (BLX-0.5)
            gamma = rng.uniform(-0.5, 1.5, dim)
            child = np.clip(gamma * p1 + (1 - gamma) * p2, lo, hi)
            # Gaussian mutation
            mask = rng.random(dim) < mut_rate
            child = np.clip(child + mask * rng.normal(0, mut_scale, dim), lo, hi)
            new.append(child)
        new = np.array(new)
        new_fit = np.array([cost_fn(ind) for ind in new])
        # elitist replacement: keep best of combined
        comb = np.vstack([pop, new])
        comb_fit = np.concatenate([fit, new_fit])
        order = np.argsort(comb_fit)[:pop_size]
        pop, fit = comb[order], comb_fit[order]
    return pop[int(np.argmin(fit))]


# --------------------------------------------------------------------------- #
# Particle Swarm Optimization                                                  #
# --------------------------------------------------------------------------- #
def pso_escape(cost_fn, x0, bounds=(0.0, np.pi), pop_size=12, iters=15,
               rng=None, w=0.7, c1=1.5, c2=1.5):
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    pos = _seed_population(x0, pop_size, dim, lo, hi, rng)
    vel = rng.normal(0, 0.1, size=(pop_size, dim))
    pbest = pos.copy()
    pbest_fit = np.array([cost_fn(p) for p in pos])
    g = int(np.argmin(pbest_fit))
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
    for _ in range(iters):
        r1, r2 = rng.random((pop_size, dim)), rng.random((pop_size, dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        fit = np.array([cost_fn(p) for p in pos])
        improved = fit < pbest_fit
        pbest[improved], pbest_fit[improved] = pos[improved], fit[improved]
        g = int(np.argmin(pbest_fit))
        if pbest_fit[g] < gbest_fit:
            gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
    return gbest


# --------------------------------------------------------------------------- #
# Ant Colony Optimization for continuous domains (ACO_R, Socha & Dorigo 2008)  #
# --------------------------------------------------------------------------- #
def aco_escape(cost_fn, x0, bounds=(0.0, np.pi), pop_size=12, iters=15,
               rng=None, q=0.1, xi=0.85):
    """Continuous ACO (ACO_R): solution archive + Gaussian kernel sampling.

    NOTE: classical ACO is for discrete/combinatorial problems; ACO_R is the
    standard continuous adaptation. We label it as such for honesty.
    """
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    arch = _seed_population(x0, pop_size, dim, lo, hi, rng)
    fit = np.array([cost_fn(a) for a in arch])
    order = np.argsort(fit)
    arch, fit = arch[order], fit[order]
    # rank-based weights for the archive
    ranks = np.arange(1, pop_size + 1)
    w = (1.0 / (q * pop_size * np.sqrt(2 * np.pi))) * \
        np.exp(-((ranks - 1) ** 2) / (2 * (q * pop_size) ** 2))
    w /= w.sum()
    # number of new ants per iteration matched so total evals ~ pop_size*iters
    n_ants = pop_size
    for _ in range(iters):
        new = np.zeros((n_ants, dim))
        for a in range(n_ants):
            k = rng.choice(pop_size, p=w)              # choose a guiding solution
            for d in range(dim):
                # sigma = xi * average distance to other archive members
                sigma = xi * np.mean(np.abs(arch[:, d] - arch[k, d])) + 1e-6
                new[a, d] = np.clip(rng.normal(arch[k, d], sigma), lo, hi)
        new_fit = np.array([cost_fn(a) for a in new])
        comb = np.vstack([arch, new])
        comb_fit = np.concatenate([fit, new_fit])
        order = np.argsort(comb_fit)[:pop_size]
        arch, fit = comb[order], comb_fit[order]
    return arch[0]


# --------------------------------------------------------------------------- #
# Whale Optimization Algorithm                                                 #
# --------------------------------------------------------------------------- #
def woa_escape(cost_fn, x0, bounds=(0.0, np.pi), pop_size=12, iters=15, rng=None):
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    pop = _seed_population(x0, pop_size, dim, lo, hi, rng)
    fit = np.array([cost_fn(p) for p in pop])
    best = pop[int(np.argmin(fit))].copy()
    best_fit = fit.min()
    for t in range(iters):
        a = 2 - 2 * t / max(iters - 1, 1)             # decreases 2 -> 0
        for i in range(pop_size):
            r = rng.random()
            A = 2 * a * r - a
            C = 2 * rng.random()
            p = rng.random()
            if p < 0.5:
                if abs(A) < 1:                         # encircling
                    D = np.abs(C * best - pop[i])
                    cand = best - A * D
                else:                                  # search (exploration)
                    rand = pop[rng.integers(pop_size)]
                    D = np.abs(C * rand - pop[i])
                    cand = rand - A * D
            else:                                      # spiral (bubble-net)
                b = 1.0
                l = rng.uniform(-1, 1, dim)
                D = np.abs(best - pop[i])
                cand = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            cand = np.clip(cand, lo, hi)
            fc = cost_fn(cand)
            if fc < fit[i]:
                pop[i], fit[i] = cand, fc
                if fc < best_fit:
                    best, best_fit = cand.copy(), fc
    return best


# --------------------------------------------------------------------------- #
# Grey Wolf Optimizer                                                          #
# --------------------------------------------------------------------------- #
def gwo_escape(cost_fn, x0, bounds=(0.0, np.pi), pop_size=12, iters=15, rng=None):
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    pop = _seed_population(x0, pop_size, dim, lo, hi, rng)
    fit = np.array([cost_fn(p) for p in pop])
    for t in range(iters):
        order = np.argsort(fit)
        alpha, beta, delta = pop[order[0]], pop[order[1]], pop[order[2]]
        a = 2 - 2 * t / max(iters - 1, 1)
        for i in range(pop_size):
            new = np.zeros(dim)
            for leader in (alpha, beta, delta):
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                D = np.abs(C * leader - pop[i])
                new += leader - A * D
            cand = np.clip(new / 3.0, lo, hi)
            fc = cost_fn(cand)
            if fc < fit[i]:
                pop[i], fit[i] = cand, fc
    return pop[int(np.argmin(fit))]


# --------------------------------------------------------------------------- #
# Harris Hawks Optimization (simplified core)                                  #
# --------------------------------------------------------------------------- #
def hho_escape(cost_fn, x0, bounds=(0.0, np.pi), pop_size=12, iters=15, rng=None):
    rng = rng or np.random.default_rng()
    dim = x0.size
    lo, hi = bounds
    pop = _seed_population(x0, pop_size, dim, lo, hi, rng)
    fit = np.array([cost_fn(p) for p in pop])
    rabbit = pop[int(np.argmin(fit))].copy()
    rabbit_fit = fit.min()
    for t in range(iters):
        E1 = 2 * (1 - t / max(iters, 1))
        for i in range(pop_size):
            E0 = 2 * rng.random() - 1
            E = E1 * E0
            if abs(E) >= 1:                            # exploration
                rand = pop[rng.integers(pop_size)]
                cand = rand - rng.random() * np.abs(rand - 2 * rng.random() * pop[i])
            else:                                      # exploitation
                if rng.random() >= 0.5:
                    cand = (rabbit - pop[i]) - E * np.abs(
                        rng.random() * rabbit - pop[i])
                else:
                    cand = rabbit - E * np.abs(rabbit - pop[i])
            cand = np.clip(cand, lo, hi)
            fc = cost_fn(cand)
            if fc < fit[i]:
                pop[i], fit[i] = cand, fc
                if fc < rabbit_fit:
                    rabbit, rabbit_fit = cand.copy(), fc
    return rabbit


# --------------------------------------------------------------------------- #
# CMA-ES (Covariance Matrix Adaptation Evolution Strategy), compact self-       #
# contained implementation. The gold-standard continuous black-box optimizer   #
# and the strongest competitor to CPO as an escape operator.                   #
# --------------------------------------------------------------------------- #
def cmaes_escape(cost_fn, x0, bounds=(0.0, np.pi), pop_size=12, iters=15,
                 rng=None, sigma0=0.3):
    """Minimal CMA-ES seeded at x0. Adapts a full covariance matrix to the local
    landscape, which is exactly what makes it strong on smooth multimodal basins
    like the QAOA (gamma, beta) surface. Budget-matched (pop_size*iters evals)
    and run under run_escape_fair's hard cap like every other operator.
    """
    rng = rng or np.random.default_rng()
    n = x0.size
    lo, hi = bounds
    lam = pop_size                      # population (offspring) size
    mu = lam // 2                       # parents
    # recombination weights (log-decreasing)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    # adaptation constants (Hansen's standard settings)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    mean = np.clip(x0.copy(), lo, hi)
    sigma = sigma0
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    best_x, best_f = mean.copy(), cost_fn(mean)

    for _ in range(iters):
        # eigen-decomposition of C for sampling
        try:
            D2, B = np.linalg.eigh(C)
            D2 = np.clip(D2, 1e-12, None)
            D = np.sqrt(D2)
        except np.linalg.LinAlgError:
            B, D = np.eye(n), np.ones(n)
        # sample lambda offspring
        Z = rng.standard_normal((lam, n))
        Y = (B @ (D[:, None] * Z.T)).T          # ~ N(0, C)
        X = np.clip(mean + sigma * Y, lo, hi)
        fvals = np.array([cost_fn(x) for x in X])
        order = np.argsort(fvals)
        if fvals[order[0]] < best_f:
            best_f, best_x = fvals[order[0]], X[order[0]].copy()
        # recombination of the mu best
        X_sel = X[order[:mu]]
        Y_sel = (X_sel - mean) / sigma
        ymean = np.sum(w[:, None] * Y_sel, axis=0)
        mean = mean + sigma * ymean
        # step-size control (cumulative path)
        C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ ymean)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = float(np.clip(sigma, 1e-6, (hi - lo)))
        # covariance update
        hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < \
            (1.4 + 2 / (n + 1)) * chiN else 0.0
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean
        rank_one = np.outer(pc, pc)
        rank_mu = np.sum([w[i] * np.outer(Y_sel[i], Y_sel[i])
                          for i in range(mu)], axis=0)
        C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
        C = np.triu(C) + np.triu(C, 1).T        # keep symmetric
    return best_x


# --------------------------------------------------------------------------- #
# Ensembles of ACO + CPO (the two strongest single operators).                 #
# BOTH are budget-neutral: they split the SAME total budget the single         #
# operators receive (via pop_size*iters), so any gain is not from extra evals.  #
# --------------------------------------------------------------------------- #
def aco_cpo_portfolio_escape(cost_fn, x0, bounds=(0.0, np.pi),
                             pop_size=12, iters=15, rng=None):
    """Portfolio: run ACO and CPO each on HALF the budget, from the same seed
    point, and return the better of the two results. Two distinct search
    dynamics explore the same neighbourhood differently; if they are
    complementary across instances, the portfolio beats either alone.
    """
    from grace_qaoa.metaheuristic.escape import cpo_escape
    rng = rng or np.random.default_rng()
    half = max(1, iters // 2)
    a = aco_escape(cost_fn, x0, bounds=bounds, pop_size=pop_size,
                   iters=half, rng=rng)
    c = cpo_escape(cost_fn, x0, bounds=bounds, pop_size=pop_size,
                   iters=iters - half, rng=rng)
    return a if cost_fn(a) <= cost_fn(c) else c


def aco_cpo_cascade_escape(cost_fn, x0, bounds=(0.0, np.pi),
                           pop_size=12, iters=15, rng=None):
    """Cascade: ACO explores first (2/3 of budget), then CPO refines from ACO's
    best (1/3 of budget). Exploration handed to ACO, exploitation/restructuring
    to CPO's cyclic dynamics.
    """
    from grace_qaoa.metaheuristic.escape import cpo_escape
    rng = rng or np.random.default_rng()
    n1 = max(1, (2 * iters) // 3)
    a = aco_escape(cost_fn, x0, bounds=bounds, pop_size=pop_size,
                   iters=n1, rng=rng)
    c = cpo_escape(cost_fn, a, bounds=bounds, pop_size=pop_size,
                   iters=iters - n1, rng=rng)
    # safety: never worse than ACO's own best
    return c if cost_fn(c) <= cost_fn(a) else a


def _coordinate_descent(cost_fn, x, bounds, rng, n_evals, step0=0.25):
    """Greedy coordinate descent local search from x, using a fixed eval budget.

    Cheap, derivative-free refinement: probe each coordinate +/- step, move if it
    improves, shrink the step when a full sweep yields no gain. Standard local
    search; budget-bounded so the caller controls total cost.
    """
    lo, hi = bounds
    x = np.clip(np.asarray(x, dtype=float).copy(), lo, hi)
    fx = cost_fn(x)
    used = 1
    dim = x.size
    step = step0
    while used < n_evals:
        improved = False
        order = rng.permutation(dim)
        for d in order:
            for sign in (+1.0, -1.0):
                if used >= n_evals:
                    break
                cand = x.copy()
                cand[d] = np.clip(cand[d] + sign * step, lo, hi)
                fc = cost_fn(cand)
                used += 1
                if fc < fx:
                    x, fx = cand, fc
                    improved = True
                    break          # greedy: accept first improving move
        if not improved:
            step *= 0.5            # contract neighbourhood
            if step < 1e-3:
                break
    return x, fx


def aco_ls_escape(cost_fn, x0, bounds=(0.0, np.pi), pop_size=12, iters=15,
                  rng=None, ls_fraction=0.35, q=0.1, xi=0.85):
    """ACO_R with a local-search refinement stage (ACO+LS).

    Budget-neutral: ~ (1 - ls_fraction) of the pop_size*iters evaluation budget
    goes to ACO exploration, the rest to greedy coordinate descent from ACO's
    best point. This mirrors the well-known result that ACO augmented with local
    search outperforms plain ACO, while staying in the same (gamma, beta)
    continuous space and adding NO learned components (no extra GNN).
    """
    rng = rng or np.random.default_rng()
    total = pop_size * iters
    ls_budget = max(2 * x0.size, int(ls_fraction * total))
    aco_iters = max(1, int(round((total - ls_budget) / pop_size)))
    # exploration phase: standard ACO_R with reduced iters
    best = aco_escape(cost_fn, x0, bounds=bounds, pop_size=pop_size,
                      iters=aco_iters, rng=rng, q=q, xi=xi)
    # exploitation phase: local search from ACO's best
    refined, _ = _coordinate_descent(cost_fn, best, bounds, rng, ls_budget)
    # safety: never return worse than ACO's own best
    return refined if cost_fn(refined) <= cost_fn(best) else best


EXTRA_ESCAPE_OPERATORS = {
    "ga": ga_escape,
    "pso": pso_escape,
    "aco": aco_escape,
    "aco_ls": aco_ls_escape,
    "woa": woa_escape,
    "gwo": gwo_escape,
    "hho": hho_escape,
    "cmaes": cmaes_escape,
    "aco_cpo_portfolio": aco_cpo_portfolio_escape,
    "aco_cpo_cascade": aco_cpo_cascade_escape,
}
