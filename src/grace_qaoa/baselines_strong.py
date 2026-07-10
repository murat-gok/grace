"""Strong, literature-standard QAOA baselines (budget-aware).

These are the baselines a Q1 reviewer expects for a QAOA initialization /
optimization paper. Beating COBYLA is not enough; beating INTERP/FOURIER is the
real bar.

References:
  - Zhou et al., "Quantum Approximate Optimization Algorithm: Performance,
    Mechanism, and Implementation on Near-Term Devices", Phys. Rev. X 10,
    021067 (2020). INTERP = linear interpolation of optimized params from depth
    p-1 to p; FOURIER = optimize in the sine/cosine (frequency) basis.
  - Spall (1992), SPSA: simultaneous-perturbation stochastic approximation,
    a strong gradient-style optimizer that needs only 2 evals per step.

Every routine takes a CountingQAOA so quantum-evaluation cost is charged
uniformly, and returns (best_params, best_cut). Budget is enforced by the
CountingQAOA itself (raises BudgetExhausted), which callers catch.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted


# --------------------------------------------------------------------------- #
# Local optimizer used as the per-level subroutine for INTERP / parameter-fix  #
# --------------------------------------------------------------------------- #
def _optimize_from(cqaoa: CountingQAOA, x0: np.ndarray, maxiter: int = 80):
    """Local optimization (COBYLA) from x0; returns optimized params."""
    try:
        res = minimize(lambda x: cqaoa.cost(x), x0, method="COBYLA",
                       options={"maxiter": maxiter})
        return np.asarray(res.x, dtype=float)
    except BudgetExhausted:
        # Out of budget mid-optimization: return current best estimate.
        return x0


# --------------------------------------------------------------------------- #
# INTERP (Zhou et al. 2020)                                                    #
# --------------------------------------------------------------------------- #
def _interp_extend(gammas, betas):
    """Linear-interpolation extension from depth p to depth p+1 (Zhou et al. 2020).

    The standard INTERP rule: new schedule of length p+1 where
      new[i] = ((i)/(p+1)) * old[i-1] + ((p+1-i)/(p+1)) * old[i]
    for i = 1..p+1, with old indexed 1..p and out-of-range terms treated as 0.
    Implemented with safe 0-based indexing.
    """
    p = len(gammas)

    def extend(vec):
        old = np.asarray(vec, dtype=float)
        new = np.zeros(p + 1)
        for i in range(1, p + 2):                       # 1..p+1
            left = old[i - 2] if (i - 2) >= 0 else 0.0   # old[i-1] (1-based)
            right = old[i - 1] if (i - 1) < p else 0.0   # old[i]   (1-based)
            new[i - 1] = (i / (p + 1)) * left + ((p + 1 - i) / (p + 1)) * right
        return new

    return extend(gammas), extend(betas)


def interp_baseline(cqaoa: CountingQAOA, target_p: int, maxiter_per_level: int = 60,
                    seed: int = 0):
    """Layer-by-layer INTERP. Optimizes p=1, interpolates up to target_p.

    INTERP genuinely needs a depth-p circuit at each level p. The passed-in
    cqaoa is the TARGET-depth counting device (used for the final evaluation and
    the budget); intermediate levels use their own depth-p circuits that share
    the same counter so the budget is accounted globally.
    """
    from grace_qaoa.quantum.qaoa import QAOAMaxCut
    rng = np.random.default_rng(seed)
    graph = cqaoa.qaoa.graph

    def level_cost(level_qaoa, x):
        # charge the shared counter, but evaluate on the depth-`level` circuit
        cqaoa.n_evals += 1
        if cqaoa.budget is not None and cqaoa.n_evals > cqaoa.budget:
            raise BudgetExhausted(cqaoa.n_evals)
        cut = level_qaoa.expected_cut(x)
        # Track best across levels so a mid-run budget cutoff still has a value.
        # (Intermediate-depth cuts are valid lower bounds on achievable cut.)
        if cut > cqaoa.best_cut_so_far:
            cqaoa.best_cut_so_far = cut
            # Also record to the trace (for sample-efficiency curves), since
            # these evals do not pass through cqaoa.expected_cut.
            if getattr(cqaoa, "record_trace", False):
                cqaoa.trace.append((cqaoa.n_evals, cut))
        return -cut

    def opt_level(level, x0):
        lq = QAOAMaxCut(graph, p=level)
        try:
            res = minimize(lambda x: level_cost(lq, x), x0, method="COBYLA",
                           options={"maxiter": maxiter_per_level})
            return np.asarray(res.x, dtype=float)
        except BudgetExhausted:
            return x0

    best_g = np.array([rng.uniform(0, np.pi)])
    best_b = np.array([rng.uniform(0, np.pi)])
    try:
        best_cut = -np.inf
        for _ in range(3):
            x = opt_level(1, rng.uniform(0, np.pi, 2))
            lq = QAOAMaxCut(graph, p=1)
            c = lq.expected_cut(x)
            if c > best_cut:
                best_cut, best_g, best_b = c, x[:1].copy(), x[1:2].copy()
        for p in range(1, target_p):
            g0, b0 = _interp_extend(best_g, best_b)
            x = opt_level(p + 1, np.concatenate([g0, b0]))
            best_g, best_b = x[:p + 1].copy(), x[p + 1:].copy()
    except BudgetExhausted:
        pass
    params = np.concatenate([best_g, best_b])
    return _finalize(cqaoa, params, target_p)


# --------------------------------------------------------------------------- #
# FOURIER (Zhou et al. 2020)                                                   #
# --------------------------------------------------------------------------- #
def _fourier_to_angles(u, v, p):
    """Map Fourier amplitudes (u for gamma via sin, v for beta via cos) to angles."""
    q = len(u)
    gammas = np.zeros(p)
    betas = np.zeros(p)
    for i in range(1, p + 1):
        gammas[i - 1] = sum(
            u[k - 1] * np.sin((k - 0.5) * (i - 0.5) * np.pi / p) for k in range(1, q + 1))
        betas[i - 1] = sum(
            v[k - 1] * np.cos((k - 0.5) * (i - 0.5) * np.pi / p) for k in range(1, q + 1))
    return gammas, betas


def fourier_baseline(cqaoa: CountingQAOA, target_p: int, q: int | None = None,
                     maxiter: int = 120, seed: int = 0):
    """FOURIER strategy: optimize in the (u, v) frequency basis at target_p."""
    rng = np.random.default_rng(seed)
    q = q or target_p
    uv0 = rng.normal(0, 0.3, 2 * q)

    def fourier_cost(uv):
        u, v = uv[:q], uv[q:]
        g, b = _fourier_to_angles(u, v, target_p)
        return cqaoa.cost(np.concatenate([g, b]))

    try:
        res = minimize(fourier_cost, uv0, method="COBYLA",
                       options={"maxiter": maxiter})
        u, v = res.x[:q], res.x[q:]
    except BudgetExhausted:
        u, v = uv0[:q], uv0[q:]
    g, b = _fourier_to_angles(u, v, target_p)
    params = np.concatenate([g, b])
    return _finalize(cqaoa, params, target_p)


# --------------------------------------------------------------------------- #
# SPSA (Spall 1992) — strong, cheap gradient-style optimizer                   #
# --------------------------------------------------------------------------- #
def spsa_baseline(cqaoa: CountingQAOA, target_p: int, iters: int = 150,
                  a=0.2, c=0.1, seed: int = 0):
    """Simultaneous Perturbation Stochastic Approximation on the QAOA cost."""
    rng = np.random.default_rng(seed)
    dim = 2 * target_p
    theta = rng.uniform(0, np.pi, dim)
    A = 0.1 * iters
    alpha, gamma = 0.602, 0.101
    try:
        for k in range(1, iters + 1):
            ak = a / (k + A) ** alpha
            ck = c / k ** gamma
            delta = rng.choice([-1, 1], size=dim)
            f_plus = cqaoa.cost(np.clip(theta + ck * delta, 0, np.pi))
            f_minus = cqaoa.cost(np.clip(theta - ck * delta, 0, np.pi))
            ghat = (f_plus - f_minus) / (2 * ck) * delta
            theta = np.clip(theta - ak * ghat, 0, np.pi)
    except BudgetExhausted:
        pass
    return _finalize(cqaoa, theta, target_p)


# --------------------------------------------------------------------------- #
# Parameter-concentration transfer                                             #
# --------------------------------------------------------------------------- #
def transfer_baseline(cqaoa: CountingQAOA, target_p: int,
                      donor_params: np.ndarray | None = None,
                      maxiter: int = 60, seed: int = 0):
    """Initialize from donor (concentration) params, then briefly refine.

    If donor_params is None, use the well-known fixed-angle concentration values
    (gamma small positive, beta moderate) as a generic donor.
    """
    rng = np.random.default_rng(seed)
    if donor_params is None:
        g = np.linspace(0.3, 0.6, target_p)
        b = np.linspace(0.6, 0.3, target_p)
        x0 = np.concatenate([g, b])
    else:
        x0 = np.asarray(donor_params, dtype=float).reshape(-1)
    try:
        x = _optimize_from(cqaoa, x0, maxiter)
    except BudgetExhausted:
        x = x0
    return _finalize(cqaoa, x, target_p)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _eval_at_depth(cqaoa, gammas, betas):
    return cqaoa.expected_cut(np.concatenate([np.asarray(gammas), np.asarray(betas)]))


def _level_view(cqaoa, p):
    """INTERP/parameter-fix optimize a depth-p circuit. Our QAOA is fixed-depth,
    so for honest accounting we evaluate at the target depth's circuit. For the
    layer-by-layer schedule we treat the schedule length as p and pad/truncate
    in _finalize. (Kept simple: same counting device throughout.)"""
    return cqaoa


def _finalize(cqaoa, params, target_p):
    """Ensure params has length 2*target_p, then return (params, best_cut)."""
    params = np.asarray(params, dtype=float).reshape(-1)
    need = 2 * target_p
    if params.size < need:
        params = np.concatenate([params, np.zeros(need - params.size)])
    elif params.size > need:
        params = params[:need]
    try:
        cut = cqaoa.expected_cut(params)
    except BudgetExhausted:
        cut = cqaoa.best_cut_so_far
    return {"best_params": params, "best_cut": max(cut, cqaoa.best_cut_so_far),
            "n_quantum_evals": cqaoa.n_evals}
