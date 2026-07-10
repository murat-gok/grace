"""Baselines for the GRACE-QAOA comparison (Phase 6).

  - random_init        : random params, no optimization (floor)
  - cold_cobyla        : classical COBYLA from random start
  - gnn_only           : one-shot GNN warm-start, no refinement
  - rl_only            : refinement from random start (no GNN, no escape)

GRACE-QAOA = gnn warm-start + rl refine + metaheuristic escape (see controller).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from grace_qaoa.quantum.qaoa import QAOAMaxCut


def random_init(qaoa: QAOAMaxCut, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    params = rng.uniform(0, np.pi, size=2 * qaoa.p)
    return {"best_params": params, "best_cut": qaoa.expected_cut(params),
            "n_quantum_evals": 1}


def cold_cobyla(qaoa: QAOAMaxCut, seed: int = 0, maxiter: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=2 * qaoa.p)
    counter = {"n": 0}

    def obj(x):
        counter["n"] += 1
        return qaoa.cost(x)

    res = minimize(obj, x0, method="COBYLA", options={"maxiter": maxiter})
    return {"best_params": res.x, "best_cut": qaoa.expected_cut(res.x),
            "n_quantum_evals": counter["n"]}


def gnn_only(qaoa: QAOAMaxCut, gnn_model, graph) -> dict:
    params = gnn_model.predict_params(graph).reshape(-1)
    return {"best_params": params, "best_cut": qaoa.expected_cut(params),
            "n_quantum_evals": 1}
