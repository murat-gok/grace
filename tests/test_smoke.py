"""End-to-end smoke tests. Small graphs only, so they run in seconds on CPU."""
import numpy as np

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.metaheuristic.escape import cpo_escape, differential_evolution_escape
from grace_qaoa.controller.grace import GraceController
from grace_qaoa.baselines import random_init, cold_cobyla


def test_qaoa_runs():
    g = make_dataset("regular", n_graphs=1, n_nodes=6, seed=1)[0]
    qaoa = QAOAMaxCut(g, p=1)
    params = np.array([[0.5], [0.5]])
    cut = qaoa.expected_cut(params)
    assert cut >= 0
    opt = brute_force_maxcut(g)
    assert opt > 0
    assert qaoa.approximation_ratio(params, opt) <= 1.0 + 1e-6


def test_weighted_graph():
    g = make_dataset("erdos_renyi", n_graphs=1, n_nodes=7, weighted=True, seed=2)[0]
    qaoa = QAOAMaxCut(g, p=2)
    assert qaoa.expected_cut(np.random.rand(2, 2) * np.pi) >= 0


def test_escape_operators_improve_or_equal():
    g = make_dataset("regular", n_graphs=1, n_nodes=6, seed=3)[0]
    qaoa = QAOAMaxCut(g, p=1)
    rng = np.random.default_rng(0)
    x0 = rng.uniform(0, np.pi, 2)
    f0 = qaoa.cost(x0)
    for op in (cpo_escape, differential_evolution_escape):
        x = op(qaoa.cost, x0.copy(), rng=np.random.default_rng(0))
        assert qaoa.cost(x) <= f0 + 1e-6


def test_grace_controller():
    g = make_dataset("regular", n_graphs=1, n_nodes=6, seed=4)[0]
    qaoa = QAOAMaxCut(g, p=1)
    ctrl = GraceController(qaoa, escape="cpo", max_rounds=3, seed=0)
    init = np.array([0.4, 0.4])
    out = ctrl.run(init)
    assert out["best_cut"] >= qaoa.expected_cut(init) - 1e-6
    assert out["n_quantum_evals"] > 0


def test_baselines():
    g = make_dataset("regular", n_graphs=1, n_nodes=6, seed=5)[0]
    qaoa = QAOAMaxCut(g, p=1)
    assert random_init(qaoa, seed=0)["best_cut"] >= 0
    assert cold_cobyla(qaoa, seed=0, maxiter=30)["best_cut"] >= 0
