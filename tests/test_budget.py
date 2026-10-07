"""Tests for the fair-budget machinery and strong baselines."""
import numpy as np

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted
from grace_qaoa.baselines_strong import (interp_baseline, fourier_baseline,
                                         spsa_baseline, transfer_baseline)


def test_counting_increments():
    g = make_dataset("regular", n_graphs=1, n_nodes=6, seed=1)[0]
    cq = CountingQAOA(QAOAMaxCut(g, p=1))
    assert cq.n_evals == 0
    cq.expected_cut(np.array([0.5, 0.5]))
    cq.expected_cut(np.array([0.6, 0.6]))
    assert cq.n_evals == 2


def test_budget_enforced():
    g = make_dataset("regular", n_graphs=1, n_nodes=6, seed=1)[0]
    cq = CountingQAOA(QAOAMaxCut(g, p=1), budget=3)
    for _ in range(3):
        cq.expected_cut(np.array([0.5, 0.5]))
    try:
        cq.expected_cut(np.array([0.5, 0.5]))
        assert False, "should have raised"
    except BudgetExhausted as e:
        assert e.n_evals == 3


def test_target_tracking():
    g = make_dataset("regular", n_graphs=1, n_nodes=6, seed=1)[0]
    opt = brute_force_maxcut(g)
    cq = CountingQAOA(QAOAMaxCut(g, p=1))
    cq.set_target(0.1 * opt)  # easy target, should be hit quickly
    cq.expected_cut(np.array([0.5, 0.5]))
    assert cq.evals_to_target is not None


def test_strong_baselines_run_and_count():
    g = make_dataset("regular", n_graphs=1, n_nodes=8, seed=2)[0]
    opt = brute_force_maxcut(g)
    # transfer needs a donor; the other three do not.
    for fn in (interp_baseline, fourier_baseline, spsa_baseline):
        cq = CountingQAOA(QAOAMaxCut(g, p=2))
        res = fn(cq, target_p=2, seed=0)
        assert 0.0 <= res["best_cut"] / opt <= 1.0 + 1e-6
        assert res["n_quantum_evals"] > 0
    from grace_qaoa.baselines_strong import optimize_donor
    donor = _donor_graph()
    dparams, dwbar = optimize_donor(donor, target_p=2, multistart=2, maxiter=40)
    cq = CountingQAOA(QAOAMaxCut(g, p=2))
    res = transfer_baseline(cq, target_p=2, donor_params=dparams,
                            donor_w_bar=dwbar, seed=0)
    assert 0.0 <= res["best_cut"] / opt <= 1.0 + 1e-6
    assert res["n_quantum_evals"] > 0


def test_strong_baselines_respect_budget():
    g = make_dataset("regular", n_graphs=1, n_nodes=8, seed=2)[0]
    budget = 60
    for fn in (interp_baseline, fourier_baseline, spsa_baseline):
        cq = CountingQAOA(QAOAMaxCut(g, p=2), budget=budget)
        res = fn(cq, target_p=2, seed=0)
        assert res["n_quantum_evals"] <= budget + 3
    from grace_qaoa.baselines_strong import optimize_donor
    donor = _donor_graph()
    dparams, dwbar = optimize_donor(donor, target_p=2, multistart=2, maxiter=40)
    cq = CountingQAOA(QAOAMaxCut(g, p=2), budget=budget)
    res = transfer_baseline(cq, target_p=2, donor_params=dparams,
                            donor_w_bar=dwbar, seed=0)
    assert res["n_quantum_evals"] <= budget + 3


def _donor_graph():
    """A donor instance disjoint from the test graph. Tries the train split;
    falls back to a distinct seed if make_dataset has no split argument."""
    try:
        return make_dataset("regular", n_graphs=1, n_nodes=8, seed=2,
                            split="train")[0]
    except TypeError:
        return make_dataset("regular", n_graphs=1, n_nodes=8, seed=99999)[0]
