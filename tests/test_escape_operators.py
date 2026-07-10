"""Tests for the expanded escape-operator set (the 'why CPO?' ablation).

Ensures all nine operators are registered and that run_escape_fair enforces an
identical evaluation budget across them -- the fairness guarantee that makes the
ablation defensible.
"""
import numpy as np

from grace_qaoa.metaheuristic.escape import ESCAPE_OPERATORS, run_escape_fair

EXPECTED = {"de", "cpo", "random", "ga", "pso", "aco", "woa", "gwo", "hho", "cmaes", "aco_ls"}


def test_all_operators_registered():
    assert EXPECTED.issubset(set(ESCAPE_OPERATORS.keys()))


def test_identical_budget_across_operators():
    x0 = np.random.default_rng(0).uniform(0, np.pi, 10)
    MAX = 120
    used = {}
    for name in EXPECTED:
        calls = {"n": 0}

        def f(x, c=calls):
            c["n"] += 1
            return float(np.sum(x ** 2))

        out = run_escape_fair(name, f, x0.copy(),
                              np.random.default_rng(1), max_evals=MAX)
        used[name] = calls["n"]
        assert np.all(out >= 0) and np.all(out <= np.pi + 1e-9)
    # every operator must use exactly the same number of evaluations
    assert len(set(used.values())) == 1, f"budget not identical: {used}"
    assert next(iter(used.values())) == MAX


def test_operators_improve_on_multimodal():
    # On a multimodal cost, each operator should not worsen the seed point.
    x0 = np.full(6, 1.5)

    def f(x):
        return float(np.sum(x ** 2 - 5 * np.cos(2 * np.pi * x)))

    base = f(x0)
    for name in EXPECTED:
        out = run_escape_fair(name, f, x0.copy(),
                              np.random.default_rng(2), max_evals=120)
        assert f(out) <= base + 1e-6
