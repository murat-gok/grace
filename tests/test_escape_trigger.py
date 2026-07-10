"""Tests for the escape-trigger logic in the GRACE controller."""
import numpy as np

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut
from grace_qaoa.controller.grace import GraceController


def test_escape_triggers_on_stall():
    g = make_dataset("regular", n_graphs=1, n_nodes=8, seed=1)[0]
    qaoa = QAOAMaxCut(g, p=3)

    # A refiner that returns the SAME params -> guaranteed stall every round.
    def stalled_refiner(params):
        return np.asarray(params, dtype=float)

    ctrl = GraceController(qaoa, escape="cpo", stall_patience=2, max_rounds=6,
                           seed=0)
    ctrl.run(np.full(6, 0.5), rl_step_fn=stalled_refiner)
    n_escapes = sum(1 for h in ctrl.history if h["event"] == "escape")
    assert n_escapes >= 1, "escape should trigger when the refiner is stalled"


def test_escape_off_never_triggers():
    g = make_dataset("regular", n_graphs=1, n_nodes=8, seed=1)[0]
    qaoa = QAOAMaxCut(g, p=3)

    def stalled_refiner(params):
        return np.asarray(params, dtype=float)

    # stall_eps < 0 disables escape entirely (the escape-OFF ablation).
    ctrl = GraceController(qaoa, escape="cpo", stall_eps=-1.0,
                           stall_patience=2, max_rounds=6, seed=0)
    ctrl.run(np.full(6, 0.5), rl_step_fn=stalled_refiner)
    n_escapes = sum(1 for h in ctrl.history if h["event"] == "escape")
    assert n_escapes == 0, "escape must never trigger when disabled"


def test_escape_only_keeps_improvements():
    # best_cut must be monotonic non-decreasing across the run.
    g = make_dataset("erdos_renyi", n_graphs=1, n_nodes=8, weighted=True, seed=2)[0]
    qaoa = QAOAMaxCut(g, p=2)
    ctrl = GraceController(qaoa, escape="de", stall_patience=2, max_rounds=5, seed=0)
    out = ctrl.run(np.full(4, 0.4))
    # best_cut returned should be >= the initial cut
    assert out["best_cut"] >= qaoa.expected_cut(np.full(4, 0.4)) - 1e-9
