"""Tests for the RL refinement pieces. Kept tiny so they run fast on CPU."""
import numpy as np

from grace_qaoa.rl.multigraph_env import MultiGraphRefineEnv
from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut
from grace_qaoa.controller.grace import GraceController


def test_multigraph_env_steps():
    env = MultiGraphRefineEnv(families=["regular"], n_nodes=6, p=1,
                              horizon=5, pool_size=4, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (2 * 1 + 3,)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert "cut" in info
    assert trunc  # episode ends at horizon


def test_controller_with_injected_refiner():
    # A trivial 'policy' that nudges params toward a fixed point, just to verify
    # the controller accepts an rl_step_fn and uses it without error.
    g = make_dataset("regular", n_graphs=1, n_nodes=6, seed=1)[0]
    qaoa = QAOAMaxCut(g, p=1)

    def fake_refiner(params):
        return np.clip(np.asarray(params) + 0.05, 0, np.pi)

    ctrl = GraceController(qaoa, escape="de", max_rounds=3, seed=0)
    out = ctrl.run(np.array([0.3, 0.3]), rl_step_fn=fake_refiner)
    assert out["n_quantum_evals"] > 0
    assert "best_cut" in out
