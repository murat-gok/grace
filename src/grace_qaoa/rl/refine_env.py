"""RL refinement environment: an agent nudges (gamma, beta) to improve the cut.

State  : current params + landscape features (energy, gradient norm).
Action : continuous deltas applied to the params (TD3/SAC compatible).
Reward : improvement in expected cut, minus a small depth/step penalty.

Used by stable-baselines3 TD3 or SAC. The QAOA evaluation inside step() is the
compute cost; everything else is negligible.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from grace_qaoa.quantum.qaoa import QAOAMaxCut


class QAOARefineEnv(gym.Env):
    """One episode = refine the parameters of one graph for `horizon` steps."""

    metadata = {"render_modes": []}

    def __init__(self, qaoa: QAOAMaxCut, init_params: np.ndarray,
                 horizon: int = 30, step_scale: float = 0.15,
                 optimal_cut: float | None = None):
        super().__init__()
        self.qaoa = qaoa
        self.p = qaoa.p
        self.dim = 2 * self.p
        self.init_params = np.asarray(init_params, dtype=np.float32).reshape(-1)
        self.horizon = horizon
        self.step_scale = step_scale
        self.optimal_cut = optimal_cut

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.dim,), dtype=np.float32)
        # Observation: params (dim) + [normalized_cut, grad_norm, step_frac]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.dim + 3,), dtype=np.float32)
        self._reset_state()

    def _reset_state(self):
        self.params = self.init_params.copy()
        self.t = 0
        self.best_cut = self.qaoa.expected_cut(self.params)

    def _grad_norm(self) -> float:
        """Cheap finite-difference gradient norm as a landscape feature."""
        eps = 1e-2
        base = self.qaoa.expected_cut(self.params)
        g = np.zeros(self.dim)
        for k in range(self.dim):
            shifted = self.params.copy()
            shifted[k] += eps
            g[k] = (self.qaoa.expected_cut(shifted) - base) / eps
        return float(np.linalg.norm(g))

    def _obs(self) -> np.ndarray:
        cut = self.qaoa.expected_cut(self.params)
        denom = self.optimal_cut if self.optimal_cut else max(cut, 1e-6)
        feats = np.array([cut / denom, self._grad_norm(),
                          self.t / self.horizon], dtype=np.float32)
        return np.concatenate([self.params.astype(np.float32), feats])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.params = np.clip(self.params + self.step_scale * action, 0.0, np.pi)
        cut = self.qaoa.expected_cut(self.params)
        reward = float(cut - self.best_cut)          # reward = improvement
        reward -= 0.001 * self.p                      # small depth penalty
        self.best_cut = max(self.best_cut, cut)
        self.t += 1
        terminated = False
        truncated = self.t >= self.horizon
        return self._obs(), reward, terminated, truncated, {"cut": cut}

    @property
    def current_params(self) -> np.ndarray:
        return self.params.copy()
