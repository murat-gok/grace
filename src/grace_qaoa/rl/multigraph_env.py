"""Multi-graph wrapper so the RL refiner generalizes across instances.

A policy trained on a single graph overfits its landscape and is useless as a
warm-startable refiner. This env samples a fresh graph (and a fresh init) every
episode, so the learned policy is a *general* refinement operator: given any
(params, landscape features), output a good delta. That is exactly what the
GRACE controller needs as its `rl_step_fn`.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut


class MultiGraphRefineEnv(gym.Env):
    """Each episode: pick a random graph, refine its (gamma,beta) for `horizon` steps."""

    metadata = {"render_modes": []}

    def __init__(self, families, n_nodes=8, p=1, horizon=30, step_scale=0.15,
                 weighted=False, pool_size=40, seed=0):
        super().__init__()
        self.p = p
        self.dim = 2 * p
        self.horizon = horizon
        self.step_scale = step_scale
        self.rng = np.random.default_rng(seed)

        # Pre-build a pool of graphs (with cached optimal cuts) to sample from.
        self.pool = []
        per_family = max(1, pool_size // len(families))
        for fam in families:
            graphs = make_dataset(fam, n_graphs=per_family, n_nodes=n_nodes,
                                  weighted=weighted, base_seed=seed,
                                  split="train")
            for g in graphs:
                opt = brute_force_maxcut(g) if n_nodes <= 20 else None
                self.pool.append((g, opt))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.dim + 3,), dtype=np.float32)
        self._pick_new_episode()

    def _pick_new_episode(self):
        g, opt = self.pool[self.rng.integers(len(self.pool))]
        self.qaoa = QAOAMaxCut(g, p=self.p)
        self.optimal_cut = opt
        self.params = self.rng.uniform(0, np.pi, self.dim).astype(np.float32)
        self.t = 0
        self.best_cut = self.qaoa.expected_cut(self.params)
        self.last_cut = self.best_cut          # for the momentum feature
        self.stall_steps = 0                   # steps since last improvement

    def _obs(self, cut=None):
        # cut is passed in when already computed (step), else compute once.
        if cut is None:
            cut = self.qaoa.expected_cut(self.params)
        # Scale by RUNNING-BEST cut, never by the true optimum: the agent must
        # not see the answer (avoids the information-leakage objection a
        # reviewer would raise, and matches inference where opt is unknown).
        denom = max(abs(self.best_cut), 1e-6)
        momentum = (cut - self.last_cut) / denom
        stall_frac = self.stall_steps / max(self.horizon, 1)
        # normalized-cut feature uses running best as a self-referential scale
        feats = np.array([cut / denom, momentum, stall_frac], dtype=np.float32)
        return np.concatenate([self.params.astype(np.float32), feats])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pick_new_episode()
        return self._obs(self.best_cut), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.params = np.clip(self.params + self.step_scale * action,
                              0.0, np.pi).astype(np.float32)
        cut = self.qaoa.expected_cut(self.params)         # the ONLY eval per step
        reward = float(cut - self.best_cut) - 0.001 * self.p
        if cut > self.best_cut:
            self.best_cut = cut
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        obs = self._obs(cut)
        self.last_cut = cut
        self.t += 1
        truncated = self.t >= self.horizon
        return obs, reward, False, truncated, {"cut": cut}
