"""Adapter: wrap a trained SB3 policy as an `rl_step_fn(params) -> params`.

The GRACE controller calls rl_step_fn to perform one RL-refinement *round*
(several env-style steps). This adapter rolls the policy forward for
`steps_per_round` greedy steps starting from the given params, using the same
observation features the policy was trained on, and returns the refined params.

IMPORTANT (fair-budget): the observation uses only ZERO-extra-cost landscape
signals (normalized cut, recent improvement/momentum, stall fraction). It does
NOT compute a finite-difference gradient, so each refinement step costs exactly
ONE quantum evaluation (the cut itself) instead of 1 + 2p. This keeps GRACE's
budget accounting honest and competitive at high p.
"""
from __future__ import annotations

import numpy as np

from grace_qaoa.quantum.qaoa import QAOAMaxCut


class RLRefiner:
    def __init__(self, model, qaoa: QAOAMaxCut, optimal_cut=None,
                 step_scale=0.15, steps_per_round=30, horizon=30):
        self.model = model
        self.qaoa = qaoa
        self.p = qaoa.p
        self.dim = 2 * qaoa.p
        self.optimal_cut = optimal_cut
        self.step_scale = step_scale
        self.steps_per_round = steps_per_round
        self.horizon = horizon

    def _obs(self, params, cut, last_cut, best_cut, stall_steps):
        # Scale by running-best (never the true optimum) -- matches the env and
        # avoids feeding the agent the answer.
        denom = max(abs(best_cut), 1e-6)
        momentum = (cut - last_cut) / denom
        stall_frac = stall_steps / max(self.horizon, 1)
        feats = np.array([cut / denom, momentum, stall_frac], dtype=np.float32)
        return np.concatenate([params.astype(np.float32), feats])

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """One refinement round: greedy-roll the policy from `params`.

        Cost: 1 quantum eval per step (plus 1 for the initial cut) -- no
        gradient evaluations.
        """
        cur = np.asarray(params, dtype=np.float32).reshape(-1)
        cut = self.qaoa.expected_cut(cur)        # 1 eval
        best, best_cut = cur.copy(), cut
        last_cut = cut
        stall = 0
        for t in range(self.steps_per_round):
            obs = self._obs(cur, cut, last_cut, best_cut, stall)
            action, _ = self.model.predict(obs, deterministic=True)
            cur = np.clip(cur + self.step_scale * np.clip(action, -1, 1),
                          0.0, np.pi).astype(np.float32)
            last_cut = cut
            cut = self.qaoa.expected_cut(cur)    # 1 eval per step
            if cut > best_cut:
                best, best_cut = cur.copy(), cut
                stall = 0
            else:
                stall += 1
        return best
