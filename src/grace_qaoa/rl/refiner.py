"""Adapter: wrap a trained SB3 policy as an `rl_step_fn(params) -> params`.

The GRACE controller calls rl_step_fn to perform one RL-refinement *round*
(several env-style steps). This adapter rolls the policy forward for
`steps_per_round` greedy steps starting from the given params, using the same
observation features the policy was trained on, and returns the refined params.

IMPORTANT (fair-budget): the observation uses only landscape signals
(normalized cut, recent improvement/momentum, stall fraction), so no
finite-difference gradient is computed. Each refinement STEP still costs one
quantum evaluation (the cut itself). A full round therefore costs
`1 + steps_per_round` evaluations, exposed via `self.last_call_evals` so the
controller can add it to the shared budget counter. (Earlier versions did not
expose this, and the round's evaluations were invisible to the counter.)
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
        # Quantum evaluations spent in the most recent __call__, so the
        # controller can charge them to the shared fair-budget counter.
        self.last_call_evals = 0

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

        Cost: 1 quantum eval per step plus 1 for the initial cut, i.e.
        `1 + steps_per_round` evaluations, recorded in `self.last_call_evals`.
        No gradient evaluations are performed.
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
        # initial cut (1) + one cut per step
        self.last_call_evals = 1 + self.steps_per_round
        return best
