"""The GRACE-QAOA closed-loop controller.

Decides when to switch among three modes:
  1. GNN-predict      -> initial warm-start (once, at the start)
  2. RL-refine        -> online gradient-free policy steps
  3. metaheuristic-escape -> triggered when RL stalls (improvement < eps for T steps)

This threshold-based controller is the simple, defensible starting point. The
ablation 'learned controller' can replace `_should_escape` with a small policy
later; the interface stays identical.
"""
from __future__ import annotations

import numpy as np

from grace_qaoa.metaheuristic.escape import ESCAPE_OPERATORS, run_escape_fair
from grace_qaoa.quantum.qaoa import QAOAMaxCut


class GraceController:
    def __init__(self, qaoa: QAOAMaxCut, escape: str = "cpo",
                 stall_eps: float = 1e-3, stall_patience: int = 3,
                 max_rounds: int = 6, seed: int = 0, escape_evals: int = 180):
        self.qaoa = qaoa
        self.escape_name = escape
        self.escape_evals = escape_evals   # identical budget for every operator
        self.stall_eps = stall_eps
        self.stall_patience = stall_patience
        self.max_rounds = max_rounds
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []
        self.n_quantum_evals = 0
        self._rounds_since_improve = 0

    def _eval(self, params):
        self.n_quantum_evals += 1
        return self.qaoa.expected_cut(params)

    def _should_escape(self, improved_this_round: bool, best_cut: float) -> bool:
        """Trigger when the BEST-so-far has not improved meaningfully for
        `stall_patience` consecutive rounds. Relative-improvement based, so it
        is robust to a noisy refiner (unlike a raw window-width test).
        Disabled entirely when stall_eps < 0 (used for the escape-OFF ablation).
        """
        if self.stall_eps < 0:
            return False
        if improved_this_round:
            self._rounds_since_improve = 0
            return False
        self._rounds_since_improve += 1
        return self._rounds_since_improve >= self.stall_patience

    def run(self, init_params: np.ndarray, rl_step_fn=None,
            rl_steps_per_round: int = 30) -> dict:
        """Run the closed loop.

        rl_step_fn(params) -> refined_params is an injected callable so the
        controller does not hard-depend on stable-baselines3 (keeps tests light).
        If None, a simple coordinate hill-climb stands in for the RL refiner.
        """
        params = np.asarray(init_params, dtype=float).reshape(-1)
        best_params = params.copy()
        best_cut = self._eval(params)
        cut_trace = [best_cut]
        improve_eps = self.stall_eps if self.stall_eps > 0 else 1e-3

        for rnd in range(self.max_rounds):
            # --- RL-refine phase ---
            if rl_step_fn is not None:
                params = rl_step_fn(params).reshape(-1)
            else:
                params = self._hill_climb(params, rl_steps_per_round)
            cut = self._eval(params)
            cut_trace.append(cut)
            improved = cut > best_cut + improve_eps
            if cut > best_cut:
                best_cut, best_params = cut, params.copy()

            # --- escape decision (relative-improvement based) ---
            if self._should_escape(improved, best_cut):
                escaped = run_escape_fair(
                    self.escape_name,
                    cost_fn=lambda x: -self._eval(x),
                    x0=best_params.copy(),
                    rng=self.rng,
                    max_evals=self.escape_evals,
                )
                params = np.asarray(escaped, dtype=float).reshape(-1)
                cut = self._eval(params)
                cut_trace.append(cut)
                esc_improved = cut > best_cut
                if esc_improved:
                    best_cut, best_params = cut, params.copy()
                    self._rounds_since_improve = 0   # escape paid off, reset
                self.history.append({"round": rnd, "event": "escape",
                                     "cut": cut, "improved": bool(esc_improved)})

        return {
            "best_params": best_params,
            "best_cut": best_cut,
            "cut_trace": cut_trace,
            "n_quantum_evals": self.n_quantum_evals,
        }

    def _hill_climb(self, params, steps):
        """Stand-in refiner used when no RL policy is injected (and in tests)."""
        cur = params.copy()
        cur_cut = self._eval(cur)
        for _ in range(steps):
            cand = np.clip(cur + self.rng.normal(0, 0.1, cur.size), 0.0, np.pi)
            c = self._eval(cand)
            if c > cur_cut:
                cur, cur_cut = cand, c
        return cur
