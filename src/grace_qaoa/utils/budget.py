"""Quantum-evaluation counting for fair-budget comparison.

The central fairness question a Q1 reviewer will ask: does GRACE win because it
produces better cuts, or because it simply spends more circuit evaluations?
To answer it, EVERY method must be charged for each call to the quantum
expectation. This wrapper makes that accounting explicit and uniform.

A "quantum evaluation" = one call to expected_cut (one circuit expectation).
Finite-difference gradients therefore cost (1 + dim) evaluations, which is
exactly the hidden cost we must expose for the RL refiner.
"""
from __future__ import annotations

import numpy as np

from grace_qaoa.quantum.qaoa import QAOAMaxCut


class CountingQAOA:
    """Wraps a QAOAMaxCut and counts every expected_cut call.

    Use this everywhere a method touches the quantum device so the budget is
    accounted identically across baselines and GRACE.
    """

    def __init__(self, qaoa: QAOAMaxCut, budget: int | None = None,
                 record_trace: bool = False):
        self.qaoa = qaoa
        self.p = qaoa.p
        self.n_evals = 0
        self.budget = budget          # optional hard cap (raises BudgetExhausted)
        self.best_cut_so_far = -np.inf
        self.evals_to_target = None   # filled in by track_target()
        self._target = None
        # Sample-efficiency support: record (n_evals, best_cut) whenever best
        # improves, so we can read off achieved quality at any budget level.
        self.record_trace = record_trace
        self.trace = []               # list of (n_evals, best_cut_so_far)

    def set_target(self, target_cut: float):
        """Record the eval count at which we first reach target_cut."""
        self._target = target_cut

    def expected_cut(self, params) -> float:
        if self.budget is not None and self.n_evals >= self.budget:
            raise BudgetExhausted(self.n_evals)
        self.n_evals += 1
        cut = self.qaoa.expected_cut(params)
        if cut > self.best_cut_so_far:
            self.best_cut_so_far = cut
            if self.record_trace:
                self.trace.append((self.n_evals, cut))
        if (self._target is not None and self.evals_to_target is None
                and cut >= self._target):
            self.evals_to_target = self.n_evals
        return cut

    def best_cut_at_budget(self, budget: int) -> float:
        """Best cut achieved within the first `budget` evaluations (from trace)."""
        best = -np.inf
        for ne, c in self.trace:
            if ne <= budget:
                best = c
            else:
                break
        return best

    def cost(self, params) -> float:
        return -self.expected_cut(params)

    def reset_counter(self):
        self.n_evals = 0
        self.best_cut_so_far = -np.inf
        self.evals_to_target = None
        self.trace = []


class BudgetExhausted(Exception):
    """Raised when a method exceeds its allotted quantum-evaluation budget."""

    def __init__(self, n_evals):
        super().__init__(f"Quantum evaluation budget exhausted at {n_evals} evals.")
        self.n_evals = n_evals
