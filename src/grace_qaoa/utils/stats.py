"""Statistical validation (Phase 6 protocol): >=30 runs, Wilcoxon, Friedman, Holm.

These are the exact tests reviewers at Swarm & Evol. Comp. / ASC expect.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_vs_baseline(method_scores, baseline_scores):
    """Paired Wilcoxon signed-rank test. Scores are per-instance (paired)."""
    stat, p = stats.wilcoxon(method_scores, baseline_scores)
    return {"statistic": float(stat), "p_value": float(p)}


def friedman_test(score_matrix):
    """Friedman test across multiple methods.

    score_matrix: array (n_instances, n_methods). Returns chi2 stat + p-value.
    """
    arrs = [score_matrix[:, j] for j in range(score_matrix.shape[1])]
    stat, p = stats.friedmanchisquare(*arrs)
    return {"statistic": float(stat), "p_value": float(p)}


def average_ranks(score_matrix, higher_is_better: bool = True):
    """Average rank of each method across instances (1 = best)."""
    s = -score_matrix if higher_is_better else score_matrix
    ranks = np.apply_along_axis(stats.rankdata, 1, s)
    return ranks.mean(axis=0)


def holm_posthoc(score_matrix, control_idx: int, higher_is_better: bool = True):
    """Holm-corrected pairwise Wilcoxon of each method vs a control method."""
    n_methods = score_matrix.shape[1]
    raw = []
    for j in range(n_methods):
        if j == control_idx:
            raw.append((j, 1.0))
            continue
        _, p = stats.wilcoxon(score_matrix[:, control_idx], score_matrix[:, j])
        raw.append((j, float(p)))
    others = sorted([r for r in raw if r[0] != control_idx], key=lambda x: x[1])
    m = len(others)
    corrected = {}
    for rank, (j, p) in enumerate(others):
        corrected[j] = min(1.0, p * (m - rank))
    corrected[control_idx] = 1.0
    return corrected
