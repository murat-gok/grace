"""Critical-difference (CD) diagrams for the escape-operator ablation.

A CD diagram is the standard Swarm-and-Evolutionary-Computation way to show a
multi-method comparison: methods are placed on an axis by mean rank, and any set
whose mean ranks differ by less than the critical difference is joined by a bar,
meaning they are statistically indistinguishable. It answers, in one picture,
"is ACO's win real, and which operators are tied with it?"

Statistics
----------
  1. Friedman test across all operators (already computed in the ablation, but
     recomputed here from per_instance so the diagram is self-contained).
  2. Mean ranks per operator (rank 1 = best on an instance; averaged over
     instances). Ranks are computed per instance from the per_instance arrays.
  3. Nemenyi critical difference:
        CD = q_alpha * sqrt(k*(k+1) / (6*N))
     where k = #operators, N = #instances, q_alpha is the Studentized-range
     critical value / sqrt(2) at alpha=0.05. We also report Conover-Holm
     adjacent-pair significance as a stronger cross-check, since Nemenyi is
     conservative.

Draws one diagram per input depth (p=3, p=5, p=8) plus prints a table, so the
depth-dependence of the ranking (E6) is visible in the same artifact.

Usage
-----
    python scripts/make_cd_diagram.py \
        --summaries p3=results/ablate_escape11_p3/ablation_summary.json \
                    p5=results/ablate_escape11_tuned/ablation_summary.json \
                    p8=results/ablate_escape11_p8/ablation_summary.json \
        --out figures/cd
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Studentized range q_alpha divided by sqrt(2), alpha=0.05, infinite df.
# Standard Nemenyi table values (Demsar 2006), indexed by number of methods k.
Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def load_matrix(summary_path, exclude=("none",)):
    """Return (ops, matrix[N_instances, k]) of per-instance approximation ratios.

    'none' is excluded by default: it is the control, not a candidate operator,
    and including an always-last method inflates the apparent separation.
    """
    d = json.load(open(summary_path, encoding="utf-8"))
    pi = d["per_instance"]
    ops = [o for o in d["operators"] if o not in exclude and o in pi]
    N = min(len(pi[o]) for o in ops)
    mat = np.column_stack([np.array(pi[o][:N]) for o in ops])
    return ops, mat


def mean_ranks(mat):
    """Per-instance ranks (1 = best = highest approx ratio), averaged."""
    # rank within each row; higher value -> better -> lower rank number
    ranks = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        ranks[i] = stats.rankdata(-mat[i])
    return ranks.mean(axis=0), ranks


def nemenyi_cd(k, N, alpha=0.05):
    q = Q05.get(k)
    if q is None:
        return None
    return q * np.sqrt(k * (k + 1) / (6.0 * N))


def conover_holm_adjacent(mat, ops, order):
    """Conover post-hoc p-values for adjacent operators in the ranking, Holm-
    corrected. A stronger cross-check than Nemenyi for 'is the winner separated
    from #2'."""
    # pairwise Wilcoxon on the columns, adjacent pairs in `order`
    raw = {}
    for a, b in zip(order[:-1], order[1:]):
        ia, ib = ops.index(a), ops.index(b)
        x, y = mat[:, ia], mat[:, ib]
        if np.allclose(x, y):
            raw[(a, b)] = 1.0
        else:
            raw[(a, b)] = float(stats.wilcoxon(x, y).pvalue)
    # Holm
    items = sorted(raw.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (pair, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[pair] = running
    return out


def draw_cd(ops, avg_ranks, cd, title, path):
    """Classic Demsar CD diagram: horizontal rank axis, methods hung off it,
    a CD bar, and cliques (bars joining statistically-tied methods)."""
    k = len(ops)
    order = np.argsort(avg_ranks)          # best (lowest rank) first
    names = [ops[i] for i in order]
    ranks = avg_ranks[order]

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 2.6 + 0.16 * k))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1.25)
    ax.axis("off")

    # title well above everything
    ax.text((lo + hi) / 2, 1.20, title, ha="center", va="top", fontsize=11,
            fontweight="bold")

    # top axis line with ticks
    y0 = 0.80
    ax.plot([lo, hi], [y0, y0], "k-", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y0, y0 + 0.03], "k-", lw=1)
        ax.text(r, y0 + 0.055, str(r), ha="center", va="bottom", fontsize=9)
    ax.text((lo + hi) / 2, y0 + 0.15, "mean rank (1 = best)",
            ha="center", fontsize=9)

    # CD bar (above the axis, left-aligned, clear of the tick labels)
    if cd is not None:
        cdy = y0 + 0.115
        ax.plot([lo, lo + cd], [cdy, cdy], "k-", lw=2.5)
        ax.plot([lo, lo], [cdy - 0.015, cdy + 0.015], "k-", lw=2.5)
        ax.plot([lo + cd, lo + cd], [cdy - 0.015, cdy + 0.015], "k-", lw=2.5)
        ax.text(lo + cd / 2, cdy + 0.02, f"CD = {cd:.2f}", ha="center",
                fontsize=9)

    # hang methods: left half on the left, right half on the right
    half = (k + 1) // 2
    for idx, (nm, r) in enumerate(zip(names, ranks)):
        if idx < half:
            yy = y0 - 0.08 - idx * 0.09
            ax.plot([r, r], [y0, yy], "k-", lw=0.8)
            ax.plot([r, lo - 0.4], [yy, yy], "k-", lw=0.8)
            ax.text(lo - 0.45, yy, f"{nm} ({r:.2f})", ha="right", va="center",
                    fontsize=9)
        else:
            j = idx - half
            yy = y0 - 0.08 - j * 0.09
            ax.plot([r, r], [y0, yy], "k-", lw=0.8)
            ax.plot([r, hi + 0.4], [yy, yy], "k-", lw=0.8)
            ax.text(hi + 0.45, yy, f"{nm} ({r:.2f})", ha="left", va="center",
                    fontsize=9)

        # cliques: connect consecutive methods within CD of each other
        clique_y = y0 - 0.045
        i = 0
        drawn = 0
        while i < k:
            j = i
            while j + 1 < k and (ranks[j + 1] - ranks[i]) <= cd:
                j += 1
            if j > i:
                yy = clique_y - drawn * 0.02
                ax.plot([ranks[i] - 0.05, ranks[j] + 0.05], [yy, yy],
                        "k-", lw=3, solid_capstyle="round")
                drawn += 1
            i = j + 1 if j > i else i + 1

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="depth=path pairs, e.g. p5=results/.../ablation_summary.json")
    ap.add_argument("--out", default="figures/cd")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    report = {}
    for spec in args.summaries:
        label, path = spec.split("=", 1)
        ops, mat = load_matrix(path)
        N, k = mat.shape
        avg, ranks = mean_ranks(mat)

        # Friedman from per-instance ranks
        fr = stats.friedmanchisquare(*[mat[:, c] for c in range(k)])
        cd = nemenyi_cd(k, N, args.alpha)

        order_idx = np.argsort(avg)
        order = [ops[i] for i in order_idx]
        conover = conover_holm_adjacent(mat, ops, order)

        print(f"\n=== {label}: {k} operators, N={N} instances ===")
        print(f"Friedman chi2={fr.statistic:.1f}, p={fr.pvalue:.3g}")
        print(f"Nemenyi CD (alpha={args.alpha}) = "
              f"{cd:.3f}" if cd else "CD unavailable")
        print("mean ranks (best first):")
        for o in order:
            print(f"  {o:>7}: {avg[ops.index(o)]:.2f}")
        print("adjacent-pair Conover-Holm p (is each gap significant?):")
        for (a, b), p in conover.items():
            sig = "***" if p < 0.001 else "**" if p < 0.01 else \
                  "*" if p < 0.05 else "ns"
            print(f"  {a:>7} vs {b:<7}: p={p:.3g} {sig}")

        fig_path = outdir / f"cd_{label}.png"
        draw_cd(ops, avg, cd, f"Escape operators, {label} (N={N})", fig_path)
        print(f"  saved -> {fig_path}")

        report[label] = {
            "operators": ops, "N": int(N),
            "mean_ranks": {o: float(avg[ops.index(o)]) for o in ops},
            "order_best_first": order,
            "friedman": {"chi2": float(fr.statistic), "p": float(fr.pvalue)},
            "nemenyi_cd": None if cd is None else float(cd),
            "conover_holm_adjacent": {f"{a}>{b}": float(p)
                                      for (a, b), p in conover.items()},
        }

    json.dump(report, open(outdir / "cd_report.json", "w"), indent=2)
    print(f"\nSaved report -> {outdir / 'cd_report.json'}")


if __name__ == "__main__":
    main()
