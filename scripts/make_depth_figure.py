"""E6 depth-sweep figure: how the escape-operator ranking moves with QAOA depth.

A bump chart (slope/rank-trajectory plot): operators on a rank axis at p=3, 5, 8,
lines connecting each operator's rank across depths. Crossing lines are the whole
point -- they show the ranking does not transfer. Operators that lead at one
depth and fall at another stand out immediately.

Reads the three ablation_summary.json files (or the cd_report.json produced by
make_cd_diagram.py) and draws:
  - left panel : bump chart (rank vs depth), lines colour-keyed per operator
  - right panel: escape benefit (best-operator minus none) vs depth, annotated
                 with warm-start quality, showing escape rescues weak warm-starts

Usage
-----
    python scripts/make_depth_figure.py \
        --summaries p3=results/ablate_escape11_p3/ablation_summary.json \
                    p5=results/ablate_escape11_tuned/ablation_summary.json \
                    p8=results/ablate_escape11_p8/ablation_summary.json \
        --warmstart 3=0.782 5=0.882 8=0.650 \
        --out figures/depth_sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# metaheuristic operators only (exclude controls none/random from the ranking)
CANDIDATES = ["de", "ga", "pso", "aco", "woa", "gwo", "hho", "cmaes", "cpo"]


def per_instance_ranks(summary_path):
    d = json.load(open(summary_path, encoding="utf-8"))
    pi = d["per_instance"]
    ops = [o for o in CANDIDATES if o in pi]
    N = min(len(pi[o]) for o in ops)
    mat = np.column_stack([np.array(pi[o][:N]) for o in ops])
    ranks = np.zeros_like(mat)
    for i in range(N):
        ranks[i] = stats.rankdata(-mat[i])   # 1 = best
    mean_rank = {o: float(ranks[:, k].mean()) for k, o in enumerate(ops)}
    mean_ar = {o: float(mat[:, k].mean()) for k, o in enumerate(ops)}
    none_ar = float(np.mean(d["per_instance"]["none"][:N])) \
        if "none" in d["per_instance"] else None
    return ops, mean_rank, mean_ar, none_ar, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--warmstart", nargs="*",
                    default=["3=0.782", "5=0.882", "8=0.650"])
    ap.add_argument("--out", default="figures/depth_sweep")
    args = ap.parse_args()

    depth_paths = {}
    for spec in args.summaries:
        label, path = spec.split("=", 1)
        d = int(label.lstrip("p"))
        depth_paths[d] = path
    depths = sorted(depth_paths)          # [3, 5, 8]
    ws = {int(k): float(v) for k, v in (s.split("=") for s in args.warmstart)}

    ranks, ars, none_ars = {}, {}, {}
    all_ops = None
    for d in depths:
        ops, mr, mar, none_ar, N = per_instance_ranks(depth_paths[d])
        ranks[d], ars[d], none_ars[d] = mr, mar, none_ar
        all_ops = ops if all_ops is None else all_ops

    k = len(all_ops)
    # distinct colours; highlight operators that ever reach rank 1-2
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    colours = {o: cmap[i % 10] for i, o in enumerate(all_ops)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    # ---- left: bump chart ------------------------------------------------
    xs = np.arange(len(depths))

    def spread(labels_ys, min_gap=0.42):
        """Given [(op, y)], nudge y's apart by min_gap, preserving order."""
        items = sorted(labels_ys, key=lambda t: t[1])
        for i in range(1, len(items)):
            if items[i][1] - items[i - 1][1] < min_gap:
                items[i] = (items[i][0], items[i - 1][1] + min_gap)
        return dict(items)

    left_y = spread([(o, ranks[depths[0]][o]) for o in all_ops])
    right_y = spread([(o, ranks[depths[-1]][o]) for o in all_ops])

    for o in all_ops:
        ys = [ranks[d][o] for d in depths]
        tops = any(ranks[d][o] <= 1.5 for d in depths)
        axL.plot(xs, ys, "-o", color=colours[o],
                 lw=2.4 if tops else 1.2,
                 alpha=1.0 if tops else 0.5,
                 markersize=6 if tops else 4, zorder=3 if tops else 2)
        axL.text(xs[0] - 0.10, left_y[o], o, ha="right", va="center",
                 fontsize=9, color=colours[o],
                 fontweight="bold" if tops else "normal")
        axL.text(xs[-1] + 0.10, right_y[o], o, ha="left", va="center",
                 fontsize=9, color=colours[o],
                 fontweight="bold" if tops else "normal")

    axL.set_xticks(xs)
    axL.set_xticklabels([f"p={d}" for d in depths], fontsize=11)
    axL.set_ylabel("mean rank (1 = best)", fontsize=11)
    axL.set_ylim(k + 0.5, 0.5)             # invert: rank 1 at top
    axL.set_yticks(range(1, k + 1))
    axL.set_xlim(-0.9, len(depths) - 0.1)
    axL.set_title("Operator ranking vs QAOA depth\n(crossing lines = ranking "
                  "does not transfer)", fontsize=11)
    axL.grid(axis="y", ls=":", alpha=0.4)

    # ---- right: escape benefit vs warm-start -----------------------------
    best_gain, best_op = [], []
    for d in depths:
        bo = max(all_ops, key=lambda o: ars[d][o])
        gain = ars[d][bo] - none_ars[d] if none_ars[d] is not None else np.nan
        best_gain.append(gain)
        best_op.append(bo)
    wsq = [ws.get(d, np.nan) for d in depths]

    ax2 = axR
    b = ax2.bar(range(len(depths)), best_gain, color="0.6", width=0.55,
                zorder=2)
    ax2.set_xticks(range(len(depths)))
    ax2.set_xticklabels([f"p={d}\n({bo})" for d, bo in zip(depths, best_op)],
                        fontsize=10)
    ax2.set_ylabel("escape benefit  (best operator − none)", fontsize=10)
    ax2.set_title("Escape rescues weak warm-starts", fontsize=11)
    for i, (g, w) in enumerate(zip(best_gain, wsq)):
        ax2.text(i, g + 0.003, f"+{g:.3f}", ha="center", fontsize=9)
        ax2.text(i, 0.002, f"warm-start\n{w:.3f}", ha="center", va="bottom",
                 fontsize=8, color="white", fontweight="bold")
    ax2.set_ylim(0, max(best_gain) * 1.25)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    # correlation annotation
    if all(not np.isnan(w) for w in wsq):
        r = np.corrcoef(wsq, best_gain)[0, 1]
        ax2.text(0.5, 0.92, f"r(warm-start, benefit) = {r:+.2f}",
                 transform=ax2.transAxes, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.tight_layout()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{outp}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{outp}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {outp}.png / .pdf")

    # print the numbers behind the figure
    print("\nRank by depth:")
    hdr = "  operator " + " ".join(f"p={d:>2}" for d in depths)
    print(hdr)
    for o in all_ops:
        print(f"  {o:>8} " + " ".join(f"{ranks[d][o]:>4.1f}" for d in depths))
    print("\nEscape benefit (best-none) and warm-start:")
    for d, g, bo in zip(depths, best_gain, best_op):
        print(f"  p={d}: best={bo} gain=+{g:.4f}  warm-start={ws.get(d)}")


if __name__ == "__main__":
    main()
