"""Phase-3 figures: warm-start ablation (E2) and size scalability (E7).

Two publication figures from data already computed -- no new runs.

  1. warmstart_ablation : grouped bars, three warm-starts (random / fixed-angle
     table / GNN) at one evaluation and after the escape loop. Shows the GNN
     beats a graph-agnostic median-angle table by a wide margin (the learned
     warm-start does real work), and how much the escape loop adds to each.

  2. size_scaling : approximation ratio vs graph size for INTERP, FOURIER and
     GRACE. Shows the schedule heuristics degrade with n while GRACE stays
     nearly flat, so GRACE's margin widens with problem size.

Usage
-----
    python scripts/make_phase3_figures.py \
        --warmstart results/ablate_warmstart/warmstart_ablation.json \
        --size results/robust_size_aco_v2/size_scalability.json \
        --out figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_warmstart(path, outdir):
    d = json.load(open(path, encoding="utf-8"))
    m, s = d["means"], d.get("stds", {})
    starts = [("random", "Random"), ("fixed", "Fixed-angle table"),
              ("gnn", "GNN warm-start")]

    one = [m[f"{k}_only"] for k, _ in starts]
    loop = [m[f"{k}_escape"] for k, _ in starts]
    one_e = [s.get(f"{k}_only", 0) for k, _ in starts]
    loop_e = [s.get(f"{k}_escape", 0) for k, _ in starts]

    x = np.arange(len(starts))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w / 2, one, w, yerr=one_e, capsize=3,
                color="0.72", label="1 evaluation", zorder=2)
    b2 = ax.bar(x + w / 2, loop, w, yerr=loop_e, capsize=3,
                color="#3b6ea5", label="+ escape loop", zorder=2)

    for xi, (v1, v2) in enumerate(zip(one, loop)):
        ax.text(xi - w / 2, v1 + 0.012, f"{v1:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, v2 + 0.012, f"{v2:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in starts], fontsize=10)
    ax.set_ylabel("mean approximation ratio", fontsize=11)
    ax.set_ylim(0.55, 1.0)
    ax.set_title("Warm-start ablation (n=60, matched budget)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)

    # annotate the GNN-vs-fixed gap at one evaluation (the headline)
    comp = d.get("comparisons", {}).get("gnn_vs_fixed_1eval", {})
    if comp:
        gap = comp.get("delta")
        ax.annotate("", xy=(2 - w / 2, one[2]), xytext=(1 - w / 2, one[1]),
                    arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.3))
        ax.text(1.5 - w / 2 - 0.28, (one[1] + one[2]) / 2 + 0.02,
                f"GNN vs fixed:\n+{gap:.3f} (60/60)", ha="right", va="center",
                fontsize=8.5, color="crimson",
                bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.95))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"warmstart_ablation.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {outdir / 'warmstart_ablation.png'} / .pdf")


def fig_size(path, outdir):
    d = json.load(open(path, encoding="utf-8"))
    sizes = sorted(int(k) for k in d)
    methods = [("interp", "INTERP", "o", "#c07a2b"),
               ("fourier", "FOURIER", "s", "#4a8a4a"),
               ("grace", "GRACE", "D", "#3b6ea5")]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, lab, mk, col in methods:
        ys = [d[str(n)][key] for n in sizes]
        lw = 2.6 if key == "grace" else 1.6
        ax.plot(sizes, ys, marker=mk, label=lab, color=col, lw=lw,
                markersize=6, zorder=3 if key == "grace" else 2)

    ax.set_xticks(sizes)
    ax.set_xlabel("graph size  n  (vertices)", fontsize=11)
    ax.set_ylabel("mean approximation ratio", fontsize=11)
    ax.set_title("Scalability with problem size (tuned ACO escape)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(ls=":", alpha=0.4)

    # shade GRACE's lead where it is ahead of the best baseline
    best_base = [max(d[str(n)]["interp"], d[str(n)]["fourier"]) for n in sizes]
    grace = [d[str(n)]["grace"] for n in sizes]
    ax.fill_between(sizes, best_base, grace,
                    where=[g >= b for g, b in zip(grace, best_base)],
                    color="#3b6ea5", alpha=0.10, zorder=1)

    # annotate the widening gap at the largest size
    n_last = sizes[-1]
    gap = grace[-1] - best_base[-1]
    ax.annotate(f"+{gap:.3f}", xy=(n_last, grace[-1]),
                xytext=(n_last - 1.2, grace[-1] + 0.006), fontsize=9,
                color="#3b6ea5",
                arrowprops=dict(arrowstyle="->", color="#3b6ea5"))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"size_scaling.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {outdir / 'size_scaling.png'} / .pdf")

    print("\nSize scaling (mean approx ratio):")
    print(f"  {'n':>4} {'INTERP':>8} {'FOURIER':>8} {'GRACE':>8} {'lead':>7}")
    for n in sizes:
        b = max(d[str(n)]["interp"], d[str(n)]["fourier"])
        print(f"  {n:>4} {d[str(n)]['interp']:>8.4f} "
              f"{d[str(n)]['fourier']:>8.4f} {d[str(n)]['grace']:>8.4f} "
              f"{d[str(n)]['grace'] - b:>+7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_warmstart(args.warmstart, outdir)
    fig_size(args.size, outdir)


if __name__ == "__main__":
    main()
