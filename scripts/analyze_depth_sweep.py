"""E6 analysis: is the escape-operator ranking stable across QAOA depth?

Reads the three depth ablations (p=3, p=5, p=8), each an ablation_summary.json
from ablate_escape.py, and quantifies how the operator ranking moves with depth.

Outputs
-------
  1. Mean approximation ratio per operator at each depth, with the induced rank.
  2. Kendall's tau between every pair of depths (rank correlation): tau ~ 1 means
     the ranking is preserved, tau ~ 0 means unrelated, tau < 0 means reversed.
     Reported with a bootstrap 95% CI over instances so the reader sees whether
     a low tau is resolved or just noisy at n=30.
  3. Spearman rho as a second, magnitude-sensitive rank correlation.
  4. Each operator's rank trajectory across depths (who rises, who falls).
  5. The escape-benefit vs warm-start table: best-operator gain over 'none' at
     each depth against the (user-supplied) warm-start quality, to show escape
     matters most where the warm-start is weakest.

The metaheuristic operators only are ranked (excluding 'none' and 'random',
which are controls, not candidates) -- ranking controls would understate tau by
pinning two always-low entries.

Usage
-----
    python scripts/analyze_depth_sweep.py \
        --p3 results/ablate_escape11_p3/ablation_summary.json \
        --p5 results/ablate_escape11_tuned/ablation_summary.json \
        --p8 results/ablate_escape11_p8/ablation_summary.json \
        --warmstart 3=0.782 5=0.882 8=0.650
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy import stats

CANDIDATES = ["de", "ga", "pso", "aco", "woa", "gwo", "hho", "cmaes", "cpo"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rank_vector(means: dict, ops) -> np.ndarray:
    """Rank operators by mean (rank 1 = best). Returns ranks aligned to `ops`."""
    vals = np.array([means[o] for o in ops])
    # higher mean -> better -> smaller rank
    order = (-vals).argsort()
    ranks = np.empty(len(ops), dtype=float)
    ranks[order] = np.arange(1, len(ops) + 1)
    return ranks


def bootstrap_tau(pi_a: dict, pi_b: dict, ops, n_boot=2000, seed=0):
    """Bootstrap Kendall tau between two depths at the instance level.

    Both ablations must share instance indexing (same test seed range). We
    resample instances with replacement, recompute each operator's mean at both
    depths, rank, and take Kendall tau. Returns (point_tau, lo, hi).
    """
    rng = np.random.default_rng(seed)
    # per-instance matrices: rows = instances, cols = ops
    A = np.column_stack([np.array(pi_a[o]) for o in ops])
    B = np.column_stack([np.array(pi_b[o]) for o in ops])
    nA, nB = A.shape[0], B.shape[0]
    n = min(nA, nB)
    A, B = A[:n], B[:n]   # align on the shared instance count

    def one(idx):
        ra = rank_vector({o: A[idx, k].mean() for k, o in enumerate(ops)}, ops)
        rb = rank_vector({o: B[idx, k].mean() for k, o in enumerate(ops)}, ops)
        return stats.kendalltau(ra, rb).statistic

    point = one(np.arange(n))
    boots = np.array([one(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--p5", required=True)
    ap.add_argument("--p8", required=True)
    ap.add_argument("--warmstart", nargs="*", default=["3=0.782", "5=0.882",
                                                       "8=0.650"],
                    help="depth=warmstart_quality pairs for the benefit table.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/depth_sweep_analysis.json")
    args = ap.parse_args()

    depths = {3: load(args.p3), 5: load(args.p5), 8: load(args.p8)}
    ws = {int(k): float(v) for k, v in (s.split("=") for s in args.warmstart)}

    # Guard: every file must contain all candidate operators.
    for p, d in depths.items():
        missing = [o for o in CANDIDATES if o not in d["mean_approx"]]
        if missing:
            sys.exit(f"p={p} is missing operators {missing}; "
                     f"re-run its ablation with all 11 operators.")

    ops = CANDIDATES

    # -- 1. means and ranks per depth -------------------------------------
    print("Mean approximation ratio and rank (1=best), metaheuristics only")
    header = "  operator " + "".join(f"| p={p} (rank) " for p in (3, 5, 8))
    print(header)
    ranks = {}
    for p in (3, 5, 8):
        ranks[p] = rank_vector(depths[p]["mean_approx"], ops)
    for k, o in enumerate(ops):
        row = f"  {o:>8} "
        for p in (3, 5, 8):
            m = depths[p]["mean_approx"][o]
            row += f"| {m:.4f} ({int(ranks[p][k]):>2}) "
        print(row)

    # winners
    print("\nWinner by depth:")
    for p in (3, 5, 8):
        w = min(ops, key=lambda o: ranks[p][list(ops).index(o)])
        best = max(ops, key=lambda o: depths[p]["mean_approx"][o])
        print(f"  p={p}: {best}  ({depths[p]['mean_approx'][best]:.4f})")

    # -- 2. Kendall tau between depths (bootstrap CI) ----------------------
    print("\nKendall tau between depths (rank correlation, "
          "1=identical, 0=unrelated, <0=reversed)")
    pairs = [(3, 5), (5, 8), (3, 8)]
    tau_results = {}
    for a, b in pairs:
        # point tau from the full-sample ranks
        tau_pt = stats.kendalltau(ranks[a], ranks[b]).statistic
        rho = stats.spearmanr(ranks[a], ranks[b]).statistic
        # bootstrap CI if per_instance present in both
        if "per_instance" in depths[a] and "per_instance" in depths[b]:
            _, lo, hi = bootstrap_tau(depths[a]["per_instance"],
                                      depths[b]["per_instance"], ops,
                                      n_boot=args.n_boot)
            ci = f"  95% CI [{lo:+.2f}, {hi:+.2f}]"
        else:
            lo = hi = None
            ci = "  (no per-instance CI)"
        tau_results[f"p{a}_p{b}"] = {"tau": float(tau_pt), "spearman": float(rho),
                                     "ci_lo": None if lo is None else float(lo),
                                     "ci_hi": None if hi is None else float(hi)}
        print(f"  p={a} vs p={b}:  tau={tau_pt:+.2f}  rho={rho:+.2f}{ci}")

    mean_tau = np.mean([tau_results[k]["tau"] for k in tau_results])
    print(f"\n  mean pairwise Kendall tau = {mean_tau:+.2f}")
    if mean_tau < 0.3:
        print("  => ranking is essentially UNRELATED across depths: no operator")
        print("     is universally best. This is the core operator-agnostic")
        print("     finding -- the choice of operator must be made per regime.")
    elif mean_tau < 0.6:
        print("  => ranking is WEAKLY preserved: a broad tier is stable but the")
        print("     winner shifts with depth.")
    else:
        print("  => ranking is largely STABLE across depths.")

    # -- 3. rank trajectories ---------------------------------------------
    print("\nRank trajectory per operator (p3 -> p5 -> p8):")
    movers = []
    for k, o in enumerate(ops):
        r3, r5, r8 = int(ranks[3][k]), int(ranks[5][k]), int(ranks[8][k])
        swing = max(r3, r5, r8) - min(r3, r5, r8)
        movers.append((swing, o, r3, r5, r8))
    for swing, o, r3, r5, r8 in sorted(movers, reverse=True):
        arrow = f"{r3} -> {r5} -> {r8}"
        print(f"  {o:>8}: {arrow:>16}   (rank swing {swing})")

    # -- 4. escape benefit vs warm-start ----------------------------------
    print("\nEscape benefit vs warm-start quality:")
    print("  depth | warm-start | best operator | best-minus-none")
    benefit = {}
    for p in (3, 5, 8):
        d = depths[p]
        best = max(ops, key=lambda o: d["mean_approx"][o])
        gain = d["mean_approx"][best] - d["mean_approx"]["none"]
        benefit[p] = {"warmstart": ws.get(p), "best": best, "gain": float(gain)}
        wsq = ws.get(p)
        wss = f"{wsq:.3f}" if wsq is not None else "  ?  "
        print(f"    p={p} |   {wss}    | {best:>4} ({d['mean_approx'][best]:.3f}) "
              f"| {gain:+.4f}")
    # correlation of warm-start quality vs benefit, if we have all three
    if all(ws.get(p) is not None for p in (3, 5, 8)):
        wq = np.array([ws[p] for p in (3, 5, 8)])
        bg = np.array([benefit[p]["gain"] for p in (3, 5, 8)])
        r = np.corrcoef(wq, bg)[0, 1]
        print(f"\n  Pearson r(warm-start quality, escape benefit) = {r:+.2f}")
        print("  (strong negative => escape rescues exactly where the warm-start"
              " is weak.)")

    # -- controls: does escape still beat 'none' everywhere? ---------------
    print("\nControl check: does escape beat no-escape at every depth?")
    for p in (3, 5, 8):
        d = depths[p]
        worst_op = min([o for o in ops], key=lambda o: d["mean_approx"][o])
        beats = d["mean_approx"][worst_op] > d["mean_approx"]["none"]
        # random vs none, to check the 'structured > perturbation' claim
        rnd = d["mean_approx"].get("random")
        none = d["mean_approx"]["none"]
        rnd_gain = (rnd - none) if rnd is not None else float("nan")
        print(f"  p={p}: weakest metaheuristic ({worst_op}) "
              f"{'beats' if beats else 'does NOT beat'} none; "
              f"random-vs-none delta {rnd_gain:+.4f}")

    out = {
        "means": {p: {o: depths[p]["mean_approx"][o] for o in ops}
                  for p in (3, 5, 8)},
        "ranks": {p: {o: int(ranks[p][k]) for k, o in enumerate(ops)}
                  for p in (3, 5, 8)},
        "kendall_tau": tau_results,
        "mean_pairwise_tau": float(mean_tau),
        "escape_benefit": benefit,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
