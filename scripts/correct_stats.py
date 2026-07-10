"""Correct statistical analysis at the instance level (fixes pseudoreplication).

The main-experiment summary reported Wilcoxon p-values computed over all 1,800
runs. Those runs are NOT independent: 30 runs share each of 60 instances (same
graph, same optimum). The correct unit of analysis is the 60 per-instance means.
This script recomputes, from the per-run checkpoint of the main experiment:

  - per-instance mean approximation ratio for each method (n = 60)
  - paired Wilcoxon signed-rank tests on those 60 means (GRACE vs each baseline)
  - rank-biserial correlation / Cliff's delta as effect sizes
  - Holm correction across the family of comparisons

Reporting p-values on 60 independent instances (minimum attainable Wilcoxon
p ~ 2^-60 ~ 1e-18) is honest; the 1e-163 figures were an artifact of treating
dependent runs as independent.

Run:
    python scripts/correct_stats.py --checkpoint results/hard_p5_aco_n60/checkpoint.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    gt = sum((a[i] > b[j]) for i in range(n) for j in range(n))
    lt = sum((a[i] < b[j]) for i in range(n) for j in range(n))
    return (gt - lt) / (n * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="per-run checkpoint.jsonl of the main experiment")
    ap.add_argument("--target", default="grace")
    args = ap.parse_args()

    # checkpoint rows are WIDE: each row is one (instance, run) with a
    # '<method>_ar' column per method. Reduce to per-instance means per method.
    per = defaultdict(lambda: defaultdict(list))  # method -> instance -> [ar]
    with open(args.checkpoint) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            inst = r.get("instance")
            fam = r.get("family", "")
            if inst is None:
                continue
            # The 'instance' field is a within-family index (0..19); combine it
            # with 'family' so the three families' instances are not collapsed.
            inst = f"{fam}:{inst}"
            for key, val in r.items():
                if key.endswith("_ar") and val is not None:
                    method = key[:-3]  # strip '_ar'
                    per[method][inst].append(val)

    if not per:
        print("Could not parse per-run records; check checkpoint schema.")
        return

    methods = sorted(per.keys())
    # per-instance means
    inst_means = {m: {i: float(np.mean(v)) for i, v in per[m].items()}
                  for m in methods}
    common = set.intersection(*[set(inst_means[m].keys()) for m in methods])
    common = sorted(common)
    n = len(common)
    print(f"methods: {methods}")
    print(f"paired instances (unit of analysis): {n}\n")

    means = {m: np.array([inst_means[m][i] for i in common]) for m in methods}
    print(f"{'method':>10} {'mean':>8} {'std(inst)':>10}")
    for m in sorted(methods, key=lambda k: means[k].mean(), reverse=True):
        print(f"{m:>10} {means[m].mean():>8.4f} {means[m].std():>10.4f}")

    if args.target not in means:
        print(f"\nTarget '{args.target}' not found.")
        return

    tgt = means[args.target]
    others = [m for m in methods if m != args.target]
    print(f"\nPaired Wilcoxon on {n} per-instance means "
          f"({args.target} vs each), Holm-corrected:")
    raw = {}
    for m in others:
        if np.allclose(tgt, means[m]):
            raw[m] = 1.0
            continue
        _, pv = stats.wilcoxon(tgt, means[m])
        raw[m] = pv
    # Holm correction
    order = sorted(raw, key=lambda k: raw[k])
    holm = {}
    k = len(order)
    for idx, m in enumerate(order):
        holm[m] = min(1.0, raw[m] * (k - idx))
    for m in sorted(others, key=lambda k: means[k].mean(), reverse=True):
        delta = cliffs_delta(tgt, means[m])
        wins = int((tgt > means[m]).sum())
        print(f"  vs {m:>10}: p_raw={raw[m]:.3g}  p_holm={holm[m]:.3g}  "
              f"Cliff={delta:+.2f}  ({wins}/{n} instances)")

    print("\nNote: with n=60 paired instances the minimum attainable Wilcoxon "
          "p-value\nis ~2e-18; report these instance-level values, not run-level "
          "figures.")


if __name__ == "__main__":
    main()
