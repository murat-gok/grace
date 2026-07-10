"""Rank all pairwise ensemble "oracles" from an existing escape ablation.

No new experiments. Reads the per-instance scores already in an ablation
checkpoint (results/<run>/checkpoint.jsonl) and, for every pair of operators,
computes the ORACLE: the per-instance best of the two, averaged. The oracle is
the ceiling any budget-doubling ensemble of that pair could reach (and an upper
bound -- a real budget-NEUTRAL ensemble does worse).

This answers: "is there a pair whose oracle clearly beats ACO alone?" If yes,
that pair is the best ensemble candidate. If the best oracle barely exceeds the
best single operator, no ensemble is worth building.

Run:
    python scripts/rank_pair_oracles.py --checkpoint results/ablate_escape11_n60/checkpoint.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to an ablation checkpoint.jsonl")
    ap.add_argument("--top", type=int, default=15, help="how many pairs to show")
    args = ap.parse_args()

    # load per-instance scores: scores[escape][instance] = mean_ar
    scores = {}
    with open(args.checkpoint) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            scores.setdefault(r["escape"], {})[r["instance"]] = r["mean_ar"]

    operators = sorted(scores.keys())
    # instances present for ALL operators (fair, paired)
    common = set.intersection(*[set(scores[o].keys()) for o in operators])
    common = sorted(common)
    n = len(common)
    print(f"operators: {operators}")
    print(f"paired instances: {n}\n")

    # single-operator means (baseline reference)
    singles = {o: np.mean([scores[o][i] for i in common]) for o in operators}
    best_single = max(singles, key=singles.get)
    print("Single-operator means:")
    for o in sorted(singles, key=singles.get, reverse=True):
        tag = "  <- best single" if o == best_single else ""
        print(f"  {o:>20}: {singles[o]:.4f}{tag}")
    print()

    # exclude controls and ensembles from PAIR building (we pair base operators)
    skip = {"none", "random", "aco_cpo_portfolio", "aco_cpo_cascade"}
    base_ops = [o for o in operators if o not in skip]

    # oracle for every unordered pair
    rows = []
    for a, b in itertools.combinations(base_ops, 2):
        arr_a = np.array([scores[a][i] for i in common])
        arr_b = np.array([scores[b][i] for i in common])
        oracle = np.maximum(arr_a, arr_b).mean()
        # complementarity: how often each is strictly the unique winner
        a_wins = int((arr_a > arr_b).sum())
        b_wins = int((arr_b > arr_a).sum())
        gain = oracle - max(singles[a], singles[b])
        rows.append((f"{a}+{b}", oracle, gain, a_wins, b_wins))

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"Top {args.top} pairs by ORACLE (per-instance best of the two):")
    print(f"{'pair':>22} {'oracle':>8} {'gain_vs_best_single':>20} "
          f"{'split(A/B wins)':>16}")
    for name, oracle, gain, aw, bw in rows[:args.top]:
        print(f"{name:>22} {oracle:>8.4f} {gain:>+20.4f} {f'{aw}/{bw}':>16}")

    print(f"\nBest single operator: {best_single} = {singles[best_single]:.4f}")
    best_pair = rows[0]
    print(f"Best pair oracle:     {best_pair[0]} = {best_pair[1]:.4f} "
          f"({best_pair[2]:+.4f} over best single)")
    print()
    if best_pair[2] < 0.002:
        print("VERDICT: no pair's oracle meaningfully beats the best single "
              "operator.\n  Even an IDEAL (budget-doubling) ensemble would barely "
              "help; a budget-\n  neutral one will not. Recommend: use the best "
              "single operator alone.")
    else:
        # is the winning pair complementary, or does one dominate?
        name, oracle, gain, aw, bw = best_pair
        if min(aw, bw) < 0.15 * n:
            print(f"VERDICT: best pair oracle is higher, BUT it is lopsided "
                  f"({aw}/{bw}) --\n  one operator dominates, so a real ensemble "
                  f"will mostly track the\n  stronger one. Ensemble upside is "
                  f"limited.")
        else:
            print(f"VERDICT: '{name}' shows GENUINE complementarity ({aw}/{bw} "
                  f"split) and its\n  oracle clears the best single by "
                  f"{gain:+.4f}. This pair is the best\n  ensemble candidate -- "
                  f"worth a budget-neutral portfolio/cascade test.")


if __name__ == "__main__":
    main()
