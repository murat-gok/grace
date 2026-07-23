"""GATE-1 diagnostic: does restricting gamma to [0, pi] cost us anything?

Why this matters
----------------
For UNWEIGHTED MaxCut the cost Hamiltonian has integer eigenvalues, so
exp(-i*gamma*H_C) is periodic in gamma and the search can be restricted to a
short interval. This benchmark uses CONTINUOUS weights, w_ij ~ U(0.1, 1.0), so
that periodicity does not hold and the [0, pi] box used throughout the codebase
is a genuine restriction that may exclude optima.

This script answers the empirical question directly, cheaply, and fairly:
run the SAME global optimizer with the SAME evaluation budget under two boxes,

    A:  gamma in [0, pi],    beta in [0, pi]      (current code)
    B:  gamma in [0, 2*pi),  beta in [0, pi]      (correct for weighted MaxCut)

and compare the best expected cut found. It also reports how often the best
gamma found in box B actually lands in the excluded region [pi, 2pi), which is
the mechanism that would explain any gap.

Interpretation
--------------
  * B beats A materially (say > 0.002 in mean approximation ratio, consistent
    sign) -> the restriction is costing quality. Do the full fix: widen the
    bounds everywhere, retrain the GNN, re-run the tables.
  * B ~ A -> the restriction is harmless on this benchmark. Report it as a
    control experiment and a one-paragraph note; no retraining needed.

Usage
-----
    python scripts/diagnose_gamma_bounds.py --config configs/hard2gnn.yaml \
        --n-instances 6 --maxiter 60 --seed 0
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
from scipy.stats import wilcoxon

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut


def _search(qaoa, gamma_hi, maxiter, popsize, seed):
    """Budget-matched global search inside a given gamma box.

    differential_evolution is used because it respects box bounds natively and
    is a global method -- exactly the right instrument for 'are there better
    optima over there?'. polish=False keeps the evaluation budget identical
    between the two boxes.
    """
    p = qaoa.p
    n_evals = {"n": 0}

    def obj(x):
        n_evals["n"] += 1
        return qaoa.cost(x)

    bounds = [(0.0, gamma_hi)] * p + [(0.0, np.pi)] * p
    res = differential_evolution(
        obj, bounds, maxiter=maxiter, popsize=popsize, seed=seed,
        tol=0.0, mutation=(0.5, 1.0), recombination=0.7,
        polish=False, init="latinhypercube",
    )
    return np.asarray(res.x, dtype=float), qaoa.expected_cut(res.x), n_evals["n"]


def _one_instance(g, p, maxiter, popsize, seed):
    qaoa = QAOAMaxCut(g, p=p)
    opt = brute_force_maxcut(g)
    xa, cut_a, ev_a = _search(qaoa, np.pi, maxiter, popsize, seed)
    xb, cut_b, ev_b = _search(qaoa, 2 * np.pi, maxiter, popsize, seed)
    gam_b = xb[:p]
    return {
        "opt": opt,
        "ar_A": cut_a / opt,
        "ar_B": cut_b / opt,
        "evals_A": ev_a,
        "evals_B": ev_b,
        # mechanism: did box B actually use the excluded region?
        "n_gamma_above_pi": int((gam_b > np.pi).sum()),
        "max_gamma_B": float(gam_b.max()),
        "gammas_B": [float(v) for v in gam_b],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yaml")
    ap.add_argument("--n-instances", type=int, default=6,
                    help="Instances PER FAMILY (test split).")
    ap.add_argument("--maxiter", type=int, default=60,
                    help="differential_evolution generations (same for A and B).")
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/diag_gamma_bounds.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    p = cfg["qaoa_p"]
    n_nodes = cfg["n_nodes"]
    weighted = cfg.get("weighted", False)
    n_jobs = cfg.get("n_jobs", 8)

    if not weighted:
        print("NOTE: config has weighted=false. With unit weights gamma IS "
              "periodic and the [0,pi] box is justified; this diagnostic is "
              "only meaningful for weighted graphs.")

    jobs = []
    for fam in cfg["families"]:
        graphs = make_dataset(fam, n_graphs=args.n_instances, n_nodes=n_nodes,
                              weighted=weighted, split="test")
        for gi, g in enumerate(graphs):
            jobs.append((fam, gi, g))

    print(f"GATE-1 diagnostic: gamma box [0,pi] vs [0,2pi)")
    print(f"  p={p}, n={n_nodes}, weighted={weighted}, "
          f"{len(jobs)} instances, DE maxiter={args.maxiter}, "
          f"popsize={args.popsize}")

    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_instance)(g, p, args.maxiter, args.popsize,
                               args.seed + 97 * gi)
        for (fam, gi, g) in jobs)

    rows = []
    for (fam, gi, _), r in zip(jobs, out):
        rows.append({"family": fam, "instance": gi, **r})

    a = np.array([r["ar_A"] for r in rows])
    b = np.array([r["ar_B"] for r in rows])
    d = b - a
    ev_a = np.array([r["evals_A"] for r in rows])
    ev_b = np.array([r["evals_B"] for r in rows])

    print("\n=== Budget check (must be comparable for the test to be fair) ===")
    print(f"  mean evals A [0,pi]   : {ev_a.mean():.0f}")
    print(f"  mean evals B [0,2pi)  : {ev_b.mean():.0f}")

    print("\n=== Best approximation ratio found ===")
    print(f"  A  gamma in [0,pi]    : {a.mean():.6f}  (std {a.std():.6f})")
    print(f"  B  gamma in [0,2pi)   : {b.mean():.6f}  (std {b.std():.6f})")
    print(f"  mean difference B - A : {d.mean():+.6f}")
    print(f"  median difference     : {np.median(d):+.6f}")
    print(f"  B better / worse / tie: {(d > 1e-9).sum()} / "
          f"{(d < -1e-9).sum()} / {(np.abs(d) <= 1e-9).sum()}")

    if np.any(np.abs(d) > 1e-12):
        try:
            w = wilcoxon(b, a)
            print(f"  Wilcoxon p            : {w.pvalue:.4g}")
        except Exception as e:
            print(f"  Wilcoxon unavailable  : {e}")

    n_above = np.array([r["n_gamma_above_pi"] for r in rows])
    print("\n=== Mechanism: did box B use the excluded region [pi, 2pi)? ===")
    print(f"  instances with >=1 gamma above pi : {(n_above > 0).sum()}/{len(rows)}")
    print(f"  mean count of gammas above pi     : {n_above.mean():.2f} of {p}")
    print(f"  max gamma found in B              : "
          f"{max(r['max_gamma_B'] for r in rows):.4f}  (pi={np.pi:.4f})")

    print("\n=== Per family ===")
    for fam in cfg["families"]:
        sel = [r for r in rows if r["family"] == fam]
        da = np.array([r["ar_B"] - r["ar_A"] for r in sel])
        print(f"  {fam:16s} mean diff {da.mean():+.6f}  "
              f"(B better on {(da > 1e-9).sum()}/{len(sel)})")

    verdict_gap = d.mean()
    print("\n=== VERDICT ===")
    if verdict_gap > 0.002 and (d > 1e-9).sum() >= 0.7 * len(d):
        print("  Box B is MATERIALLY better. The [0,pi] restriction is costing")
        print("  quality -> do the full fix (widen bounds, retrain GNN, re-run).")
    elif verdict_gap > 0.0005:
        print("  Box B is slightly better. Borderline: widen the bounds for the")
        print("  escape operators and baselines (cheap), and decide on the GNN")
        print("  retrain after seeing whether the gap survives at n=60.")
    else:
        print("  No material difference. The [0,pi] restriction is harmless on")
        print("  this benchmark -> report as a control experiment plus a short")
        print("  note in the paper; no retraining required.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": {"p": p, "n_nodes": n_nodes, "weighted": weighted,
                          "maxiter": args.maxiter, "popsize": args.popsize,
                          "n_instances_per_family": args.n_instances,
                          "seed": args.seed},
               "rows": rows,
               "summary": {"mean_ar_A": float(a.mean()),
                           "mean_ar_B": float(b.mean()),
                           "mean_diff": float(d.mean()),
                           "n_B_better": int((d > 1e-9).sum()),
                           "n_total": len(rows),
                           "mean_evals_A": float(ev_a.mean()),
                           "mean_evals_B": float(ev_b.mean())}},
              open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}  ({round(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
