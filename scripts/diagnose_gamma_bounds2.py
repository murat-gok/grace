"""GATE-1 diagnostic v2: is the gamma in [0, pi] box ACTIVE (binding)?

Why v1 was not conclusive
-------------------------
v1 gave both boxes the same evaluation budget and found the wide box [0, 2pi)
performed WORSE. That is a statement about search efficiency, not about where
optima live: widening gamma doubles each of p dimensions, so the search volume
grows by 2^p (32x at p=5) while the budget stays fixed. The wide box loses
because it is spread thinner, which tells us nothing about whether good optima
exist beyond pi.

What this script tests instead
------------------------------
The right question is whether the constraint is BINDING. Three independent
checks, none of which depends on matching search efficiency across volumes:

  1. BOUNDARY ACTIVITY. Optimize thoroughly inside [0, pi] and look at where
     the best gammas land. If they sit strictly in the interior, the constraint
     is inactive and therefore cannot be affecting any reported result. If they
     pile up against pi, the optimizer is being clipped and the box is costing
     us something.

  2. ESCAPE TEST. Start a local optimizer FROM the restricted optimum, but let
     it run in the unrestricted box. If gamma stays below pi, the restricted
     optimum is also a local optimum of the wider problem. If it escapes above
     pi and improves, the box was hiding a better basin next door.

  3. CONVERGENCE-LIMITED COMPARISON. Give the wide box a budget scaled up by
     the volume ratio, so both searches are equally converged rather than
     equally cheap, then compare the best cut found.

Verdict logic distinguishes 'wide box better', 'no difference', and 'wide box
worse', which v1 conflated.

Usage
-----
    python scripts/diagnose_gamma_bounds2.py --config configs/hard2gnn.yaml \
        --n-instances 4 --maxiter 80
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
from scipy.optimize import differential_evolution, minimize
from scipy.stats import wilcoxon

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut


def _de(qaoa, gamma_hi, maxiter, popsize, seed):
    p = qaoa.p
    n = {"c": 0}

    def obj(x):
        n["c"] += 1
        return qaoa.cost(x)

    bounds = [(0.0, gamma_hi)] * p + [(0.0, np.pi)] * p
    res = differential_evolution(obj, bounds, maxiter=maxiter, popsize=popsize,
                                 seed=seed, tol=0.0, polish=True,
                                 init="latinhypercube")
    return np.asarray(res.x, dtype=float), qaoa.expected_cut(res.x), n["c"]


def _one_instance(g, p, maxiter, popsize, seed, wide_budget_mult):
    qaoa = QAOAMaxCut(g, p=p)
    opt = brute_force_maxcut(g)

    # --- 1. thorough search INSIDE the restricted box -------------------
    xa, cut_a, ev_a = _de(qaoa, np.pi, maxiter, popsize, seed)
    gam_a = xa[:p]

    # --- 2. escape test: local search from xa, unrestricted gamma -------
    # Wide bounds, warm-started at the restricted optimum. If the restricted
    # optimum is also a local optimum of the wider problem, gamma stays put.
    wide_bounds = [(0.0, 2 * np.pi)] * p + [(0.0, np.pi)] * p
    n_esc = {"c": 0}

    def obj_esc(x):
        n_esc["c"] += 1
        return qaoa.cost(x)

    res_esc = minimize(obj_esc, xa, method="L-BFGS-B", bounds=wide_bounds,
                       options={"maxiter": 200})
    x_esc = np.asarray(res_esc.x, dtype=float)
    cut_esc = qaoa.expected_cut(x_esc)
    gam_esc = x_esc[:p]

    # --- 3. convergence-limited wide search -----------------------------
    # Scale the budget with the volume ratio so the wide box is not simply
    # spread thinner than the narrow one.
    xb, cut_b, ev_b = _de(qaoa, 2 * np.pi, int(maxiter * wide_budget_mult),
                          popsize, seed)
    gam_b = xb[:p]

    return {
        "opt": opt,
        "ar_A": cut_a / opt,
        "ar_escape": cut_esc / opt,
        "ar_B": cut_b / opt,
        "evals_A": ev_a, "evals_escape": n_esc["c"], "evals_B": ev_b,
        # boundary activity inside the restricted box
        "gammas_A": [float(v) for v in gam_a],
        "max_gamma_A": float(gam_a.max()),
        "n_gamma_A_near_pi": int((gam_a > 0.95 * np.pi).sum()),
        # did the escape test leave the restricted box?
        "max_gamma_escape": float(gam_esc.max()),
        "n_gamma_escape_above_pi": int((gam_esc > np.pi).sum()),
        "escape_gain": float((cut_esc - cut_a) / opt),
        # where the converged wide search ended up
        "max_gamma_B": float(gam_b.max()),
        "n_gamma_B_above_pi": int((gam_b > np.pi).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yaml")
    ap.add_argument("--n-instances", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--wide-budget-mult", type=float, default=4.0,
                    help="Budget multiplier for the wide box, to offset its "
                         "larger volume (default 4x).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/diag_gamma_bounds2.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    p, n_nodes = cfg["qaoa_p"], cfg["n_nodes"]
    weighted = cfg.get("weighted", False)

    jobs = []
    for fam in cfg["families"]:
        for gi, g in enumerate(make_dataset(fam, n_graphs=args.n_instances,
                                            n_nodes=n_nodes, weighted=weighted,
                                            split="test")):
            jobs.append((fam, gi, g))

    print("GATE-1 diagnostic v2: is the [0,pi] gamma box binding?")
    print(f"  p={p}, n={n_nodes}, weighted={weighted}, {len(jobs)} instances")
    print(f"  wide-box budget multiplier: {args.wide_budget_mult}x")

    t0 = time.time()
    out = Parallel(n_jobs=cfg.get("n_jobs", 8), verbose=5)(
        delayed(_one_instance)(g, p, args.maxiter, args.popsize,
                               args.seed + 97 * gi, args.wide_budget_mult)
        for (fam, gi, g) in jobs)
    rows = [{"family": f, "instance": i, **r} for (f, i, _), r in zip(jobs, out)]

    a = np.array([r["ar_A"] for r in rows])
    e = np.array([r["ar_escape"] for r in rows])
    b = np.array([r["ar_B"] for r in rows])
    max_gam_a = np.array([r["max_gamma_A"] for r in rows])
    near_pi = np.array([r["n_gamma_A_near_pi"] for r in rows])
    esc_above = np.array([r["n_gamma_escape_above_pi"] for r in rows])
    esc_gain = np.array([r["escape_gain"] for r in rows])

    print("\n=== 1. BOUNDARY ACTIVITY (is the constraint even touched?) ===")
    print(f"  largest gamma found inside [0,pi] : {max_gam_a.max():.4f} "
          f"(pi = {np.pi:.4f})")
    print(f"  mean of per-instance max gamma    : {max_gam_a.mean():.4f}")
    print(f"  instances with any gamma > 0.95pi : {(near_pi > 0).sum()}/{len(rows)}")
    if (near_pi > 0).sum() == 0:
        print("  -> constraint is INACTIVE: optima are interior, the box cannot")
        print("     be affecting any reported result.")
    else:
        print("  -> constraint is touched on some instances; see the escape test.")

    print("\n=== 2. ESCAPE TEST (local search from the restricted optimum) ===")
    print(f"  instances where gamma escaped above pi : "
          f"{(esc_above > 0).sum()}/{len(rows)}")
    print(f"  mean gain from escaping                : {esc_gain.mean():+.6f}")
    print(f"  max  gain from escaping                : {esc_gain.max():+.6f}")
    if (esc_above > 0).sum() == 0:
        print("  -> the restricted optimum is also a local optimum of the wider")
        print("     problem: no better basin immediately next door.")

    print("\n=== 3. CONVERGENCE-LIMITED COMPARISON ===")
    ev_a = np.mean([r["evals_A"] for r in rows])
    ev_b = np.mean([r["evals_B"] for r in rows])
    print(f"  mean evals  A [0,pi]  : {ev_a:.0f}")
    print(f"  mean evals  B [0,2pi) : {ev_b:.0f}  ({ev_b/ev_a:.1f}x)")
    print(f"  best AR   A           : {a.mean():.6f}")
    print(f"  best AR   B           : {b.mean():.6f}")
    d = b - a
    print(f"  mean difference B - A : {d.mean():+.6f}")
    print(f"  B better/worse/tie    : {(d > 1e-9).sum()} / {(d < -1e-9).sum()}"
          f" / {(np.abs(d) <= 1e-9).sum()}")
    if np.any(np.abs(d) > 1e-12):
        try:
            print(f"  Wilcoxon p            : {wilcoxon(b, a).pvalue:.4g}")
        except Exception as exc:
            print(f"  Wilcoxon unavailable  : {exc}")

    print("\n=== VERDICT ===")
    constraint_inactive = (near_pi > 0).sum() == 0 and (esc_above > 0).sum() == 0
    if constraint_inactive and d.mean() <= 0.0005:
        print("  The [0,pi] box is NOT BINDING on this benchmark: optima are")
        print("  interior, local search does not escape past pi, and a better-")
        print("  converged wide search finds nothing better.")
        print("  -> No retraining needed. Report as a control experiment and a")
        print("     short note that gamma is not periodic under continuous")
        print("     weights but the restriction is empirically inactive here.")
    elif d.mean() > 0.002 or esc_gain.max() > 0.002:
        print("  The wide box finds MATERIALLY better optima. The restriction is")
        print("  costing quality -> do the full fix (widen bounds everywhere,")
        print("  retrain the GNN, re-run the tables).")
    elif (near_pi > 0).sum() > 0 or (esc_above > 0).sum() > 0:
        print("  Mixed: the constraint is touched on some instances but the gain")
        print("  from relaxing it is small. Widen the bounds for the escape")
        print("  operators and baselines (cheap, no retraining) and re-check at")
        print("  n=60 before deciding on the GNN.")
    else:
        print("  Wide box is not better; constraint appears inactive or")
        print("  irrelevant. Treat as a control experiment.")

    print("\n=== Per family (best AR inside [0,pi]) ===")
    for fam in cfg["families"]:
        sel = [r for r in rows if r["family"] == fam]
        print(f"  {fam:16s} A {np.mean([r['ar_A'] for r in sel]):.6f}   "
              f"max gamma {max(r['max_gamma_A'] for r in sel):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": {"p": p, "n_nodes": n_nodes, "weighted": weighted,
                          "maxiter": args.maxiter, "popsize": args.popsize,
                          "wide_budget_mult": args.wide_budget_mult},
               "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}  ({round(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
