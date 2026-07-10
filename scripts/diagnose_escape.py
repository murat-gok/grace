"""Diagnostic: does the metaheuristic escape operator actually fire and help at high p?

We want, before committing to a long p=5 run, to confirm:
  (a) the escape operator TRIGGERS (otherwise GRACE is just GNN+RL and the CPO
      novelty is dead on the stage);
  (b) when it triggers it IMPROVES the cut (otherwise it is wasted budget);
  (c) escape-ON beats escape-OFF on the same instances.

Run:
    python scripts/diagnose_escape.py --p 5 --n-instances 6 --escape cpo
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import numpy as np

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.controller.grace import GraceController


def run_with_escape(qaoa, init, escape, stall_eps, stall_patience, max_rounds, seed):
    ctrl = GraceController(qaoa, escape=escape, stall_eps=stall_eps,
                           stall_patience=stall_patience, max_rounds=max_rounds,
                           seed=seed)
    res = ctrl.run(init)
    n_escapes = sum(1 for h in ctrl.history if h["event"] == "escape")
    # measure improvement attributable to escapes from the trace + history
    return res, n_escapes


def run_without_escape(qaoa, init, max_rounds, seed):
    # max_rounds rounds of hill-climb only, escape disabled via impossible eps.
    ctrl = GraceController(qaoa, escape="cpo", stall_eps=-1.0,  # never triggers
                           stall_patience=10**9, max_rounds=max_rounds, seed=seed)
    return ctrl.run(init)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-instances", type=int, default=6)
    ap.add_argument("--n-nodes", type=int, default=14)
    ap.add_argument("--escape", default="cpo", choices=["cpo", "de"])
    ap.add_argument("--stall-eps", type=float, default=1e-3)
    ap.add_argument("--stall-patience", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--weighted", action="store_true", default=True)
    args = ap.parse_args()

    graphs = []
    for fam in ["regular", "erdos_renyi", "watts_strogatz"]:
        graphs += make_dataset(fam, n_graphs=max(1, args.n_instances // 3),
                               n_nodes=args.n_nodes, weighted=args.weighted,
                               base_seed=0)

    total_escapes = 0
    on_ars, off_ars = [], []
    print(f"p={args.p}, {len(graphs)} weighted instances, escape={args.escape}")
    print(f"{'inst':>4} {'opt':>7} {'esc_on':>8} {'esc_off':>8} {'#esc':>5} {'delta':>7}")
    for gi, g in enumerate(graphs):
        opt = brute_force_maxcut(g) if g.number_of_nodes() <= 20 else None
        init = np.random.default_rng(gi).uniform(0, np.pi, 2 * args.p)

        qaoa_on = QAOAMaxCut(g, p=args.p)
        res_on, n_esc = run_with_escape(
            qaoa_on, init, args.escape, args.stall_eps, args.stall_patience,
            args.max_rounds, seed=gi)

        qaoa_off = QAOAMaxCut(g, p=args.p)
        res_off = run_without_escape(qaoa_off, init, args.max_rounds, seed=gi)

        ar_on = res_on["best_cut"] / opt if opt else res_on["best_cut"]
        ar_off = res_off["best_cut"] / opt if opt else res_off["best_cut"]
        on_ars.append(ar_on); off_ars.append(ar_off)
        total_escapes += n_esc
        print(f"{gi:>4} {opt:>7.2f} {ar_on:>8.4f} {ar_off:>8.4f} {n_esc:>5} "
              f"{ar_on - ar_off:>+7.4f}")

    print("-" * 45)
    print(f"mean approx: escape_ON={np.mean(on_ars):.4f}  "
          f"escape_OFF={np.mean(off_ars):.4f}  "
          f"delta={np.mean(on_ars) - np.mean(off_ars):+.4f}")
    print(f"total escapes triggered: {total_escapes} "
          f"(avg {total_escapes/len(graphs):.1f} per instance)")
    if total_escapes == 0:
        print("WARNING: escape never triggered -> loosen stall_eps or raise "
              "max_rounds / lower stall_patience.")


if __name__ == "__main__":
    main()
