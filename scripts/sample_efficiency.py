"""Sample-efficiency: achieved approximation ratio vs quantum-evaluation budget.

This answers the cost objection head-on: GRACE uses ~1216 evals, gnn_only uses 1,
INTERP ~437. A reviewer asks "is GRACE's quality worth the cost?". The honest
answer is a CURVE: quality as a function of budget, for every method. If GRACE
dominates at every budget level, the story is "more efficient", not "more
expensive". If a baseline wins at small budgets, we report that too.

Method: run each method ONCE per instance with trace recording on, then read off
the best cut achieved within each budget checkpoint. gnn_only is a flat line
(1 eval). GRACE includes the GNN warm-start as its first eval, so its curve
starts where gnn_only is and climbs.

Crash-safe: each (instance, method) trace summary is checkpointed.

Run:
    python scripts/sample_efficiency.py --config configs/hard2gnn.yml \
        --gnn-model models/gnn_warmstart_p5_gcn.pt \
        --rl-model models/rl_refiner_td3 --n-instances 12 --run-name sampeff
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.gnn.loader import load_gnn_warmstart
from grace_qaoa.controller.grace import GraceController
from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted
from grace_qaoa.baselines_strong import (interp_baseline, fourier_baseline,
                                         spsa_baseline)
from grace_qaoa.utils.checkpoint import CheckpointStore

# Budget checkpoints (quantum evaluations) at which we read achieved quality.
BUDGETS = [10, 25, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3000]


def trace_for_method(g, cfg, method, gnn, seed, escape, escape_kwargs):
    """Run one method with trace recording; return best-cut-at-budget for each
    budget checkpoint, normalized by the optimum (approximation ratio)."""
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)

    if method == "gnn_only":
        # single evaluation: a flat line at the warm-start's quality.
        init = gnn.predict_params(g).reshape(-1)
        ar = QAOAMaxCut(g, p=p).expected_cut(init) / opt
        return {b: ar for b in BUDGETS}

    cap = max(BUDGETS)
    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cap, record_trace=True)
    try:
        if method == "grace":
            init = gnn.predict_params(g).reshape(-1)
            ctrl = GraceController(cq, escape=escape,
                                   stall_eps=cfg.get("stall_eps", 1e-3),
                                   stall_patience=cfg.get("stall_patience", 2),
                                   max_rounds=cfg["grace_rounds"], seed=seed,
                                   escape_kwargs=escape_kwargs)
            ctrl.run(init, refiner="none")   # paper's two-stage method
        elif method == "interp":
            interp_baseline(cq, target_p=p, seed=seed,
                            maxiter_per_level=cfg.get("interp_maxiter", 60))
        elif method == "fourier":
            fourier_baseline(cq, target_p=p, seed=seed,
                             maxiter=cfg.get("fourier_maxiter", 120))
        elif method == "spsa":
            spsa_baseline(cq, target_p=p, iters=cfg.get("spsa_iters", 400),
                          seed=seed)
    except BudgetExhausted:
        pass

    return {b: (cq.best_cut_at_budget(b) / opt if cq.best_cut_at_budget(b) > 0
                else 0.0) for b in BUDGETS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yaml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--escape", default=None,
                    help="Escape operator (overrides config). Use 'aco'.")
    ap.add_argument("--tuned", default=None,
                    help="Path to tuned_operators.json for the chosen escape.")
    ap.add_argument("--n-instances", type=int, default=12)
    ap.add_argument("--n-runs", type=int, default=5,
                    help="runs per (instance, method); curves averaged.")
    ap.add_argument("--run-name", default="sampeff")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    gnn = load_gnn_warmstart(args.gnn_model)
    escape = args.escape or cfg["escape"]
    escape_kwargs = {}
    if args.tuned:
        escape_kwargs = json.load(open(args.tuned)).get("tuned", {}).get(escape, {})
    print(f"Escape: {escape}  tuned kwargs: {escape_kwargs}")

    # gnn_only is the strongest single-evaluation competitor and MUST be shown:
    # it is the flat line GRACE starts from, and it reveals that GRACE's low-
    # budget advantage comes from the warm-start, not the escape loop.
    methods = ["interp", "fourier", "spsa", "gnn_only", "grace"]
    graphs = []
    per_fam = max(1, args.n_instances // len(cfg["families"]))
    for fam in cfg["families"]:
        for g in make_dataset(fam, n_graphs=per_fam, n_nodes=cfg["n_nodes"],
                              weighted=cfg.get("weighted", True), split="test"):
            graphs.append(g)

    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / args.run_name)
    store.write_provenance(cfg=cfg, extra={"script": "sample_efficiency",
                                           "escape": escape,
                                           "escape_kwargs": escape_kwargs})
    if store.n_completed:
        print(f"Resuming '{args.run_name}': {store.n_completed} cells done.")

    total = len(graphs) * len(methods)
    for gi, g in enumerate(graphs):
        for m in methods:
            cid = f"inst{gi}|{m}"
            if store.is_done(cid):
                continue
            # deterministic methods run once; stochastic ones averaged.
            nr = 1 if m in ("interp", "fourier", "gnn_only") else args.n_runs
            curves = [trace_for_method(g, cfg, m, gnn, seed=1000 * gi + r,
                                       escape=escape, escape_kwargs=escape_kwargs)
                      for r in range(nr)]
            avg = {b: float(np.mean([c[b] for c in curves])) for b in BUDGETS}
            store.append(cid, {"instance": gi, "method": m, "curve": avg})
            print(f"  [{store.n_completed}/{total}] inst{gi} {m}: "
                  f"AR@max={avg[max(BUDGETS)]:.4f}")

    # --- aggregate curves across instances ---
    rows = store.load_all_records()
    by_method = {m: {b: [] for b in BUDGETS} for m in methods}
    for r in rows:
        for b in BUDGETS:
            by_method[r["method"]][b].append(r["curve"][str(b)]
                                             if str(b) in r["curve"]
                                             else r["curve"][b])

    print(f"\n{'budget':>8} " + " ".join(f"{m:>9}" for m in methods))
    table = {m: [] for m in methods}
    for b in BUDGETS:
        line = f"{b:>8} "
        for m in methods:
            v = float(np.mean(by_method[m][b])) if by_method[m][b] else 0.0
            table[m].append(v)
            line += f"{v:>9.4f} "
        print(line)

    summary = {"budgets": BUDGETS,
               "curves": {m: table[m] for m in methods},
               "escape": escape, "escape_kwargs": escape_kwargs,
               "n_instances": len(set(r["instance"] for r in rows))}
    json.dump(summary, open(store.run_dir / "sample_efficiency.json", "w"),
              indent=2)
    print(f"\nSaved -> {store.run_dir / 'sample_efficiency.json'}")

    # Where does GRACE overtake gnn_only? Below that budget the two are equal
    # (GRACE IS the warm-start until the first escape fires), so the low-budget
    # advantage over the iterative baselines is the GNN's, not the escape's.
    others = ["interp", "fourier", "spsa", "gnn_only"]
    print(f"\nGRACE vs gnn_only by budget (equal until escape fires):")
    for i, b in enumerate(BUDGETS):
        eq = abs(table["grace"][i] - table["gnn_only"][i]) < 1e-6
        marker = "  (= gnn_only)" if eq else ""
        print(f"  budget {b:>4}: grace={table['grace'][i]:.4f}"
              f"  gnn_only={table['gnn_only'][i]:.4f}{marker}")
    dominates = all(table["grace"][i] >= max(table[m][i] for m in others)
                    for i in range(len(BUDGETS)))
    print("\nGRACE >= every baseline at EVERY budget level." if dominates
          else "\nGRACE is overtaken at some budget (see table above).")


if __name__ == "__main__":
    main()
