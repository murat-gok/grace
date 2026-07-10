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
from grace_qaoa.rl.refiner import RLRefiner
from grace_qaoa.baselines_strong import (interp_baseline, fourier_baseline,
                                         spsa_baseline)
from grace_qaoa.utils.checkpoint import CheckpointStore

# Budget checkpoints (quantum evaluations) at which we read achieved quality.
BUDGETS = [10, 25, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3000]


def trace_for_method(g, cfg, method, gnn, rl_model, seed):
    """Run one method with trace recording; return best-cut-at-budget for each
    budget checkpoint, normalized by the optimum (approximation ratio)."""
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)
    cap = max(BUDGETS)
    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cap, record_trace=True)

    try:
        if method == "grace":
            init = gnn.predict_params(g).reshape(-1)
            rl_step = (RLRefiner(rl_model, cq, steps_per_round=cfg.get(
                "rl_steps_per_round", 30)) if rl_model else None)
            ctrl = GraceController(cq, escape=cfg["escape"],
                                   stall_eps=cfg.get("stall_eps", 1e-3),
                                   stall_patience=cfg.get("stall_patience", 2),
                                   max_rounds=cfg["grace_rounds"], seed=seed)
            ctrl.run(init, rl_step_fn=rl_step)
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
    ap.add_argument("--config", default="configs/hard2gnn.yml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--rl-model", default=None)
    ap.add_argument("--rl-algo", default="td3")
    ap.add_argument("--n-instances", type=int, default=12)
    ap.add_argument("--n-runs", type=int, default=5,
                    help="runs per (instance, method); curves averaged.")
    ap.add_argument("--run-name", default="sampeff")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    gnn = load_gnn_warmstart(args.gnn_model)
    rl_model = None
    if args.rl_model:
        from stable_baselines3 import SAC, TD3
        rl_model = {"td3": TD3, "sac": SAC}[args.rl_algo].load(
            args.rl_model, device="cpu")

    methods = ["interp", "fourier", "spsa", "grace"]
    graphs = []
    per_fam = max(1, args.n_instances // len(cfg["families"]))
    for fam in cfg["families"]:
        for g in make_dataset(fam, n_graphs=per_fam, n_nodes=cfg["n_nodes"],
                              weighted=cfg.get("weighted", True), split="test"):
            graphs.append(g)

    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / args.run_name)
    if store.n_completed:
        print(f"Resuming '{args.run_name}': {store.n_completed} cells done.")

    total = len(graphs) * len(methods)
    for gi, g in enumerate(graphs):
        for m in methods:
            cid = f"inst{gi}|{m}"
            if store.is_done(cid):
                continue
            # Use the SAME seed pattern as run_experiment (1000*gi+run) so the
            # curves are computed on the same conditions as the main table and
            # cannot disagree with it due to a lucky/unlucky instance subset.
            nr = 1 if m in ("interp", "fourier") else args.n_runs
            curves = [trace_for_method(g, cfg, m, gnn, rl_model, seed=1000 * gi + r)
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
               "n_instances": len(set(r["instance"] for r in rows))}
    json.dump(summary, open(store.run_dir / "sample_efficiency.json", "w"),
              indent=2)
    print(f"\nSaved -> {store.run_dir / 'sample_efficiency.json'}")
    # quick verdict: does GRACE dominate at every budget?
    dominates = all(table["grace"][i] >= max(table[m][i] for m in
                    ["interp", "fourier", "spsa"]) for i in range(len(BUDGETS)))
    print("GRACE dominates at EVERY budget level." if dominates
          else "GRACE does not dominate at all budgets (see crossover above).")


if __name__ == "__main__":
    main()
