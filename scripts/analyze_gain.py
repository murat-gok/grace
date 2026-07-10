"""Per-instance decomposition: how much does GRACE add on top of GNN warm-start?

The decisive question after finding GNN warm-start alone is strong (~0.87 on
unseen graphs): is GRACE's extra machinery (RL refine + metaheuristic escape)
worth it, and WHERE does its value come from?

This script computes, per test instance:
  - gnn_only approx ratio (1 quantum eval)
  - grace    approx ratio (full closed loop)
  - the GAIN = grace - gnn_only
and reports the DISTRIBUTION of the gain, not just the mean. That distribution
decides the narrative:
  * gain spread thin & uniform  -> "GNN is the engine, closed loop is polish"
  * gain concentrated on a few   -> "escape rescues hard instances" (tail story)

It also splits the gain by the quality of the GNN start: does GRACE help most
exactly where the GNN start was WEAKEST? That is the strongest possible story
for the closed loop.

Run (uses the trained GNN; RL optional via --rl-model):
    python scripts/analyze_gain.py --config configs/hard2gnn.yml \
        --gnn-model models/gnn_warmstart_p5_gcn.pt \
        --rl-model models/rl_refiner_td3 --n-instances 8
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import numpy as np
import yaml

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.gnn.loader import load_gnn_warmstart
from grace_qaoa.controller.grace import GraceController
from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted
from grace_qaoa.rl.refiner import RLRefiner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--rl-model", default=None)
    ap.add_argument("--rl-algo", default="td3")
    ap.add_argument("--n-instances", type=int, default=8)
    ap.add_argument("--n-runs", type=int, default=10,
                    help="GRACE runs per instance (GRACE has RNG; average them).")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    p = cfg["qaoa_p"]
    budget = cfg.get("quantum_budget")

    gnn = load_gnn_warmstart(args.gnn_model)
    rl_model = None
    if args.rl_model:
        from stable_baselines3 import SAC, TD3
        rl_model = {"td3": TD3, "sac": SAC}[args.rl_algo].load(
            args.rl_model, device="cpu")
        print(f"Using trained RL refiner: {args.rl_model}")
    else:
        print("No RL model -> GRACE uses hill-climb stand-in (lower bound on GRACE).")

    # Unseen TEST graphs, across all families.
    graphs = []
    per_fam = max(1, args.n_instances // len(cfg["families"]))
    for fam in cfg["families"]:
        for g in make_dataset(fam, n_graphs=per_fam, n_nodes=cfg["n_nodes"],
                              weighted=cfg.get("weighted", True), split="test"):
            graphs.append((fam, g))

    rows = []
    print(f"\n{'inst':>4} {'family':>14} {'gnn_only':>9} {'grace':>8} {'gain':>8}")
    for gi, (fam, g) in enumerate(graphs):
        opt = brute_force_maxcut(g)
        # GNN-only (deterministic, 1 eval)
        gnn_init = gnn.predict_params(g).reshape(-1)
        gnn_ar = QAOAMaxCut(g, p=p).expected_cut(gnn_init) / opt

        # GRACE averaged over runs (it has RNG in escape + RL is greedy)
        grace_ars = []
        for run in range(args.n_runs):
            cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=budget)
            init = gnn.predict_params(g).reshape(-1)
            rl_step = (RLRefiner(rl_model, cq, steps_per_round=cfg.get(
                "rl_steps_per_round", 30)) if rl_model else None)
            ctrl = GraceController(cq, escape=cfg["escape"],
                                   stall_eps=cfg.get("stall_eps", 1e-3),
                                   stall_patience=cfg.get("stall_patience", 2),
                                   max_rounds=cfg["grace_rounds"], seed=run)
            try:
                res = ctrl.run(init, rl_step_fn=rl_step)
                grace_ars.append(res["best_cut"] / opt)
            except BudgetExhausted:
                grace_ars.append(cq.best_cut_so_far / opt)
        grace_ar = float(np.mean(grace_ars))
        gain = grace_ar - gnn_ar
        rows.append({"family": fam, "gnn": gnn_ar, "grace": grace_ar, "gain": gain})
        print(f"{gi:>4} {fam:>14} {gnn_ar:>9.4f} {grace_ar:>8.4f} {gain:>+8.4f}")

    gnns = np.array([r["gnn"] for r in rows])
    gains = np.array([r["gain"] for r in rows])
    print("-" * 50)
    print(f"mean gnn_only = {gnns.mean():.4f} | mean grace = "
          f"{np.mean([r['grace'] for r in rows]):.4f} | "
          f"mean gain = {gains.mean():+.4f}")
    print(f"gain distribution: min={gains.min():+.4f} median="
          f"{np.median(gains):+.4f} max={gains.max():+.4f} std={gains.std():.4f}")
    # The key correlation: does GRACE help most where the GNN start was weakest?
    if len(gnns) > 2 and gnns.std() > 1e-9:
        corr = np.corrcoef(gnns, gains)[0, 1]
        print(f"corr(gnn_quality, gain) = {corr:+.3f}")
        if corr < -0.3:
            print("  => GRACE rescues WEAK GNN starts (tail story; strong for the paper).")
        elif abs(corr) <= 0.3:
            print("  => gain roughly uniform (GNN is the engine, closed loop is polish).")
        else:
            print("  => GRACE helps most where GNN already good (unusual; investigate).")
    n_positive = int((gains > 0.001).sum())
    print(f"instances where GRACE beats GNN by >0.001: {n_positive}/{len(gains)}")


if __name__ == "__main__":
    main()
