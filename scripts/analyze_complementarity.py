"""Complementarity analysis for the ACO+CPO ensemble decision.

Before spending hours on a full ensemble ablation, answer the prerequisite
question: are ACO and CPO COMPLEMENTARY (each best on different instances, so an
oracle that picks the better per instance beats both) or does ACO simply
dominate (best almost everywhere, so an ensemble only approaches ACO)?

For each test instance we run ACO and CPO under the SAME fair budget and record
which wins. Then we compute the "oracle" (per-instance best of the two) -- the
theoretical ceiling any ensemble could reach. If the oracle is meaningfully
above ACO alone, an ensemble is worth building. If not, just use ACO.

Run:
    python scripts/analyze_complementarity.py --config configs/hard2gnn.yml \
        --gnn-model models/gnn_warmstart_p5_gcn.pt \
        --rl-model models/rl_refiner_td3 --n-instances 30 --run-name complem
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
from grace_qaoa.utils.checkpoint import CheckpointStore


def grace_with_escape(g, cfg, gnn, rl_model, escape, seed):
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)
    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cfg.get("quantum_budget"))
    init = gnn.predict_params(g).reshape(-1)
    rl_step = (RLRefiner(rl_model, cq, steps_per_round=cfg.get(
        "rl_steps_per_round", 30)) if rl_model else None)
    ctrl = GraceController(cq, escape=escape,
                           stall_eps=cfg.get("stall_eps", 1e-3),
                           stall_patience=cfg.get("stall_patience", 2),
                           max_rounds=cfg["grace_rounds"], seed=seed)
    try:
        return ctrl.run(init, rl_step_fn=rl_step)["best_cut"] / opt
    except BudgetExhausted:
        return cq.best_cut_so_far / opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--rl-model", default=None)
    ap.add_argument("--rl-algo", default="td3")
    ap.add_argument("--n-instances", type=int, default=30)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--run-name", default="complem")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    gnn = load_gnn_warmstart(args.gnn_model)
    rl_model = None
    if args.rl_model:
        from stable_baselines3 import SAC, TD3
        rl_model = {"td3": TD3, "sac": SAC}[args.rl_algo].load(
            args.rl_model, device="cpu")

    graphs = []
    per_fam = max(1, args.n_instances // len(cfg["families"]))
    for fam in cfg["families"]:
        for g in make_dataset(fam, n_graphs=per_fam, n_nodes=cfg["n_nodes"],
                              weighted=cfg.get("weighted", True), split="test"):
            graphs.append(g)

    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / args.run_name)
    if store.n_completed:
        print(f"Resuming '{args.run_name}': {store.n_completed} cells done.")

    total = len(graphs) * 2
    for gi, g in enumerate(graphs):
        for esc in ("aco", "cpo"):
            cid = f"inst{gi}|{esc}"
            if store.is_done(cid):
                continue
            ars = [grace_with_escape(g, cfg, gnn, rl_model, esc, seed=r)
                   for r in range(args.n_runs)]
            store.append(cid, {"instance": gi, "escape": esc,
                               "ar": float(np.mean(ars))})
            print(f"  [{store.n_completed}/{total}] inst{gi} {esc}: {np.mean(ars):.4f}")

    rows = store.load_all_records()
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["escape"]] = r["ar"]
    complete = [i for i, d in by.items() if "aco" in d and "cpo" in d]
    aco = np.array([by[i]["aco"] for i in complete])
    cpo = np.array([by[i]["cpo"] for i in complete])
    oracle = np.maximum(aco, cpo)            # per-instance best of the two

    aco_wins = int((aco > cpo).sum())
    cpo_wins = int((cpo > aco).sum())
    ties = len(complete) - aco_wins - cpo_wins

    print("\n--- Complementarity report ---")
    print(f"instances: {len(complete)}")
    print(f"ACO better on {aco_wins}, CPO better on {cpo_wins}, ties {ties}")
    print(f"mean   ACO = {aco.mean():.4f}")
    print(f"mean   CPO = {cpo.mean():.4f}")
    print(f"mean ORACLE = {oracle.mean():.4f}  (ceiling any ensemble could reach)")
    gain_over_aco = oracle.mean() - aco.mean()
    print(f"oracle gain over ACO alone = {gain_over_aco:+.4f}")
    print()
    if cpo_wins == 0:
        print("VERDICT: ACO dominates (CPO never wins). An ensemble can at best")
        print("         approach ACO; it will NOT beat it. Recommend: use ACO alone.")
    elif gain_over_aco < 0.002:
        print("VERDICT: weak complementarity. Oracle barely above ACO. An ensemble")
        print("         is unlikely to be worth the complexity. Lean: use ACO alone.")
    else:
        print("VERDICT: genuine complementarity. Oracle clearly above ACO -- an")
        print("         ensemble that picks well per instance could beat both.")
        print("         Worth building the portfolio/cascade ablation.")
    json.dump({"instances": len(complete), "aco_wins": aco_wins,
               "cpo_wins": cpo_wins, "mean_aco": float(aco.mean()),
               "mean_cpo": float(cpo.mean()), "mean_oracle": float(oracle.mean()),
               "oracle_gain_over_aco": float(gain_over_aco)},
              open(store.run_dir / "complementarity.json", "w"), indent=2)
    print(f"\nSaved -> {store.run_dir / 'complementarity.json'}")


if __name__ == "__main__":
    main()
