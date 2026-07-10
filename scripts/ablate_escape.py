"""Escape-operator ablation: CPO vs DE vs random-restart vs none.

The novelty wedge of GRACE is the Crested Porcupine Optimizer as the escape
operator. A reviewer will ask two things, and this script answers both on the
SAME test instances (paired), with the SAME GNN warm-start and RL refiner:

  1. Does escape help at all?            -> compare 'none' vs the rest
  2. Is CPO better than alternatives?    -> compare 'cpo' vs 'de' vs 'random'

If CPO does not beat random-restart under matched budget, the CPO story is not
defensible -- better to know now. If it does, that is the headline ablation.

Run:
    python scripts/ablate_escape.py --config configs/hard2gnn.yml \
        --gnn-model models/gnn_warmstart_p5_gcn.pt \
        --rl-model models/rl_refiner_td3 --n-instances 8 --n-runs 10
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.gnn.loader import load_gnn_warmstart
from grace_qaoa.controller.grace import GraceController
from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted
from grace_qaoa.rl.refiner import RLRefiner
from grace_qaoa.utils.checkpoint import CheckpointStore

ESCAPES = ["none", "random", "de", "ga", "pso", "aco", "aco_ls", "woa", "gwo",
           "hho", "cmaes", "aco_cpo_portfolio", "aco_cpo_cascade", "cpo"]


def run_grace(g, cfg, gnn, rl_model, escape, seed):
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)
    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cfg.get("quantum_budget"))
    init = gnn.predict_params(g).reshape(-1)
    rl_step = (RLRefiner(rl_model, cq, steps_per_round=cfg.get(
        "rl_steps_per_round", 30)) if rl_model else None)
    # 'none' = escape disabled (stall_eps < 0 never triggers)
    stall_eps = -1.0 if escape == "none" else cfg.get("stall_eps", 1e-3)
    esc_name = "cpo" if escape == "none" else escape  # any valid key; never fires
    ctrl = GraceController(cq, escape=esc_name, stall_eps=stall_eps,
                           stall_patience=cfg.get("stall_patience", 2),
                           max_rounds=cfg["grace_rounds"], seed=seed)
    try:
        res = ctrl.run(init, rl_step_fn=rl_step)
        return res["best_cut"] / opt
    except BudgetExhausted:
        return cq.best_cut_so_far / opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--rl-model", default=None)
    ap.add_argument("--rl-algo", default="td3")
    ap.add_argument("--n-instances", type=int, default=8)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--run-name", default="ablate_escape",
                    help="Checkpoint folder under results/. Reuse to resume.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Restrict to these operators (e.g. --only none aco aco_ls). "
                         "Default: run all.")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    global ESCAPES
    if args.only:
        invalid = [e for e in args.only if e not in ESCAPES]
        if invalid:
            raise SystemExit(f"Unknown operators in --only: {invalid}. "
                             f"Choose from {ESCAPES}")
        ESCAPES = list(args.only)
        print(f"Restricting ablation to: {ESCAPES}")

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

    # --- crash-safe: checkpoint every (instance, escape) result to disk ---
    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / args.run_name)
    if store.n_completed:
        print(f"Resuming '{args.run_name}': {store.n_completed} cells done.")

    total = len(graphs) * len(ESCAPES)
    for gi, g in enumerate(graphs):
        for e in ESCAPES:
            cell_id = f"inst{gi}|{e}"
            if store.is_done(cell_id):
                continue
            ars = [run_grace(g, cfg, gnn, rl_model, e, seed=run)
                   for run in range(args.n_runs)]
            store.append(cell_id, {"instance": gi, "escape": e,
                                   "mean_ar": float(np.mean(ars))})
            print(f"  [{store.n_completed}/{total}] inst{gi} {e}: "
                  f"{np.mean(ars):.4f}")

    # --- rebuild results from checkpoint (single source of truth) ---
    rows = store.load_all_records()
    results = {e: [] for e in ESCAPES}
    by_inst = {}
    for r in rows:
        by_inst.setdefault(r["instance"], {})[r["escape"]] = r["mean_ar"]
    # keep only instances that have ALL escapes (paired comparison)
    complete = [i for i, d in sorted(by_inst.items())
                if all(e in d for e in ESCAPES)]
    for i in complete:
        for e in ESCAPES:
            results[e].append(by_inst[i][e])

    print(f"\n{'escape':>8} {'mean_approx':>12} {'std':>8}  "
          f"(n={len(complete)} instances)")
    for e in ESCAPES:
        arr = np.array(results[e])
        print(f"{e:>8} {arr.mean():>12.4f} {arr.std():>8.4f}")

    # paired Wilcoxon: reference operator vs each other variant.
    # Prefer 'cpo' for continuity; if it's not in the run, use the best-mean op.
    ref = "cpo" if "cpo" in results else max(
        results, key=lambda k: np.mean(results[k]) if results[k] else -1)
    print(f"\nPaired Wilcoxon ({ref} vs ...):")
    ref_arr = np.array(results[ref])
    for e in [e for e in ESCAPES if e != ref]:
        other = np.array(results[e])
        try:
            if np.allclose(ref_arr, other):
                print(f"  vs {e:>7}: identical values (no difference to test)")
                continue
            stat, pval = stats.wilcoxon(ref_arr, other)
            better = (ref_arr > other).sum()
            print(f"  vs {e:>7}: p={pval:.4g}  ({ref} better on {better}/{len(ref_arr)} "
                  f"instances, mean delta {ref_arr.mean()-other.mean():+.4f})")
        except ValueError as ex:
            print(f"  vs {e:>7}: {ex}")

    # save a tidy summary next to the checkpoint
    import json
    summary = {
        "n_instances": len(complete),
        "n_runs": args.n_runs,
        "mean_approx": {e: float(np.mean(results[e])) for e in ESCAPES},
        "std_approx": {e: float(np.std(results[e])) for e in ESCAPES},
    }
    json.dump(summary, open(store.run_dir / "ablation_summary.json", "w"), indent=2)
    print(f"\nSaved -> {store.run_dir / 'ablation_summary.json'}")
    print("\nInterpretation:")
    print("  - 'none' worse than the rest  => escape helps.")
    print("  - 'cpo' >= 'random'           => the metaheuristic earns its place.")
    print("  - 'cpo' vs 'de'               => which operator wins (novelty wedge).")


if __name__ == "__main__":
    main()
