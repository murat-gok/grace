"""Robustness study (two analyses, one script):

  MODE 'hparam' : escape-trigger sensitivity. Sweep stall_patience and stall_eps
                  and report GRACE's mean approx ratio for each setting. If the
                  surface is flat, the chosen values are not cherry-picked.

  MODE 'size'   : graph-size scalability. Run GRACE and the strong baselines at
                  several n_nodes and check whether GRACE's advantage holds as
                  the problem grows (not just at n_nodes=14).

Both are crash-safe (each cell checkpointed) and use the disjoint TEST split.

Examples:
  python scripts/robustness.py --mode hparam --config configs/hard2gnn.yml \
      --gnn-model models/gnn_warmstart_p5_gcn.pt --rl-model models/rl_refiner_td3 \
      --n-instances 12 --run-name robust_hparam

  python scripts/robustness.py --mode size --config configs/hard2gnn.yml \
      --gnn-model models/gnn_warmstart_p5_gcn.pt --rl-model models/rl_refiner_td3 \
      --sizes 10 12 14 16 --n-instances 9 --run-name robust_size

Note on 'size': the GNN/RL were trained at the config's n_nodes. Testing at other
sizes is ALSO an out-of-distribution generalization check (warm-start trained at
one size, applied at another) -- report it as such.
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
from grace_qaoa.baselines_strong import interp_baseline, fourier_baseline
from grace_qaoa.utils.checkpoint import CheckpointStore

PATIENCES = [1, 2, 3, 5]
EPSILONS = [0.0005, 0.001, 0.005, 0.01]


def grace_ar(g, cfg, gnn, rl_model, seed, stall_eps=None, stall_patience=None):
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)
    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cfg.get("quantum_budget"))
    init = gnn.predict_params(g).reshape(-1)
    rl_step = (RLRefiner(rl_model, cq, steps_per_round=cfg.get(
        "rl_steps_per_round", 30)) if rl_model else None)
    ctrl = GraceController(
        cq, escape=cfg["escape"],
        stall_eps=stall_eps if stall_eps is not None else cfg.get("stall_eps", 1e-3),
        stall_patience=(stall_patience if stall_patience is not None
                        else cfg.get("stall_patience", 2)),
        max_rounds=cfg["grace_rounds"], seed=seed)
    try:
        return ctrl.run(init, rl_step_fn=rl_step)["best_cut"] / opt
    except BudgetExhausted:
        return cq.best_cut_so_far / opt


def load_models(args, cfg):
    gnn = load_gnn_warmstart(args.gnn_model)
    rl_model = None
    if args.rl_model:
        from stable_baselines3 import SAC, TD3
        rl_model = {"td3": TD3, "sac": SAC}[args.rl_algo].load(
            args.rl_model, device="cpu")
    return gnn, rl_model


def test_graphs(cfg, n_instances, n_nodes):
    graphs = []
    per_fam = max(1, n_instances // len(cfg["families"]))
    for fam in cfg["families"]:
        for g in make_dataset(fam, n_graphs=per_fam, n_nodes=n_nodes,
                              weighted=cfg.get("weighted", True), split="test"):
            graphs.append(g)
    return graphs


def run_hparam(args, cfg, gnn, rl_model, store):
    graphs = test_graphs(cfg, args.n_instances, cfg["n_nodes"])
    total = len(PATIENCES) * len(EPSILONS) * len(graphs)
    for pat in PATIENCES:
        for eps in EPSILONS:
            for gi, g in enumerate(graphs):
                cid = f"pat{pat}|eps{eps}|inst{gi}"
                if store.is_done(cid):
                    continue
                ars = [grace_ar(g, cfg, gnn, rl_model, seed=r,
                                stall_eps=eps, stall_patience=pat)
                       for r in range(args.n_runs)]
                store.append(cid, {"patience": pat, "eps": eps, "instance": gi,
                                   "ar": float(np.mean(ars))})
                if store.n_completed % 10 == 0:
                    print(f"  [{store.n_completed}/{total}] pat={pat} eps={eps}")

    rows = store.load_all_records()
    print(f"\nGRACE mean approx ratio by (stall_patience, stall_eps):")
    header_label = "pat\\eps"
    print(f"{header_label:>8} " + " ".join(f"{e:>8}" for e in EPSILONS))
    grid = {}
    for pat in PATIENCES:
        line = f"{pat:>8} "
        for eps in EPSILONS:
            vals = [r["ar"] for r in rows
                    if r["patience"] == pat and abs(r["eps"] - eps) < 1e-12]
            v = float(np.mean(vals)) if vals else float("nan")
            grid[f"{pat}|{eps}"] = v
            line += f"{v:>8.4f} "
        print(line)
    allv = [v for v in grid.values() if not np.isnan(v)]
    print(f"\nspread across all settings: min={min(allv):.4f} "
          f"max={max(allv):.4f} range={max(allv)-min(allv):.4f}")
    print("=> flat surface (small range) means the choice is NOT cherry-picked.")
    json.dump({"grid": grid, "patiences": PATIENCES, "epsilons": EPSILONS},
              open(store.run_dir / "hparam_sensitivity.json", "w"), indent=2)
    print(f"Saved -> {store.run_dir / 'hparam_sensitivity.json'}")


def run_size(args, cfg, gnn, rl_model, store):
    methods = ["interp", "fourier", "grace"]
    total = len(args.sizes) * args.n_instances * len(methods)
    for n_nodes in args.sizes:
        graphs = test_graphs(cfg, args.n_instances, n_nodes)
        for gi, g in enumerate(graphs):
            opt = brute_force_maxcut(g)
            for m in methods:
                cid = f"n{n_nodes}|inst{gi}|{m}"
                if store.is_done(cid):
                    continue
                if m == "grace":
                    ars = [grace_ar(g, cfg, gnn, rl_model, seed=r)
                           for r in range(args.n_runs)]
                    val = float(np.mean(ars))
                else:
                    cq = CountingQAOA(QAOAMaxCut(g, p=cfg["qaoa_p"]),
                                      budget=cfg.get("quantum_budget"))
                    fn = interp_baseline if m == "interp" else fourier_baseline
                    kw = (dict(maxiter_per_level=cfg.get("interp_maxiter", 60))
                          if m == "interp" else dict(maxiter=cfg.get("fourier_maxiter", 120)))
                    try:
                        fn(cq, target_p=cfg["qaoa_p"], seed=0, **kw)
                    except BudgetExhausted:
                        pass
                    val = cq.best_cut_so_far / opt
                store.append(cid, {"n_nodes": n_nodes, "instance": gi,
                                   "method": m, "ar": val})
        print(f"  size {n_nodes}: done ({store.n_completed}/{total} cells)")

    rows = store.load_all_records()
    print(f"\nMean approx ratio by graph size:")
    print(f"{'n_nodes':>8} " + " ".join(f"{m:>9}" for m in methods))
    out = {}
    for n_nodes in args.sizes:
        line = f"{n_nodes:>8} "
        out[n_nodes] = {}
        for m in methods:
            vals = [r["ar"] for r in rows
                    if r["n_nodes"] == n_nodes and r["method"] == m]
            v = float(np.mean(vals)) if vals else float("nan")
            out[n_nodes][m] = v
            line += f"{v:>9.4f} "
        print(line)
    holds = all(out[n]["grace"] >= max(out[n]["interp"], out[n]["fourier"])
                for n in args.sizes)
    print("\n=> GRACE leads at EVERY size." if holds
          else "\n=> GRACE does not lead at all sizes (see above).")
    json.dump(out, open(store.run_dir / "size_scalability.json", "w"), indent=2)
    print(f"Saved -> {store.run_dir / 'size_scalability.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hparam", "size"], required=True)
    ap.add_argument("--config", default="configs/hard2gnn.yml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--rl-model", default=None)
    ap.add_argument("--rl-algo", default="td3")
    ap.add_argument("--n-instances", type=int, default=12)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 12, 14, 16])
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    gnn, rl_model = load_models(args, cfg)

    run_name = args.run_name or f"robust_{args.mode}"
    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / run_name)
    if store.n_completed:
        print(f"Resuming '{run_name}': {store.n_completed} cells done.")

    if args.mode == "hparam":
        run_hparam(args, cfg, gnn, rl_model, store)
    else:
        run_size(args, cfg, gnn, rl_model, store)


if __name__ == "__main__":
    main()
