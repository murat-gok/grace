"""GNN architecture ablation: GCN vs GAT vs GraphSAGE.

Answers the reviewer question "why GCN?". For each architecture we measure, on
the SAME unseen test graphs (paired):
  - gnn_only : warm-start quality of that architecture alone (1 quantum eval)
  - grace    : full closed loop on top of that architecture

If results are similar across architectures, the warm-start is ROBUST (a
strength: not cherry-picked). If one dominates, we report and justify it.

Prerequisites: train one GNN per architecture first, e.g.
    python scripts/pretrain_gnn.py --config configs/hard2gnn.yml --conv gcn  --epochs 300 --finetune-epochs 30
    python scripts/pretrain_gnn.py --config configs/hard2gnn.yml --conv gat  --epochs 300 --finetune-epochs 30
    python scripts/pretrain_gnn.py --config configs/hard2gnn.yml --conv sage --epochs 300 --finetune-epochs 30

Then:
    python scripts/ablate_gnn_arch.py --config configs/hard2gnn.yml \
        --rl-model models/rl_refiner_td3 --n-instances 12 --n-runs 10 --run-name ablate_arch

Crash-safe: every (instance, arch, method) cell is checkpointed to disk.
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
from scipy import stats

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.gnn.loader import load_gnn_warmstart
from grace_qaoa.controller.grace import GraceController
from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted
from grace_qaoa.rl.refiner import RLRefiner
from grace_qaoa.utils.checkpoint import CheckpointStore

ARCHS = ["gcn", "gat", "sage"]


def grace_ar(g, cfg, gnn, rl_model, seed):
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)
    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cfg.get("quantum_budget"))
    init = gnn.predict_params(g).reshape(-1)
    rl_step = (RLRefiner(rl_model, cq, steps_per_round=cfg.get(
        "rl_steps_per_round", 30)) if rl_model else None)
    ctrl = GraceController(cq, escape=cfg["escape"],
                           stall_eps=cfg.get("stall_eps", 1e-3),
                           stall_patience=cfg.get("stall_patience", 2),
                           max_rounds=cfg["grace_rounds"], seed=seed)
    try:
        return ctrl.run(init, rl_step_fn=rl_step)["best_cut"] / opt
    except BudgetExhausted:
        return cq.best_cut_so_far / opt


def gnn_only_ar(g, cfg, gnn):
    opt = brute_force_maxcut(g)
    init = gnn.predict_params(g).reshape(-1)
    return QAOAMaxCut(g, p=cfg["qaoa_p"]).expected_cut(init) / opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yml")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--rl-model", default=None)
    ap.add_argument("--rl-algo", default="td3")
    ap.add_argument("--n-instances", type=int, default=12)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--run-name", default="ablate_arch")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    p = cfg["qaoa_p"]

    # load the three GNNs (skip any that are missing, with a warning)
    gnns = {}
    for arch in ARCHS:
        path = Path(args.model_dir) / f"gnn_warmstart_p{p}_{arch}.pt"
        if path.exists():
            gnns[arch] = load_gnn_warmstart(path)
        else:
            print(f"WARNING: {path} not found -> skipping {arch}. "
                  f"Train it with: python scripts/pretrain_gnn.py "
                  f"--config {args.config} --conv {arch} --epochs 300 --finetune-epochs 30")
    if not gnns:
        print("No GNN models found. Train at least one architecture first.")
        return

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

    archs = list(gnns.keys())
    total = len(graphs) * len(archs) * 2   # gnn_only + grace per arch
    for gi, g in enumerate(graphs):
        for arch in archs:
            cid_g = f"inst{gi}|{arch}|gnn_only"
            if not store.is_done(cid_g):
                ar = gnn_only_ar(g, cfg, gnns[arch])
                store.append(cid_g, {"instance": gi, "arch": arch,
                                     "method": "gnn_only", "ar": float(ar)})
                print(f"  [{store.n_completed}/{total}] inst{gi} {arch} gnn_only: {ar:.4f}")
            cid_r = f"inst{gi}|{arch}|grace"
            if not store.is_done(cid_r):
                ars = [grace_ar(g, cfg, gnns[arch], rl_model, seed=run)
                       for run in range(args.n_runs)]
                store.append(cid_r, {"instance": gi, "arch": arch,
                                     "method": "grace", "ar": float(np.mean(ars))})
                print(f"  [{store.n_completed}/{total}] inst{gi} {arch} grace:    "
                      f"{np.mean(ars):.4f}")

    # --- rebuild and report ---
    rows = store.load_all_records()
    table = {arch: {"gnn_only": [], "grace": []} for arch in archs}
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[(r["arch"], r["method"])] = r["ar"]
    complete = [i for i, d in sorted(by.items())
                if all((a, m) in d for a in archs for m in ("gnn_only", "grace"))]
    for i in complete:
        for arch in archs:
            table[arch]["gnn_only"].append(by[i][(arch, "gnn_only")])
            table[arch]["grace"].append(by[i][(arch, "grace")])

    print(f"\n{'arch':>6} {'gnn_only':>10} {'grace':>10} {'gain':>8}  "
          f"(n={len(complete)})")
    summary = {"n_instances": len(complete), "n_runs": args.n_runs, "archs": {}}
    for arch in archs:
        go = np.array(table[arch]["gnn_only"])
        gr = np.array(table[arch]["grace"])
        print(f"{arch:>6} {go.mean():>10.4f} {gr.mean():>10.4f} "
              f"{(gr-go).mean():>+8.4f}")
        summary["archs"][arch] = {
            "gnn_only_mean": float(go.mean()), "gnn_only_std": float(go.std()),
            "grace_mean": float(gr.mean()), "grace_std": float(gr.std()),
            "gain_mean": float((gr - go).mean())}

    # paired Wilcoxon between architectures on GRACE
    if len(archs) > 1 and len(complete) > 1:
        print("\nPaired Wilcoxon (GRACE, arch vs arch):")
        for i in range(len(archs)):
            for j in range(i + 1, len(archs)):
                a, b = archs[i], archs[j]
                xa, xb = np.array(table[a]["grace"]), np.array(table[b]["grace"])
                if np.allclose(xa, xb):
                    print(f"  {a} vs {b}: identical")
                    continue
                _, pv = stats.wilcoxon(xa, xb)
                print(f"  {a} vs {b}: p={pv:.4g} (mean delta {xa.mean()-xb.mean():+.4f})")

    json.dump(summary, open(store.run_dir / "arch_ablation_summary.json", "w"),
              indent=2)
    print(f"\nSaved -> {store.run_dir / 'arch_ablation_summary.json'}")


if __name__ == "__main__":
    main()
