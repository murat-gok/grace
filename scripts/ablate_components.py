"""Component ablation: isolate the contribution of each GRACE stage.

Addresses the reviewer concern that the RL refiner may add nothing: in the
escape ablation, the 'none' row (GNN+RL, escape off) equalled gnn_only. This
script measures four configurations on the SAME instances, per instance, so the
marginal contribution of each component is explicit and paired:

  gnn_only    : GNN warm-start only (1 evaluation, no RL, no escape)
  gnn_rl      : GNN + RL refiner, escape OFF
  gnn_escape  : GNN + escape, RL OFF
  grace_full  : GNN + RL + escape (the full loop)

Reports per-instance means, the marginal gains (rl over gnn_only; escape over
gnn_only; full over gnn_rl), paired Wilcoxon tests on the 60 per-instance means
(the correct unit of analysis -- NOT the 1,800 dependent runs), and Cliff's
delta effect sizes. This directly answers "does RL help, and if so where?".

Run:
    python scripts/ablate_components.py --config configs/hard2gnn.yml \
        --gnn-model models/gnn_warmstart_p5_gcn.pt \
        --rl-model models/rl_refiner_td3 --n-instances 60 --n-runs 10 \
        --run-name ablate_components
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

CONFIGS = ["gnn_only", "gnn_rl", "gnn_escape", "grace_full"]


def cliffs_delta(a, b):
    """Cliff's delta effect size for paired samples a vs b."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    gt = sum((a[i] > b[j]) for i in range(n) for j in range(n))
    lt = sum((a[i] < b[j]) for i in range(n) for j in range(n))
    return (gt - lt) / (n * n)


def run_config(g, cfg, gnn, rl_model, which, seed):
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)
    init = gnn.predict_params(g).reshape(-1)

    if which == "gnn_only":
        # single evaluation, no loop
        return QAOAMaxCut(g, p=p).expected_cut(init) / opt

    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cfg.get("quantum_budget"))
    use_rl = which in ("gnn_rl", "grace_full")
    use_escape = which in ("gnn_escape", "grace_full")
    rl_step = (RLRefiner(rl_model, cq, steps_per_round=cfg.get(
        "rl_steps_per_round", 30)) if (use_rl and rl_model) else None)
    stall_eps = cfg.get("stall_eps", 1e-3) if use_escape else -1.0
    ctrl = GraceController(cq, escape=cfg["escape"], stall_eps=stall_eps,
                           stall_patience=cfg.get("stall_patience", 2),
                           max_rounds=cfg["grace_rounds"], seed=seed)
    try:
        return ctrl.run(init, rl_step_fn=rl_step,
                        refiner=("rl" if use_rl else "none"))["best_cut"] / opt
    except BudgetExhausted:
        return cq.best_cut_so_far / opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--rl-model", default=None)
    ap.add_argument("--rl-algo", default="td3")
    ap.add_argument("--n-instances", type=int, default=60)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--run-name", default="ablate_components")
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

    from grace_qaoa.utils.checkpoint import CheckpointStore
    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / args.run_name)
    if store.n_completed:
        print(f"Resuming '{args.run_name}': {store.n_completed} cells done.")

    total = len(graphs) * len(CONFIGS)
    for gi, g in enumerate(graphs):
        for which in CONFIGS:
            cid = f"inst{gi}|{which}"
            if store.is_done(cid):
                continue
            if which == "gnn_only":
                val = run_config(g, cfg, gnn, rl_model, which, 0)
            else:
                val = float(np.mean([run_config(g, cfg, gnn, rl_model, which, s)
                                     for s in range(args.n_runs)]))
            store.append(cid, {"instance": gi, "config": which, "ar": val})
            print(f"  [{store.n_completed}/{total}] inst{gi} {which}: {val:.4f}")

    # aggregate per-instance
    rows = store.load_all_records()
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["config"]] = r["ar"]
    complete = [i for i, d in sorted(by.items())
                if all(c in d for c in CONFIGS)]
    data = {c: np.array([by[i][c] for i in complete]) for c in CONFIGS}

    print(f"\n{'config':>12} {'mean':>8} {'std':>8}  (n={len(complete)} instances)")
    for c in CONFIGS:
        print(f"{c:>12} {data[c].mean():>8.4f} {data[c].std():>8.4f}")

    def paired(a_name, b_name):
        a, b = data[a_name], data[b_name]
        if np.allclose(a, b):
            return None, 0.0, 0.0, (a > b).sum(), (b > a).sum()
        stat, pv = stats.wilcoxon(a, b)
        return pv, (a.mean() - b.mean()), cliffs_delta(a, b), \
            int((a > b).sum()), int((b > a).sum())

    print("\nMarginal contributions (paired Wilcoxon on per-instance means, "
          "n={}):".format(len(complete)))
    comparisons = [
        ("gnn_rl", "gnn_only", "RL over GNN-only"),
        ("gnn_escape", "gnn_only", "escape over GNN-only"),
        ("grace_full", "gnn_rl", "escape over GNN+RL"),
        ("grace_full", "gnn_escape", "RL over GNN+escape"),
        ("grace_full", "gnn_only", "full over GNN-only"),
    ]
    summary = {"n_instances": len(complete),
               "means": {c: float(data[c].mean()) for c in CONFIGS},
               "stds": {c: float(data[c].std()) for c in CONFIGS},
               "comparisons": {}}
    for a, b, label in comparisons:
        pv, dmean, delta, aw, bw = paired(a, b)
        pstr = "identical" if pv is None else f"p={pv:.3g}"
        print(f"  {label:>22}: delta={dmean:+.4f}  {pstr}  "
              f"Cliff={delta:+.2f}  ({aw}/{bw} win/lose)")
        summary["comparisons"][label] = {
            "delta_mean": float(dmean), "p_value": (None if pv is None else float(pv)),
            "cliffs_delta": float(delta), "a_wins": int(aw), "b_wins": int(bw)}

    json.dump(summary, open(store.run_dir / "component_ablation.json", "w"),
              indent=2)
    print(f"\nSaved -> {store.run_dir / 'component_ablation.json'}")
    # verdict on RL
    rl_gain = data["gnn_rl"].mean() - data["gnn_only"].mean()
    rl_in_full = data["grace_full"].mean() - data["gnn_escape"].mean()
    print("\nRL verdict:")
    print(f"  RL alone over GNN-only:        {rl_gain:+.4f}")
    print(f"  RL within full (vs GNN+escape): {rl_in_full:+.4f}")
    if abs(rl_gain) < 0.001 and abs(rl_in_full) < 0.001:
        print("  => RL adds no measurable value; reposition framework as "
              "GNN+escape.")
    else:
        print("  => RL contributes; keep three-component framing with this "
              "evidence.")


if __name__ == "__main__":
    main()
