"""Main experiment runner for GRACE-QAOA.

Tuned for the office Xeon E5-1660 v3 (8C/16T) + 32GB, CPU-only.

Parallelism strategy (IMPORTANT to avoid oversubscription):
  - We parallelize ACROSS runs/instances with joblib.
  - Each worker pins the simulator to a SINGLE thread (OMP_NUM_THREADS=1)
    so the N workers do not fight over cores. Set N_JOBS <= physical cores.

Usage:
    python scripts/run_experiment.py --config configs/quick.yaml
"""
from __future__ import annotations

# --- pin threads BEFORE numpy / pennylane import (must come first) ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.controller.grace import GraceController
from grace_qaoa.rl.refiner import RLRefiner
from grace_qaoa.baselines import random_init, cold_cobyla
from grace_qaoa.baselines_strong import (interp_baseline, fourier_baseline,
                                         spsa_baseline, transfer_baseline)
from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted
from grace_qaoa.utils.stats import (friedman_test, average_ranks,
                                    holm_posthoc, wilcoxon_vs_baseline)
from grace_qaoa.utils.checkpoint import CheckpointStore, job_id


# Methods compared. GRACE is the proposal; the rest are baselines, with
# interp/fourier/spsa/transfer being the STRONG ones a Q1 reviewer expects.
# gnn_only is the key ablation: GNN warm-start with NO refinement, NO escape.
METHODS = ["random", "cobyla", "spsa", "interp", "fourier", "transfer",
           "gnn_only", "grace"]


def random_init_counting(cq, seed=0):
    """Random params under the counting device (1 eval)."""
    rng = np.random.default_rng(seed)
    params = rng.uniform(0, np.pi, size=2 * cq.p)
    try:
        cut = cq.expected_cut(params)
    except BudgetExhausted:
        cut = cq.best_cut_so_far
    return {"best_params": params, "best_cut": max(cut, cq.best_cut_so_far),
            "n_quantum_evals": cq.n_evals}


def cobyla_counting(cq, seed=0, maxiter=100):
    """COBYLA from a random start under the counting device."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=2 * cq.p)
    try:
        res = minimize(lambda x: cq.cost(x), x0, method="COBYLA",
                       options={"maxiter": maxiter})
        best = res.x
    except BudgetExhausted:
        best = x0
    return {"best_params": best, "best_cut": cq.best_cut_so_far,
            "n_quantum_evals": cq.n_evals}


def _load_rl_model(cfg):
    """Load a trained SB3 policy if cfg points to one, else None (-> hill-climb)."""
    path = cfg.get("rl_model")
    if not path:
        return None
    from stable_baselines3 import SAC, TD3
    algo = cfg.get("rl_algo", "td3")
    cls = {"td3": TD3, "sac": SAC}[algo]
    model = cls.load(path, device="cpu")
    # Guard: the policy's observation size encodes p (= (obs_dim - 3)/2).
    obs_dim = model.observation_space.shape[0]
    model_p = (obs_dim - 3) // 2
    if model_p != cfg["qaoa_p"]:
        raise ValueError(
            f"RL model was trained for p={model_p} but config qaoa_p="
            f"{cfg['qaoa_p']}. Train a matching model: "
            f"python scripts/train_rl.py --config <this config>.")
    return model


def run_one_instance(graph, cfg, seed, rl_model=None, gnn_model=None):
    """Run all methods on one graph under a fair quantum-evaluation budget.

    Returns, per method: approximation ratio (quality) AND evals-to-target
    (cost). The fair-budget design is the core of the comparison: every method
    is capped at the SAME number of quantum evaluations, so a higher ratio
    cannot be bought with more circuit calls.
    """
    p = cfg["qaoa_p"]
    budget = cfg.get("quantum_budget")          # None => uncapped
    opt = brute_force_maxcut(graph) if graph.number_of_nodes() <= 22 else None
    # Target for the cost-to-target metric: a fraction of optimal.
    target_frac = cfg.get("target_frac", 0.95)
    target_cut = (target_frac * opt) if opt else None

    out = {}

    def record(name, res, cq):
        cut = res["best_cut"]
        out[f"{name}_ar"] = (cut / opt) if opt else cut
        out[f"{name}_evals"] = res.get("n_quantum_evals", cq.n_evals)
        out[f"{name}_evals_to_target"] = cq.evals_to_target  # None if never hit

    # --- weak baselines (kept for continuity) ---
    cq = CountingQAOA(QAOAMaxCut(graph, p=p), budget=budget)
    if target_cut is not None:
        cq.set_target(target_cut)
    r = random_init_counting(cq, seed=seed)
    record("random", r, cq)

    cq = CountingQAOA(QAOAMaxCut(graph, p=p), budget=budget)
    if target_cut is not None: cq.set_target(target_cut)
    r = cobyla_counting(cq, seed=seed, maxiter=cfg["cobyla_maxiter"])
    record("cobyla", r, cq)

    # --- strong baselines (the real bar) ---
    for name, fn, kwargs in [
        ("spsa", spsa_baseline, dict(iters=cfg.get("spsa_iters", 150))),
        ("interp", interp_baseline, dict(maxiter_per_level=cfg.get("interp_maxiter", 60))),
        ("fourier", fourier_baseline, dict(maxiter=cfg.get("fourier_maxiter", 120))),
        ("transfer", transfer_baseline, dict(maxiter=cfg.get("transfer_maxiter", 60))),
    ]:
        cq = CountingQAOA(QAOAMaxCut(graph, p=p), budget=budget)
        if target_cut is not None:
            cq.set_target(target_cut)
        r = fn(cq, target_p=p, seed=seed, **kwargs)
        record(name, r, cq)

    # --- gnn_only ablation: GNN warm-start with no refinement/escape ---
    cq = CountingQAOA(QAOAMaxCut(graph, p=p), budget=budget)
    if target_cut is not None:
        cq.set_target(target_cut)
    if gnn_model is not None:
        gnn_init = gnn_model.predict_params(graph).reshape(-1)
        try:
            gcut = cq.expected_cut(gnn_init)
        except BudgetExhausted:
            gcut = cq.best_cut_so_far
        record("gnn_only", {"best_cut": max(gcut, cq.best_cut_so_far),
                            "n_quantum_evals": cq.n_evals}, cq)
    else:
        # No GNN available: fall back to random (keeps the column populated).
        r = random_init_counting(cq, seed=seed + 7)
        record("gnn_only", r, cq)

    # --- GRACE (proposal), under the SAME budget and counter ---
    cq = CountingQAOA(QAOAMaxCut(graph, p=p), budget=budget)
    if target_cut is not None:
        cq.set_target(target_cut)
    # Warm-start: GNN prediction if a model is provided, else random init.
    if gnn_model is not None:
        init = gnn_model.predict_params(graph).reshape(-1)
    else:
        init = np.random.default_rng(seed).uniform(0, np.pi, 2 * p)
    rl_step_fn = None
    if rl_model is not None:
        rl_step_fn = RLRefiner(rl_model, cq, optimal_cut=opt,
                               steps_per_round=cfg.get("rl_steps_per_round", 30))
    grace = GraceController(cq, escape=cfg["escape"],
                            stall_eps=cfg.get("stall_eps", 1e-3),
                            stall_patience=cfg.get("stall_patience", 2),
                            max_rounds=cfg["grace_rounds"], seed=seed)
    try:
        gres = grace.run(init, rl_step_fn=rl_step_fn)
    except BudgetExhausted:
        gres = {"best_cut": cq.best_cut_so_far, "n_quantum_evals": cq.n_evals}
    record("grace", gres, cq)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--run-name", default=None,
                    help="Checkpoint folder name under results/. "
                         "Reuse the SAME name to resume after a crash.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Jobs per parallel batch; checkpoint flushed after each "
                         "batch. Defaults to 4x n_jobs.")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    t0 = time.time()
    rl_model = _load_rl_model(cfg)
    if rl_model is not None:
        print(f"Loaded RL refiner: {cfg.get('rl_model')} ({cfg.get('rl_algo','td3')})")
    else:
        print("No rl_model in config -> GRACE uses hill-climb stand-in refiner.")

    gnn_model = None
    if cfg.get("gnn_model"):
        from grace_qaoa.gnn.loader import load_gnn_warmstart
        gnn_model = load_gnn_warmstart(cfg["gnn_model"])
        print(f"Loaded GNN warm-start: {cfg['gnn_model']}")
    else:
        print("No gnn_model in config -> GRACE uses random init (no warm-start).")

    # --- set up crash-safe checkpoint store ---
    out_dir = Path(cfg.get("out_dir", "results"))
    run_name = args.run_name or f"run_{Path(args.config).stem}"
    store = CheckpointStore(out_dir / run_name)
    if store.n_completed:
        print(f"Resuming '{run_name}': {store.n_completed} jobs already done, "
              f"skipping those.")
    else:
        print(f"Starting fresh run '{run_name}'.")

    # --- build the full job list (deterministic order) ---
    all_jobs = []      # (family, instance, run, graph)
    for family in cfg["families"]:
        weighted = cfg.get("weighted", False)
        graphs = make_dataset(family, n_graphs=cfg["n_instances"],
                              n_nodes=cfg["n_nodes"], weighted=weighted)
        for gi, g in enumerate(graphs):
            for run in range(cfg["n_runs"]):
                all_jobs.append((family, gi, run, g))

    # --- filter out already-completed jobs (resume) ---
    pending = [(fam, gi, run, g) for (fam, gi, run, g) in all_jobs
               if not store.is_done(job_id(fam, gi, run))]
    print(f"Total jobs: {len(all_jobs)} | pending: {len(pending)}")

    # --- run pending jobs in batches; checkpoint after every batch ---
    batch_size = args.batch_size or (4 * cfg["n_jobs"])
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        out = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(
            delayed(run_one_instance)(g, cfg, seed=1000 * gi + run,
                                      rl_model=rl_model, gnn_model=gnn_model)
            for (fam, gi, run, g) in batch)
        for (fam, gi, run, g), res in zip(batch, out):
            res.update({"family": fam, "instance": gi, "run": run})
            store.append(job_id(fam, gi, run), res)   # flushed + fsync'd
        done = store.n_completed
        print(f"  checkpoint: {done}/{len(all_jobs)} jobs saved "
              f"({round(time.time()-t0)}s elapsed)")

    # --- aggregate from the checkpoint file (single source of truth) ---
    all_rows = store.load_all_records()
    methods = METHODS
    grace_idx = methods.index("grace")

    # Approximation-ratio matrix (quality), instance-paired across methods.
    ar_matrix = np.array([[row[f"{m}_ar"] for m in methods] for row in all_rows])

    # Cost-to-target: median evals to reach target (None -> treated as budget cap
    # for reporting "fraction that reached target").
    def reached_frac(m):
        vals = [row.get(f"{m}_evals_to_target") for row in all_rows]
        hit = [v for v in vals if v is not None]
        return len(hit) / len(vals) if vals else 0.0

    def median_evals_to_target(m):
        vals = [row.get(f"{m}_evals_to_target") for row in all_rows]
        hit = [v for v in vals if v is not None]
        return float(np.median(hit)) if hit else None

    summary = {
        "config": cfg,
        "run_name": run_name,
        "n_samples": len(all_rows),
        "methods": methods,
        "mean_approx_ratio": {m: float(ar_matrix[:, j].mean())
                              for j, m in enumerate(methods)},
        "std_approx_ratio": {m: float(ar_matrix[:, j].std())
                             for j, m in enumerate(methods)},
        "mean_evals_used": {m: float(np.mean([row[f"{m}_evals"] for row in all_rows]))
                            for m in methods},
        "frac_reached_target": {m: reached_frac(m) for m in methods},
        "median_evals_to_target": {m: median_evals_to_target(m) for m in methods},
        "average_ranks": dict(zip(methods, average_ranks(ar_matrix).tolist())),
        "friedman": friedman_test(ar_matrix),
        # GRACE vs each STRONG baseline, instance-paired Wilcoxon.
        "wilcoxon_grace_vs": {
            m: wilcoxon_vs_baseline(ar_matrix[:, grace_idx], ar_matrix[:, j])
            for j, m in enumerate(methods) if m != "grace"
        },
        "holm_vs_grace": {methods[k]: v for k, v in
                          holm_posthoc(ar_matrix, control_idx=grace_idx).items()},
        "runtime_sec": round(time.time() - t0, 1),
    }

    out_file = store.run_dir / "summary.json"
    json.dump(summary, open(out_file, "w"), indent=2)
    print("\n=== Mean approximation ratio (fair budget) ===")
    for m in methods:
        print(f"  {m:10} {summary['mean_approx_ratio'][m]:.4f}  "
              f"(evals~{summary['mean_evals_used'][m]:.0f}, "
              f"reached {summary['frac_reached_target'][m]*100:.0f}%)")
    print(f"Friedman p = {summary['friedman']['p_value']:.4g}")
    print("GRACE vs strong baselines (Wilcoxon p):")
    for m in ["spsa", "interp", "fourier", "transfer"]:
        print(f"  vs {m:10} p = {summary['wilcoxon_grace_vs'][m]['p_value']:.4g}")
    print(f"Saved -> {out_file}  ({summary['runtime_sec']}s)")


if __name__ == "__main__":
    main()
