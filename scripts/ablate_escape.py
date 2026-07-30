"""Escape-operator ablation: which metaheuristic, and does escape help at all?

Runs the identical GRACE loop (GNN warm-start + escape, refiner="none" -- the
method described in the paper) on the SAME test instances (paired), swapping
only the escape operator, each under an identical per-escape evaluation budget.

Two questions a reviewer will ask, both answered here:
  1. Does escape help at all?         -> 'none' vs every operator
  2. Which operator wins?             -> paired tests vs the best-mean operator

The reference operator for post-hoc tests is the one with the highest mean
approximation ratio (the empirical winner), reported explicitly -- not fixed a
priori, which would beg the question. Effect sizes use matched-pairs
rank-biserial correlation (paired data), alongside Cliff's delta.

The 11 operators of the main comparison are:
    none, random, de, ga, pso, aco, woa, gwo, hho, cmaes, cpo
The local-search hybrid (aco_ls) and the two ACO-CPO ensembles
(aco_cpo_portfolio, aco_cpo_cascade) are reported separately as negative
results; pass them via --only for that ablation.

Run (main 11-operator comparison, RL disabled):
    python scripts/ablate_escape.py --config configs/hard2gnn.yaml \
        --gnn-model models/gnn_warmstart_p5_gcn.pt \
        --only none random de ga pso aco woa gwo hho cmaes cpo \
        --n-instances 60 --n-runs 10 --run-name ablate_escape11_v2
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
from grace_qaoa.utils.checkpoint import CheckpointStore

# Main comparison = the 11 operators reported in the paper.
ESCAPES = ["none", "random", "de", "ga", "pso", "aco", "woa", "gwo",
           "hho", "cmaes", "cpo"]
# Extra operators for the separate negative-result ablation (pass via --only):
#   aco_ls, aco_cpo_portfolio, aco_cpo_cascade
ALL_KNOWN = ESCAPES + ["aco_ls", "aco_cpo_portfolio", "aco_cpo_cascade"]


def run_grace(g, cfg, gnn, escape, seed):
    """One GRACE run with the given escape operator, refiner disabled.

    refiner="none" is the two-stage method the paper describes: warm-start, then
    escape when the loop stalls. With no refiner the params are unchanged each
    round, so the escape fires on a fixed schedule and every operator receives
    the identical evaluation budget -- which is what makes this ablation fair.
    """
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)
    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cfg.get("quantum_budget"))
    init = gnn.predict_params(g).reshape(-1)
    # 'none' = escape disabled (stall_eps < 0 never triggers)
    stall_eps = -1.0 if escape == "none" else cfg.get("stall_eps", 1e-3)
    esc_name = "cpo" if escape == "none" else escape  # any valid key; never fires
    ctrl = GraceController(cq, escape=esc_name, stall_eps=stall_eps,
                           stall_patience=cfg.get("stall_patience", 2),
                           max_rounds=cfg["grace_rounds"], seed=seed)
    try:
        res = ctrl.run(init, refiner="none")
        return res["best_cut"] / opt
    except BudgetExhausted:
        return cq.best_cut_so_far / opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
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
        invalid = [e for e in args.only if e not in ALL_KNOWN]
        if invalid:
            raise SystemExit(f"Unknown operators in --only: {invalid}. "
                             f"Choose from {ALL_KNOWN}")
        ESCAPES = list(args.only)
        print(f"Restricting ablation to: {ESCAPES}")

    gnn = load_gnn_warmstart(args.gnn_model)

    graphs = []
    per_fam = max(1, args.n_instances // len(cfg["families"]))
    for fam in cfg["families"]:
        for g in make_dataset(fam, n_graphs=per_fam, n_nodes=cfg["n_nodes"],
                              weighted=cfg.get("weighted", True), split="test"):
            graphs.append(g)

    # --- crash-safe: checkpoint every (instance, escape) result to disk ---
    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / args.run_name)
    store.write_provenance(cfg=cfg, extra={"script": "ablate_escape",
                                           "operators": ESCAPES})
    if store.n_completed:
        print(f"Resuming '{args.run_name}': {store.n_completed} cells done.")

    total = len(graphs) * len(ESCAPES)
    for gi, g in enumerate(graphs):
        for e in ESCAPES:
            cell_id = f"inst{gi}|{e}"
            if store.is_done(cell_id):
                continue
            ars = [run_grace(g, cfg, gnn, e, seed=run)
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

    # operators sorted by mean, ascending (as the paper's Table 3)
    order = sorted(ESCAPES, key=lambda e: np.mean(results[e]))
    print(f"\n{'escape':>8} {'mean_approx':>12} {'std':>8}  "
          f"(n={len(complete)} instances, sorted ascending)")
    for e in order:
        arr = np.array(results[e])
        print(f"{e:>8} {arr.mean():>12.4f} {arr.std():>8.4f}")

    def rank_biserial(a, b):
        """Matched-pairs rank-biserial correlation (the effect size for a paired
        Wilcoxon; Cliff's delta is for independent samples)."""
        d = np.asarray(a) - np.asarray(b)
        nz = d[d != 0]
        if len(nz) == 0:
            return 0.0
        r = stats.rankdata(np.abs(nz))
        return float(2.0 * r[nz > 0].sum() / r.sum() - 1.0)

    def cliffs_delta(a, b):
        a, b = np.asarray(a), np.asarray(b)
        n = len(a)
        gt = sum((a[i] > b[j]) for i in range(n) for j in range(n))
        lt = sum((a[i] < b[j]) for i in range(n) for j in range(n))
        return (gt - lt) / (n * n)

    # ---- Q1: does escape help? every operator vs 'none' -------------------
    none_tests = {}
    if "none" in results:
        none_arr = np.array(results["none"])
        print("\nDoes escape help?  (each operator vs 'none', paired Wilcoxon)")
        raw = {}
        for e in [e for e in order if e != "none"]:
            other = np.array(results[e])
            if np.allclose(other, none_arr):
                raw[e] = 1.0
                none_tests[e] = {"p_raw": 1.0, "delta": 0.0, "r_rb": 0.0,
                                 "wins": 0}
            else:
                pv = float(stats.wilcoxon(other, none_arr).pvalue)
                raw[e] = pv
                none_tests[e] = {"p_raw": pv,
                                 "delta": float(other.mean() - none_arr.mean()),
                                 "r_rb": rank_biserial(other, none_arr),
                                 "wins": int((other > none_arr).sum())}
        # Holm correction across the operator-vs-none family
        holm = _holm(raw)
        for e in [e for e in order if e != "none"]:
            t = none_tests[e]
            t["p_holm"] = holm[e]
            print(f"  {e:>7} vs none: delta={t['delta']:+.4f}  "
                  f"p_holm={holm[e]:.3g}  r_rb={t['r_rb']:+.2f}  "
                  f"({t['wins']}/{len(none_arr)} better)")

    # ---- Q2: which operator wins? best-mean vs the rest --------------------
    ref = max([e for e in ESCAPES if e != "none"],
              key=lambda k: np.mean(results[k]))
    print(f"\nBest-mean operator: '{ref}'.  Paired tests vs the others:")
    ref_arr = np.array(results[ref])
    ref_tests = {}
    raw = {}
    for e in [e for e in order if e not in ("none", ref)]:
        other = np.array(results[e])
        if np.allclose(ref_arr, other):
            raw[e] = 1.0
            ref_tests[e] = {"p_raw": 1.0, "delta": 0.0, "r_rb": 0.0, "wins": 0}
        else:
            pv = float(stats.wilcoxon(ref_arr, other).pvalue)
            raw[e] = pv
            ref_tests[e] = {"p_raw": pv,
                            "delta": float(ref_arr.mean() - other.mean()),
                            "r_rb": rank_biserial(ref_arr, other),
                            "cliffs": float(cliffs_delta(ref_arr, other)),
                            "wins": int((ref_arr > other).sum())}
    holm = _holm(raw)
    for e in [e for e in order if e not in ("none", ref)]:
        t = ref_tests[e]
        t["p_holm"] = holm[e]
        print(f"  {ref} vs {e:>7}: delta={t['delta']:+.4f}  "
              f"p_holm={holm[e]:.3g}  r_rb={t['r_rb']:+.2f}  "
              f"({t['wins']}/{len(ref_arr)})")

    # spread among the stronger operators (the 'operator-agnostic' claim)
    top5 = order[-5:]
    spread5 = np.mean([results[e] for e in top5], axis=1)
    print(f"\nTop-5 operators: {top5}")
    print(f"  mean-AR spread across top 5: {spread5.max()-spread5.min():.4f} "
          f"(small => benefit is robust to operator choice)")

    # ---- Friedman across all operators ------------------------------------
    mat = np.column_stack([results[e] for e in ESCAPES])
    fr = stats.friedmanchisquare(*[mat[:, k] for k in range(mat.shape[1])])
    print(f"\nFriedman across {len(ESCAPES)} operators: "
          f"chi2={fr.statistic:.1f}, p={fr.pvalue:.3g}")

    import json
    summary = {
        "n_instances": len(complete), "n_runs": args.n_runs,
        "operators": ESCAPES,
        "mean_approx": {e: float(np.mean(results[e])) for e in ESCAPES},
        "std_approx": {e: float(np.std(results[e])) for e in ESCAPES},
        "order_ascending": order,
        "best_mean_operator": ref,
        "friedman": {"chi2": float(fr.statistic), "p": float(fr.pvalue)},
        "vs_none": none_tests,
        "vs_best": ref_tests,
        "top5_spread": float(spread5.max() - spread5.min()),
        "per_instance": {e: [float(x) for x in results[e]] for e in ESCAPES},
    }
    json.dump(summary, open(store.run_dir / "ablation_summary.json", "w"),
              indent=2)
    print(f"\nSaved -> {store.run_dir / 'ablation_summary.json'}")
    print("  (per_instance arrays saved for the critical-difference diagram.)")


def _holm(pmap: dict) -> dict:
    """Holm-Bonferroni step-down correction. Input/return: {label: p}."""
    items = sorted(pmap.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)   # enforce monotonicity
        out[k] = running
    return out


if __name__ == "__main__":
    main()
