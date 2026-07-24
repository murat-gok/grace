"""GATE-2: warm-start ablation -- is the GNN necessary?

The question
------------
The paper's headline is that a GNN predicts good QAOA angles from graph
structure in a single evaluation. But parameter concentration says optimal
angles are approximately shared across instances of a family. If that is the
whole story, a fixed table of angles -- the MEDIAN of optimal angles over the
training graphs -- should match the GNN at the same cost (one evaluation), and
the entire learned component would be unnecessary.

No experiment in the manuscript rules this out. This one does.

Design: 3 warm-starts x 2 loop settings, on the same test instances
-------------------------------------------------------------------
    random_only    uniform random angles, 1 evaluation
    fixed_only     median of TRAIN-split optimal angles, 1 evaluation
    gnn_only       GNN prediction, 1 evaluation

    random_escape  random angles  + escape loop
    fixed_escape   median angles  + escape loop
    gnn_escape     GNN prediction + escape loop   (= GRACE, refiner="none")

All three loop configurations run the identical controller with the identical
budget, so any difference is attributable to the warm-start alone. The escape
schedule is deterministic under refiner="none" (no refinement -> a stall every
round -> an escape every `stall_patience` rounds), so the three loop variants
consume the same number of evaluations by construction. This is verified and
reported rather than assumed.

Fixed angles come from the TRAIN split (disjoint seed range), so this baseline
sees exactly the same data the GNN was trained on -- no more, no less. That is
what makes it a fair rival rather than a straw man.

Reading the result
------------------
  * fixed_only ~ gnn_only  -> the GNN is a lookup table with extra steps.
                              Reposition the paper around the escape study, or
                              justify the GNN on out-of-distribution transfer
                              (cross-family / cross-size), not on this benchmark.
  * gnn_only >> fixed_only -> the GNN genuinely reads graph structure. The
                              warm-start claim stands and this table proves it.
  * random_escape ~ gnn_escape -> the warm-start does not survive the loop; the
                              escape operator is doing all the work.

Run
---
    python scripts/ablate_warmstart.py --config configs/hard2gnn.yaml \
        --gnn-model models/gnn_warmstart_p5_gcn.pt \
        --n-instances 60 --n-runs 10 --run-name ablate_warmstart
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from scipy.optimize import minimize

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.gnn.loader import load_gnn_warmstart
from grace_qaoa.controller.grace import GraceController
from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted
from grace_qaoa.utils.checkpoint import CheckpointStore

CONFIGS = ["random_only", "fixed_only", "gnn_only",
           "random_escape", "fixed_escape", "gnn_escape"]

WARM_STARTS = {"random_only": "random", "random_escape": "random",
               "fixed_only": "fixed", "fixed_escape": "fixed",
               "gnn_only": "gnn", "gnn_escape": "gnn"}


# --------------------------------------------------------------------------- #
# fixed-angle table (the GNN's rival)
# --------------------------------------------------------------------------- #
def compute_fixed_angles(cfg, n_train_per_family, multistart, maxiter,
                         cache_path, seed=0):
    """Median optimal angles over TRAIN-split graphs -- one table, reused for
    every test instance at a cost of one evaluation.

    This mirrors how the GNN's training targets are produced (multi-start
    COBYLA), so the two warm-starts are built from the same information; the
    only difference is that the GNN can condition on the individual graph while
    this table cannot. That contrast is exactly what we want to measure.

    The median (not the mean) is used because angle distributions across
    instances are skewed and the median is the robust summary; it is also what
    a practitioner would tabulate.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        d = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded fixed-angle table from {cache_path}: "
              f"{np.round(d['angles'], 4).tolist()}")
        return np.asarray(d["angles"], dtype=float)

    p = cfg["qaoa_p"]
    rng = np.random.default_rng(seed)
    all_params = []
    print(f"Building fixed-angle table from TRAIN split "
          f"({n_train_per_family} graphs/family, {multistart} restarts each)...")
    for fam in cfg["families"]:
        graphs = make_dataset(fam, n_graphs=n_train_per_family,
                              n_nodes=cfg["n_nodes"],
                              weighted=cfg.get("weighted", True), split="train")
        for g in graphs:
            qaoa = QAOAMaxCut(g, p=p)
            best_x, best_cut = None, -np.inf
            for _ in range(multistart):
                x0 = rng.uniform(0, np.pi, 2 * p)
                res = minimize(lambda x: qaoa.cost(x), x0, method="COBYLA",
                               options={"maxiter": maxiter})
                cut = qaoa.expected_cut(res.x)
                if cut > best_cut:
                    best_cut, best_x = cut, np.asarray(res.x, dtype=float)
            all_params.append(best_x)
        print(f"  {fam}: done")

    arr = np.vstack(all_params)
    angles = np.median(arr, axis=0)
    # Report dispersion: tight spread is direct evidence of parameter
    # concentration and tells us how strong this baseline should be.
    iqr = np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)
    print(f"  median angles : {np.round(angles, 4).tolist()}")
    print(f"  IQR per angle : {np.round(iqr, 4).tolist()}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"angles": angles.tolist(), "iqr": iqr.tolist(),
         "n_graphs": int(arr.shape[0]), "p": p,
         "multistart": multistart, "maxiter": maxiter}, indent=2),
        encoding="utf-8")
    return angles


# --------------------------------------------------------------------------- #
# one (instance, config) cell
# --------------------------------------------------------------------------- #
def run_cell(g, cfg, gnn, fixed_angles, which, seed):
    p = cfg["qaoa_p"]
    opt = brute_force_maxcut(g)
    kind = WARM_STARTS[which]

    if kind == "random":
        init = np.random.default_rng(seed).uniform(0, np.pi, 2 * p)
    elif kind == "fixed":
        init = fixed_angles.copy()
    else:
        init = gnn.predict_params(g).reshape(-1)

    if which.endswith("_only"):
        return QAOAMaxCut(g, p=p).expected_cut(init) / opt, 1

    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cfg.get("quantum_budget"))
    ctrl = GraceController(cq, escape=cfg["escape"],
                           stall_eps=cfg.get("stall_eps", 1e-3),
                           stall_patience=cfg.get("stall_patience", 2),
                           max_rounds=cfg["grace_rounds"], seed=seed)
    try:
        res = ctrl.run(init, refiner="none")   # the paper's two-stage method
        return res["best_cut"] / opt, res["n_quantum_evals"]
    except BudgetExhausted:
        return cq.best_cut_so_far / opt, cq.n_evals


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    gt = sum((a[i] > b[j]) for i in range(n) for j in range(n))
    lt = sum((a[i] < b[j]) for i in range(n) for j in range(n))
    return (gt - lt) / (n * n)


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation -- the effect size that belongs
    with a Wilcoxon signed-rank test (Cliff's delta is for independent samples).
    """
    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    total = ranks.sum()
    return float(2.0 * ranks[nz > 0].sum() / total - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yaml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--n-instances", type=int, default=60,
                    help="Total test instances (split evenly across families).")
    ap.add_argument("--n-runs", type=int, default=10,
                    help="Runs per instance for the stochastic configurations.")
    ap.add_argument("--n-train", type=int, default=20,
                    help="TRAIN graphs per family for the fixed-angle table.")
    ap.add_argument("--multistart", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--run-name", default="ablate_warmstart")
    ap.add_argument("--fixed-cache", default="models/fixed_angles_p5.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    t0 = time.time()

    gnn = load_gnn_warmstart(args.gnn_model)
    fixed_angles = compute_fixed_angles(
        cfg, args.n_train, args.multistart, args.maxiter, args.fixed_cache)

    graphs = []
    per_fam = max(1, args.n_instances // len(cfg["families"]))
    for fam in cfg["families"]:
        for g in make_dataset(fam, n_graphs=per_fam, n_nodes=cfg["n_nodes"],
                              weighted=cfg.get("weighted", True), split="test"):
            graphs.append((fam, g))

    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / args.run_name)
    store.write_provenance(cfg=cfg, extra={"script": "ablate_warmstart",
                                           "fixed_angles": fixed_angles.tolist()})
    if store.n_completed:
        print(f"Resuming '{args.run_name}': {store.n_completed} cells done.")

    total = len(graphs) * len(CONFIGS)
    for gi, (fam, g) in enumerate(graphs):
        for which in CONFIGS:
            cid = f"inst{gi}|{which}"
            if store.is_done(cid):
                continue
            if which in ("fixed_only", "gnn_only"):
                # deterministic: one run suffices
                val, ev = run_cell(g, cfg, gnn, fixed_angles, which, 0)
            else:
                vals, evs = zip(*[run_cell(g, cfg, gnn, fixed_angles, which, s)
                                  for s in range(args.n_runs)])
                val, ev = float(np.mean(vals)), float(np.mean(evs))
            store.append(cid, {"instance": gi, "family": fam,
                               "config": which, "ar": val, "evals": ev})
            print(f"  [{store.n_completed}/{total}] inst{gi} {which:>14}: "
                  f"{val:.4f}  ({ev:.0f} evals)")

    # ----------------------------------------------------------------- #
    # aggregate
    # ----------------------------------------------------------------- #
    rows = store.load_all_records()
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["config"]] = r
    complete = [i for i, d in sorted(by.items())
                if all(c in d for c in CONFIGS)]
    ar = {c: np.array([by[i][c]["ar"] for i in complete]) for c in CONFIGS}
    ev = {c: np.array([by[i][c]["evals"] for i in complete]) for c in CONFIGS}
    n = len(complete)

    print(f"\n{'config':>15} {'mean AR':>9} {'std':>8} {'mean evals':>11}"
          f"   (n={n} instances)")
    for c in CONFIGS:
        print(f"{c:>15} {ar[c].mean():>9.4f} {ar[c].std():>8.4f} "
              f"{ev[c].mean():>11.0f}")

    # budget parity check -- the claim that the loop variants are matched
    loop = ["random_escape", "fixed_escape", "gnn_escape"]
    spread = max(ev[c].mean() for c in loop) - min(ev[c].mean() for c in loop)
    print(f"\nBudget parity across loop variants: spread = {spread:.1f} evals "
          f"({'OK' if spread < 5 else 'CHECK -- not matched!'})")

    def compare(a_name, b_name, label):
        a, b = ar[a_name], ar[b_name]
        d = a - b
        if np.allclose(a, b):
            print(f"  {label:>34}: identical")
            return {"delta": 0.0, "p": None, "r_rb": 0.0,
                    "cliffs": 0.0, "wins": 0, "losses": 0}
        pv = float(stats.wilcoxon(a, b).pvalue)
        rec = {"delta": float(d.mean()), "p": pv,
               "r_rb": rank_biserial(a, b), "cliffs": float(cliffs_delta(a, b)),
               "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
        print(f"  {label:>34}: {rec['delta']:+.4f}  p={pv:.3g}  "
              f"r_rb={rec['r_rb']:+.2f}  ({rec['wins']}/{rec['losses']} w/l)")
        return rec

    print(f"\nPaired comparisons on the {n} per-instance means:")
    comps = {}
    comps["gnn_vs_fixed_1eval"] = compare(
        "gnn_only", "fixed_only", "GNN vs fixed table (1 eval)")
    comps["gnn_vs_random_1eval"] = compare(
        "gnn_only", "random_only", "GNN vs random (1 eval)")
    comps["fixed_vs_random_1eval"] = compare(
        "fixed_only", "random_only", "fixed table vs random (1 eval)")
    comps["gnn_vs_fixed_loop"] = compare(
        "gnn_escape", "fixed_escape", "GRACE vs fixed+escape")
    comps["gnn_vs_random_loop"] = compare(
        "gnn_escape", "random_escape", "GRACE vs random+escape")
    comps["escape_gain_gnn"] = compare(
        "gnn_escape", "gnn_only", "escape gain (GNN start)")
    comps["escape_gain_random"] = compare(
        "random_escape", "random_only", "escape gain (random start)")

    print("\n=== VERDICT ===")
    gap_1eval = ar["gnn_only"].mean() - ar["fixed_only"].mean()
    gap_loop = ar["gnn_escape"].mean() - ar["fixed_escape"].mean()
    print(f"  GNN advantage at 1 eval      : {gap_1eval:+.4f}")
    print(f"  GNN advantage after the loop : {gap_loop:+.4f}")
    if gap_1eval < 0.005:
        print("  The fixed-angle table matches the GNN at equal cost. On this")
        print("  benchmark the learned warm-start is not carrying the result;")
        print("  justify it on transfer (cross-family / cross-size) or drop it")
        print("  and reposition the paper around the escape-operator study.")
    elif gap_loop < 0.005 <= gap_1eval:
        print("  The GNN is better at 1 eval, but the escape stage erases the")
        print("  gap. The learned start buys speed, not final quality -- say so")
        print("  explicitly and support it with the sample-efficiency curve.")
    else:
        print("  The GNN beats the fixed-angle table both at 1 eval and after")
        print("  the loop: the learned warm-start is doing real work. This is")
        print("  the table that proves it.")

    print("\n=== Per family (mean AR) ===")
    fams = sorted({by[i]["gnn_only"]["family"] for i in complete})
    for fam in fams:
        sel = [i for i in complete if by[i]["gnn_only"]["family"] == fam]
        line = "  ".join(
            f"{c}={np.mean([by[i][c]['ar'] for i in sel]):.4f}"
            for c in ("fixed_only", "gnn_only", "fixed_escape", "gnn_escape"))
        print(f"  {fam:16s} {line}")

    summary = {
        "n_instances": n, "n_runs": args.n_runs,
        "fixed_angles": fixed_angles.tolist(),
        "means": {c: float(ar[c].mean()) for c in CONFIGS},
        "stds": {c: float(ar[c].std()) for c in CONFIGS},
        "mean_evals": {c: float(ev[c].mean()) for c in CONFIGS},
        "budget_spread_loop_variants": float(spread),
        "comparisons": comps,
    }
    out = store.run_dir / "warmstart_ablation.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}  ({round(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
