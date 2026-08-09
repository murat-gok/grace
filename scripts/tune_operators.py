"""E5: operator tuning parity -- remove the 'untuned rivals' objection.

Why
---
In the escape ablation every operator ran with the default coefficients from its
original paper. ACO won 60/60. A Swarm-and-Evolutionary-Computation reviewer will
object -- correctly -- that a default CMA-ES or DE at a 180-evaluation / 10-dim
budget is handicapped, so the win may reflect tuning, not the operator. This
script gives every operator the SAME tuning budget on a SEPARATE validation split,
then reports the tuned operators. Whatever the outcome, the paper is stronger:
either ACO still wins against tuned rivals (a much harder claim), or the ranking
shifts and we report that tuning changes the order (a methodological contribution
in its own right).

Fairness rules baked in
-----------------------
  * Validation graphs come from split="val", disjoint from the test set used in
    the ablation. Tuning never touches test data.
  * Every operator gets the identical tuning budget: `n_configs` random
    configurations x `n_val` instances x `n_runs`. ACO gets no more tries than
    DE.
  * pop_size and iters are NOT tuned -- they set the per-escape evaluation count,
    and the whole ablation's fairness rests on that count being identical across
    operators. Only behavioural coefficients are searched.
  * The objective is the same GRACE loop (refiner="none") the ablation uses, so
    the tuned config is optimized for exactly the setting it will be judged in.
  * Random search (not grid) so the budget buys the same coverage regardless of
    how many parameters an operator has.

Output
------
  models/tuned_operators.json : {operator: best_config} for run_escape_fair to
  consume via **op_kwargs. Also a per-operator table of default vs tuned
  validation score, so the tuning gain is visible.

PREREQUISITE (one-line change in escape.py):
    def run_escape_fair(name, cost_fn, x0, rng, max_evals, bounds=(0.0, np.pi),
                        **op_kwargs):
        ...
        op(capped, x0, bounds=bounds, rng=rng, **op_kwargs)
    so a tuned config can be passed through. WOA takes no coefficients and is
    included only as a fixed reference.

Run
---
    python scripts/tune_operators.py --config configs/hard2gnn.yaml \
        --gnn-model models/gnn_warmstart_p5_gcn.pt \
        --operators de ga pso aco cmaes gwo hho cpo \
        --n-val 6 --n-configs 20 --n-runs 3 --run-name tune_v1
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
from joblib import Parallel, delayed

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.gnn.loader import load_gnn_warmstart
from grace_qaoa.controller.grace import GraceController
from grace_qaoa.utils.budget import CountingQAOA, BudgetExhausted
from grace_qaoa.utils.checkpoint import CheckpointStore


# --------------------------------------------------------------------------- #
# per-operator tunable coefficient spaces (behavioural only; NOT pop_size/iters)
# Ranges bracket the published defaults so the default is inside the search box.
# --------------------------------------------------------------------------- #
def _sample_config(op, rng):
    if op == "de":
        return {"F": float(rng.uniform(0.3, 0.9)),
                "CR": float(rng.uniform(0.5, 0.95))}
    if op == "ga":
        return {"mut_rate": float(rng.uniform(0.1, 0.5)),
                "mut_scale": float(rng.uniform(0.05, 0.4))}
    if op == "pso":
        return {"w": float(rng.uniform(0.4, 0.9)),
                "c1": float(rng.uniform(1.0, 2.0)),
                "c2": float(rng.uniform(1.0, 2.0))}
    if op == "aco":
        return {"q": float(rng.uniform(0.02, 0.5)),
                "xi": float(rng.uniform(0.5, 0.95))}
    if op == "cmaes":
        # sigma0 is the CMA-ES step size. Keep it in a sane band: very small
        # values can stall the internal adaptation, very large ones just random-
        # walk. The published default (0.3) sits inside this range.
        return {"sigma0": float(rng.uniform(0.15, 0.6))}
    if op in ("gwo", "hho", "cpo", "woa"):
        # these are (near-)parameter-free in this codebase; only a seed spread
        # analogue is meaningful, if the operator accepts it.
        return {}
    return {}


DEFAULTS = {
    "de": {"F": 0.6, "CR": 0.9},          # scipy-ish default
    "ga": {"mut_rate": 0.3, "mut_scale": 0.2},
    "pso": {"w": 0.7, "c1": 1.5, "c2": 1.5},
    "aco": {"q": 0.1, "xi": 0.85},
    "cmaes": {},
    "gwo": {}, "hho": {}, "cpo": {}, "woa": {},
}


def _one_run(g, opt, init, op, config, cfg, seed):
    p = cfg["qaoa_p"]
    cq = CountingQAOA(QAOAMaxCut(g, p=p), budget=cfg.get("quantum_budget"))
    ctrl = GraceController(cq, escape=op,
                           stall_eps=cfg.get("stall_eps", 1e-3),
                           stall_patience=cfg.get("stall_patience", 2),
                           max_rounds=cfg["grace_rounds"], seed=seed,
                           escape_kwargs=config)
    res = ctrl.run(init.copy(), refiner="none")
    return res["best_cut"] / opt


def _score(op, config, graphs, opts, inits, cfg, n_runs, n_jobs=8):
    """Mean validation approximation ratio of GRACE using `op` with `config`.

    Parallelized across (instance, run) cells. The GNN warm-start does not depend
    on the escape operator or its config, so inits are precomputed once and
    passed in -- workers never touch the (non-picklable) GNN. The previous serial
    version ran n_val*n_runs GRACE loops back-to-back per config, which is what
    made a full sweep take many hours. The config reaches the operator through
    the controller's escape_kwargs -> run_escape_fair(**op_kwargs); a coefficient
    the operator does not accept raises TypeError (surfaced to the caller).
    """
    cells = [(g, opts[i], inits[i], run)
             for i, g in enumerate(graphs) for run in range(n_runs)]
    vals = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_run)(g, opt, init, op, config, cfg, seed)
        for (g, opt, init, seed) in cells)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hard2gnn.yaml")
    ap.add_argument("--gnn-model", default="models/gnn_warmstart_p5_gcn.pt")
    ap.add_argument("--operators", nargs="+",
                    default=["de", "ga", "pso", "aco", "cmaes", "gwo",
                             "hho", "cpo"])
    ap.add_argument("--n-val", type=int, default=6,
                    help="Validation instances per family.")
    ap.add_argument("--n-configs", type=int, default=20,
                    help="Random configs per operator (identical for all).")
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--trial-warn-sec", type=float, default=300.0,
                    help="Warn if a single (operator, config) trial exceeds this "
                         "wall-clock time -- flags a wedged config.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="tune_operators")
    ap.add_argument("--out", default="models/tuned_operators.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    gnn = load_gnn_warmstart(args.gnn_model)
    rng = np.random.default_rng(args.seed)

    # validation graphs -- DISJOINT from the test split used in the ablation
    val_graphs = []
    for fam in cfg["families"]:
        for g in make_dataset(fam, n_graphs=args.n_val, n_nodes=cfg["n_nodes"],
                              weighted=cfg.get("weighted", True), split="val"):
            val_graphs.append(g)
    print(f"Tuning on {len(val_graphs)} validation instances "
          f"(split='val', disjoint from test).")
    print(f"Budget per operator: {args.n_configs} configs x {len(val_graphs)} "
          f"instances x {args.n_runs} runs.")

    # Precompute exact optima and GNN warm-starts once: neither depends on the
    # escape operator or its config, so there is no reason to recompute them for
    # every trial (and it keeps the non-picklable GNN out of the workers).
    print("Precomputing exact optima and GNN warm-starts for validation set...")
    val_opts = [brute_force_maxcut(g) for g in val_graphs]
    val_inits = [gnn.predict_params(g).reshape(-1) for g in val_graphs]
    n_jobs = cfg.get("n_jobs", 8)

    store = CheckpointStore(Path(cfg.get("out_dir", "results")) / args.run_name)
    store.write_provenance(cfg=cfg, extra={"script": "tune_operators",
                                           "operators": args.operators,
                                           "n_configs": args.n_configs})

    # Read the checkpoint ONCE up front (not per trial): resume set + any
    # previously computed scores. Repeatedly reloading the whole JSONL inside
    # the trial loop is O(n^2) and was the cause of the runaway slowdown.
    prior = {r["id"]: r for r in store.load_all_records()}

    results = {}
    t0 = time.time()
    for op in args.operators:
        # always evaluate the published default first, so 'tuned' can never be
        # worse than default on validation (we keep the better of the two).
        trials = [("default", DEFAULTS.get(op, {}))]
        trials += [(f"cfg{c}", _sample_config(op, rng))
                   for c in range(args.n_configs)]

        best = None
        default_score = None
        for ti, (tag, config) in enumerate(trials):
            cid = f"{op}|{tag}|{json.dumps(config, sort_keys=True)}"
            if cid in prior:
                score = prior[cid]["score"]
                print(f"  [{op} {ti+1}/{len(trials)}] {tag}: cached {score:.4f}")
            else:
                ts = time.time()
                try:
                    score = _score(op, config, val_graphs, val_opts, val_inits,
                                   cfg, args.n_runs, n_jobs=n_jobs)
                except TypeError as e:
                    print(f"  [{op}] {tag} {config} rejected by operator "
                          f"signature ({e}); skipping.")
                    continue
                except Exception as e:  # a wedged operator config: skip, don't die
                    print(f"  [{op}] {tag} {config} raised {type(e).__name__}: "
                          f"{e}; skipping.")
                    continue
                dt = time.time() - ts
                rec = {"operator": op, "tag": tag, "config": config,
                       "score": score}
                store.append(cid, rec)
                prior[cid] = rec
                print(f"  [{op} {ti+1}/{len(trials)}] {tag}: {score:.4f} "
                      f"({dt:.0f}s) {config}")
                # A single trial should take seconds-to-minutes. If one blows
                # past a sane wall-clock ceiling something is wrong with that
                # config; warn so it is visible in the log.
                if dt > args.trial_warn_sec:
                    print(f"    WARNING: {op}/{tag} took {dt:.0f}s "
                          f"(> {args.trial_warn_sec}s). Consider excluding "
                          f"that config region.")
            if tag == "default":
                default_score = score
            if best is None or score > best[1]:
                best = (config, score, tag)
        results[op] = {"config": best[0], "score": best[1], "tag": best[2],
                       "default_score": default_score}
        print(f"  => {op:>6}: default={default_score if default_score else float('nan'):.4f}"
              f"  best={best[1]:.4f} ({best[2]}) config={best[0]}")

    print(f"\n{'operator':>8} {'default':>9} {'tuned':>9} {'gain':>8}  config")
    for op in args.operators:
        r = results[op]
        dflt = r["default_score"]
        gain = (r["score"] - dflt) if dflt is not None else float("nan")
        print(f"{op:>8} {(dflt if dflt else float('nan')):>9.4f} "
              f"{r['score']:>9.4f} {gain:>+8.4f}  {r['config']}")

    tuned = {op: results[op]["config"] for op in args.operators}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tuned": tuned,
               "validation": {op: {"default_score": results[op]["default_score"],
                                   "tuned_score": results[op]["score"]}
                              for op in args.operators},
               "meta": {"n_val": args.n_val, "n_configs": args.n_configs,
                        "n_runs": args.n_runs, "split": "val"}},
              open(args.out, "w"), indent=2)
    print(f"\nSaved tuned configs -> {args.out}  ({round(time.time()-t0)}s)")
    print("Next: re-run the escape ablation with --tuned models/tuned_operators.json")


if __name__ == "__main__":
    main()
