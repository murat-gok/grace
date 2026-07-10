# GRACE-QAOA

**G**raph-neural warm-start + **R**einforcement-learning refinement + met**A**heuristic es**C**ape op**E**rator for closed-loop QAOA.

A research codebase for the T3 project: closing the QAOA initialization loop by
combining a GNN warm-start, an RL refiner, and a metaheuristic (CPO/DE) escape
operator that fires when the refiner stalls. Designed to run **CPU-only on the
office Xeon E5-1660 v3 (8C/16T) + 32 GB** — no GPU required, Quadro K2200 ignored.

## What works right now

The full closed loop runs end-to-end on simulators today:
- QAOA MaxCut on `lightning.qubit` (CPU), weighted + OOD graph families
- GNN warm-start model (GCN/GAT/GraphSAGE switchable — ablation d)
- RL refinement environment (Gymnasium, TD3/SAC-ready)
- CPO and DE escape operators (ablation c)
- Threshold controller orchestrating the three modes
- Baselines (random, COBYLA, GNN-only) + Wilcoxon / Friedman / Holm stats
- Parallel experiment runner tuned for the Xeon

> Note: the experiment runner currently uses a built-in hill-climb *stand-in*
> for the RL refiner so the whole pipeline runs without a trained policy. Swap
> in a trained Stable-Baselines3 TD3/SAC policy via `rl_step_fn` to get the full
> system. This is the next implementation step (see Roadmap).

## Setup (office Xeon, CPU-only)

```bash
conda env create -f environment.yml
conda activate grace-qaoa
pip install -e .
```

Or with plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

**Important version pin:** `autoray==0.6.12`. Newer autoray removes
`NumpyMimic` and breaks PennyLane 0.38. This is already pinned in the recipes.

## Fair-budget protocol & strong baselines

The comparison is built to survive Q1 review. Two things matter:

**1. Strong baselines, not strawmen.** Beyond random/COBYLA, the suite includes
the literature-standard QAOA methods a reviewer expects:
- **INTERP** and **FOURIER** (Zhou et al., *Phys. Rev. X* 10, 021067, 2020) —
  layer-by-layer interpolation / frequency-basis initialization.
- **SPSA** (Spall 1992) — strong, cheap gradient-style optimizer.
- **Parameter-concentration transfer.**
Beating COBYLA proves little; beating INTERP/FOURIER under equal budget is the
real result.

**2. Equal quantum-evaluation budget.** Every method (including GRACE and its
finite-difference RL features) is wrapped in a `CountingQAOA` that charges one
unit per circuit expectation and enforces a hard `quantum_budget`. This answers
the decisive reviewer question — *does GRACE win on quality, or just by spending
more circuit calls?* — directly. The runner reports, per method:
- mean **approximation ratio** (quality) under the shared budget,
- mean **evaluations used**,
- **fraction of instances that reached the target** (default 95% of optimal),
- **median evaluations-to-target** (sample efficiency / cost-to-solution).

GRACE's intended story is *not* "higher optimum at any cost" but "reaches a good
cut with fewer evaluations and more reliably" — the natural payoff of the escape
operator. Statistics are instance-paired Wilcoxon of GRACE vs each baseline,
plus Friedman + Holm across all methods.

Set the budget and target in the config:

```yaml
quantum_budget: 1500   # max circuit evaluations per method per instance
target_frac: 0.95      # "reached" = >= 95% of optimal cut
```

## Power-outage resilience (checkpointing)

Long runs survive power cuts. Both the experiment runner and RL training save
progress to disk incrementally and can resume.

**Experiments.** Every completed job is appended to
`results/<run-name>/checkpoint.jsonl` and `fsync`'d to physical disk
immediately, so a power loss costs at most the few jobs in flight. To make a run
resumable, give it a name; to resume, just run the *same command again*:

```bash
python scripts/run_experiment.py --config configs/full.yaml --run-name full_p3
# ... power cut ...
python scripts/run_experiment.py --config configs/full.yaml --run-name full_p3   # resumes
```

It prints e.g. `Resuming 'full_p3': 240 jobs already done, skipping those.` and
continues from where it stopped. Jobs run in batches (default 4x n_jobs);
the checkpoint is flushed after each batch. A half-written final line from a
hard crash is detected and that single job is simply recomputed.

**RL training.** Autosaves every `--save-every` steps to
`models/ckpt_<algo>/`. Resume with `--resume`:

```bash
python scripts/train_rl.py --algo td3 --timesteps 50000 --save-every 2000
# ... power cut ...
python scripts/train_rl.py --algo td3 --timesteps 50000 --resume   # continues
```

> Tip: on the office Xeon, also consider a small UPS for graceful shutdown — but
> the checkpointing above means you lose minutes, not the whole run, either way.

## Run

```bash
# smoke test the whole pipeline (seconds)
pytest -v

# quick experiment with the hill-climb stand-in refiner
python scripts/run_experiment.py --config configs/quick.yaml

# --- full system: train both components, then run ---
# 1) GNN warm-start (MSE to optimal angles, then cut-based fine-tuning)
python scripts/pretrain_gnn.py --config configs/hard.yaml --conv gcn --epochs 300 --finetune-epochs 30
# 2) RL refiner
python scripts/train_rl.py --algo td3 --timesteps 50000 --config configs/hard.yaml
# 3) point the config at both models (see commented keys in configs/hard.yaml),
#    then run the HARD-REGIME experiment (high p, weighted/OOD)
python scripts/run_experiment.py --config configs/hard.yaml --run-name hard_p5
```

### The hard regime (where GRACE is meant to win)

`configs/hard.yaml` is the decisive setting: **p=5**, weighted + structurally
diverse (OOD) graphs. This matters because, empirically, at **p=1 the optimal
QAOA angles barely depend on graph structure** — so a GNN warm-start has almost
nothing to learn and an escape operator has no local minima to escape. The
contribution only shows up at depth, where the landscape is rugged and
one-shot warm-starts are brittle. Do not judge the method on p=1 results.

> **Key ablation built in: `gnn_only`.** The runner measures GNN warm-start with
> no refinement and no escape, so the marginal value of each component is
> explicit. (In early p=1 tests `gnn_only` was no better than random — exactly
> why the hard regime is necessary.)

## CPU parallelism (read this before big runs)

To avoid core oversubscription on the Xeon, the runner pins each simulator
worker to a single thread (`OMP_NUM_THREADS=1`, set at the top of
`run_experiment.py`) and parallelizes **across runs** with joblib. Set
`n_jobs` in the config to **<= 8** (physical cores). For one heavy
high-qubit run instead, set `n_jobs: 1` and raise the thread env vars.

## Qubit / memory budget

- Keep `n_nodes` (= qubits) <= 24 for single runs, <= 20 when running many in parallel.
- Exact state-vector memory ~ 16 bytes x 2^n: 24 qubits ~ 256 MB, 28 ~ 4 GB.
- **Noisy phase:** density-matrix memory grows as 2^(2n) — keep <= 14 qubits, or
  use shot-based sampling (`shots=...` in `QAOAMaxCut`) instead.

## Layout

```
src/grace_qaoa/
  quantum/      qaoa.py (PennyLane MaxCut), graphs.py (instance families)
  gnn/          warm_start.py (GCN/GAT/SAGE angle predictor)
  rl/           refine_env.py (Gymnasium env for TD3/SAC)
  metaheuristic/escape.py (CPO + DE escape operators)
  controller/   grace.py (closed-loop orchestrator)
  utils/        stats.py (Wilcoxon / Friedman / Holm)
  baselines.py
configs/        quick.yaml, full.yaml
scripts/        run_experiment.py
tests/          test_smoke.py
```

## Roadmap (next implementation steps)

1. **[DONE] Train the RL refiner.** `scripts/train_rl.py` (SB3 TD3/SAC),
   adapter in `rl/refiner.py`, autosave + `--resume`.
2. **[DONE] Pretrain the GNN.** `scripts/pretrain_gnn.py`: offline optimal-angle
   targets (cached), MSE phase, then cut-based fine-tuning via PennyLane's torch
   interface. Loader in `gnn/loader.py`; wired into GRACE init + `gnn_only`
   ablation in the runner.
3. **[IN PROGRESS] Hard regime.** `configs/hard.yaml` (p=5, weighted/OOD). Train
   RL + GNN *for p=5*, then run. This is where escape + warm-start should beat
   the strong baselines under equal budget.
4. **Resolve the RL gradient-feature cost.** The finite-difference gradient-norm
   feature costs ~2p evals/step and is charged honestly by CountingQAOA — but it
   eats budget. Options: cache it, approximate it, or drop it. Decide before
   final runs.
5. **Run the ablations** (escape on/off via `gnn_only` vs `grace`, CPO vs DE,
   GCN/GAT/SAGE, eps sweep) once the hard regime shows a signal.

## Open issues / honest status

- At **p=1**, `gnn_only` ≈ random and GRACE does not beat strong baselines under
  equal budget. This is expected (shallow landscape) and motivates the hard
  regime; it is not yet a publishable result.
- The GNN MSE plateaus around ~1.2 at p=1 because optimal angles are nearly
  graph-independent there; the cut-based fine-tuning helps only marginally at
  p=1. Re-evaluate at p=5 where graph structure matters.
- In easy regimes, simple `transfer` (parameter concentration) is already very
  strong (often ~0.90 approx ratio in tens of evals). GRACE must be tested where
  `transfer` breaks down (high p, weighted/OOD) — that is what `configs/hard.yaml`
  targets.

## Resolved

- **[RESOLVED — CRITICAL] GNN train/test data leakage.** Earlier, `pretrain_gnn.py`
  and `run_experiment.py` both generated graphs with `base_seed=0`, so the GNN
  was trained on the very graphs it was later tested on. This inflated
  `gnn_only` to ~0.88 (and GRACE on top of it) in the first p=5 run. Fixed with
  disjoint per-split seed offsets (`SPLIT_OFFSET`: test=0, train=1e6, val=2e6);
  training code passes `split="train"`, the experiment runner defaults to
  `split="test"`. Verified zero overlap; guarded by `tests/test_no_leakage.py`.
  **The first `hard_p5_full` result (gnn_only 0.878, grace 0.894) is invalid and
  must be re-run after retraining the GNN/RL on the train split.**
- **[RESOLVED] RL gradient-feature cost.** The refiner no longer computes a
  finite-difference gradient (which cost 1 + 2p evals/step). The observation now
  uses zero-extra-cost landscape signals — normalized cut, recent improvement
  (momentum), and stall fraction — so each refinement step costs exactly ONE
  quantum evaluation. Verified: a 30-step round at p=3 dropped from ~211 to 31
  evals. GRACE is now budget-competitive with the baselines.
- **[RESOLVED] Observation leakage.** Features are scaled by the running-best cut,
  never the true optimum, so the agent never sees the answer and training matches
  inference (where the optimum is unknown). Pre-empts the "information leakage"
  reviewer objection.
- **[RESOLVED] Escape operator rarely fired.** The trigger was a brittle
  window-width test; it now fires when the *best-so-far* fails to improve for
  `stall_patience` consecutive rounds (relative-improvement based, robust to a
  noisy refiner). Diagnosed with `scripts/diagnose_escape.py`: at p=5 on weighted
  graphs, escape went from firing on 1/6 instances to 5/6 (avg ~1.8 per
  instance), and escape-ON beat escape-OFF by +0.006 mean approx ratio (up to
  +0.018 on the hardest instance). The CPO novelty is now actually exercised.

> Diagnose escape behaviour before any long run:
> ```bash
> python scripts/diagnose_escape.py --p 5 --n-instances 6 --n-nodes 12 --escape cpo
> ```
> It reports trigger counts and escape-ON vs escape-OFF per instance. Tune
> `stall_patience` / `stall_eps` in the config if escape never fires.

> Note: RL models are **p-specific** (the observation size encodes p). Train one
> model per p; the runner raises a clear error on mismatch.

## Target venues

Quantum Machine Intelligence or Applied Soft Computing (first submission).

## Statistical protocol

>= 30 independent runs; Wilcoxon signed-rank (pairwise vs best baseline),
Friedman (multi-method ranking), Holm post-hoc. All wired in `utils/stats.py`
and reported automatically by the runner.
