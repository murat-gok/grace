# Grace: Budget-Fair, Operator-Portable QAOA Parameter Setting

Reference implementation for the paper **"Grace: A Budget-Fair,
Operator-Portable Framework Coupling Graph-Neural Warm-Starts with Metaheuristic
Escape for QAOA Parameter Setting"** (submitted to *Swarm and Evolutionary
Computation*).

Grace is a closed-loop framework for setting the variational angles of QAOA on
weighted MaxCut. It couples a **GNN warm-start** (predicts good angles in one
circuit evaluation) with a **metaheuristic escape operator** invoked when the
loop stalls, all under strict per-evaluation budget accounting. The escape stage
is *operator-portable*: eleven operators are compared under an identical budget
and at tuned coefficients; continuous ant colony optimization (ACO_R) is the
strongest default.

---

## Installation

CPU-only; no GPU or quantum hardware required (QAOA circuits are simulated).

```bash
git clone https://github.com/murat-gok/grace.git
cd grace
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the grace_qaoa package
```

Tested with Python 3.11.

---

## Reproducing the paper

All experiments are crash-safe (per-cell checkpointing): re-running the same
`--run-name` resumes where it left off. Results are written under `results/`.

The trained GNN warm-starts (`models/gnn_warmstart_p*_gcn.pt`) and the tuned
operator coefficients (`models/tuned_operators.json`) are included so the main
tables can be reproduced without retraining. To rebuild them from scratch, see
**Training** below.

### Main comparison (Table 1)

```bash
python scripts/run_experiment.py --config configs/hard2gnn.yaml \
    --run-name maintable
```

Produces the mean approximation ratios, standard deviations, average ranks,
Friedman test, and Holm-corrected Wilcoxon tests for all nine methods on the
60-instance benchmark (20 instances × 3 families, 30 runs each).

### Warm-start ablation (Table 2 / Fig. warm-start)

```bash
python scripts/ablate_warmstart.py --config configs/hard2gnn.yaml \
    --gnn-model models/gnn_warmstart_p5_gcn.pt \
    --n-instances 60 --n-runs 10 --run-name ablate_warmstart
```

Compares random / fixed-angle-table / GNN warm-starts, at one evaluation and
after the escape loop.

### Escape-operator ablation (Table 3 / CD diagram), tuned

```bash
python scripts/ablate_escape.py --config configs/hard2gnn.yaml \
    --gnn-model models/gnn_warmstart_p5_gcn.pt \
    --tuned models/tuned_operators.json \
    --only none random de ga pso aco woa gwo hho cmaes cpo \
    --n-instances 60 --n-runs 10 --run-name ablate_escape11_tuned
```

Eleven operators under an identical per-escape budget of 180 evaluations, at
their validation-tuned coefficients.

### Operator tuning parity

```bash
python scripts/tune_operators.py --config configs/hard2gnn.yaml \
    --gnn-model models/gnn_warmstart_p5_gcn.pt \
    --operators de ga pso aco cmaes gwo hho cpo \
    --n-val 6 --n-configs 20 --n-runs 3 --run-name tune_v1
```

Tunes each operator's behavioural coefficients on a disjoint validation split;
writes `models/tuned_operators.json`. (Already provided; re-run only to rebuild.)

### Depth sweep (p ∈ {3, 5, 8})

```bash
# p=3 and p=8 use their own configs and GNN warm-starts:
python scripts/ablate_escape.py --config configs/hard2gnn_p3.yaml \
    --gnn-model models/gnn_warmstart_p3_gcn.pt --tuned models/tuned_operators.json \
    --only none random de ga pso aco woa gwo hho cmaes cpo \
    --n-instances 30 --n-runs 10 --run-name ablate_escape11_p3

python scripts/ablate_escape.py --config configs/hard2gnn_p8.yaml \
    --gnn-model models/gnn_warmstart_p8_gcn.pt --tuned models/tuned_operators.json \
    --only none random de ga pso aco woa gwo hho cmaes cpo \
    --n-instances 30 --n-runs 10 --run-name ablate_escape11_p8

python scripts/analyze_depth_sweep.py \
    --p3 results/ablate_escape11_p3/ablation_summary.json \
    --p5 results/ablate_escape11_tuned/ablation_summary.json \
    --p8 results/ablate_escape11_p8/ablation_summary.json \
    --warmstart 3=0.782 5=0.882 8=0.778
```

Reports the Kendall-τ rank correlation between depths.

### Size scaling (n ∈ {10..18})

```bash
python scripts/robustness.py --mode size --config configs/hard2gnn.yaml \
    --escape aco --tuned models/tuned_operators.json \
    --sizes 10 12 14 16 18 --n-instances 20 --n-runs 20 \
    --run-name robust_size_aco
```

### Empirical Goemans–Williamson reference

```bash
python scripts/gw_baseline.py --config configs/hard2gnn.yaml \
    --n-instances 60 --rounding-restarts 200 --run-name gw_empirical
```

### Figures

```bash
python scripts/make_cd_diagram.py --summaries \
    p3=results/ablate_escape11_p3/ablation_summary.json \
    p5=results/ablate_escape11_tuned/ablation_summary.json \
    p8=results/ablate_escape11_p8/ablation_summary.json --out figures/cd
python scripts/make_depth_figure.py --summaries \
    p3=results/ablate_escape11_p3/ablation_summary.json \
    p5=results/ablate_escape11_tuned/ablation_summary.json \
    p8=results/ablate_escape11_p8/ablation_summary.json \
    --warmstart 3=0.782 5=0.882 8=0.778 --out figures/depth_sweep
python scripts/make_phase3_figures.py \
    --warmstart results/ablate_warmstart/warmstart_ablation.json \
    --size results/robust_size_aco/size_scalability.json --out figures
python scripts/make_sampeff_figure.py \
    --data results/sampeff/sample_efficiency.json --out figures/sample_efficiency
```

---

## Training the GNN warm-start

The provided models suffice to reproduce the paper. To retrain:

```bash
# p=5 (default depth). Targets are cached; the offline COBYLA budget scales
# with the search dimension (60 * 2p).
python scripts/pretrain_gnn.py --config configs/hard2gnn.yaml --conv gcn

# other depths:
python scripts/pretrain_gnn.py --config configs/hard2gnn_p3.yaml --conv gcn
python scripts/pretrain_gnn.py --config configs/hard2gnn_p8.yaml --conv gcn
```

---

## Repository layout

```
src/grace_qaoa/
  quantum/      QAOA circuit + weighted-MaxCut graph generators (seeded splits)
  gnn/          GNN warm-start model and loader
  metaheuristic/ escape operators (ACO_R, DE, PSO, GA, WOA, GWO, HHO, CMA-ES, CPO, ...)
  controller/   the Grace closed loop
  baselines*    INTERP, FOURIER, SPSA, transfer, random search
  utils/        shared budget counter, crash-safe checkpointing
scripts/        experiment, ablation, tuning, and figure scripts
configs/        hard2gnn.yaml (p=5) and depth variants (p=3, p=8)
models/         trained GNN warm-starts + tuned operator coefficients
tests/          unit tests, including the train/test seed-disjointness guard
```

---

## Notes on fairness and reproducibility

- **Budget accounting.** Every method is charged one unit per circuit
  evaluation through a shared counter with a hard cap of 3,000 evaluations.
- **No train/test leakage.** Training, validation, and test graphs use disjoint
  seed ranges; `tests/test_no_leakage.py` enforces this.
- **Deterministic.** All instances come from seeded generators, so every table
  and figure is reproducible bit-for-bit on CPU.

## Citation

```bibtex
@article{gok2026grace,
  title   = {Grace: A Budget-Fair, Operator-Portable Framework Coupling
             Graph-Neural Warm-Starts with Metaheuristic Escape for QAOA
             Parameter Setting},
  author  = {G\"ok, Murat and Gen\c{c}, Sevdanur and Cengiz, Emine and Tekin, Muhammed},
  journal = {Swarm and Evolutionary Computation (submitted)},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
