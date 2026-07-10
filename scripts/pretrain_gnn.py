"""Pretrain the GNN warm-start model.

Pipeline:
  1. Generate a pool of training graphs (in-distribution families).
  2. For each graph, find good (gamma, beta) OFFLINE with a thorough optimizer
     (multi-start COBYLA). These become regression targets.
  3. Train GNNWarmStart to map graph -> (gamma, beta) by MSE on the targets.
  4. Optionally fine-tune with a cut-based objective (maximize expected cut of
     the predicted angles) for a few epochs.

CPU-only, office Xeon. Targets are cached to disk so a rerun (or a crash) does
not recompute them.

Usage:
    python scripts/pretrain_gnn.py --config configs/quick.yaml --conv gcn --epochs 300
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import minimize
from torch_geometric.loader import DataLoader

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.quantum.qaoa import QAOAMaxCut, brute_force_maxcut
from grace_qaoa.gnn.warm_start import GNNWarmStart, graph_to_data


def compute_targets(graphs, p, multistart=8, maxiter=120, seed=0):
    """Find good (gamma, beta) for each graph via multi-start COBYLA."""
    rng = np.random.default_rng(seed)
    targets = []
    for gi, g in enumerate(graphs):
        qaoa = QAOAMaxCut(g, p=p)
        best_x, best_cut = None, -np.inf
        for _ in range(multistart):
            x0 = rng.uniform(0, np.pi, 2 * p)
            res = minimize(lambda x: qaoa.cost(x), x0, method="COBYLA",
                           options={"maxiter": maxiter})
            cut = qaoa.expected_cut(res.x)
            if cut > best_cut:
                best_cut, best_x = cut, np.asarray(res.x, dtype=float)
        opt = brute_force_maxcut(g) if g.number_of_nodes() <= 20 else None
        targets.append({"params": best_x, "cut": best_cut,
                        "approx": (best_cut / opt) if opt else None})
        if (gi + 1) % 10 == 0:
            print(f"  targets: {gi+1}/{len(graphs)} graphs done")
    return targets


def _finetune_on_cut(model, graphs, targets, p, epochs, lr=3e-4):
    """Fine-tune the GNN to maximize the expected cut of its predicted angles.

    Uses PennyLane's torch interface so gradients flow from the QAOA expectation
    back into the GNN. This directly optimizes what we care about (cut), fixing
    the well-known weakness of pure angle-MSE (good angles != unique angles).
    """
    import pennylane as qml
    from grace_qaoa.gnn.warm_start import graph_to_data

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Build a torch-interfaced QAOA expectation per graph (cached).
    qnodes = []
    for g in graphs:
        qaoa = QAOAMaxCut(g, p=p)
        dev = qml.device("default.qubit", wires=qaoa.n_qubits)

        def circuit(params, _qaoa=qaoa):
            gammas, betas = params[0], params[1]
            for w in range(_qaoa.n_qubits):
                qml.Hadamard(wires=w)
            for layer in range(_qaoa.p):
                for i, j, weight in _qaoa.edges:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * gammas[layer] * weight, wires=j)
                    qml.CNOT(wires=[i, j])
                for w in range(_qaoa.n_qubits):
                    qml.RX(2.0 * betas[layer], wires=w)
            return qml.expval(_qaoa._cost_h)

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
        qnodes.append((qnode, qaoa._max_cut_offset))

    model.train()
    for epoch in range(epochs):
        total_cut = 0.0
        for g, (qnode, offset) in zip(graphs, qnodes):
            opt.zero_grad()
            data = graph_to_data(g)
            pred = model(data).view(2, p)        # torch tensor, grad-enabled
            energy = qnode(pred)
            cut = energy + offset
            loss = -cut                          # maximize cut
            loss.backward()
            opt.step()
            total_cut += float(cut.detach())
        if (epoch + 1) % 5 == 0:
            print(f"  [cut] epoch {epoch+1}/{epochs}  "
                  f"mean_cut={total_cut/len(graphs):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--conv", choices=["gcn", "gat", "sage"], default="gcn")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--finetune-epochs", type=int, default=30,
                    help="Cut-based fine-tuning epochs after MSE phase (0=skip).")
    ap.add_argument("--pool-size", type=int, default=80)
    ap.add_argument("--out", default="models")
    ap.add_argument("--cache", default=None,
                    help="Path to cache targets; reused if present (crash-safe).")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    p = cfg["qaoa_p"]
    weighted = cfg.get("weighted", False)

    # --- build training pool ---
    graphs = []
    per_fam = max(1, args.pool_size // len(cfg["families"]))
    for fam in cfg["families"]:
        graphs += make_dataset(fam, n_graphs=per_fam, n_nodes=cfg["n_nodes"],
                               weighted=weighted, base_seed=cfg.get("seed", 0),
                               split="train")
    print(f"Training pool: {len(graphs)} graphs (p={p}, conv={args.conv})")

    # --- targets (cached) ---
    cache = Path(args.cache) if args.cache else \
        Path(args.out) / f"targets_p{p}_{args.conv}.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print(f"Loading cached targets from {cache}")
        targets = pickle.load(open(cache, "rb"))
    else:
        print("Computing optimal-angle targets offline (multi-start COBYLA)...")
        targets = compute_targets(graphs, p, seed=cfg.get("seed", 0))
        pickle.dump(targets, open(cache, "wb"))
        print(f"Cached targets -> {cache}")
    approxes = [t["approx"] for t in targets if t["approx"] is not None]
    if approxes:
        print(f"Target quality: mean approx ratio = {np.mean(approxes):.4f}")

    # --- build dataset ---
    data_list = []
    for g, t in zip(graphs, targets):
        d = graph_to_data(g)
        d.y = torch.tensor(t["params"], dtype=torch.float).view(1, -1)
        data_list.append(d)
    loader = DataLoader(data_list, batch_size=16, shuffle=True)

    # --- train (phase 1: MSE regression to optimal angles) ---
    model = GNNWarmStart(p=p, conv_type=args.conv)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)                 # (batch, 2p), already in [0,pi]
            loss = loss_fn(pred, batch.y)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
        if (epoch + 1) % 50 == 0:
            print(f"  [mse] epoch {epoch+1}/{args.epochs}  "
                  f"MSE={total/len(data_list):.4f}")

    # --- evaluate warm-start quality before fine-tuning ---
    def warmstart_quality():
        model.eval()
        ars = []
        for g, t in zip(graphs, targets):
            if t["approx"] is None:
                continue
            qaoa = QAOAMaxCut(g, p=p)
            params = model.predict_params(g).reshape(-1)
            opt_cut = t["cut"] / t["approx"]
            ars.append(qaoa.expected_cut(params) / opt_cut)
        model.train()
        return float(np.mean(ars)) if ars else None

    q = warmstart_quality()
    if q is not None:
        print(f"Warm-start quality after MSE phase: mean approx = {q:.4f}")

    # --- train (phase 2: cut-based fine-tuning) ---
    # MSE to optimal angles is a proxy; what we actually want is high expected
    # cut from the predicted angles. We fine-tune directly on a differentiable
    # surrogate: maximize predicted-angle cut via a PennyLane torch interface.
    if args.finetune_epochs > 0:
        print(f"Fine-tuning {args.finetune_epochs} epochs on cut objective...")
        _finetune_on_cut(model, graphs, targets, p, args.finetune_epochs)
        q2 = warmstart_quality()
        if q2 is not None:
            print(f"Warm-start quality after fine-tune: mean approx = {q2:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"gnn_warmstart_p{p}_{args.conv}.pt"
    torch.save({"state_dict": model.state_dict(), "p": p,
                "conv_type": args.conv}, model_path)
    print(f"Saved GNN warm-start -> {model_path}")


if __name__ == "__main__":
    main()
