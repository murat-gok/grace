"""Graph instance generators for QAOA MaxCut experiments.

The key contribution of GRACE-QAOA is robustness on *weighted* and
out-of-distribution (OOD) graphs, so we deliberately provide several families.
Prior work mostly stalls on unweighted regular graphs; we go beyond that.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def _maybe_weight(g: nx.Graph, weighted: bool, rng: np.random.Generator) -> nx.Graph:
    for u, v in g.edges():
        g[u][v]["weight"] = float(rng.uniform(0.1, 1.0)) if weighted else 1.0
    return g


def regular_graph(n: int, d: int = 3, weighted: bool = False, seed: int = 0):
    """d-regular graph. The 'easy', in-distribution family."""
    rng = np.random.default_rng(seed)
    # n*d must be even for a d-regular graph to exist; nudge d if needed.
    if (n * d) % 2 != 0:
        d += 1
    d = min(d, n - 1)
    g = nx.random_regular_graph(d, n, seed=seed)
    return _maybe_weight(g, weighted, rng)


def erdos_renyi_graph(n: int, prob: float = 0.5, weighted: bool = False, seed: int = 0):
    """Erdos-Renyi random graph."""
    rng = np.random.default_rng(seed)
    g = nx.gnp_random_graph(n, prob, seed=seed)
    # Ensure connectivity-ish: add a spanning path if disconnected.
    if not nx.is_connected(g):
        nodes = list(g.nodes())
        for a, b in zip(nodes[:-1], nodes[1:]):
            if not g.has_edge(a, b):
                g.add_edge(a, b)
    return _maybe_weight(g, weighted, rng)


def watts_strogatz_graph(n: int, k: int = 4, prob: float = 0.3,
                         weighted: bool = False, seed: int = 0):
    """Small-world graph. A structurally distinct (OOD) family."""
    rng = np.random.default_rng(seed)
    g = nx.connected_watts_strogatz_graph(n, k, prob, seed=seed)
    return _maybe_weight(g, weighted, rng)


def split_summary() -> str:
    """Human-readable description of the disjoint-split design (for logs/papers)."""
    return ("Graphs use disjoint seed ranges per split: "
            f"test={SPLIT_OFFSET['test']}, train={SPLIT_OFFSET['train']}, "
            f"val={SPLIT_OFFSET['val']}. Train and test graphs never coincide, "
            "preventing the GNN-warm-start data leakage that would otherwise "
            "inflate results. For OOD evaluation, train and test on different "
            "graph families.")


# Disjoint seed offsets per split so train and test graphs NEVER overlap.
# This prevents the data leakage that would otherwise inflate GNN warm-start.
SPLIT_OFFSET = {"test": 0, "train": 1_000_000, "val": 2_000_000}


def make_dataset(family: str, n_graphs: int, n_nodes: int,
                 weighted: bool = False, base_seed: int = 0, split: str = "test",
                 **kwargs):
    """Produce a list of graphs from a named family.

    family : 'regular' | 'erdos_renyi' | 'watts_strogatz'
    split  : 'train' | 'val' | 'test' -- selects a DISJOINT seed range so that
             graphs generated for training can never coincide with test graphs.
    """
    # Allow callers to pass `seed=` as an alias for `base_seed`.
    if "seed" in kwargs:
        base_seed = kwargs.pop("seed")
    if split not in SPLIT_OFFSET:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(SPLIT_OFFSET)}.")
    offset = SPLIT_OFFSET[split]
    generators = {
        "regular": regular_graph,
        "erdos_renyi": erdos_renyi_graph,
        "watts_strogatz": watts_strogatz_graph,
    }
    if family not in generators:
        raise ValueError(f"Unknown family '{family}'. Choose from {list(generators)}.")
    gen = generators[family]
    graphs = []
    for i in range(n_graphs):
        graphs.append(gen(n=n_nodes, weighted=weighted,
                          seed=offset + base_seed + i, **kwargs))
    return graphs
