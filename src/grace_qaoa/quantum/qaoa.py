"""QAOA quantum subroutine for MaxCut, built on PennyLane lightning.qubit (CPU).

The whole project bottleneck is here: each (gamma, beta) evaluation runs a
state-vector simulation. On the office Xeon we use 'lightning.qubit', the C++
CPU backend. Keep n_qubits (= graph nodes) <= 24 for comfortable single-run
speed; <= 20 when running many parallel runs.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pennylane as qml


class QAOAMaxCut:
    """Builds and evaluates a depth-p QAOA circuit for the MaxCut of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Problem graph. Edge weights (attribute 'weight', default 1.0) are used,
        so weighted / OOD graphs are supported out of the box.
    p : int
        QAOA depth (number of cost+mixer layers). Paper sweeps p = 1..5.
    shots : int | None
        None -> exact expectation (fast, deterministic). An int -> shot-based
        sampling (use this for the noisy-evaluation phase to keep memory low).
    """

    def __init__(self, graph: nx.Graph, p: int = 1, shots: int | None = None):
        self.graph = graph
        self.n_qubits = graph.number_of_nodes()
        self.p = p
        self.shots = shots
        # Map arbitrary node labels to wire indices 0..n-1.
        self._node_to_wire = {node: i for i, node in enumerate(graph.nodes())}
        self.edges = [
            (self._node_to_wire[u], self._node_to_wire[v],
             float(d.get("weight", 1.0)))
            for u, v, d in graph.edges(data=True)
        ]
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits, shots=shots)
        self._cost_h, self._max_cut_offset = self._build_cost_hamiltonian()
        self._qnode = qml.QNode(self._circuit, self.dev)

    def _build_cost_hamiltonian(self):
        """MaxCut cost: sum_{(i,j)} w_ij * 0.5 * (1 - Z_i Z_j).

        We return the operator form 0.5*w*(I - Z_i Z_j) summed, and the constant
        offset so that <C> directly equals expected cut weight.
        """
        coeffs, ops = [], []
        offset = 0.0
        for i, j, w in self.edges:
            coeffs.append(-0.5 * w)
            ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
            offset += 0.5 * w
        if not ops:  # graph with no edges
            return qml.Hamiltonian([0.0], [qml.Identity(0)]), 0.0
        return qml.Hamiltonian(coeffs, ops), offset

    def _circuit(self, params):
        """Standard QAOA ansatz. params shape = (2, p): [gammas; betas]."""
        gammas, betas = params[0], params[1]
        for w in range(self.n_qubits):
            qml.Hadamard(wires=w)
        for layer in range(self.p):
            # Cost unitary
            for i, j, weight in self.edges:
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * gammas[layer] * weight, wires=j)
                qml.CNOT(wires=[i, j])
            # Mixer unitary
            for w in range(self.n_qubits):
                qml.RX(2.0 * betas[layer], wires=w)
        return qml.expval(self._cost_h)

    def expected_cut(self, params: np.ndarray) -> float:
        """Return expected cut weight for given params. Higher is better."""
        params = np.asarray(params, dtype=float).reshape(2, self.p)
        energy = float(self._qnode(params))
        return self._max_cut_offset + energy

    def cost(self, params: np.ndarray) -> float:
        """Minimization objective (negative expected cut) for optimizers."""
        return -self.expected_cut(params)

    def approximation_ratio(self, params: np.ndarray,
                            optimal_cut: float | None = None) -> float:
        """Cut achieved / optimal cut. Needs the true optimum (small graphs)."""
        if optimal_cut is None:
            optimal_cut = brute_force_maxcut(self.graph)
        if optimal_cut == 0:
            return 1.0
        return self.expected_cut(params) / optimal_cut


def brute_force_maxcut(graph: nx.Graph) -> float:
    """Exact MaxCut by enumeration. Only for small graphs (<= ~22 nodes)."""
    nodes = list(graph.nodes())
    n = len(nodes)
    idx = {node: k for k, node in enumerate(nodes)}
    best = 0.0
    for assignment in range(1 << n):
        cut = 0.0
        for u, v, d in graph.edges(data=True):
            if ((assignment >> idx[u]) & 1) != ((assignment >> idx[v]) & 1):
                cut += float(d.get("weight", 1.0))
        best = max(best, cut)
    return best
