"""GNN warm-start: predict initial (gamma, beta) from the QUBO/MaxCut graph.

Ablation (d) GCN vs GAT vs GraphSAGE is wired via the `conv_type` argument.
Training target: the (gamma, beta) that maximize expected cut, obtained offline
by a thorough optimizer on a set of training graphs (see scripts/pretrain_gnn.py).

This module intentionally keeps the model tiny -- in T3 the graphs are small and
the GNN is NOT the compute bottleneck (the QAOA simulation is).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
from torch_geometric.utils import from_networkx

_CONV = {"gcn": GCNConv, "gat": GATConv, "sage": SAGEConv}


def graph_to_data(graph: nx.Graph):
    """Convert a networkx graph to a PyG Data object with simple node features."""
    g = graph.copy()
    deg = dict(g.degree(weight="weight"))
    max_deg = max(deg.values()) if deg else 1.0
    for node in g.nodes():
        # Node features: normalized degree + constant bias term.
        g.nodes[node]["x"] = [deg[node] / max_deg if max_deg else 0.0, 1.0]
    data = from_networkx(g, group_node_attrs=["x"])
    data.x = data.x.float()
    return data


class GNNWarmStart(nn.Module):
    """Predicts 2*p QAOA angles (gammas then betas) from a graph."""

    def __init__(self, p: int = 1, hidden: int = 64, conv_type: str = "gcn",
                 in_dim: int = 2):
        super().__init__()
        self.p = p
        conv = _CONV[conv_type]
        self.conv1 = conv(in_dim, hidden)
        self.conv2 = conv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * p),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        g = global_mean_pool(x, batch)
        raw = self.head(g)                       # (batch, 2p)
        # Squash into [0, pi] -- the natural QAOA parameter range.
        return torch.sigmoid(raw) * np.pi

    @torch.no_grad()
    def predict_params(self, graph: nx.Graph) -> np.ndarray:
        """Return a (2, p) numpy array of warm-start angles for one graph."""
        self.eval()
        data = graph_to_data(graph)
        out = self.forward(data).squeeze(0).cpu().numpy()
        return out.reshape(2, self.p)
