"""Tests for GNN warm-start model: shape, save/load roundtrip."""
import numpy as np
import torch

from grace_qaoa.quantum.graphs import make_dataset
from grace_qaoa.gnn.warm_start import GNNWarmStart, graph_to_data
from grace_qaoa.gnn.loader import load_gnn_warmstart


def test_predict_shape():
    g = make_dataset("regular", n_graphs=1, n_nodes=8, seed=1)[0]
    for conv in ("gcn", "gat", "sage"):
        model = GNNWarmStart(p=3, conv_type=conv)
        params = model.predict_params(g)
        assert params.shape == (2, 3)
        assert np.all(params >= 0) and np.all(params <= np.pi + 1e-6)


def test_save_load_roundtrip(tmp_path):
    g = make_dataset("regular", n_graphs=1, n_nodes=8, seed=2)[0]
    model = GNNWarmStart(p=2, conv_type="gcn")
    before = model.predict_params(g)
    path = tmp_path / "gnn.pt"
    torch.save({"state_dict": model.state_dict(), "p": 2, "conv_type": "gcn"}, path)
    loaded = load_gnn_warmstart(path)
    after = loaded.predict_params(g)
    assert np.allclose(before, after, atol=1e-5)


def test_graph_to_data_weighted():
    g = make_dataset("erdos_renyi", n_graphs=1, n_nodes=7, weighted=True, seed=3)[0]
    d = graph_to_data(g)
    assert d.x.shape[0] == 7
    assert d.x.shape[1] == 2
