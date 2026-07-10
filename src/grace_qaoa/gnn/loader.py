"""Load a trained GNNWarmStart model from a checkpoint produced by pretrain_gnn.py."""
from __future__ import annotations

from pathlib import Path

import torch

from grace_qaoa.gnn.warm_start import GNNWarmStart


def load_gnn_warmstart(path: str | Path) -> GNNWarmStart:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GNNWarmStart(p=ckpt["p"], conv_type=ckpt["conv_type"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
