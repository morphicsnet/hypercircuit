from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hypercircuit.sae_io.loaders import PretrainedSAEDictionary


def test_sae_loader_npz(tmp_path: Path) -> None:
    W = np.random.randn(4, 3).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    names = np.array(["f0", "f1", "f2", "f3"])
    path = tmp_path / "sae.npz"
    np.savez(path, W_enc=W, b_enc=b, feature_names=names)

    sae = PretrainedSAEDictionary.from_path(str(path), fmt="npz")
    assert sae.n_features == 4
    assert sae.input_dim == 3
    assert sae.feature_names()[:2] == ["f0", "f1"]

    acts = torch.randn(2, 5, 3)
    feats = sae.encode_activations(acts)
    assert feats.shape == (2, 5, 4)
