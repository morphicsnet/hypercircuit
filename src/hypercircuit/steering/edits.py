from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_edit_map(weights: np.ndarray, scale: float) -> np.ndarray:
    """Compute per-feature scale factors based on surrogate weights."""
    w = np.clip(weights, 0.0, None)
    w_norm = w / (w.max() + 1e-8)
    return 1.0 - scale * w_norm


def apply_edits(X: np.ndarray, edit_map: np.ndarray, indices: List[int] | None = None) -> np.ndarray:
    """Apply per-feature scaling edits to activations."""
    out = X.copy()
    idx = indices if indices is not None else list(range(X.shape[1]))
    out[:, idx] = out[:, idx] * edit_map[idx]
    return out
