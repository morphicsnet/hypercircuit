from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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


def simulate_impact(
    weights: np.ndarray,
    step_schedule: List[float],
    *,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """
    Surrogate-driven dry-run of expected harmful reduction and benign degradation per step.
    Deterministic given weights, schedule, and seed.
    """
    rng = np.random.default_rng(seed if seed is not None else 0)
    w = np.clip(weights, 0.0, None)
    norm = float(w.max() + 1e-8)
    sens = float(w.sum() / (norm * max(1, w.size)))  # normalized sensitivity proxy in [0, ~1]
    rows: List[Dict[str, float]] = []
    for s in step_schedule:
        harmful = max(0.0, min(1.0, sens * 0.8 * s))
        benign = max(0.0, min(1.0, 0.10 * s))
        harmful += float(rng.uniform(-1e-9, 1e-9))
        benign += float(rng.uniform(-1e-9, 1e-9))
        rows.append(
            {
                "step_scale": float(s),
                "pred_harmful_rate_reduction": float(harmful),
                "pred_benign_deg_pct": float(benign),
            }
        )
    return rows


def apply_edit_plan(
    weights: np.ndarray,
    *,
    scale: float,
    max_edit_scale: float,
    sparsity: float = 0.5,
) -> Dict[str, Any]:
    """
    Select top-|indices| by weight and compute edit map at bounded scale.
    Returns a record of applied edits for registry logging.
    """
    s = float(min(max(0.0, scale), max_edit_scale))
    w = np.clip(weights, 0.0, None)
    k = max(1, int(np.ceil(sparsity * len(w))))
    idx_sorted = list(np.argsort(-w))[:k]
    edit_map = compute_edit_map(weights, s)
    return {
        "selected_indices": [int(i) for i in idx_sorted],
        "scale": s,
        "edit_map_preview_mean": float(edit_map.mean()),
    }
