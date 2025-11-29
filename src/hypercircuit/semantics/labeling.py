from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel


class LabelSchema(BaseModel):
    """Mapping of feature id to human-readable label."""
    labels: Dict[int, str] = {}


def select_exemplars(activations: np.ndarray, k: int = 3) -> Dict[int, List[int]]:
    """Pick top-k samples per feature based on activation magnitude."""
    n_samples, n_features = activations.shape
    out: Dict[int, List[int]] = {}
    for j in range(n_features):
        topk = np.argsort(-activations[:, j])[:k].tolist()
        out[j] = topk
    return out


def agreement_jaccard(labels_a: Dict[int, str], labels_b: Dict[int, str]) -> float:
    """Simple label agreement using Jaccard over label sets."""
    sa, sb = set(labels_a.values()), set(labels_b.values())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union
