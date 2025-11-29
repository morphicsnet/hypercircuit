from __future__ import annotations

from typing import List, Tuple

import numpy as np


def align_token_window(seq_len: int, window: int) -> Tuple[int, int]:
    """Return a centered window [start, end) over a sequence length."""
    start = max(0, (seq_len - window) // 2)
    end = min(seq_len, start + window)
    return start, end


def apply_intervention(acts: np.ndarray, indices: List[int], scale: float) -> np.ndarray:
    """Scale selected feature activations by a factor."""
    out = acts.copy()
    out[:, indices] *= scale
    return out
