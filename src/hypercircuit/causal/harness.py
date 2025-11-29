from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class CausalHarness:
    """Mock harness for ensemble ablations, transfers, and patching."""
    ablation_strength: float = 1.0

    def ablate(self, X: np.ndarray, features: List[int]) -> np.ndarray:
        X2 = X.copy()
        X2[:, features] *= (1.0 - self.ablation_strength)
        return X2

    def transfer(self, X_src: np.ndarray, X_tgt: np.ndarray, features: List[int]) -> np.ndarray:
        X2 = X_tgt.copy()
        X2[:, features] = X_src[:, features]
        return X2

    def evaluate_delta(self, y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, float]:
        """Return simple mock causal deltas."""
        mse_a = float(np.mean((y_true - y_pred_a) ** 2))
        mse_b = float(np.mean((y_true - y_pred_b) ** 2))
        return {"mse_a": mse_a, "mse_b": mse_b, "delta": mse_b - mse_a}
