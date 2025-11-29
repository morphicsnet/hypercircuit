from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class MonotoneCombiner:
    """Interpretable combiner with non-negative weights and optional thresholds.

    This is a lightweight, numpy-based scaffold. Fitting uses a clipped least squares.
    """
    nonneg: bool = True
    thresholds: Optional[np.ndarray] = None
    weights_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MonotoneCombiner":
        X_fit = X.copy()
        if self.thresholds is not None:
            X_fit = np.clip(X_fit - self.thresholds, 0.0, None)
        # Add bias column for intercept
        A = np.c_[X_fit, np.ones((X_fit.shape[0], 1))]
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        w_main = w[:-1]
        b = w[-1]
        if self.nonneg:
            w_main = np.clip(w_main, 0.0, None)
            b = float(np.mean(y) - np.mean(X_fit, axis=0) @ w_main)
        self.weights_ = w_main
        self.intercept_ = float(b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model not fit.")
        Xp = X.copy()
        if self.thresholds is not None:
            Xp = np.clip(Xp - self.thresholds, 0.0, None)
        return Xp @ self.weights_ + self.intercept_

    def to_dict(self) -> Dict[str, object]:
        return {
            "nonneg": self.nonneg,
            "thresholds": self.thresholds.tolist() if self.thresholds is not None else None,
            "weights": self.weights_.tolist() if self.weights_ is not None else None,
            "intercept": self.intercept_,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "MonotoneCombiner":
        m = cls(nonneg=bool(d.get("nonneg", True)))
        th = d.get("thresholds")
        if th is not None:
            m.thresholds = np.array(th, dtype=float)
        w = d.get("weights")
        if w is not None:
            m.weights_ = np.array(w, dtype=float)
        m.intercept_ = float(d.get("intercept", 0.0))
        return m

    # Stable contract: state_dict/load_state_dict (back-compat with to_dict/from_dict)
    def state_dict(self) -> Dict[str, object]:
        """Return a serializable model state."""
        return self.to_dict()

    def load_state_dict(self, state: Dict[str, object]) -> "MonotoneCombiner":
        """Load model parameters from a state dict in-place and return self."""
        th = state.get("thresholds")
        self.thresholds = np.array(th, dtype=float) if th is not None else None
        w = state.get("weights")
        self.weights_ = np.array(w, dtype=float) if w is not None else None
        self.intercept_ = float(state.get("intercept", 0.0))
        self.nonneg = bool(state.get("nonneg", True))
        return self
