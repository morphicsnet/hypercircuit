from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


class SAEAdapter:
    """Abstract adapter for loading SAE dictionaries and activations."""

    def feature_names(self) -> List[str]:
        raise NotImplementedError

    def sample_activations(self, n_samples: int) -> np.ndarray:
        """Return mock activations [n_samples, n_features]."""
        raise NotImplementedError


@dataclass
class FakeSAEDictionary(SAEAdapter):
    """Simple fake SAE dictionary for smoke tests.

    Generates Bernoulli(approx sparsity) activations with small Gaussian noise.
    Deterministic given the numpy RNG state (controlled via utils.seed).
    """
    n_features: int
    sparsity: float = 0.1

    def feature_names(self) -> List[str]:
        return [f"f{i}" for i in range(self.n_features)]

    def sample_activations(self, n_samples: int) -> np.ndarray:
        base = (np.random.rand(n_samples, self.n_features) < self.sparsity).astype(float)
        noise = 0.05 * np.random.randn(n_samples, self.n_features)
        return np.clip(base + noise, 0.0, None)
