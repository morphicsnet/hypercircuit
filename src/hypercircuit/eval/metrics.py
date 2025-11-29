from __future__ import annotations

from typing import Dict

import numpy as np


def causal_impact(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(1.0 / (1.0 + np.mean((y_true - y_pred) ** 2)))


def minimality(weights: np.ndarray) -> float:
    nz = np.count_nonzero(weights)
    return float(1.0 / (1 + nz))


def sufficiency(X: np.ndarray, y: np.ndarray, feats: list[int]) -> float:
    s = X[:, feats].sum(axis=1)
    return float(np.corrcoef(s, y)[0, 1])


def stability(scores: np.ndarray) -> float:
    return float(1.0 - np.std(scores))


def specificity(X: np.ndarray, feats: list[int]) -> float:
    others = np.setdiff1d(np.arange(X.shape[1]), feats)
    return float(np.mean(X[:, feats]) - np.mean(X[:, others]))


def coherence(labels: dict[int, str]) -> float:
    return float(len(set(labels.values())) / (len(labels) or 1))
