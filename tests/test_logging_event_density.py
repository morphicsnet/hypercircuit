from __future__ import annotations

import pytest

from hypercircuit.utils.config import Config, load_config
from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.utils.seed import seed_context


DATASETS = [
    "configs/datasets/synthetic_parity.yaml",
    "configs/datasets/roman_numeral.yaml",
]


@pytest.mark.parametrize("overlay", DATASETS)
def test_event_density_and_determinism(overlay: str) -> None:
    cfg: Config = load_config(["configs/base.yaml", "configs/logging.yaml", overlay])
    low, high = cfg.logging.expected_event_density_range
    enabled = cfg.logging.node_types.model_dump()
    seed = 1234

    # Build logger under fixed seed context (deterministic)
    with seed_context(seed):
        logger = ActivationLogger(
            tokens_per_sample=cfg.logging.tokens_per_sample,
            threshold=cfg.logging.threshold,
            hysteresis=cfg.logging.hysteresis,
            sparsity=cfg.logging.sparsity,
            n_features=cfg.dataset.n_features,
            seed=seed,
            instrumented_layers=cfg.logging.instrumented_layers,
            token_window=cfg.logging.token_window,
            thresholds=cfg.logging.thresholds,
            node_types=enabled,
        )
        metrics = logger.compute_metrics(cfg.dataset.n_samples)

    # Density within configured band
    ept = float(metrics["events_per_token"])
    assert low <= ept <= high, f"events_per_token {ept} not in [{low},{high}]"

    # Per-node-type counts are non-zero for all enabled types
    counts = metrics.get("events_by_node_type")
    assert isinstance(counts, dict)
    for nt, on in enabled.items():
        if on:
            assert counts.get(nt, 0) > 0, f"no events for enabled node type {nt}"

    # Layer coverage fraction and layer count
    assert metrics.get("n_layers") == len(cfg.logging.instrumented_layers) == 12
    cov = float(metrics["layer_coverage_fraction"])
    assert cov > 0.8, f"layer_coverage_fraction too low: {cov}"

    # Determinism: same seed yields identical metrics
    with seed_context(seed):
        logger2 = ActivationLogger(
            tokens_per_sample=cfg.logging.tokens_per_sample,
            threshold=cfg.logging.threshold,
            hysteresis=cfg.logging.hysteresis,
            sparsity=cfg.logging.sparsity,
            n_features=cfg.dataset.n_features,
            seed=seed,
            instrumented_layers=cfg.logging.instrumented_layers,
            token_window=cfg.logging.token_window,
            thresholds=cfg.logging.thresholds,
            node_types=enabled,
        )
        metrics2 = logger2.compute_metrics(cfg.dataset.n_samples)

    assert metrics == metrics2, "Metrics must be deterministic given fixed seed"