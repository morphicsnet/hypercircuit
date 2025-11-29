from __future__ import annotations

from hypercircuit.utils.config import Config, load_config


def test_config_validation() -> None:
    cfg = load_config(["configs/base.yaml", "configs/logging.yaml", "configs/discovery.yaml"])
    assert isinstance(cfg, Config)
    # legacy run_dir present for back-compat
    assert cfg.run.run_dir
    # new run metadata
    assert isinstance(cfg.run.seeds, list) and len(cfg.run.seeds) >= 1
    # logging baselines
    assert getattr(cfg.logging, "token_window", 0) == 160
    assert hasattr(cfg.logging, "instrumented_layers")
    assert len(cfg.logging.instrumented_layers) == 12
    assert hasattr(cfg.logging, "thresholds")
    # dataset and discovery
    assert cfg.dataset.n_samples > 0
    assert cfg.discovery.min_support >= 0
