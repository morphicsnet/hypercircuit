from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.discovery.aggregate import run_week2_screening
from hypercircuit.utils.config import load_config
from hypercircuit.utils.io import read_json
from hypercircuit.utils.registry import start_run, finalize_run


def _make_logs_for_family(base_runs: Path, overlay_path: str, seed: int = 1234) -> Path:
    """
    Create a deterministic mock logging run for a given dataset overlay and
    return the path to logs.jsonl. The manifest will include task_family/split.
    """
    cfg = load_config(["configs/base.yaml", "configs/logging.yaml", overlay_path])
    ds = cfg.dataset
    fam_name = (ds.task_family or ds.name or "unknown").strip()
    split = ds.split

    # Prepare minimal run dict for registry with per-family metadata
    cfg_dict = {
        "run": {
            "output_dir": str(base_runs),
            "run_id": f"{fam_name}-logs",
            "seeds": [seed],
            "stage": None,
            "model_tag": "week2-test",
            "task_family": fam_name,
            "split": split,
        },
        "dataset": {
            "name": ds.name,
            "n_samples": ds.n_samples,
            "n_features": ds.n_features,
            "variant": ds.variant,
            "task_family": ds.task_family,
            "split": ds.split,
        },
        "logging": cfg.logging.model_dump() if hasattr(cfg.logging, "model_dump") else {},
    }

    run_id, run_dir = start_run(cfg_dict, stage_name="logging", config_paths=["configs/base.yaml", overlay_path])
    logs_path = run_dir / "logs.jsonl"

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
        node_types=cfg.logging.node_types.model_dump(),
    )
    logger.run(out_path=logs_path, n_samples=cfg.dataset.n_samples)
    assert logs_path.exists() and logs_path.stat().st_size > 0, f"logs missing for {fam_name}"
    finalize_run(status="success", metrics_dict={"n_samples": cfg.dataset.n_samples})
    return logs_path


def test_week2_aggregator_multi_replicate_determinism(tmp_path: Path) -> None:
    # Arrange: generate deterministic logs for four families
    base_runs = tmp_path / "runs"
    overlays = [
        "configs/datasets/sycophancy.yaml",
        "configs/datasets/jailbreak.yaml",
        "configs/datasets/deceptive_reasoning.yaml",
        "configs/datasets/truthfulness_qa.yaml",
    ]
    for ov in overlays:
        _make_logs_for_family(base_runs, ov, seed=77)

    # Act: run aggregator with replicates_k=3 (multi-replicate stability)
    overrides = [
        f"run.output_dir={base_runs}",
        "discovery.week2_screening.replicates_k=3",
        "discovery.week2_screening.paraphrase_replicates=2",
    ]
    report_path = run_week2_screening(
        config_paths=["configs/base.yaml", "configs/discovery.yaml"],
        overlay_paths=overlays,
        overrides=overrides,
    )
    assert report_path.exists(), "week2_synergy_report.json should be created"
    rep = read_json(report_path)

    families = rep.get("families", {})
    # Presence for all families
    expected = ["sycophancy", "jailbreak", "deceptive_reasoning", "truthfulness_qa"]
    for fam in expected:
        assert fam in families, f"Missing family block: {fam}"
        block = families[fam]
        assert "counts" in block and "top_exemplars" in block, f"Missing counts/exemplars for {fam}"
        assert "stability" in block and "stability_multi" in block["stability"], f"Missing stability_multi for {fam}"
        assert "top10_signature_md5" in block, f"Missing checksum for {fam}"

    # Acceptance flags present for configured top families
    acceptance = rep.get("acceptance", {})
    for fam in ["sycophancy", "jailbreak"]:
        assert fam in acceptance, f"Acceptance flag missing for top family {fam}"

    # Determinism: same seed/config yields identical top-10 signature checksums
    report_path2 = run_week2_screening(
        config_paths=["configs/base.yaml", "configs/discovery.yaml"],
        overlay_paths=overlays,
        overrides=overrides,
    )
    rep2 = read_json(report_path2)
    fam2 = rep2.get("families", {})

    for fam in expected:
        c1 = families[fam]["top10_signature_md5"]
        c2 = fam2[fam]["top10_signature_md5"]
        assert c1 == c2, f"Checksum mismatch for {fam} under same seed/config"