from __future__ import annotations

from pathlib import Path

import numpy as np

from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.utils.io import load_jsonl, read_json
from hypercircuit.utils.registry import finalize_run, log_artifact, start_run
from hypercircuit.utils.seed import seed_context


def test_registry_and_seeds(tmp_path: Path) -> None:
    # Arrange minimal config for registry
    out_base = tmp_path / "runs"
    cfg_dict = {
        "run": {
            "output_dir": str(out_base),
            "run_id": "test-run",
            "seeds": [0, 1, 2, 3, 4],
            "stage": None,
            "model_tag": "debug",
            "split": "dev",
        },
        "dataset": {"name": "synthetic_parity"},
    }

    # Start run and verify manifest
    run_id, run_dir = start_run(cfg_dict, stage_name="logging", config_paths=["configs/base.yaml"])
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists(), "Manifest file should be created"
    manifest = read_json(manifest_path)
    assert manifest.get("run_id") == run_id
    assert manifest.get("stage") == "logging"
    assert manifest.get("seeds") == [0, 1, 2, 3, 4]

    # Log a dummy artifact and verify artifacts log
    dummy_file = run_dir / "dummy.txt"
    dummy_file.write_text("ok")
    log_artifact(dummy_file, kind="text", metadata={"note": "dummy"})
    artifacts_path = run_dir / "artifacts.jsonl"
    assert artifacts_path.exists(), "artifacts.jsonl should be created"
    artifacts = load_jsonl(artifacts_path)
    assert len(artifacts) >= 1, "Should have at least one artifact record"

    # Produce a tiny mock logging run and ensure the output is recorded
    logs_path = run_dir / "logs.jsonl"
    logger = ActivationLogger(
        tokens_per_sample=8,
        threshold=0.5,
        hysteresis=0.05,
        sparsity=0.1,
        n_features=8,
        seed=0,
    )
    logger.run(out_path=logs_path, n_samples=10)
    assert logs_path.exists() and logs_path.stat().st_size > 0
    log_artifact(logs_path, kind="events", metadata={"n_samples": 10})

    # Finalize run
    finalize_run(status="success", metrics_dict={"ok": 1})

    # Seed context determinism check: same seed produces identical draws across contexts
    with seed_context(123):
        a1 = np.random.rand(5, 3)
    with seed_context(123):
        a2 = np.random.rand(5, 3)
    assert np.allclose(a1, a2), "seed_context should yield deterministic arrays across separate uses"