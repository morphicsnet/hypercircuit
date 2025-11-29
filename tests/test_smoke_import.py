from __future__ import annotations

from pathlib import Path

from hypercircuit import __version__
from hypercircuit.cli.run_log import main as log_main
from hypercircuit.cli.run_discovery import main as disc_main
from hypercircuit.cli.run_surrogate import main as surv_main
from hypercircuit.cli.run_causal_eval import main as causal_main
from hypercircuit.cli.run_edit_eval import main as edit_main


def test_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Use base configs; each script appends its own stage config
    log_main(["--config", "configs/base.yaml", "configs/logging.yaml", "--run-dir", str(run_dir)])
    assert (run_dir / "logs.jsonl").exists(), "logs.jsonl not created"

    disc_main(["--config", "configs/base.yaml", "configs/discovery.yaml", "--run-dir", str(run_dir)])
    assert (run_dir / "candidates.jsonl").exists(), "candidates.jsonl not created"

    surv_main(["--config", "configs/base.yaml", "configs/surrogate.yaml", "--run-dir", str(run_dir)])
    assert (run_dir / "surrogate.json").exists(), "surrogate.json not created"

    causal_main(["--config", "configs/base.yaml", "configs/causal.yaml", "--run-dir", str(run_dir)])
    assert (run_dir / "causal.json").exists(), "causal.json not created"

    edit_main(["--config", "configs/base.yaml", "configs/editing.yaml", "--run-dir", str(run_dir)])
    assert (run_dir / "edits.json").exists(), "edits.json not created"
