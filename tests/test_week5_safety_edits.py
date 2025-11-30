from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import json

from hypercircuit.utils.io import load_jsonl, write_json
from hypercircuit.cli.run_week5_safety import main as week5_main


def _write_surrogate_params(run_dir: Path, families: List[str]) -> Path:
    """
    Create a tiny deterministic surrogates_params.jsonl to drive sensitivity-based ranking.
    """
    path = run_dir / "surrogates_params.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    # Provide distinct weight magnitudes per family to induce a stable ranking
    base_weights = {
        "sycophancy": [0.9, 0.8, 0.1],
        "jailbreak": [0.6, 0.4, 0.1],
        "deceptive_reasoning": [0.5, 0.3, 0.2],
    }
    for fam in families:
        w = base_weights.get(fam, [0.2, 0.2, 0.2])
        rows.append(
            {
                "ensemble_id": f"{fam}-e0",
                "family": fam,
                "cv_score": 0.5,
                "calibration_error": 0.1,
                "model_state": {"weights": w, "intercept": 0.0},
                "hyperparams": {"alpha": 0.0},
            }
        )
    # Minimal JSONL writer
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _artifact_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "plans": run_dir / "safety_edit_plans.jsonl",
        "results": run_dir / "safety_eval_results.jsonl",
        "gate3": run_dir / "gate3_report.json",
    }


def test_week5_cli_artifacts_and_determinism(tmp_path: Path) -> None:
    # Arrange: choose a stable run_dir and pre-create surrogate params
    run_dir = tmp_path / "runs" / "week5_mock"
    run_dir.mkdir(parents=True, exist_ok=True)
    families = ["sycophancy", "jailbreak", "deceptive_reasoning"]
    _ = _write_surrogate_params(run_dir, families)

    # Act: run the Week 5 CLI in mock mode with legacy run_dir override
    args = [
        "--config",
        "configs/base.yaml",
        "configs/editing.yaml",
        "configs/causal.yaml",
        "configs/dictionary.yaml",
        "-o",
        f"run.run_dir={run_dir}",
    ]
    week5_main(args)

    paths = _artifact_paths(run_dir)
    assert paths["plans"].exists() and paths["plans"].stat().st_size > 0, "safety_edit_plans.jsonl should exist"
    assert paths["results"].exists() and paths["results"].stat().st_size > 0, "safety_eval_results.jsonl should exist"
    assert paths["gate3"].exists() and paths["gate3"].stat().st_size > 0, "gate3_report.json should exist"

    # Validate plan schema contents
    plans = load_jsonl(paths["plans"])
    assert len(plans) >= 1, "At least one plan row expected"
    plan0 = plans[0]
    for k in ["families", "proposed_scales", "applied_scales_per_step", "ranking", "seed"]:
        assert k in plan0, f"Missing plan key: {k}"
    assert set(plan0["families"]) == set(families), "Families in plan must match configured risky families"

    # Validate eval results schema fields present
    rows = load_jsonl(paths["results"])
    assert len(rows) > 0, "At least one evaluation row expected"
    r0 = rows[0]
    for k in ["family", "step_index", "step_scale", "harmful_rate_reduction", "benign_deg_pct", "specificity_ratio"]:
        assert k in r0, f"Missing eval results key: {k}"

    # Validate Gate 3 report structure
    with paths["gate3"].open("r") as f:
        rep = json.load(f)
    assert "families" in rep and isinstance(rep["families"], dict), "Gate 3 families block missing"
    assert "accept_all" in rep and isinstance(rep["accept_all"], bool), "Gate 3 accept_all flag missing"
    fam_block = next(iter(rep["families"].values()))
    for k in ["harmful_rate_reduction", "benign_deg_pct", "specificity_ratio", "accept"]:
        assert k in fam_block, f"Per-family Gate 3 field missing: {k}"

    # Determinism: re-run with the same config and seed; compare plan selections and summary metrics
    # Clean artifacts to avoid accumulation
    for p in paths.values():
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    week5_main(args)

    plans2 = load_jsonl(paths["plans"])
    plan1 = plans2[0]
    assert plan0["ranking"] == plan1["ranking"], "Ranking must be identical under same seed/config"
    assert plan0["proposed_scales"] == plan1["proposed_scales"], "Proposed scales must be deterministic"

    # Compare Gate 3 family-level summary within tight tolerance
    with paths["gate3"].open("r") as f:
        rep2 = json.load(f)
    fams0 = rep["families"]
    fams1 = rep2["families"]
    for fam in fams0.keys():
        a = fams0[fam]
        b = fams1[fam]
        for key in ["harmful_rate_reduction", "benign_deg_pct", "specificity_ratio"]:
            assert abs(float(a[key]) - float(b[key])) <= 1e-9, f"Metric {key} must be deterministic for {fam}"