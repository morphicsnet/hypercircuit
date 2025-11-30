from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

from hypercircuit.cli.run_week6_gate4 import main as week6_main


def _write_mock_logs(run_dir: Path, n_samples: int = 48) -> Path:
    """
    Deterministic tiny mock of logs.jsonl with only the fields needed by matrix evaluator:
    - sample_id
    - node_type
    """
    node_types = ["sae_features", "attn_heads", "mlp_channels", "residual_dirs"]
    path = run_dir / "logs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n_samples):
            # Alternate patterns deterministically
            if i % 4 == 0:
                present = node_types[:2]
            elif i % 4 == 1:
                present = node_types[2:]
            elif i % 4 == 2:
                present = node_types[::2]
            else:
                present = node_types[1::2]
            for nt in present:
                f.write(json.dumps({"sample_id": i, "node_type": nt}) + "\n")
    return path


def _write_minimal_dictionary_and_surrogates(run_dir: Path, families: List[str], ensembles_per_family: int = 3) -> Dict[str, Path]:
    """
    Write minimal ensembles.jsonl and surrogates_params.jsonl sufficient for matrix and necessity stages.
    Ensures:
      - For each family, a few ensembles with 2 members chosen from node_types
      - Surrogate params include non-negative weights aligned to member count
    """
    node_types = ["sae_features", "attn_heads", "mlp_channels"]
    ensembles_path = run_dir / "ensembles.jsonl"
    sur_params_path = run_dir / "surrogates_params.jsonl"
    ensembles_path.parent.mkdir(parents=True, exist_ok=True)

    with ensembles_path.open("w") as fe, sur_params_path.open("w") as fp:
        for fam in families:
            for j in range(ensembles_per_family):
                members = [node_types[j % len(node_types)], node_types[(j + 1) % len(node_types)]]
                eid = f"{fam}-e{j}"
                ens_row = {
                    "id": eid,
                    "family": fam,
                    "members": members,
                    "size": len(members),
                    "synergy_score": 0.5 + 0.1 * j,
                    "stability_score": 0.6 + 0.05 * j,
                }
                fe.write(json.dumps(ens_row) + "\n")

                # Keep weights aligned with member count; positive for causal deltas
                w0 = float(0.4 + 0.1 * j)
                w1 = float(0.2)
                par_row = {
                    "ensemble_id": eid,
                    "family": fam,
                    "cv_score": 0.5 + 0.05 * j,
                    "calibration_error": 0.1,
                    "model_state": {"weights": [w0, w1], "intercept": 0.01},
                    "hyperparams": {"alpha": 0.0},
                }
                fp.write(json.dumps(par_row) + "\n")
    return {"ensembles": ensembles_path, "surrogates": sur_params_path}


def _artifact_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "remaining": run_dir / "remaining_matrix_results.jsonl",
        "matrix": run_dir / "matrix_results.jsonl",
        "rob_rows": run_dir / "matrix_robustness.jsonl",
        "rob_sum": run_dir / "robustness_summary.json",
        "nec": run_dir / "necessity_results.jsonl",
        "g4": run_dir / "gate4_report.json",
    }


def test_week6_gate4_end_to_end_robustness_and_necessity(tmp_path: Path) -> None:
    # Arrange
    run_dir = tmp_path / "runs" / "week6_mock"
    run_dir.mkdir(parents=True, exist_ok=True)
    families = ["sycophancy", "jailbreak", "truthfulness_qa"]

    _ = _write_mock_logs(run_dir, n_samples=48)
    _ = _write_minimal_dictionary_and_surrogates(run_dir, families, ensembles_per_family=3)

    # Act: run Week 6 CLI with deterministic, reduced counts
    args = [
        "--config",
        "configs/base.yaml",
        "configs/surrogate.yaml",
        "configs/causal.yaml",
        "configs/dictionary.yaml",
        "configs/matrix.yaml",
        "-o",
        f"run.run_dir={run_dir}",
        "-o",
        "matrix.per_cell.n_seeds=2",
        "-o",
        "matrix.per_cell.n_prompts=64",
        "-o",
        "causal.gate4.robustness.seeds=3",
        "-o",
        "causal.gate4.robustness.n_paraphrases=2",
        "-o",
        "causal.gate4.robustness.n_adversarial=2",
        "-o",
        "causal.n_ensembles_per_family=3",
    ]
    week6_main(args)

    paths = _artifact_paths(run_dir)
    # remaining or matrix results must exist (remaining preferred)
    assert (paths["remaining"].exists() or paths["matrix"].exists()), "Matrix results file missing"
    assert paths["rob_rows"].exists() and paths["rob_rows"].stat().st_size > 0, "matrix_robustness.jsonl should exist"
    assert paths["rob_sum"].exists() and paths["rob_sum"].stat().st_size > 0, "robustness_summary.json should exist"
    assert paths["nec"].exists() and paths["nec"].stat().st_size > 0, "necessity_results.jsonl should exist"
    assert paths["g4"].exists() and paths["g4"].stat().st_size > 0, "gate4_report.json should exist"

    # Validate robustness summary schema
    with paths["rob_sum"].open("r") as f:
        rs = json.load(f)
    for k in ["n_cells", "n_seeds", "effect_persistence_median", "seed_stability_median", "ood_delta_median"]:
        assert k in rs, f"Missing robustness summary key: {k}"

    # Validate Gate 4 report structure and acceptance flags
    with paths["g4"].open("r") as f:
        g4 = json.load(f)
    assert "acceptance" in g4 and isinstance(g4["acceptance"], dict), "Gate 4 acceptance block missing"
    acc = g4["acceptance"]
    for k in ["effect_persistence_ok", "seed_stability_ok", "necessity_drop_ok", "accept_gate4", "reasons"]:
        assert k in acc, f"Missing acceptance key: {k}"
    assert isinstance(acc["accept_gate4"], bool), "accept_gate4 must be boolean"
    assert isinstance(acc["reasons"], list), "reasons must be a list"
    assert "determinism" in g4 and isinstance(g4["determinism"], dict), "determinism signatures missing"
    det_a = g4["determinism"]
    for k in ["remaining_cells_md5", "robustness_signature_md5", "necessity_signature_md5"]:
        assert k in det_a, f"Missing determinism key: {k}"

    # Determinism: re-run and compare signatures and acceptance
    # Clean artifacts before re-run
    for p in paths.values():
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    week6_main(args)

    with paths["g4"].open("r") as f:
        g4_b = json.load(f)

    det_b = g4_b["determinism"]
    # Signatures must match for identical seeds/config
    for k in ["remaining_cells_md5", "robustness_signature_md5", "necessity_signature_md5"]:
        assert det_a[k] == det_b[k], f"Determinism checksum mismatch for {k}"

    # Acceptance stable
    acc_b = g4_b["acceptance"]
    assert acc["accept_gate4"] == acc_b["accept_gate4"], "accept_gate4 should be deterministic"