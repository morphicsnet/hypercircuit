from __future__ import annotations

from pathlib import Path
import json

from hypercircuit.cli.run_build_dictionary import main as build_dict_main
from hypercircuit.cli.run_week7_labels_dash import main as week7_main
from hypercircuit.cli.run_week8_release import main as week8_main


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_week8_release_end_to_end(tmp_path: Path) -> None:
    """
    End-to-end Week 8:
    - Build (possibly empty) dictionary artifacts deterministically.
    - Produce labels and dashboard summary (Week 7) for inclusion in final report.
    - Run Week 8 release CLI twice with fixed run_dir and configs to ensure determinism.
    - Assert presence and schema of frozen dictionary, release manifest, final report and summary, and scaling doc.
    - Verify snapshot_id and release_bundle_checksum are deterministic across runs.
    """
    run_dir = tmp_path / "runs" / "week8_release_e2e"

    # 1) Build dictionary artifacts (mock discovery may be absent; builder still emits header + index)
    build_dict_main([
        "--config", "configs/base.yaml", "configs/discovery.yaml", "configs/dictionary.yaml",
        "-o", f"run.run_dir={run_dir}",
    ])

    # 2) Week 7 labeling + dashboards to provide label_report.json and dashboard_summary.json
    week7_main([
        "--config", "configs/base.yaml", "configs/discovery.yaml", "configs/dictionary.yaml",
        "-o", f"run.run_dir={run_dir}",
    ])

    # 3) Week 8 release — first run
    week8_args = [
        "--config", "configs/base.yaml", "configs/discovery.yaml", "configs/dictionary.yaml",
        "-o", f"run.run_dir={run_dir}",
    ]
    week8_main(week8_args)

    # Artifact paths
    frozen_path = run_dir / "ensemble_dictionary_frozen.json"
    manifest_path = run_dir / "release_manifest.json"
    final_report_path = run_dir / "final_report.json"
    final_summary_path = run_dir / "final_summary.md"
    scaling_doc_path = run_dir / "docs" / "SCALING_RECOMMENDATION.md"

    # Existence checks
    for p in [frozen_path, manifest_path, final_report_path, final_summary_path, scaling_doc_path]:
        assert p.exists() and p.stat().st_size > 0, f"{p.name} should exist and be non-empty"

    # Structure checks
    frozen = _read_json(frozen_path)
    manifest = _read_json(manifest_path)
    final_rep = _read_json(final_report_path)

    assert "snapshot" in frozen and "snapshot_id" in frozen["snapshot"], "Frozen dictionary must include snapshot.snapshot_id"
    assert "artifacts" in manifest and isinstance(manifest["artifacts"], list), "release_manifest artifacts list missing"
    assert "release_bundle_checksum" in manifest and isinstance(manifest["release_bundle_checksum"], str), "release_manifest checksum missing"
    assert "gates" in final_rep and "accept_release" in final_rep, "final_report must include gates and accept_release"

    snap1 = str(frozen["snapshot"]["snapshot_id"])
    chk1 = str(manifest["release_bundle_checksum"])

    # 4) Week 8 release — second run for determinism
    week8_main(week8_args)
    frozen2 = _read_json(frozen_path)
    manifest2 = _read_json(manifest_path)

    snap2 = str(frozen2["snapshot"]["snapshot_id"])
    chk2 = str(manifest2["release_bundle_checksum"])

    assert snap1 == snap2, "snapshot_id should be deterministic under fixed seed and inputs"
    assert chk1 == chk2, "release_bundle_checksum should be deterministic across runs"

    # Sanity: acceptance flags present (may be false in mock if upstream gates missing)
    gates = final_rep.get("gates", {})
    for k in ["accept_gate1", "accept_gate2", "accept_gate3", "accept_gate4"]:
        assert k in gates, f"Missing {k} in final_report.gates"

    # Ensure scaling note references snapshot/checksum
    txt = scaling_doc_path.read_text()
    assert snap1 in txt, "Scaling recommendation should include snapshot_id"
    assert chk1 in txt or "bundle_checksum" in txt, "Scaling recommendation should include bundle checksum reference"