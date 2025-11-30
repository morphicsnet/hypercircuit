from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import hashlib
import json

from hypercircuit.utils.io import load_jsonl, read_json
from hypercircuit.cli.run_log import main as run_log_main
from hypercircuit.cli.run_week7_labels_dash import main as week7_main


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _md5_json_sanitized(path: Path) -> str:
    """
    Load JSON, sanitize volatile fields (timestamps), and return MD5 of stable dump.
    """
    obj = json.loads(path.read_text())
    # Remove/normalize provenance timestamps that are intentionally dynamic
    prov = obj.get("provenance")
    if isinstance(prov, dict):
        prov["timestamp"] = "0"
    dump = json.dumps(obj, sort_keys=True)
    return hashlib.md5(dump.encode("utf-8")).hexdigest()


def test_week7_logging_24_layer_expansion(tmp_path: Path) -> None:
    """
    Run logging CLI with top_behaviors_only and a top family overlay.
    Assert n_layers expands to 24 and event density within configured band.
    """
    run_dir = tmp_path / "runs" / "week7_logging_24"
    args = [
        "--config",
        "configs/base.yaml",
        "configs/logging.yaml",
        "configs/datasets/sycophancy.yaml",
        "-o",
        f"run.run_dir={run_dir}",
        "-o",
        "logging.top_behaviors_only=true",
    ]
    run_log_main(args)

    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists(), "Manifest should be created by logging CLI"
    manifest = read_json(manifest_path)
    summary = manifest.get("summary", {})
    assert int(summary.get("n_layers", 0)) == 24 or int(summary.get("n_layers_used", 0)) == 24, "n_layers should be 24 for top family"
    assert bool(summary.get("within_band", False)), "Event density should be within configured band"
    # Sanity: logs.jsonl present and non-empty
    logs_path = run_dir / "logs.jsonl"
    assert logs_path.exists() and logs_path.stat().st_size > 0, "logs.jsonl should exist and be non-empty"


def test_week7_labels_and_dashboards(tmp_path: Path) -> None:
    """
    Run Week 7 labeling + dashboards CLI twice, assert artifacts exist, contain required sections,
    and deterministic checksums (ignoring timestamps in dashboard files).
    """
    run_dir = tmp_path / "runs" / "week7_labels_dash"
    args = [
        "--config",
        "configs/base.yaml",
        "configs/discovery.yaml",
        "configs/dictionary.yaml",
        "-o",
        f"run.run_dir={run_dir}",
    ]
    week7_main(args)

    # Artifacts existence
    labels = run_dir / "labels.jsonl"
    label_report = run_dir / "label_report.json"
    dash_ens = run_dir / "dashboard_ensembles.json"
    dash_lbl = run_dir / "dashboard_labels.json"
    dash_sum = run_dir / "dashboard_summary.json"

    for p in [labels, label_report, dash_ens, dash_lbl, dash_sum]:
        assert p.exists() and p.stat().st_size > 0, f"{p.name} should exist and be non-empty"

    # Validate labels.jsonl schema (first row)
    rows = load_jsonl(labels)
    assert len(rows) > 0, "labels.jsonl should contain at least one row"
    r0 = rows[0]
    for key in [
        "ensemble_id",
        "family",
        "label_text",
        "evidence_tokens",
        "context_summary",
        "confidence",
        "uncertainty",
        "exemplars",
        "annotator_stub",
        "seed",
        "schema_version",
    ]:
        assert key in r0, f"Missing required label field: {key}"

    # Validate label_report.json structure
    rep = read_json(label_report)
    assert "coverage" in rep and isinstance(rep["coverage"], dict), "label_report coverage missing"
    assert "agreement" in rep and "kappa_mock" in rep["agreement"], "label_report agreement.kappa_mock missing"

    # Validate dashboards sections
    ens_obj = read_json(dash_ens)
    lbl_obj = read_json(dash_lbl)
    sum_obj = read_json(dash_sum)

    assert "ensembles" in ens_obj and isinstance(ens_obj["ensembles"], list), "dashboard_ensembles ensembles missing"
    assert "labels" in lbl_obj and isinstance(lbl_obj["labels"], list), "dashboard_labels labels missing"
    assert "summary" in sum_obj and isinstance(sum_obj["summary"], dict), "dashboard_summary summary missing"
    assert "sections_emitted" in sum_obj["summary"], "dashboard_summary should include sections_emitted"

    # Determinism: run again and compare stable checksums
    c1_ens = _md5_json_sanitized(dash_ens)
    c1_lbl = _md5_json_sanitized(dash_lbl)
    c1_sum = _md5_json_sanitized(dash_sum)
    c1_labels_rows = hashlib.md5(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()

    week7_main(args)

    rows2 = load_jsonl(labels)
    c2_ens = _md5_json_sanitized(dash_ens)
    c2_lbl = _md5_json_sanitized(dash_lbl)
    c2_sum = _md5_json_sanitized(dash_sum)
    c2_labels_rows = hashlib.md5(json.dumps(rows2, sort_keys=True).encode("utf-8")).hexdigest()

    assert c1_ens == c2_ens, "dashboard_ensembles checksum should be deterministic"
    assert c1_lbl == c2_lbl, "dashboard_labels checksum should be deterministic"
    assert c1_sum == c2_sum, "dashboard_summary checksum should be deterministic"
    assert c1_labels_rows == c2_labels_rows, "labels selection should be deterministic across runs"

    # Finalize metrics presence
    manifest = read_json(run_dir / "manifest.json")
    summary = manifest.get("summary", {})
    assert int(summary.get("dashboard_sections_emitted", 0)) >= 3, "Should emit all dashboard sections"
    assert "agreement_kappa_mock" in summary, "Finalize summary should include agreement_kappa_mock"