from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from blt.export import run_analysis as _blt_run_analysis
    from mair.manifest import load_manifest
except Exception as exc:  # pragma: no cover - exercised in integration usage
    _blt_run_analysis = None  # type: ignore
    load_manifest = None  # type: ignore
    _IMPORT_ERROR = exc

from hypercircuit.utils.io import write_json


def run_blt_analysis(manifest_path: str | Path, output_dir: str | Path | None = None) -> Path:
    if _blt_run_analysis is None or load_manifest is None:
        raise RuntimeError(
            "BLT analysis requires installed blt and mair packages. "
            "Run python -m pip install -e '/Volumes/128/MAIR[dev]' -e '/Volumes/128/BLT[dev]'"
        ) from _IMPORT_ERROR
    manifest_file = Path(manifest_path)
    output_root = Path(output_dir) if output_dir is not None else manifest_file.parent
    updated_manifest = _blt_run_analysis(manifest_file, output_dir=output_root)
    manifest = load_manifest(updated_manifest)
    artifact_by_type = {artifact["artifact_type"]: artifact for artifact in manifest["artifacts"]}
    grouped = json.loads((output_root / artifact_by_type["grouped_clt_bundle"]["path"]).read_text(encoding="utf-8"))
    sweep_count = sum(1 for _ in (output_root / artifact_by_type["intervention_sweep"]["path"]).open("r", encoding="utf-8") if _.strip())
    report = {
        "trace_id": manifest["trace_id"],
        "artifact_count": len(manifest["artifacts"]),
        "group_count": len(grouped["groups"]),
        "intervention_count": sweep_count,
        "summary_metrics": grouped["summary_metrics"],
    }
    report_path = output_root / "blt_analysis_report.json"
    write_json(report_path, report)
    return updated_manifest
