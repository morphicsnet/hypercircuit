from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("blt")
pytest.importorskip("mair")

from blt.export import run_trace
from hypercircuit.blt_io import run_blt_analysis


def test_hypercircuit_blt_analysis_emits_report(tmp_path: Path) -> None:
    manifest_path = run_trace("Hypercircuit delegates grouped CLT.", "trace-hc-1", tmp_path, backend="mock")
    updated_manifest = run_blt_analysis(manifest_path)
    report = json.loads((tmp_path / "blt_analysis_report.json").read_text(encoding="utf-8"))
    assert updated_manifest.exists()
    assert report["group_count"] >= 1
    assert report["intervention_count"] >= 1
