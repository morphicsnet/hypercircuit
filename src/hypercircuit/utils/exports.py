from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from hypercircuit.logging.schema import EVENT_SCHEMA_VERSION
from hypercircuit.utils.io import load_jsonl, write_json


def export_feature_events_bundle(events_path: Path, out_path: Path, manifest: Mapping[str, Any]) -> Path:
    rows = load_jsonl(events_path) if events_path.exists() else []
    payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "kind": "feature_events",
        "manifest": dict(manifest),
        "events": rows,
    }
    write_json(out_path, payload)
    return out_path


def export_candidates_bundle(candidates_path: Path, out_path: Path, manifest: Mapping[str, Any]) -> Path:
    rows = load_jsonl(candidates_path) if candidates_path.exists() else []
    payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "kind": "candidates",
        "manifest": dict(manifest),
        "candidates": rows,
    }
    write_json(out_path, payload)
    return out_path


def export_evidence_bundle(results_path: Path, out_path: Path, manifest: Mapping[str, Any]) -> Path:
    rows = load_jsonl(results_path) if results_path.exists() else []
    payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "kind": "causal_evidence",
        "manifest": dict(manifest),
        "results": rows,
    }
    write_json(out_path, payload)
    return out_path

