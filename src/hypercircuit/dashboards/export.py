from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from pathlib import Path
import json


def export_flat_json(artifacts: Mapping[str, object], out_path: Path, provenance: Mapping[str, object]) -> None:
    payload = {
        "provenance": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **provenance,
        },
        **artifacts,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def export_dictionary(dictionary_path: Path, ensembles_path: Path | None, out_path: Path, provenance: Mapping[str, object] | None = None) -> None:
    """
    Export a dashboard-ready flat JSON for the ensemble dictionary.

    Inputs:
      - dictionary_path: path to ensemble_dictionary.json (header with counts/index)
      - ensembles_path: optional path to ensembles.jsonl (entries)
      - out_path: destination flat JSON
      - provenance: optional extra provenance fields

    The exporter is lightweight and keeps payloads small for dashboards.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dictionary header
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Dictionary header not found: {dictionary_path}")
    try:
        import json as _json
        header = _json.loads(dictionary_path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to read {dictionary_path}: {e}") from e

    # Sample up to 20 ensembles for preview (optional)
    sample: list[dict] = []
    if ensembles_path and ensembles_path.exists():
        try:
            with ensembles_path.open("r") as f:
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample.append(_json.loads(line))
                    except Exception:
                        # best-effort in mock
                        pass
        except Exception:
            sample = []

    payload = {
        "provenance": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(dict(provenance or {})),
        },
        "dictionary": {
            "schema_version": header.get("schema_version"),
            "created_at": header.get("created_at"),
            "config": header.get("config", {}),
            "counts_by_family": header.get("counts_by_family", {}),
            "total": header.get("total", 0),
            # Keep index minimal for dashboards; present but potentially large
            "index_size": len(header.get("index", {}) or {}),
        },
        "ensembles_preview": [
            {
                "id": e.get("id"),
                "family": e.get("family"),
                "size": e.get("size"),
                "synergy": e.get("synergy_score"),
                "stability": e.get("stability_score"),
                "members": e.get("members"),
            }
            for e in sample
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
