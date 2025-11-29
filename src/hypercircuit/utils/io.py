from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping

SCHEMA_VERSION = "0.1.0"


def save_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write an iterable of JSON-serializable mappings to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append records to a JSONL file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    """Load a JSONL file to a list of dictionaries."""
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    """Write a single JSON mapping with schema version tag."""
    payload = {"_schema_version": SCHEMA_VERSION, **obj}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())
