from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def hash_json(obj: Mapping[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hash_bytes(payload)


def file_checksum(path: Path) -> Optional[str]:
    try:
        return _hash_bytes(path.read_bytes())
    except Exception:
        return None


def dependency_snapshot(root: Path) -> Dict[str, Optional[str]]:
    """Best-effort checksum snapshot of dependency lock/config files."""
    files = [
        root / "pyproject.toml",
        root / "poetry.lock",
        root / "uv.lock",
        root / "requirements.txt",
        root / "requirements-dev.txt",
    ]
    snap: Dict[str, Optional[str]] = {}
    for p in files:
        if p.exists():
            snap[str(p.name)] = file_checksum(p)
    return snap

