from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .io import write_json, append_jsonl

# Simple in-process registry state (single-run context)
_CURRENT: Optional["RunContext"] = None


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    manifest_path: Path
    artifacts_path: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_run_id() -> str:
    # timestamp + 6-char uuid for determinism/readability
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    short = uuid.uuid4().hex[:6]
    return f"{ts}-{short}"


def _best_effort_git_sha(root: Path) -> Optional[str]:
    try:
        head = (root / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            ref_path = root / ".git" / ref
            if ref_path.exists():
                return ref_path.read_text().strip()[:12]
        # detached head
        if len(head) >= 12:
            return head[:12]
    except Exception:
        pass
    return None


def _relative_to(child: Path, parent: Path) -> str:
    try:
        return str(child.resolve().relative_to(parent.resolve()))
    except Exception:
        return str(child)


def start_run(config_dict: Mapping[str, Any], stage_name: str, config_paths: Optional[Iterable[str]] = None) -> Tuple[str, Path]:
    """
    Initialize a run registry entry and open a run context.

    Writes: runs/{run_id}/manifest.json and prepares artifacts.jsonl

    Returns:
        (run_id, run_dir)
    """
    global _CURRENT
    run_cfg = dict(config_dict.get("run", {}))  # shallow copy
    output_dir = Path(run_cfg.get("output_dir") or "runs")
    run_id = run_cfg.get("run_id")
    if not run_id:
        run_id = _generate_run_id()

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    artifacts_path = run_dir / "artifacts.jsonl"

    # optional metadata
    model_tag = run_cfg.get("model_tag")
    task_family = run_cfg.get("task_family") or (config_dict.get("dataset", {}) or {}).get("name")
    split = run_cfg.get("split")
    seeds = run_cfg.get("seeds") or [0, 1, 2, 3, 4]
    versions = {
        "schema": "0.1.0",
        "python": os.getenv("PYTHON_VERSION") or None,
        "package": "hypercircuit",
    }
    git_sha = _best_effort_git_sha(Path("."))

    manifest = {
        "run_id": run_id,
        "created_at": _now_iso(),
        "stage": stage_name,
        "config_paths": list(config_paths or []),
        "model_tag": model_tag,
        "task_family": task_family,
        "split": split,
        "seeds": seeds,
        "versions": versions,
        "git_sha": git_sha,
        "status": "running",
        "summary": {},
    }
    write_json(manifest_path, manifest)

    # initialize artifacts file (touch)
    if not artifacts_path.exists():
        artifacts_path.write_text("")

    _CURRENT = RunContext(run_id=run_id, run_dir=run_dir, manifest_path=manifest_path, artifacts_path=artifacts_path)
    return run_id, run_dir


def current_run() -> Tuple[str, Path]:
    """Return the current (run_id, run_dir)."""
    if _CURRENT is None:
        raise RuntimeError("No active run context. Call start_run() first.")
    return _CURRENT.run_id, _CURRENT.run_dir


def log_artifact(path: Path | str, kind: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
    """Append an artifact record to artifacts.jsonl, storing relative path."""
    if _CURRENT is None:
        raise RuntimeError("No active run context. Call start_run() first.")
    p = Path(path)
    rel = _relative_to(p, _CURRENT.run_dir)
    rec: Dict[str, Any] = {
        "path": rel,
        "kind": kind,
        "created_at": _now_iso(),
        "meta": dict(metadata or {}),
    }
    append_jsonl(_CURRENT.artifacts_path, [rec])


def finalize_run(status: str, metrics_dict: Optional[Mapping[str, Any]] = None) -> None:
    """Update manifest with final status and summary metrics."""
    if _CURRENT is None:
        raise RuntimeError("No active run context. Call start_run() first.")
    try:
        manifest = json.loads(_CURRENT.manifest_path.read_text())
        # remove internal schema tag if present
        if isinstance(manifest, dict) and "_schema_version" in manifest:
            manifest.pop("_schema_version", None)
    except Exception:
        manifest = {}
    manifest.update(
        {
            "status": status,
            "finalized_at": _now_iso(),
            "summary": dict(metrics_dict or {}),
        }
    )
    write_json(_CURRENT.manifest_path, manifest)