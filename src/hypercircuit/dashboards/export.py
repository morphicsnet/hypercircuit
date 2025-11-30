from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from pathlib import Path
import json
import hashlib
import re


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


# ----------------------------
# Week 7: Dashboards integration
# ----------------------------
from typing import Any, Dict, List, Mapping, DefaultDict, Sequence
from collections import defaultdict
from hypercircuit.utils.io import load_jsonl


def export_dashboard_ensembles(
    labels_path: Path,
    out_path: Path,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """
    Emit dashboard_ensembles.json with ensemble membership, stability/synergy (mock),
    causal and safety summaries, and links to labels.
    """
    rows = load_jsonl(labels_path) if labels_path.exists() else []
    by_ens: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
    labels_by_ens: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        eid = str(r.get("ensemble_id"))
        fam = str(r.get("family"))
        labels_by_ens[eid].append({"label_text": r.get("label_text"), "confidence": r.get("confidence")})
        if "id" not in by_ens[eid]:
            # Mock ensemble membership and scores (deterministic placeholders)
            members = [f"feature_{i}" for i in range(3)]
            by_ens[eid] = {
                "id": eid,
                "family": fam,
                "members": members,
                "stability": 0.9,
                "synergy": 0.1,
                "causal_effect_summary": {"delta_mean": 0.1, "delta_ci": [0.05, 0.15]},
                "safety_summary": {"benign_deg_pct": 0.1, "specificity_ratio": 1.2},
                "labels_linked": [],
            }

    for eid, rec in by_ens.items():
        rec["labels_linked"] = labels_by_ens.get(eid, [])

    payload = {
        "provenance": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(dict(provenance or {})),
        },
        "ensembles": list(by_ens.values()),
        "counts": {"n_ensembles": len(by_ens)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def export_dashboard_labels(
    labels_path: Path,
    out_path: Path,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """
    Emit dashboard_labels.json with label text, exemplars, confidence, and cross-links.
    """
    rows = load_jsonl(labels_path) if labels_path.exists() else []
    labels_out: List[Dict[str, Any]] = []
    for r in rows:
        labels_out.append(
            {
                "ensemble_id": r.get("ensemble_id"),
                "family": r.get("family"),
                "label_text": r.get("label_text"),
                "exemplars": r.get("exemplars", []),
                "confidence": r.get("confidence"),
                "uncertainty": r.get("uncertainty"),
            }
        )
    payload = {
        "provenance": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(dict(provenance or {})),
        },
        "labels": labels_out,
        "counts": {"total_labels": len(labels_out)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


# ----------------------------
# Week 8: Release bundle checksum
# ----------------------------
def _stable_json_checksum(path: Path) -> str:
    """
    Compute a stable checksum for a JSON file by sanitizing volatile fields.
    - Zero top-level created_at if present
    - If top-level contains {"provenance": {"timestamp": ...}}, zero it.
    - Dump with sort_keys=True before hashing.
    Fallback to raw bytes MD5 if JSON parsing fails.
    """
    try:
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            # Zero volatile fields commonly emitted by our writers
            if "created_at" in obj:
                obj["created_at"] = "0"
            prov = obj.get("provenance")
            if isinstance(prov, dict) and "timestamp" in prov:
                prov = dict(prov)
                prov["timestamp"] = "0"
                obj["provenance"] = prov
        dump = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.md5(dump).hexdigest()
    except Exception:
        try:
            return hashlib.md5(path.read_bytes()).hexdigest()
        except Exception:
            return ""


def _stable_text_checksum(path: Path) -> str:
    """
    Stable checksum for text/markdown by removing ISO8601 timestamps and normalizing whitespace.
    Falls back to raw bytes MD5 if decoding fails.
    """
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        # Replace ISO8601 timestamps like 2025-11-30T15:40:07.234Z or with timezone offsets
        txt = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?", "0", txt)
        # Collapse multiple spaces
        txt = re.sub(r"\s+", " ", txt).strip()
        return hashlib.md5(txt.encode("utf-8")).hexdigest()
    except Exception:
        try:
            return hashlib.md5(path.read_bytes()).hexdigest()
        except Exception:
            return ""


def compute_release_bundle_checksum(paths: Sequence[Path]) -> str:
    """
    Compute an aggregate MD5 over a set of artifact checksums in a stable manner.
    The aggregate payload is constructed from sorted entries of "relpath:name_md5".
    Missing files contribute empty checksums.
    """
    parts: List[str] = []
    for p in sorted(paths, key=lambda x: str(x)):
        if p.exists() and p.is_file():
            ext = p.suffix.lower()
            if ext in {".json", ".jsonl"}:
                ch = _stable_json_checksum(p)
            else:
                ch = _stable_text_checksum(p)
        else:
            ch = ""
        parts.append(f"{str(p)}:{ch}")
    payload = "|".join(parts).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def export_release_bundle(paths: Sequence[Path], out_path: Path, provenance: Mapping[str, object] | None = None) -> Dict[str, Any]:
    """
    Export a small JSON with per-file stable checksums and an aggregate bundle checksum.
    Returns a dict with {"out_path": Path, "bundle_checksum": str}.
    """
    entries: List[Dict[str, Any]] = []
    for p in sorted(paths, key=lambda x: str(x)):
        exists = p.exists() and p.is_file()
        size = int(p.stat().st_size) if exists else 0
        checksum = _stable_json_checksum(p) if exists else ""
        entries.append(
            {
                "path": str(p),
                "exists": bool(exists),
                "bytes": size,
                "checksum_md5": checksum,
            }
        )
    bundle_checksum = compute_release_bundle_checksum(paths)
    payload: Dict[str, Any] = {
        "provenance": {"timestamp": datetime.now(timezone.utc).isoformat(), **(dict(provenance or {}))},
        "entries": entries,
        "bundle_checksum": bundle_checksum,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return {"out_path": out_path, "bundle_checksum": bundle_checksum}


def export_dashboard_summary(
    labels_path: Path,
    out_path: Path,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """
    Emit dashboard_summary.json aggregating key metrics for quick load.
    """
    rows = load_jsonl(labels_path) if labels_path.exists() else []
    fams = sorted({str(r.get("family")) for r in rows}) if rows else []
    payload = {
        "provenance": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(dict(provenance or {})),
        },
        "summary": {
            "families": fams,
            "n_families": len(fams),
            "total_labels": len(rows),
            "highlights": {
                "stability_median": 0.9,
                "synergy_median": 0.1,
                "agreement_kappa_mock": 0.66,
            },
            "sections_emitted": ["ensembles", "labels", "summary"],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
