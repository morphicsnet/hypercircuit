from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hypercircuit.utils.config import Config
from hypercircuit.utils.io import SCHEMA_VERSION, load_jsonl, read_json, write_json
from hypercircuit.utils.registry import current_run
from hypercircuit.utils.seed import active_seed
from hypercircuit.utils.config import stage_path
from hypercircuit.dashboards.export import compute_release_bundle_checksum


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_bucket_from_iso(iso_str: Optional[str]) -> str:
    """
    Return YYYYMMDD from ISO timestamp, or '0' if unavailable.
    """
    if not iso_str:
        return "0"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y%m%d")
    except Exception:
        return "0"


def _stable_entry_key(e: Mapping[str, Any]) -> Tuple:
    return (
        str(e.get("family") or ""),
        -float(e.get("synergy_score", 0.0)),
        -float(e.get("stability_score", 0.0)),
        tuple(e.get("members", [])),
    )


def _families_from_cfg(cfg: Config) -> List[str]:
    """
    Priority:
    - cfg.release.include_families
    - cfg.dictionary.families
    - cfg.discovery.week2_screening.top_families
    - [cfg.dataset.task_family or cfg.dataset.name]
    """
    rel = getattr(cfg, "release", None)
    if rel and getattr(rel, "include_families", None):
        return list(rel.include_families or [])
    dic = getattr(cfg, "dictionary", None)
    if dic and getattr(dic, "families", None):
        return list(dic.families or [])
    week2 = getattr(cfg.discovery, "week2_screening", None)
    if week2 and getattr(week2, "top_families", None):
        fams = list(week2.top_families or [])
        if fams:
            return fams
    return [cfg.run.task_family or cfg.dataset.task_family or cfg.dataset.name]


@dataclass
class DictionarySource:
    run_id: Optional[str]
    run_path: Optional[Path]
    dictionary_path: Optional[Path]
    ensembles_path: Optional[Path]
    header: Mapping[str, Any]


def _find_dictionary_source(run_base: Path, prefer_dir: Path | None = None) -> Optional[DictionarySource]:
    """
    Locate a dictionary header + ensembles with best-effort heuristics:
    1) Prefer paths in prefer_dir if present.
    2) Else scan run_base/* for a directory containing ensemble_dictionary.json and ensembles.jsonl,
       selecting the most recently modified dictionary.
    """
    # 1) Prefer current run_dir for direct usage
    cand_paths: List[Tuple[float, Path, Path, Optional[Path], Mapping[str, Any]]] = []

    def _add_candidate(d: Path) -> None:
        dct = d / "ensemble_dictionary.json"
        ens = d / "ensembles.jsonl"
        if not (dct.exists() and ens.exists()):
            return
        try:
            hdr = read_json(dct)
        except Exception:
            return
        try:
            mtime = dct.stat().st_mtime
        except Exception:
            mtime = 0.0
        man = d / "manifest.json"
        # Attach manifest read only for run_id if available
        return cand_paths.append((mtime, d, dct, man if man.exists() else None, hdr))

    if prefer_dir:
        _add_candidate(prefer_dir)

    if run_base.exists():
        for sub in sorted(run_base.iterdir()):
            if sub.is_dir():
                _add_candidate(sub)

    if not cand_paths:
        return None
    # Choose by latest dictionary mtime; if prefer_dir present and valid, it will sort among candidates
    mtime, d, dct, man, hdr = sorted(cand_paths, key=lambda x: x[0], reverse=True)[0]
    run_id = None
    try:
        if man is not None and man.exists():
            m = read_json(man)
            run_id = str(m.get("run_id")) if m.get("run_id") is not None else None
    except Exception:
        run_id = None
    return DictionarySource(run_id=run_id, run_path=d, dictionary_path=dct, ensembles_path=d / "ensembles.jsonl", header=hdr)


def _filter_and_sort_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    families: Sequence[str],
    stability_min: float,
) -> List[Mapping[str, Any]]:
    fam_set = set(families)
    filt = []
    for e in entries:
        fam = str(e.get("family"))
        if fam_set and fam not in fam_set:
            continue
        if float(e.get("stability_score", 0.0)) < float(stability_min):
            continue
        filt.append(e)
    # Deduplicate by id or member set
    seen_ids: set[str] = set()
    seen_members: set[Tuple[str, ...]] = set()
    out: List[Mapping[str, Any]] = []
    for e in sorted(filt, key=_stable_entry_key):
        eid = str(e.get("id"))
        mem = tuple(e.get("members", []))
        if eid in seen_ids or mem in seen_members:
            continue
        out.append(
            {
                "id": eid,
                "family": str(e.get("family")),
                "members": list(mem),
                "size": int(e.get("size", len(mem))),
                "synergy_score": float(e.get("synergy_score", 0.0)),
                "stability_score": float(e.get("stability_score", 0.0)),
            }
        )
        seen_ids.add(eid)
        seen_members.add(mem)
    return out


def _compute_snapshot_id(entries: Sequence[Mapping[str, Any]], *, seed: int, timestamp_bucket: str) -> str:
    """
    snapshot_id = sha1(f"{timestamp_bucket}|{seed}|sorted_entries")[:12]
    where sorted_entries concatenates "family:id:member1,member2,..."
    """
    parts: List[str] = []
    for e in sorted(entries, key=lambda x: (str(x.get("family") or ""), str(x.get("id") or ""))):
        fam = str(e.get("family"))
        eid = str(e.get("id"))
        mem = ",".join(str(m) for m in (e.get("members") or []))
        parts.append(f"{fam}:{eid}:{mem}")
    payload = f"{timestamp_bucket}|{int(seed)}|{'|'.join(parts)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def freeze_dictionary(*, cfg: Config, run_dir: Path, families: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Load latest accepted ensembles from the current dictionary, apply canonical sorting,
    de-duplication, and stability gating, and emit a frozen snapshot with immutable metadata.

    Writes:
      - ensemble_dictionary_frozen.json under run_dir (name configurable via cfg.release.out_dir_names.frozen_dictionary)
    Returns:
      {
        "frozen_path": Path,
        "snapshot_id": str,
        "n_ensembles": int,
        "source": {dictionary_run_id, dictionary_path, ensembles_path},
        "frozen_header": Mapping[str, Any]
      }
    """
    cfg_dict = cfg.model_dump()
    run_meta = cfg_dict.get("run", {}) or {}
    run_base = Path(run_meta.get("output_dir") or "runs")

    fams = list(families) if families else _families_from_cfg(cfg)

    # Resolve dictionary source (prefer current run_dir first)
    src = _find_dictionary_source(run_base=run_base, prefer_dir=run_dir)
    entries: List[Mapping[str, Any]] = []
    timestamp_bucket = "0"
    source_info: Dict[str, Any] = {"dictionary_run_id": None, "dictionary_path": None, "ensembles_path": None}
    if src and src.ensembles_path and src.dictionary_path:
        try:
            raw = load_jsonl(src.ensembles_path)
        except Exception:
            raw = []
        header = dict(src.header or {})
        timestamp_bucket = _date_bucket_from_iso(header.get("created_at"))
        source_info = {
            "dictionary_run_id": src.run_id,
            "dictionary_path": str(src.dictionary_path),
            "ensembles_path": str(src.ensembles_path),
        }
        entries = _filter_and_sort_entries(
            raw,
            families=fams,
            stability_min=float(getattr(cfg.dictionary, "stability_min", 0.5)),
        )

    # Deterministic snapshot id
    seed_val = active_seed(cfg.run.seeds, stage_idx=0)
    snapshot_id = _compute_snapshot_id(entries, seed=int(seed_val), timestamp_bucket=timestamp_bucket)

    counts_by_family: Dict[str, int] = {}
    for e in entries:
        f = str(e.get("family"))
        counts_by_family[f] = counts_by_family.get(f, 0) + 1

    # Compose frozen snapshot payload
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "snapshot": {
            "snapshot_id": snapshot_id,
            "snapshot_tag": getattr(getattr(cfg, "release", None), "snapshot_tag", "v1"),
            "timestamp_bucket": timestamp_bucket,
            "seed": int(seed_val),
            "parent": {
                "run_id": source_info["dictionary_run_id"],
                "dictionary_path": source_info["dictionary_path"],
                "ensembles_path": source_info["ensembles_path"],
            },
            "dictionary_version": (src.header.get("schema_version") if src else None),
            "manifold_version": getattr(getattr(cfg, "release", None), "manifold_version", None),
        },
        "families": sorted(set(fams)),
        "counts_by_family": {k: int(v) for k, v in counts_by_family.items()},
        "total": int(len(entries)),
        "ensembles": entries,
        "acceptance_annotation": {
            # informational; final acceptance emitted by final_report
            "stability_min": float(getattr(cfg.dictionary, "stability_min", 0.5)),
        },
        "provenance": {
            "run_id": current_run()[0] if _safe_has_current_run() else None,
        },
    }

    out_name = getattr(getattr(cfg, "release", None), "out_dir_names", None)
    fname = getattr(out_name, "frozen_dictionary", "ensemble_dictionary_frozen.json") if out_name else "ensemble_dictionary_frozen.json"
    out_path = stage_path(run_dir, fname)
    write_json(out_path, frozen)

    return {
        "frozen_path": out_path,
        "snapshot_id": snapshot_id,
        "n_ensembles": int(len(entries)),
        "source": source_info,
        "frozen_header": frozen,
    }


def _safe_has_current_run() -> bool:
    try:
        current_run()
        return True
    except Exception:
        return False


def _size_bytes(p: Path) -> int:
    try:
        return int(p.stat().st_size)
    except Exception:
        return 0


def _stable_file_checksum(path: Path) -> str:
    """
    Stable per-file checksum:
    - JSON/JSONL: load and zero provenance.timestamp if present, then sort_keys dump.
    - Text/Markdown: strip ISO8601 timestamps and collapse whitespace.
    - Fallback: raw bytes MD5.
    """
    try:
        if path.suffix.lower() in {".json", ".jsonl"}:
            obj = json.loads(path.read_text())
            if isinstance(obj, dict):
                prov = obj.get("provenance")
                if isinstance(prov, dict) and "timestamp" in prov:
                    prov = dict(prov)
                    prov["timestamp"] = "0"
                    obj["provenance"] = prov
            dump = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.md5(dump).hexdigest()
        # text-like
        txt = path.read_text(encoding="utf-8", errors="ignore")
        txt = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?", "0", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return hashlib.md5(txt.encode("utf-8")).hexdigest()
    except Exception:
        try:
            return hashlib.md5(path.read_bytes()).hexdigest()
        except Exception:
            return ""


def assemble_release_manifest(
    *,
    cfg: Config,
    run_dir: Path,
    snapshot_id: str,
    artifacts: Mapping[str, Path],
    acceptance_flags: Mapping[str, bool],
    reasons: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    Compile artifact list, checksums, provenance, versions, and acceptance flags into a manifest.

    Writes:
      - release_manifest.json under run_dir (name configurable via cfg.release.out_dir_names.release_manifest)
    Returns:
      {
        "manifest_path": Path,
        "release_bundle_checksum": str,
        "manifest": Mapping[str, Any]
      }
    """
    names_sorted = sorted(artifacts.keys())
    paths_sorted = [artifacts[k] for k in names_sorted]

    bundle_checksum = compute_release_bundle_checksum(paths_sorted)

    entries: List[Mapping[str, Any]] = []
    for k in names_sorted:
        p = artifacts[k]
        entries.append(
            {
                "name": k,
                "path": str(p),
                "bytes": _size_bytes(p),
                "checksum_md5": _stable_file_checksum(p) if p.exists() and p.is_file() else "",
            }
        )

    run_id = None
    try:
        run_id = current_run()[0]
    except Exception:
        run_id = None

    versions = {
        "schema": SCHEMA_VERSION,
        "package": "hypercircuit",
        "python": None,
    }

    accept_release = bool(
        acceptance_flags.get("accept_gate1", False)
        and acceptance_flags.get("accept_gate2", False)
        and acceptance_flags.get("accept_gate3", False)
        and acceptance_flags.get("accept_gate4", False)
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "artifacts": entries,
        "acceptance": {
            **{k: bool(v) for k, v in acceptance_flags.items()},
            "accept_release": bool(accept_release),
        },
        "release_bundle_checksum": bundle_checksum,
        "reasons": list(reasons or []),
        "versions": versions,
    }

    out_name = getattr(getattr(cfg, "release", None), "out_dir_names", None)
    fname = getattr(out_name, "release_manifest", "release_manifest.json") if out_name else "release_manifest.json"
    out_path = stage_path(run_dir, fname)
    write_json(out_path, payload)
    return {"manifest_path": out_path, "release_bundle_checksum": bundle_checksum, "manifest": payload}