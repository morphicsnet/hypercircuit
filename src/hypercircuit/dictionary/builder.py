from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hypercircuit.utils.io import save_jsonl, write_json, read_json, load_jsonl, SCHEMA_VERSION
from hypercircuit.utils.seed import seed_context
from hypercircuit.utils.config import DictionaryConfig


Member = str


def _jaccard(a: Sequence[Member], b: Sequence[Member]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return 0.0 if union == 0 else inter / union


def _stable_id(family: str, members: Sequence[Member]) -> str:
    # Deterministic id from family + sorted members
    key = f"{family}|{','.join(sorted(members))}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def select_and_dedup_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    synergy_min: float,
    stability_min: float,
    dedup_jaccard_min: float,
    max_per_family: int,
) -> Tuple[List[Mapping[str, Any]], int, int]:
    """
    Filter by thresholds, then deduplicate by Jaccard over member sets, and cap.
    Returns (selected, pre_dedup_count, dedup_skipped_count).
    """
    # Threshold filter first
    thresh: List[Mapping[str, Any]] = []
    for c in candidates:
        if float(c.get("synergy_score", 0.0)) < float(synergy_min):
            continue
        if float(c.get("stability_score", 0.0)) < float(stability_min):
            continue
        if c.get("redundancy_flag"):
            continue
        thresh.append(c)
    pre_count = len(thresh)

    # Deterministic sort for stable selection
    thresh.sort(
        key=lambda d: (
            -float(d.get("synergy_score", 0.0)),
            -float(d.get("stability_score", 0.0)),
            -float(d.get("weighted_support", 0.0)),
            tuple(d.get("members", [])),
        )
    )

    selected: List[Mapping[str, Any]] = []
    seen: List[List[Member]] = []
    for c in thresh:
        members = list(c.get("members", []))
        if any(_jaccard(members, prev) >= float(dedup_jaccard_min) for prev in seen):
            continue
        selected.append(c)
        seen.append(members)
        if len(selected) >= int(max_per_family):
            break

    dedup_skipped = pre_count - len(selected)
    return selected, pre_count, dedup_skipped


def compute_provenance(
    entry: Mapping[str, Any],
    *,
    family: str,
    split: Optional[str],
    run_id: Optional[str],
    source_artifact_paths: Sequence[Path | str],
    discovered_at: Optional[str] = None,
) -> Dict[str, Any]:
    members = list(entry.get("members", []))
    pid = _stable_id(family, members)
    # Ensure paths are strings relative or absolute; we keep as string in payload
    src_paths = [str(p) for p in source_artifact_paths]

    # Coverage placeholders (mock)
    layer_cov = None
    token_cov = {"window_span": int(entry.get("window_span", 0))} if "window_span" in entry else None

    return {
        "id": pid,
        "members": members,
        "size": int(entry.get("size", len(members))),
        "synergy_score": float(entry.get("synergy_score", 0.0)),
        "stability_score": float(entry.get("stability_score", 0.0)),
        "weighted_support": float(entry.get("weighted_support", 0.0)),
        "family": family,
        "split": split,
        "run_id": run_id,
        "source_artifact_paths": src_paths,
        "discovered_at": discovered_at or _now_iso(),
        "layer_coverage": layer_cov,
        "token_coverage": token_cov,
        # Semantic placeholders
        "semantics": {"label": None, "evidence": None, "uncertainty": 1.0},
    }


def assemble_dictionary_entries(
    *,
    by_family_selected: Mapping[str, Sequence[Mapping[str, Any]]],
    by_family_meta: Mapping[str, Mapping[str, Any]],
    exemplars_top_k: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build final entries with provenance and exemplars.
    Returns (entries, counts_by_family).
    """
    entries: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for fam, sel in by_family_selected.items():
        meta = by_family_meta.get(fam, {})
        split = meta.get("split")
        run_id = meta.get("run_id")
        src_paths = meta.get("source_artifact_paths", [])
        disc_at = meta.get("discovered_at")
        fam_count = 0
        for s in sel:
            prov = compute_provenance(
                s,
                family=fam,
                split=split,
                run_id=run_id,
                source_artifact_paths=src_paths,
                discovered_at=disc_at,
            )
            pid = prov["id"]
            k = int(exemplars_top_k)
            exemplars = [{"rank": i + 1, "window": f"{pid}:w{i}"} for i in range(max(0, k))]
            entries.append({**prov, "exemplars": exemplars})
            fam_count += 1
        counts[fam] = fam_count
    # Deterministic ordering across families and entries
    entries.sort(
        key=lambda d: (
            d.get("family") or "",
            -float(d.get("synergy_score", 0.0)),
            -float(d.get("stability_score", 0.0)),
            tuple(d.get("members", [])),
        )
    )
    return entries, counts


def emit_go_no_go(
    *,
    counts_by_family: Mapping[str, int],
    families_to_check: Sequence[str],
    min_passed_per_top_family: int,
    artifact_paths: Sequence[Path],
) -> Dict[str, Any]:
    """Compute Gate 1 go/no-go flags following mock Week 2 rules."""
    # counts_ok: all top families meet the per-family minimum
    fams = list(families_to_check)
    per_ok: Dict[str, bool] = {
        fam: int(counts_by_family.get(fam, 0)) >= int(min_passed_per_top_family) for fam in fams
    }
    counts_ok = all(per_ok.values()) if fams else True

    # storage_ok: all declared artifacts exist and are non-empty
    storage_ok = True
    for p in artifact_paths:
        try:
            if (not p.exists()) or p.stat().st_size <= 0:
                storage_ok = False
                break
        except Exception:
            storage_ok = False
            break

    flags = {"counts_ok": bool(counts_ok), "storage_ok": bool(storage_ok)}
    if all(flags.values()):
        final = "go"
        reasons: List[str] = []
    else:
        final = "no_go"
        reasons = []
        if not flags["counts_ok"]:
            reasons.append(
                f"Counts below minimum for one or more families (min={int(min_passed_per_top_family)})."
            )
        if not flags["storage_ok"]:
            reasons.append("One or more artifacts missing or empty.")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "flags": flags,
        "families_checked": fams,
        "counts_by_family": {k: int(v) for k, v in counts_by_family.items()},
        "final": final,
        "reasons": reasons,
    }


@dataclass
class BuildResult:
    ensembles_path: Path
    dictionary_path: Path
    go_no_go_path: Path
    counts_by_family: Dict[str, int]
    selected_total: int
    synergy_stats: Dict[str, float]
    stability_stats: Dict[str, float]
    pre_threshold_counts: Dict[str, int]
    dedup_skipped_counts: Dict[str, int]
    decision: Dict[str, Any]


def build_ensemble_dictionary(
    *,
    inputs_by_family: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
    config: DictionaryConfig | Mapping[str, Any],
    run_dir: Path,
    seed: Optional[int] = 0,
    families_to_evaluate: Optional[Sequence[str]] = None,
) -> BuildResult:
    """
    Construct an ensemble dictionary from discovery candidates across families.

    inputs_by_family: {
      family: {
        "candidates": List[Mapping[str, Any]],
        "split": Optional[str],
        "run_id": Optional[str],
        "source_artifact_paths": Sequence[Path|str],
        "discovered_at": Optional[str]
      }
    }
    """
    # Normalize config (allow mapping for tests)
    if isinstance(config, DictionaryConfig):
        cfg = config
    else:
        # Construct with defaults + provided mapping
        cfg = DictionaryConfig(**dict(config))

    with seed_context(seed):
        # Selection and per-family metadata capture
        by_family_selected: Dict[str, List[Mapping[str, Any]]] = {}
        pre_counts: Dict[str, int] = {}
        dedup_skipped: Dict[str, int] = {}
        by_family_meta: Dict[str, Dict[str, Any]] = {}
        for fam, meta in inputs_by_family.items():
            cands = list(meta.get("candidates", []))
            sel, pre, skipped = select_and_dedup_candidates(
                cands,
                synergy_min=cfg.synergy_min,
                stability_min=cfg.stability_min,
                dedup_jaccard_min=cfg.dedup_jaccard_min,
                max_per_family=cfg.max_per_family,
            )
            by_family_selected[fam] = sel
            pre_counts[fam] = pre
            dedup_skipped[fam] = skipped
            by_family_meta[fam] = {
                "split": meta.get("split"),
                "run_id": meta.get("run_id"),
                "source_artifact_paths": meta.get("source_artifact_paths", []),
                "discovered_at": meta.get("discovered_at"),
            }

        entries, counts_by_family = assemble_dictionary_entries(
            by_family_selected=by_family_selected,
            by_family_meta=by_family_meta,
            exemplars_top_k=cfg.exemplars_top_k,
        )

        # Write ensembles.jsonl
        ensembles_path = run_dir / "ensembles.jsonl"
        save_jsonl(ensembles_path, entries)

        # Build index/header for ensemble_dictionary.json
        index = {
            e["id"]: {
                "family": e["family"],
                "size": e["size"],
                "synergy": e["synergy_score"],
                "stability": e["stability_score"],
                "members": e["members"],
            }
            for e in entries
        }
        synergy_vals = [float(e["synergy_score"]) for e in entries]
        stability_vals = [float(e["stability_score"]) for e in entries]

        header = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _now_iso(),
            "config": {
                "synergy_min": float(cfg.synergy_min),
                "stability_min": float(cfg.stability_min),
                "max_per_family": int(cfg.max_per_family),
                "dedup_jaccard_min": float(cfg.dedup_jaccard_min),
                "exemplars_top_k": int(cfg.exemplars_top_k),
                "min_passed_per_top_family": int(cfg.min_passed_per_top_family),
                "families": list(cfg.families) if cfg.families else None,
            },
            "counts_by_family": {k: int(v) for k, v in counts_by_family.items()},
            "total": int(len(entries)),
            "index": index,
        }
        dictionary_path = run_dir / "ensemble_dictionary.json"
        write_json(dictionary_path, header)

        # Decision/go-no-go
        fams_to_check: List[str] = []
        if cfg.families:
            fams_to_check = list(cfg.families)
        elif families_to_evaluate is not None:
            fams_to_check = list(families_to_evaluate)
        else:
            fams_to_check = sorted(counts_by_family.keys())

        decision = emit_go_no_go(
            counts_by_family=counts_by_family,
            families_to_check=fams_to_check,
            min_passed_per_top_family=cfg.min_passed_per_top_family,
            artifact_paths=[ensembles_path, dictionary_path],
        )
        go_no_go_path = run_dir / "gate1_go_no_go.json"
        write_json(go_no_go_path, decision)

    synergy_stats = {
        "min": float(min(synergy_vals)) if synergy_vals else 0.0,
        "max": float(max(synergy_vals)) if synergy_vals else 0.0,
        "median": float(_median(synergy_vals)) if synergy_vals else 0.0,
    }
    stability_stats = {
        "min": float(min(stability_vals)) if stability_vals else 0.0,
        "max": float(max(stability_vals)) if stability_vals else 0.0,
        "median": float(_median(stability_vals)) if stability_vals else 0.0,
    }

    return BuildResult(
        ensembles_path=ensembles_path,
        dictionary_path=dictionary_path,
        go_no_go_path=go_no_go_path,
        counts_by_family={k: int(v) for k, v in counts_by_family.items()},
        selected_total=int(len(entries)),
        synergy_stats=synergy_stats,
        stability_stats=stability_stats,
        pre_threshold_counts={k: int(v) for k, v in pre_counts.items()},
        dedup_skipped_counts={k: int(v) for k, v in dedup_skipped.items()},
        decision=decision,
    )