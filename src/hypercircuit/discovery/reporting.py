from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Optional


def _size_bytes(paths: Sequence[Path]) -> int:
    total = 0
    for p in paths:
        try:
            if p and p.exists():
                total += p.stat().st_size
        except Exception:
            # Best-effort; ignore inaccessible files in mock mode
            pass
    return int(total)


def _sorted_candidates(cands: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    # Deterministic sort used across pipeline (weighted_support desc, synergy desc, members lex)
    return sorted(
        cands,
        key=lambda d: (
            -float(d.get("weighted_support", 0.0)),
            -float(d.get("synergy_score", 0.0)),
            tuple(d.get("members", [])),
        ),
    )


def assemble_gate1_report(
    *,
    all_candidates: Sequence[Mapping[str, Any]],
    after_synergy: Sequence[Mapping[str, Any]],
    after_stability: Sequence[Mapping[str, Any]],
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    thresholds: Mapping[str, Any],
    run_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble Gate 1 acceptance report with counts, storage metrics, exemplars, and flags.

    Returns a JSON-serializable mapping. For convenience and test friendliness,
    top-level count fields are duplicated under the "counts" block.
    """
    # Counts
    n_total = len(all_candidates)
    n_size2 = sum(1 for c in all_candidates if int(c.get("size", 0)) == 2)
    n_size3 = sum(1 for c in all_candidates if int(c.get("size", 0)) == 3)
    n_synergy = len(after_synergy)
    n_stability = len(after_stability)

    counts = {
        "n_candidates_total": int(n_total),
        "n_candidates_size2": int(n_size2),
        "n_candidates_size3": int(n_size3),
        "n_passed_synergy": int(n_synergy),
        "n_passed_stability": int(n_stability),
    }

    # Storage
    storage_input = _size_bytes(list(input_paths))
    storage_output = _size_bytes(list(output_paths))
    storage = {
        "input_bytes": int(storage_input),
        "output_bytes": int(storage_output),
    }

    # Exemplars
    pool = after_stability if after_stability else (after_synergy if after_synergy else all_candidates)
    top = _sorted_candidates(pool)[:10]
    exemplars = [
        {
            "members": list(c.get("members", [])),
            "size": int(c.get("size", 0)),
            "support": int(c.get("support", 0)),
            "weighted_support": float(c.get("weighted_support", 0.0)),
            "synergy_score": float(c.get("synergy_score", 0.0)),
            "stability_score": float(c.get("stability_score", 0.0)),
        }
        for c in top
    ]

    # Acceptance flags (informational in mock)
    counts_ok = n_stability >= 15
    storage_ok = (storage_input > 0) and (storage_output > 0)
    acceptance = {
        "flags": {
            "counts_ok": bool(counts_ok),
            "storage_ok": bool(storage_ok),
        },
        "notes": "Week 1 mock acceptance is informational; thresholds configurable in configs/discovery.yaml.",
    }

    # Flattened top-level copies for ease of testing
    report: Dict[str, Any] = {
        **counts,
        "counts": counts,
        "storage": storage,
        "acceptance": acceptance,
        "top_exemplars": exemplars,
        "thresholds": dict(thresholds),
        "run": dict(run_meta or {}),
    }
    return report


def assemble_week2_synergy_report(
    *,
    families: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    week2: Mapping[str, Any],
    run_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble a consolidated Week 2 synergy report across multiple task families.

    Top-level schema:
      - families: {family: {counts, stability, top_exemplars, thresholds, top10_signature_md5}}
      - week2: {replicates_k, paraphrase_replicates, top_families}
      - acceptance: {family: bool} for those present in top_families
      - thresholds: discovery thresholds used
      - run: optional metadata for provenance
    """
    import hashlib
    import json

    tf_list = week2.get("top_families", [])
    top_families = list(tf_list) if isinstance(tf_list, (list, tuple)) else []

    families_block: Dict[str, Any] = {}
    acceptance: Dict[str, bool] = {}

    for fam, data in families.items():
        all_c = list(data.get("all_candidates", []))
        after_syn = list(data.get("after_synergy", []))
        after_stab = list(data.get("after_stability", []))
        k = int(data.get("replicates_k", 2))

        # Counts
        n_total = len(all_c)
        n_size2 = sum(1 for c in all_c if int(c.get("size", 0)) == 2)
        n_size3 = sum(1 for c in all_c if int(c.get("size", 0)) == 3)
        n_syn = len(after_syn)
        n_stab = len(after_stab)

        counts = {
            "n_candidates_total": int(n_total),
            "n_candidates_size2": int(n_size2),
            "n_candidates_size3": int(n_size3),
            "n_passed_synergy": int(n_syn),
            "n_passed_stability": int(n_stab),
        }

        # Exemplars
        pool = after_stab if after_stab else (after_syn if after_syn else all_c)
        top = _sorted_candidates(pool)[:10]
        exemplars = [
            {
                "members": list(c.get("members", [])),
                "size": int(c.get("size", 0)),
                "support": int(c.get("support", 0)),
                "weighted_support": float(c.get("weighted_support", 0.0)),
                "synergy_score": float(c.get("synergy_score", 0.0)),
                "stability_score": float(c.get("stability_score", 0.0)),
            }
            for c in top
        ]

        # Determinism checksum of top-10 signature (members, ws, syn)
        sig = [
            (
                tuple(x.get("members", [])),
                float(x.get("weighted_support", 0.0)),
                float(x.get("synergy_score", 0.0)),
            )
            for x in exemplars
        ]
        sig_bytes = json.dumps(sig, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        checksum = hashlib.md5(sig_bytes).hexdigest()

        fam_block = {
            "counts": counts,
            "stability": {
                "replicates_k": k,
                "stability_multi": float(
                    data.get(
                        "stability_multi",
                        exemplars[0].get("stability_score", 0.0) if exemplars else 0.0,
                    )
                ),
            },
            "top_exemplars": exemplars,
            "thresholds": dict(thresholds),
            "top10_signature_md5": checksum,
        }
        families_block[fam] = fam_block

        if fam in top_families:
            # Informational acceptance flag for CI dashboards
            acceptance[fam] = counts["n_passed_stability"] > 0

    report: Dict[str, Any] = {
        "families": families_block,
        "week2": {
            "replicates_k": int(week2.get("replicates_k", 2)),
            "paraphrase_replicates": int(week2.get("paraphrase_replicates", 1)),
            "top_families": top_families,
        },
        "acceptance": acceptance,
        "thresholds": dict(thresholds),
        "run": dict(run_meta or {}),
    }
    return report

def load_candidates_for_family(run_base: Path, family: str, split: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Locate the latest discovery run for a given family (and optional split) and
    load its candidates with minimal provenance for dictionary building.

    Returns:
        {
          "candidates": List[Mapping[str, Any]],
          "split": Optional[str],
          "run_id": Optional[str],
          "source_artifact_paths": List[str],
          "discovered_at": Optional[str],
          "family": str,
          "run_path": str
        }
        or None if not found.
    """
    from hypercircuit.utils.io import read_json, load_jsonl  # lazy import to avoid cycles

    if not run_base.exists():
        return None

    best_mtime: Optional[float] = None
    best_run_dir: Optional[Path] = None
    best_manifest: Optional[Mapping[str, Any]] = None
    best_cand: Optional[Path] = None
    best_report: Optional[Path] = None

    for sub in sorted(run_base.iterdir()):
        if not sub.is_dir():
            continue
        manifest = sub / "manifest.json"
        cand = sub / "candidates.jsonl"
        rep = sub / "gate1_report.json"
        if not (manifest.exists() and cand.exists()):
            continue
        try:
            m = read_json(manifest)
        except Exception:
            continue
        if family and m.get("task_family") and m.get("task_family") != family:
            continue
        if split and m.get("split") and m.get("split") != split:
            continue
        mt = cand.stat().st_mtime
        if best_mtime is None or mt > best_mtime:
            best_mtime = mt
            best_run_dir = sub
            best_manifest = m
            best_cand = cand
            best_report = rep if rep.exists() else None

    if best_cand is None or best_run_dir is None or best_manifest is None:
        return None

    cands = load_jsonl(best_cand)
    src_paths = [str(best_cand)]
    if best_report is not None:
        src_paths.append(str(best_report))

    return {
        "candidates": cands,
        "split": best_manifest.get("split"),
        "run_id": best_manifest.get("run_id"),
        "source_artifact_paths": src_paths,
        "discovered_at": best_manifest.get("created_at"),
        "family": family,
        "run_path": str(best_run_dir),
    }