from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hypercircuit.discovery.coactivation import mine_rank_weighted_coactivations
from hypercircuit.discovery.synergy import (
    score_candidates,
    filter_candidates,
    compute_stability_multi,
)
from hypercircuit.discovery.reporting import assemble_week2_synergy_report
from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.io import load_jsonl, write_json, read_json, save_jsonl
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run


def _find_latest_logs(run_base: Path, task_family: Optional[str], split: Optional[str]) -> Optional[Path]:
    """
    Search runs/* for a logs.jsonl whose manifest matches task_family/split.
    Choose the most recent by mtime.
    """
    if not run_base.exists():
        return None
    best: Optional[Tuple[float, Path]] = None
    for sub in sorted(run_base.iterdir()):
        if not sub.is_dir():
            continue
        manifest = sub / "manifest.json"
        logs = sub / "logs.jsonl"
        if not (manifest.exists() and logs.exists()):
            continue
        try:
            m = read_json(manifest)
        except Exception:
            continue
        if task_family and m.get("task_family") and m.get("task_family") != task_family:
            continue
        if split and m.get("split") and m.get("split") != split:
            continue
        mt = logs.stat().st_mtime
        if best is None or mt > best[0]:
            best = (mt, logs)
    return best[1] if best else None


def _partition_events_k(events: Iterable[Mapping[str, Any]], k: int) -> List[List[Mapping[str, Any]]]:
    parts: List[List[Mapping[str, Any]]] = [[] for _ in range(max(2, k))]
    for ev in events:
        sid = int(ev.get("sample_id", 0))
        parts[sid % len(parts)].append(ev)
    return parts


def _thresholds_from_disc(disc: Any) -> Dict[str, Any]:
    return {
        "min_weighted_support": float(getattr(disc, "min_weighted_support", 0.0)),
        "synergy_threshold": float(getattr(disc, "synergy_threshold", 0.0)),
        "stability_score_min": float(getattr(disc, "stability_score_min", 0.0)),
        "dedup_jaccard_min": float(getattr(disc, "dedup_jaccard_min", 0.5)),
        "caps": {
            "size2": int(getattr(disc.candidate_caps, "size2", 1000)),
            "size3": int(getattr(disc.candidate_caps, "size3", 300)),
            "temporal_span": int(getattr(disc.candidate_caps, "temporal_span", 3)),
        },
    }


def run_week2_screening(
    config_paths: Optional[Sequence[str]] = None,
    overlay_paths: Optional[Sequence[str]] = None,
    overrides: Optional[Sequence[str]] = None,
) -> Path:
    """
    Aggregator for Week 2 synergy screening.

    Steps per overlay (task family):
      - Locate latest logs via registry manifests (task_family/split match)
      - Mine rank-weighted coactivations (global candidates)
      - Build k replicate supports via sample_id % k partition if replicates_k > 2
      - Score synergy and stability; filter by thresholds
      - Collect per-family metrics and exemplars
    Writes:
      - week2_synergy_report.json (consolidated)
      - Optional per-family candidates JSONL
    """
    cfg_paths = list(config_paths or ["configs/base.yaml", "configs/discovery.yaml"])
    cfg: Config = load_config(cfg_paths, overrides or [])

    # Prepare run registry payload (respect legacy run_dir if provided)
    cfg_dict = cfg.model_dump()
    run_sec = cfg_dict.setdefault("run", {})
    legacy = run_sec.get("run_dir")
    if legacy and not run_sec.get("run_id"):
        lp = Path(legacy)
        run_sec["output_dir"] = str(lp.parent)
        run_sec["run_id"] = lp.name

    run_id, run_dir = start_run(cfg_dict, stage_name="week2_screening", config_paths=cfg_paths)
    run_base = Path(run_sec.get("output_dir") or "runs")

    disc = cfg.discovery
    week2 = getattr(disc, "week2_screening", None)
    top_families: List[str] = []
    replicates_k: int = 2
    paraphrase_replicates: int = 1
    if week2 is not None:
        # pydantic model or mapping
        tf = getattr(week2, "top_families", None)
        if tf is None and isinstance(week2, Mapping):
            top_families = list(week2.get("top_families", []))
        else:
            top_families = list(tf or [])
        replicates_k = int(getattr(week2, "replicates_k", 2))
        paraphrase_replicates = int(getattr(week2, "paraphrase_replicates", 1))

    thresholds = _thresholds_from_disc(disc)

    # Default overlays if not provided
    overlays = list(
        overlay_paths
        or [
            "configs/datasets/sycophancy.yaml",
            "configs/datasets/jailbreak.yaml",
            "configs/datasets/deceptive_reasoning.yaml",
            "configs/datasets/truthfulness_qa.yaml",
        ]
    )

    families_block: Dict[str, Dict[str, Any]] = {}
    processed: List[str] = []

    for overlay in overlays:
        # Load overlay to get task family metadata
        fam_cfg: Config = load_config([*cfg_paths, overlay], overrides or [])
        ds = fam_cfg.dataset
        fam_name = (ds.task_family or ds.name or "unknown").strip()
        split = ds.split

        # Find logs
        logs_path = _find_latest_logs(run_base, fam_name, split)
        if logs_path is None or not logs_path.exists():
            # Skip silently; aggregator should be resilient in CI
            continue

        events = load_jsonl(logs_path)

        # Mine on full events for candidate set and global WS
        cands, ws_index, rep_ws_ab = mine_rank_weighted_coactivations(
            events,
            min_weighted_support=disc.min_weighted_support,
            candidate_caps=disc.candidate_caps.model_dump()
            if hasattr(disc.candidate_caps, "model_dump")
            else dict(disc.candidate_caps),  # type: ignore[arg-type]
            temporal_span=disc.candidate_caps.temporal_span,
            dedup_jaccard_min=getattr(disc, "dedup_jaccard_min", 0.5),
            max_set_size=3,
        )

        # Build replicate supports
        stability_multi_val: float = 0.0
        replicate_ws: Dict[str, Dict[frozenset[str], float]]
        if replicates_k > 2:
            parts = _partition_events_k(events, replicates_k)
            replicate_ws = {}
            for i, part in enumerate(parts):
                # Mine supports for each replicate subset
                _c, ws_i, _rep_i = mine_rank_weighted_coactivations(
                    part,
                    min_weighted_support=disc.min_weighted_support,
                    candidate_caps=disc.candidate_caps.model_dump()
                    if hasattr(disc.candidate_caps, "model_dump")
                    else dict(disc.candidate_caps),  # type: ignore[arg-type]
                    temporal_span=disc.candidate_caps.temporal_span,
                    dedup_jaccard_min=getattr(disc, "dedup_jaccard_min", 0.5),
                    max_set_size=3,
                )
                replicate_ws[f"R{i}"] = ws_i
            # Compute a stability summary for report convenience
            keys = [frozenset(c["members"]) for c in cands]
            vectors = [
                [float(replicate_ws.get(f"R{i}", {}).get(k, 0.0)) for k in keys] for i in range(replicates_k)
            ]
            stability_multi_val = compute_stability_multi(vectors)
        else:
            replicate_ws = rep_ws_ab

        # Score and filter
        scored = score_candidates(cands, ws_index, replicate_ws, replicates_k=replicates_k)
        after_synergy = [
            c
            for c in scored
            if (not c.get("redundancy_flag")) and float(c.get("synergy_score", 0.0)) >= disc.synergy_threshold
        ]
        after_stability = filter_candidates(
            scored,
            synergy_threshold=disc.synergy_threshold,
            stability_score_min=getattr(disc, "stability_score_min", 0.5),
        )

        # Optional per-family dump for inspection
        fam_cand_path = stage_path(run_dir, f"{fam_name}_candidates.jsonl")
        save_jsonl(fam_cand_path, scored)
        log_artifact(fam_cand_path, kind="candidates", metadata={"family": fam_name, "count": len(scored)})

        families_block[fam_name] = {
            "all_candidates": scored,
            "after_synergy": after_synergy,
            "after_stability": after_stability,
            "replicates_k": replicates_k,
            "stability_multi": float(stability_multi_val),
        }
        processed.append(fam_name)

    report = assemble_week2_synergy_report(
        families=families_block,
        thresholds=thresholds,
        week2={
            "replicates_k": replicates_k,
            "paraphrase_replicates": paraphrase_replicates,
            "top_families": top_families,
        },
        run_meta={
            "task_list": processed,
            "replicates_k": replicates_k,
            "paraphrase_replicates": paraphrase_replicates,
        },
    )
    report_path = stage_path(run_dir, "week2_synergy_report.json")
    write_json(report_path, report)
    log_artifact(report_path, kind="report", metadata={"families": processed, "replicates_k": replicates_k})

    # Finalize with aggregate metrics
    fam_metrics = {
        f"{fam}.n_candidates_total": int(val["counts"]["n_candidates_total"])
        if "counts" in val
        else len(val.get("all_candidates", []))
        for fam, val in report.get("families", {}).items()
    }
    finalize_run(
        status="success",
        metrics_dict={
            "families_processed": len(processed),
            "replicates_k": replicates_k,
            "top_families": top_families,
            **fam_metrics,
        },
    )
    return report_path