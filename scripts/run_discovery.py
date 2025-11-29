from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple, Dict

from hypercircuit.discovery.coactivation import mine_rank_weighted_coactivations
from hypercircuit.discovery.synergy import score_candidates, filter_candidates
from hypercircuit.discovery.reporting import assemble_gate1_report
from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.io import load_jsonl, save_jsonl, write_json, read_json
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discovery: mine co-activations and compile Gate 1 report.")
    p.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/discovery.yaml"])
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Optional runs/<run_id> override (bypasses run.output_dir/run_id).")
    p.add_argument("--logs-path", default=None, help="Optional direct path to logs.jsonl (skips search).")
    return p.parse_args(argv)


def _prepare_run(cfg: Config, args: argparse.Namespace) -> Tuple[Dict, Path, str, Path]:
    cfg_dict = cfg.model_dump()
    run_sec = cfg_dict.setdefault("run", {})
    if args.run_dir:
        p = Path(args.run_dir)
        run_sec["output_dir"] = str(p.parent)
        run_sec["run_id"] = p.name
    else:
        legacy = run_sec.get("run_dir")
        if legacy and not run_sec.get("run_id"):
            lp = Path(legacy)
            run_sec["output_dir"] = str(lp.parent)
            run_sec["run_id"] = lp.name

    # thread dataset metadata into run metadata if present
    ds = cfg.dataset
    run_sec["task_family"] = run_sec.get("task_family") or ds.task_family or ds.name
    if ds.split is not None:
        run_sec["split"] = ds.split

    run_id, run_dir = start_run(cfg_dict, stage_name="discovery", config_paths=args.config)
    return cfg_dict, run_dir, run_id, Path(run_sec["output_dir"])


def _find_latest_logs(run_base: Path, task_family: Optional[str], split: Optional[str]) -> Optional[Path]:
    """Search runs/* for a logs.jsonl whose manifest matches task_family/split. Choose most recent by mtime."""
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


def run_discovery(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)
    cfg_dict, run_dir, run_id, run_base = _prepare_run(cfg, args)

    # Resolve input logs
    logs_path: Optional[Path] = Path(args.logs_path) if args.logs_path else None
    if logs_path is None:
        tf = cfg_dict.get("run", {}).get("task_family")
        sp = cfg_dict.get("run", {}).get("split")
        logs_path = _find_latest_logs(run_base, tf, sp)
    if logs_path is None or not logs_path.exists():
        raise FileNotFoundError("Could not locate logs.jsonl. Provide --logs-path or run logging first.")

    events = load_jsonl(logs_path)
    # Mine rank-weighted co-activations (mock weights)
    disc = cfg.discovery
    cands, ws_index, rep_ws = mine_rank_weighted_coactivations(
        events,
        min_weighted_support=disc.min_weighted_support,
        candidate_caps=disc.candidate_caps.model_dump() if hasattr(disc.candidate_caps, "model_dump") else dict(disc.candidate_caps),  # type: ignore[arg-type]
        temporal_span=disc.candidate_caps.temporal_span,
        dedup_jaccard_min=getattr(disc, "dedup_jaccard_min", 0.5),
        max_set_size=3,
    )

    # Score synergy and stability, then filter
    scored = score_candidates(cands, ws_index, rep_ws)
    passed_synergy = [c for c in scored if c.get("synergy_score", 0.0) >= disc.synergy_threshold and not c.get("redundancy_flag")]  # type: ignore[truthy-function]
    passed_stability = filter_candidates(
        scored,
        synergy_threshold=disc.synergy_threshold,
        stability_score_min=getattr(disc, "stability_score_min", 0.5),
    )

    # Write artifacts
    cand_path = stage_path(run_dir, "candidates.jsonl")
    save_jsonl(cand_path, scored)
    log_artifact(cand_path, kind="candidates", metadata={"count": len(scored)})

    report = assemble_gate1_report(
        all_candidates=scored,
        after_synergy=passed_synergy,
        after_stability=passed_stability,
        input_paths=[logs_path],
        output_paths=[cand_path],
        thresholds={
            "min_weighted_support": disc.min_weighted_support,
            "synergy_threshold": disc.synergy_threshold,
            "stability_score_min": getattr(disc, "stability_score_min", 0.5),
            "dedup_jaccard_min": getattr(disc, "dedup_jaccard_min", 0.5),
            "caps": {
                "size2": disc.candidate_caps.size2,
                "size3": disc.candidate_caps.size3,
                "temporal_span": disc.candidate_caps.temporal_span,
            },
        },
        run_meta={
            "task_family": cfg_dict.get("run", {}).get("task_family"),
            "split": cfg_dict.get("run", {}).get("split"),
        },
    )
    report_path = stage_path(run_dir, "gate1_report.json")
    write_json(report_path, report)
    log_artifact(report_path, kind="report", metadata={"n_candidates_total": report.get("counts", {}).get("n_candidates_total", 0)})

    # Finalize with summary metrics
    storage_in = sum(p.stat().st_size for p in [logs_path] if p and p.exists())
    storage_out = sum(p.stat().st_size for p in [cand_path, report_path] if p.exists())
    finalize_run(
        status="success",
        metrics_dict={
            "n_candidates_total": int(report["counts"]["n_candidates_total"]),
            "n_candidates_size2": int(report["counts"]["n_candidates_size2"]),
            "n_candidates_size3": int(report["counts"]["n_candidates_size3"]),
            "n_passed_synergy": int(report["counts"]["n_passed_synergy"]),
            "n_passed_stability": int(report["counts"]["n_passed_stability"]),
            "top_candidate_example": scored[0] if scored else None,
            "storage_input_bytes": int(storage_in),
            "storage_output_bytes": int(storage_out),
            "synergy_threshold": float(disc.synergy_threshold),
            "stability_score_min": float(getattr(disc, "stability_score_min", 0.5)),
            "task_family": cfg_dict.get("run", {}).get("task_family"),
            "split": cfg_dict.get("run", {}).get("split"),
        },
    )


def main(argv: Optional[List[str]] = None) -> None:
    run_discovery(argv)


if __name__ == "__main__":
    main()
