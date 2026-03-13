from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from hypercircuit.discovery.coactivation import mine_rank_weighted_coactivations
from hypercircuit.discovery.synergy import score_candidates, filter_candidates
from hypercircuit.discovery.reporting import assemble_gate1_report
from hypercircuit.logging.schema import EVENT_SCHEMA_VERSION
from hypercircuit.utils.config import Config, load_config, stage_path, apply_legacy_run_dir
from hypercircuit.utils.io import load_jsonl, save_jsonl, write_json, read_json
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.utils.compat import assert_schema_version, assert_feature_space_compat
from hypercircuit.utils.failures import classify_exception


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discovery: mine co-activations and compile Gate 1 report.")
    p.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/discovery.yaml"])
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Optional runs/<run_id> override (bypasses run.output_dir/run_id).")
    p.add_argument("--logs-path", default=None, help="Optional direct path to logs.jsonl (skips search).")
    return p.parse_args(argv)


def _prepare_run(cfg: Config, args: argparse.Namespace) -> Tuple[Dict, Path]:
    cfg_dict = cfg.model_dump()
    run_sec = cfg_dict.setdefault("run", {})
    if args.run_dir:
        p = Path(args.run_dir)
        run_sec["output_dir"] = str(p.parent)
        run_sec["run_id"] = p.name
    else:
        apply_legacy_run_dir(run_sec)

    # thread dataset metadata into run metadata if present
    ds = cfg.dataset
    run_sec["task_family"] = run_sec.get("task_family") or ds.task_family or ds.name
    if ds.split is not None:
        run_sec["split"] = ds.split

    _, run_dir = start_run(cfg_dict, stage_name="discovery", config_paths=args.config)
    return cfg_dict, run_dir


def _find_latest_logs(run_base: Path, task_family: Optional[str], split: Optional[str]) -> Optional[Path]:
    """Search runs/* for logs.jsonl whose manifest matches task_family/split. Choose most recent by mtime."""
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


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)
    cfg_dict, run_dir = _prepare_run(cfg, args)

    run_meta = cfg_dict.get("run", {})
    run_base = Path(run_meta.get("output_dir") or "runs")

    # Resolve input logs
    logs_path: Optional[Path] = Path(args.logs_path) if args.logs_path else None
    if logs_path is None:
        tf = run_meta.get("task_family")
        sp = run_meta.get("split")
        logs_path = _find_latest_logs(run_base, tf, sp)
    if logs_path is None or not logs_path.exists():
        raise FileNotFoundError("Could not locate logs.jsonl. Provide --logs-path or run logging first.")

    try:
        events = load_jsonl(logs_path)

        # Optional manifest-level compatibility checks
        manifest_path = logs_path.parent / "events_manifest.json"
        if not manifest_path.exists():
            alt_manifest = logs_path.parent / "events_reconciled_manifest.json"
            if alt_manifest.exists():
                manifest_path = alt_manifest
        member_granularity = cfg.logging.member_granularity
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            assert_schema_version(manifest, key="schema_version", supported=[EVENT_SCHEMA_VERSION])
            member_granularity = str(manifest.get("member_granularity") or member_granularity)
            assert_feature_space_compat(
                manifest,
                expected_id=cfg.sae.feature_space_id,
                expected_version=cfg.sae.feature_space_version,
            )

        # Mine rank-weighted co-activations (mock weights)
        disc = cfg.discovery
        cands, ws_index, rep_ws = mine_rank_weighted_coactivations(
            events,
            min_weighted_support=disc.min_weighted_support,
            candidate_caps=disc.candidate_caps.model_dump() if hasattr(disc.candidate_caps, "model_dump") else dict(disc.candidate_caps),  # type: ignore[arg-type]
            temporal_span=disc.candidate_caps.temporal_span,
            dedup_jaccard_min=getattr(disc, "dedup_jaccard_min", 0.5),
            max_set_size=3,
            member_granularity=member_granularity,
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

        # Candidate manifest for downstream compatibility checks
        cand_manifest = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "member_granularity": member_granularity,
            "candidate_type": "coactivation",
            "scores": ["coactivation_score", "synergy_score", "stability_score", "final_rank_score"],
        }
        cand_manifest_path = stage_path(run_dir, "candidates_manifest.json")
        write_json(cand_manifest_path, cand_manifest)
        log_artifact(cand_manifest_path, kind="candidates_manifest")

        # Calibration artifact (real-path friendly)
        if scored:
            ws_vals = [float(c.get("weighted_support", 0.0)) for c in scored]
            syn_vals = [float(c.get("synergy_score", 0.0)) for c in scored]
            stab_vals = [float(c.get("stability_score", 0.0)) for c in scored]
        else:
            ws_vals = []
            syn_vals = []
            stab_vals = []
        def _quantiles(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
            vs = sorted(vals)
            def _q(p: float) -> float:
                idx = int(round((len(vs) - 1) * p))
                return float(vs[idx])
            return {"p05": _q(0.05), "p50": _q(0.50), "p95": _q(0.95)}

        calibration = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "dataset_slice": {
                "n_events": len(events),
                "n_candidates": len(scored),
            },
            "thresholds": {
                "min_weighted_support": disc.min_weighted_support,
                "synergy_threshold": disc.synergy_threshold,
                "stability_score_min": getattr(disc, "stability_score_min", 0.5),
            },
            "score_distributions": {
                "weighted_support": _quantiles(ws_vals),
                "synergy_score": _quantiles(syn_vals),
                "stability_score": _quantiles(stab_vals),
            },
        }
        calib_path = stage_path(run_dir, "calibration.json")
        write_json(calib_path, calibration)
        log_artifact(calib_path, kind="calibration")

        report = assemble_gate1_report(
            all_candidates=scored,
            after_synergy=passed_synergy,
            after_stability=passed_stability,
            input_paths=[Path(logs_path)],
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
            run_meta={"task_family": run_meta.get("task_family"), "split": run_meta.get("split")},
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
                "task_family": run_meta.get("task_family"),
                "split": run_meta.get("split"),
            },
            manifest_updates={"provenance": {"member_granularity": member_granularity}},
        )
    except Exception as exc:
        fail = classify_exception(exc)
        finalize_run(
            status="failed",
            metrics_dict={"failure_mode": fail.mode, "error": fail.message},
            manifest_updates={"provenance": {"failure_mode": fail.mode}},
        )
        raise


if __name__ == "__main__":
    main()
