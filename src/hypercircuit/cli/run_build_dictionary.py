from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from hypercircuit.utils.config import Config, load_config
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.utils.seed import active_seed
from hypercircuit.discovery.reporting import load_candidates_for_family
from hypercircuit.dictionary.builder import build_ensemble_dictionary
from hypercircuit.dashboards.export import export_dictionary


def _parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ensemble dictionary from discovery outputs (mock, deterministic).")
    p.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/discovery.yaml", "configs/dictionary.yaml"],
        help="Base + discovery + dictionary config files (merged in order).",
    )
    p.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Config overrides in dotted.key=value form (e.g., dictionary.max_per_family=25).",
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Explicit list of families to include. If omitted, uses dictionary.families or discovery.week2_screening.top_families.",
    )
    p.add_argument(
        "--export-flat",
        default=None,
        help="Optional path to export dashboard-ready flat JSON (e.g., runs/mock/dictionary_flat.json).",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Optional runs/<run_id> override (bypasses run.output_dir/run_id from config).",
    )
    return p.parse_args(argv)


def _prepare_run(cfg: Config, args: argparse.Namespace) -> Tuple[Dict, Path]:
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

    # Thread dataset metadata into run metadata if present
    ds = cfg.dataset
    run_sec["task_family"] = run_sec.get("task_family") or ds.task_family or ds.name
    if ds.split is not None:
        run_sec["split"] = ds.split

    _, run_dir = start_run(cfg_dict, stage_name="build_dictionary", config_paths=args.config)
    return cfg_dict, run_dir


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)
    cfg_dict, run_dir = _prepare_run(cfg, args)

    run_meta = cfg_dict.get("run", {})
    run_base = Path(run_meta.get("output_dir") or "runs")

    # Resolve families
    families: List[str]
    if args.families:
        families = list(args.families)
    elif cfg.dictionary.families:
        families = list(cfg.dictionary.families)  # type: ignore[arg-type]
    else:
        families = list(cfg.discovery.week2_screening.top_families)

    # Load latest discovery outputs per family
    inputs_by_family: Dict[str, Mapping[str, object]] = {}
    resolved: List[str] = []
    for fam in families:
        pack = load_candidates_for_family(run_base, fam, split=None)
        if not pack:
            # Skip if discovery artifacts missing for this family
            continue
        inputs_by_family[fam] = {
            "candidates": pack.get("candidates", []),
            "split": pack.get("split"),
            "run_id": pack.get("run_id"),
            "source_artifact_paths": pack.get("source_artifact_paths", []),
            "discovered_at": pack.get("discovered_at"),
        }
        resolved.append(fam)

    # Build dictionary (deterministic via active seed)
    seed = active_seed(run_meta.get("seeds"), stage_idx=0)
    result = build_ensemble_dictionary(
        inputs_by_family=inputs_by_family,
        config=cfg.dictionary,
        run_dir=run_dir,
        seed=seed,
        families_to_evaluate=resolved,
    )

    # Log artifacts
    log_artifact(result.ensembles_path, kind="ensembles", metadata={"total": result.selected_total})
    log_artifact(result.dictionary_path, kind="dictionary", metadata={"families": resolved})
    log_artifact(result.go_no_go_path, kind="decision", metadata={"final": result.decision.get("final")})

    # Optional dashboard export
    if args.export_flat:
        flat_path = Path(args.export_flat)
        export_dictionary(result.dictionary_path, result.ensembles_path, flat_path, provenance={"run_id": cfg_dict["run"]["run_id"]})
        log_artifact(flat_path, kind="dashboard", metadata={"note": "flat_dictionary"})

    # Finalize metrics
    finalize_run(
        status="success",
        metrics_dict={
            "families_resolved": resolved,
            "counts_by_family": result.counts_by_family,
            "selected_total": result.selected_total,
            "synergy_min": float(cfg.dictionary.synergy_min),
            "stability_min": float(cfg.dictionary.stability_min),
            "exemplars_top_k": int(cfg.dictionary.exemplars_top_k),
            "min_passed_per_top_family": int(cfg.dictionary.min_passed_per_top_family),
            "synergy_min_max_median": result.synergy_stats,
            "stability_min_max_median": result.stability_stats,
            "pre_threshold_counts": result.pre_threshold_counts,
            "dedup_skipped_counts": result.dedup_skipped_counts,
            "go_no_go": result.decision.get("final"),
            "go_no_go_flags": result.decision.get("flags"),
            "go_no_go_reasons": result.decision.get("reasons"),
        },
    )


if __name__ == "__main__":
    main()