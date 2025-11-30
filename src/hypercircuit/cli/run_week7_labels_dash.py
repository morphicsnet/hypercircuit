from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.utils.io import read_json, load_jsonl
from hypercircuit.semantics.labeling import finalize_labels
from hypercircuit.dashboards.export import (
    export_dashboard_ensembles,
    export_dashboard_labels,
    export_dashboard_summary,
)


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Week 7: Semantic labeling finalization and dashboards export (mock/deterministic)."
    )
    p.add_argument(
        "--config",
        nargs="+",
        required=False,
        default=["configs/base.yaml", "configs/discovery.yaml", "configs/dictionary.yaml"],
    )
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument(
        "--run-dir",
        default=None,
        help="Legacy path form runs/<run_id> (overrides run.output_dir/run_id).",
    )
    p.add_argument("--families", nargs="*", default=None, help="Optional explicit families list (else from config).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)

    # Legacy run-dir handling (consistent with other CLIs)
    cfg_dict = cfg.model_dump()
    run_sec = cfg_dict.setdefault("run", {})
    if args.run_dir:
        p = Path(args.run_dir)
        run_sec["output_dir"] = str(p.parent)
        run_sec["run_id"] = p.name
    else:
        legacy = run_sec.get("run_dir")
        if legacy:
            lp = Path(legacy)
            run_sec["output_dir"] = str(lp.parent)
            run_sec["run_id"] = lp.name

    # Prefer dataset metadata (informational)
    ds = cfg.dataset
    run_sec["task_family"] = run_sec.get("task_family") or ds.task_family or ds.name
    if ds.split is not None:
        run_sec["split"] = ds.split

    # Start registry stage
    run_id, run_dir = start_run(cfg_dict, stage_name="week7_labels_dash", config_paths=args.config)

    # 1) Label finalization (deterministic)
    out = finalize_labels(cfg=cfg, run_dir=run_dir, families=args.families)
    labels_path = Path(out["labels_path"])
    report_path = Path(out["report_path"])
    log_artifact(labels_path, kind="labels")
    log_artifact(report_path, kind="label_report")

    # 2) Dashboards export (controlled by cfg.dashboard include flags)
    prov = {"run_id": run_id}
    out_ensembles = stage_path(run_dir, getattr(cfg.dashboard, "out_ensembles", "dashboard_ensembles.json"))
    out_labels = stage_path(run_dir, getattr(cfg.dashboard, "out_labels", "dashboard_labels.json"))
    out_summary = stage_path(run_dir, getattr(cfg.dashboard, "out_summary", "dashboard_summary.json"))

    sections_emitted = 0
    if getattr(cfg.dashboard, "include_ensembles", True):
        export_dashboard_ensembles(labels_path=labels_path, out_path=out_ensembles, provenance=prov)
        log_artifact(out_ensembles, kind="dashboard_ensembles")
        sections_emitted += 1
    if getattr(cfg.dashboard, "include_labels", True):
        export_dashboard_labels(labels_path=labels_path, out_path=out_labels, provenance=prov)
        log_artifact(out_labels, kind="dashboard_labels")
        sections_emitted += 1
    if getattr(cfg.dashboard, "include_summary", True):
        export_dashboard_summary(labels_path=labels_path, out_path=out_summary, provenance=prov)
        log_artifact(out_summary, kind="dashboard_summary")
        sections_emitted += 1

    # 3) Finalize registry with acceptance metrics
    rep = read_json(report_path)
    kappa = float(rep.get("agreement", {}).get("kappa_mock", 0.0))
    coverage = rep.get("coverage", {})
    label_rows = load_jsonl(labels_path)

    # Acceptance rule (mock): non-empty labels, kappa above target, all sections emitted
    kappa_min = 0.6
    at = getattr(cfg.labeling, "agreement_targets", None)
    # agreement_targets is Optional[Mapping[str, float]]
    try:
        if at and "kappa_min" in at:
            kappa_min = float(at["kappa_min"])  # type: ignore[index]
    except Exception:
        pass

    accept_week7 = (len(label_rows) > 0) and (kappa >= kappa_min) and (sections_emitted >= 3)

    finalize_run(
        status="success",
        metrics_dict={
            "label_coverage": coverage,
            "agreement_kappa_mock": kappa,
            "total_labels": len(label_rows),
            "dashboard_sections_emitted": sections_emitted,
            "accept_week7": bool(accept_week7),
        },
    )


if __name__ == "__main__":
    main()