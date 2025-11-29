from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from hypercircuit.surrogate.aggregate_train import train_all_families
from hypercircuit.eval.matrix import run_matrix_evaluation
from hypercircuit.eval.reporting import assemble_interim_report
from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.registry import start_run, finalize_run, log_artifact
from hypercircuit.utils.io import read_json, load_jsonl


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 4: Train surrogates for all families, run half-matrix evaluation, and assemble interim report.")
    p.add_argument(
        "--config",
        nargs="+",
        required=False,
        default=["configs/base.yaml", "configs/surrogate.yaml", "configs/causal.yaml", "configs/dictionary.yaml", "configs/matrix.yaml"],
    )
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Legacy path form runs/<run_id> (overrides run.output_dir/run_id).")
    p.add_argument("--families", nargs="*", default=None, help="Optional explicit families list (else from config).")
    p.add_argument("--methods", nargs="*", default=None, help="Optional explicit methods list (else from matrix config).")
    p.add_argument("--full-matrix", action="store_true", help="Evaluate full matrix instead of half (deterministic parity).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)

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

    # Wire dataset metadata (task_family, split) into run for registry
    ds = cfg.dataset
    run_sec["task_family"] = run_sec.get("task_family") or ds.task_family or ds.name
    if ds.split is not None:
        run_sec["split"] = ds.split

    # Start registry stage
    run_id, run_dir = start_run(cfg_dict, stage_name="week4_interim", config_paths=args.config)

    # 1) Train surrogates across all families
    agg_res = train_all_families(cfg=cfg, run_dir=run_dir, families=args.families)
    idx_all_path = agg_res["paths"]["index_all"]
    log_artifact(idx_all_path, kind="surrogates_index_all")

    # 2) Run matrix evaluation (half by default unless --full-matrix)
    mat_summary = run_matrix_evaluation(
        cfg=cfg,
        run_dir=run_dir,
        families=args.families,
        methods=args.methods,
        half_matrix=(not args.full_matrix),
    )
    matrix_path = mat_summary["matrix_path"]
    log_artifact(matrix_path, kind="matrix_results")

    # 3) Assemble interim report
    rep = assemble_interim_report(cfg=cfg, run_dir=run_dir, matrix_path=matrix_path, seeds=cfg.run.seeds)
    report_path = rep["report_path"]
    log_artifact(report_path, kind="interim_report")

    # Finalize with summary metrics + acceptance flags
    report = rep["report"]
    acceptance = report.get("acceptance", {})
    finalize_run(
        status="success",
        metrics_dict={
            "n_cells_evaluated": int(mat_summary.get("n_cells_evaluated", 0)),
            "coverage_ratio": float(mat_summary.get("coverage_ratio", 0.0)),
            "median_effect_hypercircuit": float(mat_summary.get("median_effect_hypercircuit", 0.0)),
            "pass_rate_improvement": float(mat_summary.get("pass_rate_improvement", 0.0)),
            "fdr_level": float(mat_summary.get("fdr_level", 0.10)),
            "coverage_ok": bool(acceptance.get("coverage_ok", False)),
            "effect_ok": bool(acceptance.get("effect_ok", False)),
            "stability_ok": bool(acceptance.get("stability_ok", False)),
        },
    )


if __name__ == "__main__":
    main()