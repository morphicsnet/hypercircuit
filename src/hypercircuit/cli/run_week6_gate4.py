from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from hypercircuit.eval.matrix import run_matrix_evaluation
from hypercircuit.eval.robustness import run_robustness_evaluation
from hypercircuit.causal.necessity import run_higher_order_necessity
from hypercircuit.causal.gate4_reporting import assemble_gate4_report
from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.utils.io import load_jsonl, read_json, write_json


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 6: Finish remaining matrix cells, robustness & necessity, Gate 4 report (mock/deterministic).")
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
    p.add_argument("--no-remaining-only", action="store_true", help="If set, evaluate parity-0 (default), not the remaining parity-1 set.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)

    cfg_dict = cfg.model_dump()
    run_sec = cfg_dict.setdefault("run", {})
    # Legacy run-dir handling (consistent with other CLIs)
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

    # Wire dataset metadata (task_family, split) into run for registry
    ds = cfg.dataset
    run_sec["task_family"] = run_sec.get("task_family") or ds.task_family or ds.name
    if ds.split is not None:
        run_sec["split"] = ds.split

    # Start registry stage
    run_id, run_dir = start_run(cfg_dict, stage_name="week6_gate4", config_paths=args.config)

    # 1) Evaluate remaining matrix cells (parity-1) by default
    mat = run_matrix_evaluation(
        cfg=cfg,
        run_dir=run_dir,
        families=args.families,
        methods=args.methods,
        half_matrix=True,
        remaining_only=(not args.no_remaining_only),
    )
    # Resolve which artifact was written
    remaining_path = stage_path(run_dir, "remaining_matrix_results.jsonl")
    matrix_path = remaining_path if remaining_path.exists() else stage_path(run_dir, "matrix_results.jsonl")
    log_artifact(matrix_path, kind="remaining_matrix_results" if matrix_path == remaining_path else "matrix_results")

    # 2) Robustness evaluation over the selected cells
    rob = run_robustness_evaluation(cfg=cfg, run_dir=run_dir, families=args.families, methods=args.methods, use_remaining=True)
    rob_rows_path = rob["robustness_path"]
    rob_sum_path = rob["summary_path"]
    log_artifact(rob_rows_path, kind="matrix_robustness")
    log_artifact(rob_sum_path, kind="robustness_summary")

    # 3) Higher-order necessity ablation (mock)
    nec = run_higher_order_necessity(cfg=cfg, run_dir=run_dir, families=args.families)
    nec_path = nec["necessity_path"]
    log_artifact(nec_path, kind="necessity_results")

    # 4) Gate 4 final report
    g4 = assemble_gate4_report(cfg=cfg, run_dir=run_dir, families=args.families, methods=args.methods)
    g4_path = g4["report_path"]
    log_artifact(g4_path, kind="gate4_report")

    # Finalize registry summary metrics
    # Pull metrics from artifacts where possible
    g4_report = read_json(g4_path)
    acc = g4_report.get("acceptance", {})
    remaining = g4_report.get("remaining_cells", {})
    robust = g4_report.get("robustness", {})
    necess = g4_report.get("necessity", {})

    finalize_run(
        status="success",
        metrics_dict={
            "coverage_remaining_cells": float(remaining.get("coverage_remaining_cells", 0.0)),
            "effect_persistence_median": float(robust.get("effect_persistence_median", 0.0)),
            "seed_stability_median": float(robust.get("seed_stability_median", 0.0)),
            "necessity_drop_median": float(necess.get("causal_impact_drop_median", 0.0)),
            "accept_gate4": bool(acc.get("accept_gate4", False)),
            "reasons": list(acc.get("reasons", [])),
        },
    )


if __name__ == "__main__":
    main()