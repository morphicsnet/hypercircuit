from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.io import load_jsonl, save_jsonl, write_json
from hypercircuit.utils.registry import finalize_run, log_artifact, start_run
from hypercircuit.steering.policies import apply_edit_schedule, propose_edit_plan, rollback_point
from hypercircuit.steering.monitor import monitor_loop
from hypercircuit.causal.safety_reporting import assemble_gate3_report
from hypercircuit.steering.edits import simulate_impact


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 5: Safety edit planning, monitoring, and Gate 3 report (mock/deterministic).")
    p.add_argument(
        "--config",
        nargs="+",
        required=False,
        default=["configs/base.yaml", "configs/editing.yaml", "configs/causal.yaml", "configs/dictionary.yaml"],
    )
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Legacy path form runs/<run_id> (overrides run.output_dir/run_id).")
    p.add_argument("--families", nargs="*", default=None, help="Optional explicit families list (else from config).")
    return p.parse_args(argv)


def _resolve_families(cfg: Config, families_arg: Optional[List[str]]) -> List[str]:
    if families_arg:
        fams = list(dict.fromkeys(families_arg))
    elif getattr(cfg.editing, "families", None):
        fams = list(dict.fromkeys(cfg.editing.families or []))
    elif getattr(cfg.dictionary, "families", None):
        fams = list(dict.fromkeys(cfg.dictionary.families or []))
    else:
        fams = list(dict.fromkeys(getattr(cfg.discovery.week2_screening, "top_families", []) or []))
    if not fams:
        fams = ["sycophancy", "jailbreak"]
    return sorted(fams)


def _thresholds_from_cfg(cfg: Config) -> Dict[str, float]:
    return {
        "benign_deg_pct_max": float(getattr(cfg.editing, "benign_deg_pct_max", 0.5)),
        "specificity_min_ratio": float(getattr(cfg.editing, "specificity_min_ratio", 1.2)),
        "target_reduction_pct": float(getattr(cfg.editing, "target_reduction_pct", 0.20)),
    }


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
    run_id, run_dir = start_run(cfg_dict, stage_name="week5_safety", config_paths=args.config)

    # Resolve families
    families = _resolve_families(cfg, args.families)

    # Propose edit plan (deterministic ranking and scales)
    plan = propose_edit_plan(cfg=cfg, run_dir=run_dir, families=families)
    step_schedule = list(getattr(cfg.editing, "step_schedule", [0.1, 0.2, 0.3]))
    max_edit_scale = float(getattr(cfg.editing, "max_edit_scale", 0.3))
    applied = apply_edit_schedule(
        proposed_scales=plan.proposed_scales,
        step_schedule=step_schedule,
        max_edit_scale=max_edit_scale,
    )

    # Optional pre-edit checkpoint for rollback
    checkpoint = rollback_point(run_dir)
    chk_path = stage_path(run_dir, "pre_edit_checkpoint.json")
    write_json(chk_path, checkpoint)
    log_artifact(chk_path, kind="checkpoint")

    # Simulate (dry-run) predicted impact per step per family (mock)
    sim_by_family: Dict[str, List[Dict[str, float]]] = {}
    n_feat = int(getattr(cfg.dataset, "n_features", 8))
    dummy_w = np.ones((n_feat,), dtype=float)
    for fam in plan.families:
        sim_by_family[fam] = simulate_impact(dummy_w, step_schedule=step_schedule, seed=plan.seed)

    # Write safety_edit_plans.jsonl
    plans_path = stage_path(run_dir, "safety_edit_plans.jsonl")
    plan_record: Dict[str, Any] = {
        "families": plan.families,
        "proposed_scales": plan.proposed_scales,
        "ranking": plan.ranking,
        "applied_scales_per_step": applied,
        "step_schedule": step_schedule,
        "seed": plan.seed,
        "simulation": sim_by_family,
    }
    save_jsonl(plans_path, [plan_record])
    log_artifact(plans_path, kind="safety_edit_plans", metadata={"n_families": len(plan.families)})

    # Monitor loop (writes safety_eval_results.jsonl internally)
    mon_sum = monitor_loop(
        cfg=cfg,
        run_dir=run_dir,
        families=plan.families,
        applied_scales_per_step=applied,
    )
    results_path = Path(mon_sum.get("results_path") or stage_path(run_dir, "safety_eval_results.jsonl"))
    log_artifact(results_path, kind="safety_eval_results")

    # Assemble Gate 3 report
    rows = load_jsonl(results_path) if results_path.exists() else []
    thr = _thresholds_from_cfg(cfg)
    g3 = assemble_gate3_report(
        cfg_snapshot=cfg_dict,
        seeds=cfg.run.seeds,
        results_rows=rows,
        thresholds=thr,
    )
    g3_path = stage_path(run_dir, "gate3_report.json")
    write_json(g3_path, g3)
    log_artifact(g3_path, kind="gate3_report")

    # Finalize registry with summary metrics
    finalize_run(
        status="success",
        metrics_dict={
            "n_families": int(len(plan.families)),
            "n_steps": int(mon_sum.get("n_steps", 0)),
            "gate3_accept_all": bool(g3.get("accept_all", False)),
            "breached": bool(mon_sum.get("breached", False)),
        },
    )


if __name__ == "__main__":
    main()