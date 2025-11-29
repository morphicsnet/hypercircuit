from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.io import write_json
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.surrogate.train import fit_surrogates_for_family


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train interpretable surrogates with CV+calibration.")
    p.add_argument("--config", nargs="+", required=False, default=["configs/base.yaml", "configs/surrogate.yaml", "configs/dictionary.yaml"])
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Legacy path form runs/<run_id> (overrides run.output_dir/run_id).")
    p.add_argument("--families", nargs="*", default=None, help="Families to train (defaults to discovery.week2_screening.top_families)")
    return p.parse_args(argv)


def _resolve_families(cfg: Config, families_cli: Optional[Sequence[str]]) -> List[str]:
    if families_cli:
        return list(families_cli)
    week2 = getattr(cfg.discovery, "week2_screening", None)
    if week2 and getattr(week2, "top_families", None):
        return list(week2.top_families)
    ds_name = cfg.dataset.task_family or cfg.dataset.name
    return [ds_name]


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

    # Stage metadata
    run_id, run_dir = start_run(cfg_dict, stage_name="surrogate_train", config_paths=args.config)
    families = _resolve_families(cfg, args.families)

    # Train surrogates per family; orchestrator handles reading ensembles + logs (mock)
    result = fit_surrogates_for_family(run_dir=run_dir, cfg=cfg, families=families)

    # Log artifacts
    params_path = result["paths"]["params"]
    index_path = result["paths"]["index"]
    log_artifact(params_path, kind="surrogates_params")
    log_artifact(index_path, kind="surrogates_index")

    # Persist a small summary snapshot to run_dir for convenience
    write_json(stage_path(run_dir, "surrogate_summary.json"), result["summary"])

    # Finalize with metrics
    finalize_run(status="success", metrics_dict={**result["summary"], "task_family": run_sec.get("task_family"), "split": run_sec.get("split")})
