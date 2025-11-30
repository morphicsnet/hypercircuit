from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.utils.seed import active_seed


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run mock activation logging.")
    p.add_argument("--config", nargs="+", required=False, default=["configs/base.yaml", "configs/logging.yaml"])
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Legacy path form runs/<run_id> (overrides run.output_dir/run_id).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)

    # Prepare config dict for registry with output_dir & run_id
    cfg_dict = cfg.model_dump()
    run_sec = cfg_dict.setdefault("run", {})
    if args.run_dir:
        p = Path(args.run_dir)
        run_sec["output_dir"] = str(p.parent)
        run_sec["run_id"] = p.name
    else:
        # fallback from legacy run_dir if present
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

    run_id, run_dir = start_run(cfg_dict, stage_name="logging", config_paths=args.config)
    out_path = stage_path(run_dir, "logs.jsonl")

    # Seed policy: choose active seed index 0 for Week 1 baseline
    seed_val = active_seed(run_sec.get("seeds"), stage_idx=0)

    # Resolve dataset toggles for Week 7 expansion
    ds_family = run_sec.get("task_family")
    ds_top = bool(getattr(cfg.dataset, "top_behaviors", False))
    top_fams = list(getattr(cfg.logging, "top_behaviors_families", []) or getattr(cfg.discovery.week2_screening, "top_families", []))
    layers24 = list(getattr(cfg.logging, "layers_profile_24", []) or [])

    logger = ActivationLogger(
        tokens_per_sample=cfg.logging.tokens_per_sample,
        threshold=cfg.logging.threshold,
        hysteresis=cfg.logging.hysteresis,
        sparsity=cfg.logging.sparsity,
        n_features=cfg.dataset.n_features,
        seed=seed_val,
        instrumented_layers=cfg.logging.instrumented_layers,
        token_window=cfg.logging.token_window,
        thresholds=cfg.logging.thresholds,
        node_types=cfg.logging.node_types.model_dump(),
        # Week 7 toggles
        top_behaviors_only=bool(getattr(cfg.logging, "top_behaviors_only", False)),
        top_behaviors_families=top_fams,
        layers_profile_24=layers24,
        dataset_family=ds_family,
        dataset_top_behaviors=ds_top,
    )
    metrics = logger.run(out_path=out_path, n_samples=cfg.dataset.n_samples)

    # Event density sanity check line
    expected = cfg.logging.expected_event_density_range
    density = float(metrics["events_per_token"])
    within = (density >= expected[0]) and (density <= expected[1])
    print(f"[sanity] events_per_token={density:.2f} expected=[{expected[0]:.2f},{expected[1]:.2f}] within={within}")

    log_artifact(out_path, kind="events", metadata={"n_samples": cfg.dataset.n_samples})
    # Record extended metrics to manifest summary
    finalize_run(
        status="success",
        metrics_dict={
            **metrics,
            "expected_low": expected[0],
            "expected_high": expected[1],
            "within_band": within,
            "task_family": run_sec.get("task_family"),
            "split": run_sec.get("split"),
            "n_samples": cfg.dataset.n_samples,
            "n_features": cfg.dataset.n_features,
            "seed": seed_val,
            # Week 7 acceptance metric
            "n_layers_used": int(metrics.get("n_layers", 0)),
        },
    )


if __name__ == "__main__":
    main()
