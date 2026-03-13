from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.logging.real_activations import RealActivationLogger
from hypercircuit.logging.schema import EventContext, EVENT_SCHEMA_VERSION
from hypercircuit.utils.config import Config, load_config, stage_path, apply_legacy_run_dir
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.utils.failures import classify_exception
from hypercircuit.utils.provenance import hash_json
from hypercircuit.utils.io import write_json
from hypercircuit.utils.seed import active_seed
from hypercircuit.utils.validate import validate_config_for_stage


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
        apply_legacy_run_dir(run_sec)

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

    mode = str(getattr(cfg.logging, "mode", "mock"))
    if mode == "mock" and not bool(getattr(cfg.run, "mock", True)):
        mode = "real"
    source_kind = cfg.logging.source_kind or ("mock" if mode == "mock" else "hf_local")
    event_ctx = EventContext(
        run_id=run_id,
        source_kind=source_kind,
        feature_space_id=cfg.sae.feature_space_id or ("mock" if mode == "mock" else "sae-unknown"),
        feature_space_version=cfg.sae.feature_space_version,
        dictionary_id=cfg.sae.dictionary_id,
        dictionary_version=cfg.sae.dictionary_version,
        dictionary_type=cfg.sae.dictionary_type if mode == "real" else "mock",
        run_intent=cfg.run.intent,
        task_family=run_sec.get("task_family"),
        prompt_family=cfg.dataset.prompt_family,
        split=run_sec.get("split"),
        capability_tag=cfg.dataset.capability_tag,
        safety_tag=cfg.dataset.safety_tag,
    )

    try:
        validate_config_for_stage(cfg, stage="logging")
        if mode == "mock":
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
                event_ctx=event_ctx,
            )
            metrics = logger.run(out_path=out_path, n_samples=cfg.dataset.n_samples)
        else:
            metrics = RealActivationLogger(
                model_cfg=cfg.model.model_dump(),
                sae_cfg=cfg.sae.model_dump(),
                dataset_cfg=cfg.dataset.model_dump(),
                logging_cfg={**cfg.logging.model_dump(), "run_id": run_id, "run_intent": cfg.run.intent},
                seed=seed_val,
                event_ctx=event_ctx,
            ).run(out_path=out_path)

        # Event density sanity check line (mock only)
        expected = cfg.logging.expected_event_density_range
        density = float(metrics["events_per_token"])
        within = (density >= expected[0]) and (density <= expected[1])
        print(f"[sanity] events_per_token={density:.2f} expected=[{expected[0]:.2f},{expected[1]:.2f}] within={within}")

        # Write events manifest
        events_manifest = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "run_id": run_id,
            "run_intent": cfg.run.intent,
            "source_kind": source_kind,
            "member_granularity": cfg.logging.member_granularity,
            "feature_space": metrics.get("feature_space")
            or {
                "feature_space_id": event_ctx.feature_space_id,
                "feature_space_version": event_ctx.feature_space_version,
                "feature_space_type": event_ctx.dictionary_type or "mock",
            },
            "model": metrics.get("model_info"),
            "tokenizer": metrics.get("tokenizer_info"),
            "dataset": {
                "source": cfg.dataset.source,
                "name": cfg.dataset.name,
                "hf_name": cfg.dataset.hf_name,
                "hf_split": cfg.dataset.hf_split,
                "path": cfg.dataset.path,
                "text_field": cfg.dataset.text_field,
                "label_field": cfg.dataset.label_field,
                "max_samples": cfg.dataset.max_samples,
            },
            "tags": {
                "task_family": run_sec.get("task_family"),
                "prompt_family": cfg.dataset.prompt_family,
                "split": run_sec.get("split"),
                "capability_tag": cfg.dataset.capability_tag,
                "safety_tag": cfg.dataset.safety_tag,
            },
        }
        manifest_path = stage_path(run_dir, "events_manifest.json")
        write_json(manifest_path, events_manifest)
        log_artifact(manifest_path, kind="events_manifest")

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
            manifest_updates={
                "provenance": {
                    "model": metrics.get("model_info"),
                    "tokenizer": metrics.get("tokenizer_info"),
                    "feature_space": metrics.get("feature_space"),
                    "dataset": {
                        "source": cfg.dataset.source,
                        "name": cfg.dataset.name,
                        "hf_name": cfg.dataset.hf_name,
                        "hf_split": cfg.dataset.hf_split,
                        "path": cfg.dataset.path,
                    },
                    "seeds": run_sec.get("seeds"),
                    "batching": {
                        "batch_size": cfg.model.batch_size,
                        "max_length": cfg.model.max_length,
                    },
                    "activation_targets": [t.model_dump() for t in (cfg.model.targets or [])],
                    "config_checksum": hash_json(cfg_dict),
                }
            },
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
