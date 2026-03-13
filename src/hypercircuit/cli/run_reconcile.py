from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from hypercircuit.feature_io.reconcile import reconcile_feature_space
from hypercircuit.logging.schema import EVENT_SCHEMA_VERSION
from hypercircuit.utils.config import Config, load_config, stage_path, apply_legacy_run_dir
from hypercircuit.utils.io import read_json, write_json
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.utils.failures import classify_exception


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconcile feature spaces into stable IDs.")
    p.add_argument("--config", nargs="+", required=False, default=["configs/base.yaml"])
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Legacy path form runs/<run_id> (overrides run.output_dir/run_id).")
    p.add_argument("--events-path", default=None, help="Optional direct path to logs.jsonl.")
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
        apply_legacy_run_dir(run_sec)

    _, run_dir = start_run(cfg_dict, stage_name="reconcile_features", config_paths=args.config)

    try:
        events_path = Path(args.events_path) if args.events_path else stage_path(run_dir, "logs.jsonl")
        if not events_path.exists():
            raise FileNotFoundError(f"events_path not found: {events_path}")

        method = cfg.reconciliation.method
        result = reconcile_feature_space(
            events_path=events_path,
            out_dir=run_dir,
            method=method,
            out_events=cfg.reconciliation.out_events,
        )
        for p in result.values():
            log_artifact(p, kind="reconciliation")

        # Emit a reconciled events manifest for downstream compatibility checks.
        if "events" in result:
            src_manifest_path = events_path.parent / "events_manifest.json"
            base_manifest = {}
            if src_manifest_path.exists():
                try:
                    base_manifest = dict(read_json(src_manifest_path))
                    base_manifest.pop("_schema_version", None)
                except Exception:
                    base_manifest = {}
            rec_manifest = {
                **base_manifest,
                "schema_version": EVENT_SCHEMA_VERSION,
                "reconciled": True,
                "member_granularity": "group",
                "source_events": str(events_path),
                "reconciliation": {
                    "method": method,
                    "alignment_path": str(result.get("alignment")),
                    "consensus_path": str(result.get("consensus")),
                },
            }
            out_manifest = stage_path(run_dir, cfg.reconciliation.out_manifest)
            write_json(out_manifest, rec_manifest)
            log_artifact(out_manifest, kind="events_manifest")

        finalize_run(
            status="success",
            metrics_dict={"method": method, "n_outputs": len(result)},
        )
    except Exception as exc:
        fail = classify_exception(exc)
        finalize_run(status="failed", metrics_dict={"failure_mode": fail.mode, "error": fail.message})
        raise


if __name__ == "__main__":
    main()
