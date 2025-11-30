from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run
from hypercircuit.utils.io import read_json
from hypercircuit.dictionary.freeze import freeze_dictionary, assemble_release_manifest
from hypercircuit.eval.final_report import assemble_final_report


def _parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 8: Freeze dictionary, assemble final reports, and emit release manifest (mock/deterministic).")
    p.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/discovery.yaml", "configs/dictionary.yaml"],
        help="Config files merged in order (base, discovery, dictionary).",
    )
    p.add_argument("-o", "--override", action="append", default=[], help="Config overrides dotted.key=value.")
    p.add_argument("--run-dir", default=None, help="Optional runs/<run_id> override (bypasses run.output_dir/run_id from config).")
    p.add_argument("--families", nargs="*", default=None, help="Optional families subset for freeze (default resolves from config).")
    return p.parse_args(argv)


def _prepare_run(cfg: Config, args: argparse.Namespace) -> tuple[Dict[str, Any], Path]:
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

    # Wire dataset metadata (task_family, split)
    ds = cfg.dataset
    run_sec["task_family"] = run_sec.get("task_family") or ds.task_family or ds.name
    if ds.split is not None:
        run_sec["split"] = ds.split

    _, run_dir = start_run(cfg_dict, stage_name="week8_release", config_paths=args.config)
    return cfg_dict, run_dir


def _write_scaling_doc(*, run_dir: Path, fname: str, metrics: Mapping[str, Any]) -> Path:
    """
    Deterministic, templated scaling recommendation doc under run_dir/docs/.
    """
    out_path = run_dir / "docs" / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_id = str(metrics.get("snapshot_id", "n/a"))
    n_ensembles = metrics.get("n_ensembles_frozen")
    n_ensembles_s = "n/a" if n_ensembles is None else str(int(n_ensembles))
    accept_flags = metrics.get("flags", {})
    accept_release = bool(metrics.get("accept_release", False))
    bundle_checksum = str(metrics.get("release_bundle_checksum", ""))

    budgets = metrics.get("budgets", {})
    storage_bytes = int(budgets.get("storage_bytes", 0))
    storage_mb = storage_bytes / (1024 * 1024) if storage_bytes else 0.0
    compute_stub = str(budgets.get("compute_stub", "deterministic-mock"))

    reasons = metrics.get("reasons", []) or []
    reasons_txt = "- " + "\n- ".join(reasons) if reasons else "None"

    content = f"""# Scaling Recommendation: V1 → V1.5 → V2

Snapshot
- snapshot_id: {snapshot_id}
- n_ensembles_frozen: {n_ensembles_s}
- release_bundle_checksum: {bundle_checksum}

Gate acceptance (V1 status)
- Gate 1: {"PASS" if accept_flags.get("accept_gate1") else "FAIL"}
- Gate 2: {"PASS" if accept_flags.get("accept_gate2") else "FAIL"}
- Gate 3: {"PASS" if accept_flags.get("accept_gate3") else "FAIL"}
- Gate 4: {"PASS" if accept_flags.get("accept_gate4") else "FAIL"}
- Release decision: {"ACCEPT" if accept_release else "REJECT"}

Budgets (mock/deterministic)
- Storage (current run artifacts): {storage_mb:.2f} MB
- Compute envelope (stub): {compute_stub}

Observed outcomes and gaps
{reasons_txt}

Migration steps
- V1 → V1.5:
  - Expand evaluation coverage to all configured families; address failing gates from reasons.
  - Tighten stability/necessity thresholds as needed; re-run Gate 4 to confirm persistence.
  - Grow dashboard surface: add per-family drill-down and error budgets.
- V1.5 → V2:
  - Scale dataset variants and seeds; increase CV folds for surrogates.
  - Add regression guardrails: freeze API + manifests, enforce bundle checksums in CI.
  - Productionize run registry with artifact lineage queries and retention policy.

Resource envelopes
- Data: maintain 2× overhead on storage for concurrent runs and snapshots.
- Compute: allocate capacity for 2× current cell grid with half-matrix parity.
- People/time: reserve 1 sprint for remediation of reasons above.

Provenance
- This note is generated deterministically during Week 8 release orchestration.
- snapshot_id: {snapshot_id}
- bundle_checksum: {bundle_checksum}
"""
    out_path.write_text(content)
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse(argv)
    cfg: Config = load_config(args.config, args.override)
    cfg_dict, run_dir = _prepare_run(cfg, args)

    # 1) Freeze dictionary
    fr = freeze_dictionary(cfg=cfg, run_dir=run_dir, families=args.families)
    frozen_path = Path(fr["frozen_path"])
    snapshot_id = str(fr.get("snapshot_id"))
    n_frozen = int(fr.get("n_ensembles", 0))
    log_artifact(frozen_path, kind="dictionary_frozen", metadata={"snapshot_id": snapshot_id, "n_ensembles": n_frozen})

    # 2) Final report + summary (aggregates Gates 1-4 + labels + dashboard summary)
    fin = assemble_final_report(cfg=cfg, run_dir=run_dir, snapshot_id=snapshot_id, n_ensembles_frozen=n_frozen)
    final_report_path = Path(fin["report_path"])
    final_summary_path = Path(fin["summary_path"])
    flags: Mapping[str, bool] = fin["flags"]
    reasons = list(fin.get("reasons", []))

    log_artifact(final_report_path, kind="final_report")
    log_artifact(final_summary_path, kind="final_summary")

    # 3) Scaling recommendation doc (templated)
    scaling_name = getattr(getattr(cfg, "release", None), "out_dir_names", None)
    scaling_fname = getattr(scaling_name, "scaling_doc", "SCALING_RECOMMENDATION.md") if scaling_name else "SCALING_RECOMMENDATION.md"

    # quick budgets: size of key artifacts
    storage_bytes = 0
    for p in [frozen_path, final_report_path, final_summary_path]:
        if p.exists():
            try:
                storage_bytes += p.stat().st_size
            except Exception:
                pass
    scaling_metrics = {
        "snapshot_id": snapshot_id,
        "n_ensembles_frozen": n_frozen,
        "flags": flags,
        "accept_release": bool(flags.get("accept_release", False)),
        "release_bundle_checksum": "",  # populated post-manifest
        "budgets": {"storage_bytes": storage_bytes, "compute_stub": "deterministic-mock"},
        "reasons": reasons,
    }
    scaling_path = _write_scaling_doc(run_dir=run_dir, fname=scaling_fname, metrics=scaling_metrics)
    log_artifact(scaling_path, kind="scaling_recommendation")

    # 4) Assemble release manifest (with bundle checksum)
    artifacts: Dict[str, Path] = {
        "ensemble_dictionary_frozen": frozen_path,
        "final_report": final_report_path,
        "final_summary": final_summary_path,
        "scaling_recommendation": scaling_path,
    }
    acceptance_flags = {
        "accept_gate1": bool(flags.get("accept_gate1", False)),
        "accept_gate2": bool(flags.get("accept_gate2", False)),
        "accept_gate3": bool(flags.get("accept_gate3", False)),
        "accept_gate4": bool(flags.get("accept_gate4", False)),
    }
    man = assemble_release_manifest(cfg=cfg, run_dir=run_dir, snapshot_id=snapshot_id, artifacts=artifacts, acceptance_flags=acceptance_flags, reasons=reasons)
    manifest_path = Path(man["manifest_path"])
    bundle_checksum = str(man.get("release_bundle_checksum", ""))
    log_artifact(manifest_path, kind="release_manifest", metadata={"bundle_checksum": bundle_checksum})

    # Update scaling doc checksum info (best-effort)
    try:
        scaling_metrics["release_bundle_checksum"] = bundle_checksum
        _ = _write_scaling_doc(run_dir=run_dir, fname=scaling_fname, metrics=scaling_metrics)
    except Exception:
        pass

    # 5) Finalize registry with release metrics
    finalize_run(
        status="success",
        metrics_dict={
            "snapshot_id": snapshot_id,
            "n_ensembles_frozen": n_frozen,
            "accept_gate1": bool(acceptance_flags["accept_gate1"]),
            "accept_gate2": bool(acceptance_flags["accept_gate2"]),
            "accept_gate3": bool(acceptance_flags["accept_gate3"]),
            "accept_gate4": bool(acceptance_flags["accept_gate4"]),
            "accept_release": bool(flags.get("accept_release", False)),
            "release_bundle_checksum": bundle_checksum,
            "reasons": reasons,
        },
    )


if __name__ == "__main__":
    main()