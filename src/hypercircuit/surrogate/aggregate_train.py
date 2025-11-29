from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hypercircuit.surrogate.train import fit_surrogates_for_family
from hypercircuit.utils.config import Config, stage_path
from hypercircuit.utils.io import SCHEMA_VERSION, write_json
from hypercircuit.utils.registry import current_run, log_artifact


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_all_families(cfg: Config) -> List[str]:
    # Priority: matrix.families -> dictionary.families -> discovery.week2_screening.top_families -> dataset.{task_family|name}
    mx = getattr(cfg, "matrix", None)
    if mx and getattr(mx, "families", None):
        return list(mx.families or [])
    dic = getattr(cfg, "dictionary", None)
    if dic and getattr(dic, "families", None):
        return list(dic.families or [])
    week2 = getattr(cfg.discovery, "week2_screening", None)
    if week2 and getattr(week2, "top_families", None):
        return list(week2.top_families or [])
    # fallback: dataset family/name
    fam = cfg.run.task_family or cfg.dataset.task_family or cfg.dataset.name
    return [fam]


def train_all_families(*, cfg: Config, run_dir: Path, families: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Orchestrate surrogate training for all configured families and emit a consolidated index.

    Steps:
      - Resolve families from config (matrix/dictionary/discovery fallbacks)
      - Call fit_surrogates_for_family() once with full family list (writes surrogates_params.jsonl, surrogates_index.json)
      - Collate per-family summaries into surrogates_index_all.json with run/paths metadata
      - Log consolidated artifact via registry

    Returns:
      {
        "families": [...],
        "paths": {"index_all": Path},
        "summary": {returned from fit_surrogates_for_family()}
      }
    """
    fams: List[str] = list(families) if families else _resolve_all_families(cfg)

    # Train surrogates across all families in one pass for deterministic artifacts
    train_res = fit_surrogates_for_family(run_dir=run_dir, cfg=cfg, families=fams)
    per_family = train_res.get("summary", {}).get("families", {})  # type: ignore[assignment]
    total_trained = int(train_res.get("summary", {}).get("total_trained", 0))  # type: ignore[arg-type]

    # Consolidated index with minimal provenance
    run_id, _ = current_run()
    params_path = train_res.get("paths", {}).get("params")
    index_path = train_res.get("paths", {}).get("index")

    rows: List[Dict[str, Any]] = []
    for fam in sorted(per_family.keys()):
        fb: Dict[str, Any] = dict(per_family.get(fam, {}))
        rows.append(
            {
                "family": fam,
                "n_surrogates_trained": int(fb.get("n_surrogates_trained", 0)),
                "median_cv_score": float(fb.get("median_cv_score", 0.0)),
                "median_calibration_error": float(fb.get("median_calibration_error", 0.0)),
                "run_ids": [run_id],
                "artifact_paths": {
                    "params": str(params_path) if params_path else None,
                    "index": str(index_path) if index_path else None,
                },
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "families": rows,
        "families_list": fams,
        "total_trained": total_trained,
    }
    out_path = stage_path(run_dir, "surrogates_index_all.json")
    write_json(out_path, payload)
    log_artifact(out_path, kind="surrogates_index_all", metadata={"families": fams, "total_trained": total_trained})

    return {"families": fams, "paths": {"index_all": out_path}, "summary": train_res.get("summary", {})}


__all__ = ["train_all_families"]