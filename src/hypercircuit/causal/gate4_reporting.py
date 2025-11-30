from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from hypercircuit.utils.config import Config, stage_path
from hypercircuit.utils.io import SCHEMA_VERSION, load_jsonl, read_json, write_json
from hypercircuit.utils.registry import log_artifact


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(x) for x in vals)
    n = len(s)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return float(0.5 * (s[m - 1] + s[m]))


def _configured_counts(cfg: Config, fams: Sequence[str], meths: Sequence[str]) -> Dict[str, int]:
    total = len(fams) * len(meths)
    # For remaining-half coverage expectation, use floor(total / 2) as the other parity half
    expected_remaining = total // 2
    return {"total": total, "expected_remaining": expected_remaining}


def _remaining_cells_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    parts = [f"{str(r.get('family'))}|{str(r.get('method'))}|{str(r.get('checksum_id'))}" for r in rows]
    payload = ",".join(sorted(parts)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _robustness_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    parts: List[str] = []
    for r in rows:
        fam = str(r.get("family"))
        met = str(r.get("method"))
        ep = float(r.get("effect_persistence", 0.0))
        ss = float(r.get("seed_stability", 0.0))
        od = float(r.get("ood_delta", 0.0))
        parts.append(f"{fam}|{met}|{ep:.8f}|{ss:.8f}|{od:.8f}")
    payload = ",".join(sorted(parts)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _necessity_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    parts: List[str] = []
    for r in rows:
        fam = str(r.get("family"))
        eid = str(r.get("ensemble_id"))
        cid = float(r.get("causal_impact_drop", 0.0))
        parts.append(f"{fam}|{eid}|{cid:.8f}")
    payload = ",".join(sorted(parts)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def assemble_gate4_report(
    *,
    cfg: Config,
    run_dir: Path,
    families: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Assemble Gate 4 report by consolidating:
      - remaining_matrix_results.jsonl (coverage, checksum)
      - matrix_robustness.jsonl + robustness_summary.json (persistence, stability)
      - necessity_results.jsonl (necessity drops, irreducibility)
    And compute acceptance according to cfg.causal.gate4.acceptance thresholds.
    """
    # Load remaining matrix results (optional if absent)
    rem_path = stage_path(run_dir, "remaining_matrix_results.jsonl")
    rem_rows = load_jsonl(rem_path) if rem_path.exists() else []
    fams = sorted({str(r.get("family")) for r in rem_rows}) if rem_rows else (
        list(getattr(getattr(cfg.dictionary, "families", None), "__iter__", lambda: [])()) or
        list(getattr(getattr(cfg.discovery, "week2_screening", None), "top_families", []) or [])
    )
    meths = sorted({str(r.get("method")) for r in rem_rows}) if rem_rows else (
        list(getattr(getattr(cfg.matrix, "methods", None), "__iter__", lambda: [])()) or []
    )
    counts = _configured_counts(cfg, fams, meths)
    n_cells = len(rem_rows)
    coverage_remaining = float(0.0 if counts["expected_remaining"] == 0 else n_cells / counts["expected_remaining"])
    remaining_cells_md5 = _remaining_cells_checksum(rem_rows) if rem_rows else ""

    # Load robustness artifacts
    rob_rows_path = stage_path(run_dir, "matrix_robustness.jsonl")
    rob_sum_path = stage_path(run_dir, "robustness_summary.json")
    rob_rows = load_jsonl(rob_rows_path) if rob_rows_path.exists() else []
    rob_sum = read_json(rob_sum_path) if rob_sum_path.exists() else {}
    ep_med = float(rob_sum.get("effect_persistence_median", 0.0))
    ss_med = float(rob_sum.get("seed_stability_median", 0.0))
    od_med = float(rob_sum.get("ood_delta_median", 0.0))
    robustness_md5 = _robustness_signature(rob_rows) if rob_rows else ""

    # Load necessity results
    nec_path = stage_path(run_dir, "necessity_results.jsonl")
    nec_rows = load_jsonl(nec_path) if nec_path.exists() else []
    nec_pred_med = _median([float(r.get("predictive_alignment_drop", 0.0)) for r in nec_rows])
    nec_causal_med = _median([float(r.get("causal_impact_drop", 0.0)) for r in nec_rows])
    irr_med = _median([float(r.get("irreducibility_score", 0.0)) for r in nec_rows])
    necessity_md5 = _necessity_signature(nec_rows) if nec_rows else ""

    # Thresholds
    acc = getattr(getattr(cfg.causal, "gate4", None), "acceptance", None)
    thr_drop = float(getattr(acc, "necessity_drop_min", 0.10) if acc else 0.10)
    thr_seed = float(getattr(acc, "seed_stability_min", 0.8) if acc else 0.8)
    thr_pers = float(getattr(acc, "effect_persistence_min", 0.7) if acc else 0.7)

    # Acceptance
    flags = {
        "effect_persistence_ok": bool(ep_med >= thr_pers),
        "seed_stability_ok": bool(ss_med >= thr_seed),
        "necessity_drop_ok": bool(nec_causal_med >= thr_drop),
    }
    accept_gate4 = bool(all(flags.values()))
    reasons: List[str] = []
    if not flags["effect_persistence_ok"]:
        reasons.append("effect_persistence_below_min")
    if not flags["seed_stability_ok"]:
        reasons.append("seed_stability_below_min")
    if not flags["necessity_drop_ok"]:
        reasons.append("necessity_drop_below_min")

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": None,
        "config": cfg.model_dump(),
        "families": fams,
        "methods": meths,
        "remaining_cells": {
            "n_cells": int(n_cells),
            "expected_cells": int(counts["expected_remaining"]),
            "coverage_remaining_cells": float(coverage_remaining),
            "remaining_cells_md5": remaining_cells_md5,
        },
        "robustness": {
            "effect_persistence_median": ep_med,
            "seed_stability_median": ss_med,
            "ood_delta_median": od_med,
            "signature_md5": robustness_md5,
        },
        "necessity": {
            "predictive_alignment_drop_median": float(nec_pred_med),
            "causal_impact_drop_median": float(nec_causal_med),
            "irreducibility_median": float(irr_med),
            "signature_md5": necessity_md5,
        },
        "acceptance": {
            **flags,
            "accept_gate4": bool(accept_gate4),
            "thresholds": {
                "necessity_drop_min": thr_drop,
                "seed_stability_min": thr_seed,
                "effect_persistence_min": thr_pers,
            },
            "reasons": reasons,
        },
        "determinism": {
            "remaining_cells_md5": remaining_cells_md5,
            "robustness_signature_md5": robustness_md5,
            "necessity_signature_md5": necessity_md5,
        },
    }
    out_path = stage_path(run_dir, "gate4_report.json")
    write_json(out_path, report)
    log_artifact(out_path, kind="gate4_report")
    return {"report_path": out_path, "report": report}