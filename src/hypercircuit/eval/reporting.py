from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hypercircuit.utils.config import Config, stage_path
from hypercircuit.utils.io import load_jsonl, write_json
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


def _families_from_results(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fams = sorted({str(r.get("family")) for r in rows if r.get("family") is not None})
    return fams


def _methods_from_results(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    meths = sorted({str(r.get("method")) for r in rows if r.get("method") is not None})
    return meths


def _bh_threshold(pvals: List[float], fdr: float) -> Tuple[float, List[bool]]:
    """
    Benjamini–Hochberg procedure (two-sided mock).
    Returns (threshold, discoveries_mask).
    """
    m = len(pvals)
    if m == 0:
        return 0.0, []
    idx = np.argsort(pvals)
    ps = np.array(sorted(pvals))
    thresh = 0.0
    k_star = -1
    for i, p in enumerate(ps, start=1):
        crit = (i / m) * fdr
        if p <= crit:
            k_star = i
            thresh = crit
    discover = [False] * m
    if k_star > 0:
        # all p <= p_(k*) are discoveries
        p_star = ps[k_star - 1]
        for j in range(m):
            discover[j] = pvals[j] <= p_star
    return float(thresh), discover


def _configured_grid_counts(cfg: Config, fams: Sequence[str], meths: Sequence[str]) -> Tuple[int, int]:
    config_total = len(fams) * len(meths)
    mx = getattr(cfg, "matrix", None)
    half = bool(getattr(mx, "half_matrix", True) if mx else True)
    expected = (config_total + 1) // 2 if half else config_total
    return config_total, expected


def _checksum_for_cellset(rows: Sequence[Mapping[str, Any]]) -> str:
    parts = [f"{str(r.get('family'))}|{str(r.get('method'))}|{str(r.get('checksum_id'))}" for r in rows]
    payload = ",".join(sorted(parts)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _top_cell_checksum_by_family(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, str], str]:
    by_fam: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        fam = str(r.get("family"))
        by_fam.setdefault(fam, []).append(r)
    per: Dict[str, str] = {}
    for fam, lst in by_fam.items():
        if not lst:
            continue
        top = max(lst, key=lambda d: float(d.get("effect_size", 0.0)))
        per[fam] = str(top.get("checksum_id") or "")
    # Aggregate checksum for convenience
    payload = ",".join(f"{k}:{v}" for k, v in sorted(per.items())).encode("utf-8")
    agg = hashlib.md5(payload).hexdigest()
    return per, agg


def assemble_interim_report(
    *,
    cfg: Config,
    run_dir: Path,
    matrix_path: Optional[Path] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Consolidate matrix_results.jsonl into interim_report.json with acceptance flags, FDR, and determinism checksums.

    Writes:
      - interim_report.json under run_dir

    Returns:
      {
        "report_path": Path,
        "summary": {...},
        "acceptance": {...},
        "families": {...}
      }
    """
    # Load results
    mpath = matrix_path or stage_path(run_dir, "matrix_results.jsonl")
    rows = load_jsonl(mpath) if Path(mpath).exists() else []

    fams = _families_from_results(rows)
    meths = _methods_from_results(rows)
    _, expected_cells = _configured_grid_counts(cfg, fams, meths)

    n_cells = len(rows)
    coverage_ratio = float(0.0 if expected_cells == 0 else n_cells / expected_cells)
    coverage_ok = bool(n_cells >= expected_cells)

    # Method medians
    by_method_vals: Dict[str, List[float]] = {}
    for r in rows:
        m = str(r.get("method"))
        by_method_vals.setdefault(m, []).append(float(r.get("effect_size", 0.0)))
    med_by_method = {m: _median(v) for m, v in by_method_vals.items()}

    med_hc = float(med_by_method.get("hypercircuit_full", 0.0))
    base_vals: List[float] = [v for m, v in med_by_method.items() if m != "hypercircuit_full"]
    base_med = float(_median(base_vals)) if base_vals else 0.0
    delta_effect = float(med_hc - base_med)
    # Specificity placeholder: map delta to [0,1] with soft clamp
    delta_specificity = float(max(0.0, min(1.0, 0.5 + 0.1 * delta_effect)))

    # Improvement rate across families
    improved = 0
    fam_set = sorted(set(fams))
    for fam in fam_set:
        vals_hc = [float(r.get("effect_size", 0.0)) for r in rows if r.get("family") == fam and r.get("method") == "hypercircuit_full"]
        vals_base = [float(r.get("effect_size", 0.0)) for r in rows if r.get("family") == fam and r.get("method") != "hypercircuit_full"]
        if vals_hc and vals_base and (_median(vals_hc) > _median(vals_base)):
            improved += 1
    pass_rate_improvement = float(0.0 if not fam_set else improved / len(fam_set))

    # Stability
    stab_vals = [float(r.get("stability", 0.0)) for r in rows]
    med_stability = _median(stab_vals)
    stability_ok = bool(med_stability >= 0.8)

    # FDR (BH) across all cells
    mx = getattr(cfg, "matrix", None)
    fdr_level = float(getattr(getattr(mx, "per_cell", None), "fdr", 0.10) if mx else 0.10)
    pvals = [float(r.get("p_value", 1.0)) for r in rows]
    bh_thr, mask = _bh_threshold(pvals, fdr=fdr_level)
    discoveries_idx = [i for i, flag in enumerate(mask) if flag]
    discoveries = [
        {"family": str(rows[i].get("family")), "method": str(rows[i].get("method")), "checksum_id": str(rows[i].get("checksum_id"))}
        for i in discoveries_idx
    ]

    # Acceptance flags
    effect_ok = bool((med_hc > 0.0) and (pass_rate_improvement >= 0.5))

    # Per-family summaries
    per_family: Dict[str, Any] = {}
    for fam in fam_set:
        frows = [r for r in rows if r.get("family") == fam]
        methods = sorted(set(str(r.get("method")) for r in frows))
        summary_methods: Dict[str, Any] = {}
        for m in methods:
            mvals = [float(r.get("effect_size", 0.0)) for r in frows if r.get("method") == m]
            mpass = [bool(r.get("passed", False)) for r in frows if r.get("method") == m]
            summary_methods[m] = {
                "median_effect": _median(mvals),
                "pass_rate": float(sum(1 for b in mpass if b) / len(mpass)) if mpass else 0.0,
            }
        per_family[fam] = {
            "methods": summary_methods,
            "n_cells": int(len(frows)),
        }

    # Determinism checksums
    cellset_md5 = _checksum_for_cellset(rows)
    top_per_fam, topagg_md5 = _top_cell_checksum_by_family(rows)

    # Compose report
    report = {
        "created_at": None,
        "config": cfg.model_dump(),
        "seeds": list(seeds or (cfg.run.seeds or [])),
        "families": per_family,
        "methods": sorted(meths),
        "n_cells": n_cells,
        "coverage_ratio": coverage_ratio,
        "topline": {
            "median_effect_hypercircuit": med_hc,
            "median_effect_baselines": base_med,
            "delta_effect": delta_effect,
            "delta_specificity": delta_specificity,
        },
        "fdr": {
            "level": fdr_level,
            "bh_threshold": float(bh_thr),
            "discoveries": discoveries,
            "n_discoveries": int(len(discoveries)),
        },
        "acceptance": {
            "coverage_ok": bool(coverage_ok),
            "effect_ok": bool(effect_ok),
            "stability_ok": bool(stability_ok),
        },
        "determinism": {
            "evaluated_set_md5": cellset_md5,
            "top_cells_by_family": top_per_fam,
            "top_cells_by_family_md5": topagg_md5,
        },
    }

    out_path = stage_path(run_dir, "interim_report.json")
    write_json(out_path, report)
    log_artifact(out_path, kind="interim_report", metadata={"n_cells": n_cells})

    return {"report_path": out_path, "report": report}