from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hypercircuit.utils.io import SCHEMA_VERSION


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(x) for x in vals)
    n = len(s)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return float(0.5 * (s[m - 1] + s[m]))


def _rate_true(bools: Sequence[bool]) -> float:
    n = len(bools)
    if n == 0:
        return 0.0
    return float(sum(1 for b in bools if b) / n)


def _top10_checksum(ids: Sequence[str]) -> str:
    payload = ",".join(ids).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _family_list(items: Sequence[Mapping[str, Any]], key: str = "family") -> List[str]:
    fams: List[str] = []
    for it in items:
        fam = str(it.get(key)) if it.get(key) is not None else None
        if fam is not None:
            fams.append(fam)
    return sorted(set(fams))


def assemble_gate2_report(
    *,
    results: Sequence[Mapping[str, Any]],
    params: Sequence[Mapping[str, Any]],
    ensembles: Sequence[Mapping[str, Any]],
    cfg_snapshot: Mapping[str, Any],
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Build a Gate 2 report summarizing surrogate training and targeted causal evals.

    Returns a JSON-serializable mapping with per-family metrics, acceptance flags,
    determinism checksums, and configuration snapshot.
    """
    # Families observed across artifacts
    fams = sorted(set(_family_list(results) + _family_list(params) + _family_list(ensembles)))

    # Index helpers
    by_family_params: Dict[str, List[Mapping[str, Any]]] = {f: [] for f in fams}
    for r in params:
        f = str(r.get("family"))
        if f in by_family_params:
            by_family_params[f].append(r)

    by_family_results: Dict[str, List[Mapping[str, Any]]] = {f: [] for f in fams}
    for r in results:
        f = str(r.get("family"))
        if f in by_family_results:
            by_family_results[f].append(r)

    families_block: Dict[str, Any] = {}
    acceptance_flags: Dict[str, bool] = {}

    for fam in fams:
        p_list = by_family_params.get(fam, [])
        r_list = by_family_results.get(fam, [])

        # Training summaries
        n_sur_trained = len(p_list)
        cv_vals = [float(x.get("cv_score", 0.0)) for x in p_list]
        cal_vals = [float(x.get("calibration_error", 0.0)) for x in p_list]
        med_cv = _median(cv_vals)
        med_cal = _median(cal_vals)

        # Eval summaries
        n_eval = len(r_list)
        deltas = [float(x.get("causal_impact_delta", 0.0)) for x in r_list]
        med_delta = _median(deltas)
        suff_rate = _rate_true([float(x.get("sufficiency_reinstatement", 0.0)) > 0.0 for x in r_list])
        min_rate = _rate_true([bool(x.get("minimality_pass", False)) for x in r_list])

        # Determinism: top-10 ensemble ids by CV (from params)
        sorted_by_cv = sorted(p_list, key=lambda d: -float(d.get("cv_score", 0.0)))
        top_ids = [str(x.get("ensemble_id")) for x in sorted_by_cv[:10] if x.get("ensemble_id") is not None]
        checksum = _top10_checksum(top_ids)

        # Acceptance (mock criteria)
        accept = (n_eval >= 10) and (med_delta > 0.0) and (suff_rate >= 0.7)
        acceptance_flags[fam] = bool(accept)

        families_block[fam] = {
            "n_surrogates_trained": int(n_sur_trained),
            "median_cv_score": float(med_cv),
            "median_calibration_error": float(med_cal),
            "n_ensembles_evaluated": int(n_eval),
            "median_causal_impact_delta": float(med_delta),
            "sufficiency_reinstatement_rate": float(suff_rate),
            "minimality_pass_rate": float(min_rate),
            "top10_signature_md5": checksum,
            "top10_ids": top_ids,
            "accept": bool(accept),
        }

    accept_all = all(acceptance_flags.values()) if families_block else False

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "families": families_block,
        "acceptance": acceptance_flags,
        "accept_all": bool(accept_all),
        "config": dict(cfg_snapshot),
        "seeds": list(seeds or []),
    }
    return report


__all__ = ["assemble_gate2_report"]