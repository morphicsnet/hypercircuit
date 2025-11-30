from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List
from datetime import datetime, timezone

from hypercircuit.utils.config import Config, stage_path
from hypercircuit.utils.io import read_json, write_json, SCHEMA_VERSION


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_if_exists(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _gate1_flags(g1: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    if not g1:
        return False, ["gate1_missing"]
    final = str(g1.get("final") or "").lower()
    ok = final == "go"
    return bool(ok), ([] if ok else ["gate1_no_go"])


def _gate2_flags(g2: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    if not g2:
        return False, ["gate2_missing"]
    ok = bool(g2.get("accept_all", False))
    return bool(ok), ([] if ok else ["gate2_accept_all_false"])


def _gate3_flags(g3: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    if not g3:
        return False, ["gate3_missing"]
    ok = bool(g3.get("accept_all", False))
    return bool(ok), ([] if ok else ["gate3_accept_all_false"])


def _gate4_flags(g4: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    if not g4:
        return False, ["gate4_missing"]
    acc = g4.get("acceptance", {})
    ok = bool(acc.get("accept_gate4", g4.get("accept_gate4", False)))
    return bool(ok), ([] if ok else ["gate4_accept_false"])


def _mk_summary_md(
    *,
    snapshot_id: Optional[str],
    n_ensembles_frozen: Optional[int],
    kappa: Optional[float],
    sections_emitted: int,
    flags: Mapping[str, bool],
    accept_release: bool,
    reasons: Sequence[str],
) -> str:
    sid = snapshot_id or "n/a"
    nef = str(n_ensembles_frozen) if n_ensembles_frozen is not None else "n/a"
    kap = f"{kappa:.2f}" if isinstance(kappa, (float, int)) else "n/a"
    rz = ", ".join(reasons) if reasons else "none"
    lines = [
        f"# Week 8 Final Summary",
        "",
        f"- Snapshot ID: {sid}",
        f"- Ensembles Frozen: {nef}",
        f"- Label Agreement (kappa_mock): {kap}",
        f"- Dashboard sections emitted: {sections_emitted}",
        "",
        "## Gate Acceptance",
        f"- Gate 1 (discovery): {'PASS' if flags.get('accept_gate1') else 'FAIL'}",
        f"- Gate 2 (surrogate/causal): {'PASS' if flags.get('accept_gate2') else 'FAIL'}",
        f"- Gate 3 (safety edits): {'PASS' if flags.get('accept_gate3') else 'FAIL'}",
        f"- Gate 4 (robustness & necessity): {'PASS' if flags.get('accept_gate4') else 'FAIL'}",
        "",
        f"## Release Decision: {'ACCEPT' if accept_release else 'REJECT'}",
        f"- Reasons (if any): {rz}",
        "",
        f"_Generated at {_now_iso()}_",
    ]
    return "\n".join(lines) + "\n"


def assemble_final_report(
    *,
    cfg: Config,
    run_dir: Path,
    snapshot_id: Optional[str] = None,
    n_ensembles_frozen: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Consolidate Gate 1–4 outputs, labeling, and dashboard summaries into:
      - final_report.json
      - final_summary.md (concise overview)

    Returns:
      {
        "report_path": Path,
        "summary_path": Path,
        "flags": {"accept_gate1": bool, ..., "accept_release": bool},
        "reasons": List[str],
        "metrics": {...}
      }
    """
    # Locate expected upstream artifacts under run_dir (graceful if missing)
    g1_path = stage_path(run_dir, "gate1_go_no_go.json")
    g2_path = stage_path(run_dir, "gate2_report.json")
    g3_path = stage_path(run_dir, "gate3_report.json")
    g4_path = stage_path(run_dir, "gate4_report.json")
    lbl_path = stage_path(run_dir, "label_report.json")
    dash_summary_path = stage_path(run_dir, "dashboard_summary.json")

    g1 = _load_json_if_exists(g1_path)
    g2 = _load_json_if_exists(g2_path)
    g3 = _load_json_if_exists(g3_path)
    g4 = _load_json_if_exists(g4_path)
    lbl = _load_json_if_exists(lbl_path)
    dash = _load_json_if_exists(dash_summary_path)

    acc1, r1 = _gate1_flags(g1)
    acc2, r2 = _gate2_flags(g2)
    acc3, r3 = _gate3_flags(g3)
    acc4, r4 = _gate4_flags(g4)

    flags = {
        "accept_gate1": bool(acc1),
        "accept_gate2": bool(acc2),
        "accept_gate3": bool(acc3),
        "accept_gate4": bool(acc4),
    }

    reasons: List[str] = []
    for r in (r1 + r2 + r3 + r4):
        if r not in reasons:
            reasons.append(r)

    # Labeling metrics
    kappa = None
    try:
        kappa = float(lbl.get("agreement", {}).get("kappa_mock", None)) if lbl else None
    except Exception:
        kappa = None

    # Dashboard summary
    sections_emitted = 0
    try:
        summ = dash.get("summary", {}) if dash else {}
        seclist = summ.get("sections_emitted", []) or []
        if isinstance(seclist, (list, tuple)):
            sections_emitted = len(seclist)
    except Exception:
        sections_emitted = 0

    # Final decision: all gates True
    accept_release = bool(flags["accept_gate1"] and flags["accept_gate2"] and flags["accept_gate3"] and flags["accept_gate4"])

    # Compose JSON report
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "snapshot_id": snapshot_id,
        "gates": flags,
        "accept_release": bool(accept_release),
        "reasons": reasons,
        "labels": {
            "agreement_kappa_mock": kappa,
        },
        "dashboards": {
            "sections_emitted": sections_emitted,
        },
        "frozen": {
            "n_ensembles_frozen": int(n_ensembles_frozen) if n_ensembles_frozen is not None else None,
        },
        "paths": {
            "gate1_go_no_go": str(g1_path),
            "gate2_report": str(g2_path),
            "gate3_report": str(g3_path),
            "gate4_report": str(g4_path),
            "label_report": str(lbl_path),
            "dashboard_summary": str(dash_summary_path),
        },
    }

    out_json = stage_path(run_dir, "final_report.json")
    write_json(out_json, report)

    # Markdown summary
    md = _mk_summary_md(
        snapshot_id=snapshot_id,
        n_ensembles_frozen=n_ensembles_frozen,
        kappa=kappa,
        sections_emitted=sections_emitted,
        flags=flags,
        accept_release=accept_release,
        reasons=reasons,
    )
    out_md = stage_path(run_dir, "final_summary.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    return {
        "report_path": out_json,
        "summary_path": out_md,
        "flags": {**flags, "accept_release": bool(accept_release)},
        "reasons": reasons,
        "metrics": {
            "agreement_kappa_mock": kappa,
            "sections_emitted": sections_emitted,
            "n_ensembles_frozen": n_ensembles_frozen,
        },
    }