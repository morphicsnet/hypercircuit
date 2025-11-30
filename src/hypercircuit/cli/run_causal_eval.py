from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from hypercircuit.causal.harness import CausalHarness
from hypercircuit.causal.reporting import assemble_gate2_report
from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.io import load_jsonl, read_json, write_json, save_jsonl
from hypercircuit.utils.registry import start_run, log_artifact, finalize_run


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run targeted causal evaluations and emit Gate 2 report.")
    p.add_argument("--config", nargs="+", required=False, default=["configs/base.yaml", "configs/causal.yaml", "configs/dictionary.yaml"])
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--run-dir", default=None, help="Legacy path form runs/<run_id> (overrides run.output_dir/run_id).")
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

    run_id, run_dir = start_run(cfg_dict, stage_name="causal_eval", config_paths=args.config)

    # Load artifacts
    events_path = stage_path(run_dir, "logs.jsonl")
    ensembles_path = stage_path(run_dir, "ensembles.jsonl")
    sur_params_path = stage_path(run_dir, "surrogates_params.jsonl")

    events = load_jsonl(events_path) if events_path.exists() else []
    ensembles = load_jsonl(ensembles_path) if ensembles_path.exists() else []
    params = load_jsonl(sur_params_path) if sur_params_path.exists() else []

    # Build an index: best CV per ensemble
    by_ensemble: Dict[str, Dict[str, Any]] = {}
    for r in params:
        eid = r.get("ensemble_id")
        if not eid:
            continue
        cur = by_ensemble.get(eid)
        if (cur is None) or (float(r.get("cv_score", 0.0)) > float(cur.get("cv_score", 0.0))):
            by_ensemble[eid] = r

    # Select top-K per family by CV
    n_k = int(getattr(cfg.causal, "n_ensembles_per_family", 10))
    selected: List[Dict[str, Any]] = []
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for e in params:
        fam = e.get("family")
        by_family.setdefault(fam, []).append(e)
    for fam, recs in by_family.items():
        recs.sort(key=lambda d: -float(d.get("cv_score", 0.0)))
        selected.extend(recs[:n_k])

    # Assemble mock held-out data (reuse same events deterministically)
    n_samples = len(events)
    # Build simple per-member presence matrix per sample (node_type scoped)
    def _sample_features_for_members(members: List[str]) -> np.ndarray:
        X = np.zeros((n_samples, len(members)), dtype=float)
        # Map sample -> set of node_types present
        by_sample: Dict[int, set] = {}
        for ev in events:
            s = int(ev.get("sample_id", 0))
            nt = str(ev.get("node_type", ""))
            by_sample.setdefault(s, set()).add(nt)
        for i in range(n_samples):
            present = by_sample.get(i, set())
            for j, m in enumerate(members):
                X[i, j] = 1.0 if m in present else 0.0
        return X

    harness = CausalHarness(ablation_strength=cfg.causal.ablation_strength)
    results: List[Dict[str, Any]] = []
    for rec in selected:
        eid = rec.get("ensemble_id")
        fam = rec.get("family")
        members = []
        # recover members from ensembles.jsonl
        for e in ensembles:
            if e.get("id") == eid:
                members = list(e.get("members", []))
                break
        if not members:
            continue
        X = _sample_features_for_members(members)
        y = X.sum(axis=1) + 0.01  # synthetic behavior
        # reconstruct surrogate
        state = rec.get("model_state") or {}
        weights = np.array(state.get("weights") or [0.0] * len(members), dtype=float)
        intercept = float(state.get("intercept", 0.0))
        y_pred = X @ weights + intercept
        # ablation on non-zero features
        nz = [i for i, w in enumerate(weights) if w > 0]
        X_abl = harness.ablate(X, features=nz) if nz else X
        y_pred_abl = X_abl @ weights + intercept
        delta = float(np.mean((y - y_pred_abl) ** 2) - np.mean((y - y_pred) ** 2))
        # transfer (mock) - copy nz columns from a "source" (first half) to second half
        X_src = X[: max(1, n_samples // 2)]
        X_tgt = X[max(1, n_samples // 2) :]
        if len(X_tgt) == 0:
            suff_reinst = 0.0
        else:
            Xt = harness.transfer(X_src, X_tgt, features=nz) if nz else X_tgt
            yt_pred = Xt @ weights + intercept
            yb = X_tgt @ weights + intercept
            suff_reinst = float(np.mean(yt_pred) - np.mean(yb))
        # minimality: any single-feature ablation causes increase in error
        minimality_pass = bool(any(weights > 0)) and any(
            (np.mean((y - (harness.ablate(X, [i]) @ weights + intercept)) ** 2) - np.mean((y - y_pred) ** 2)) > 0.0
            for i in nz
        )
        results.append(
            {
                "ensemble_id": eid,
                "family": fam,
                "causal_impact_delta": delta,
                "sufficiency_reinstatement": suff_reinst,
                "minimality_pass": bool(minimality_pass),
                "n_features": int(len(members)),
                "cv_score": float(rec.get("cv_score", 0.0)),
            }
        )

    # Write results and Gate 2 report
    res_path = stage_path(run_dir, "causal_eval_results.jsonl")
    save_jsonl(res_path, results)
    log_artifact(res_path, kind="causal_eval_results")

    gate2 = assemble_gate2_report(results=results, params=params, ensembles=ensembles, cfg_snapshot=cfg_dict, seeds=run_sec.get("seeds"))
    g2_path = stage_path(run_dir, "gate2_report.json")
    write_json(g2_path, gate2)
    log_artifact(g2_path, kind="gate2_report")
    # Back-compat artifact expected by smoke test
    write_json(stage_path(run_dir, "causal.json"), {"status": "ok"})

    # Finalize
    fam_metrics = gate2.get("families", {})
    summary = {
        "n_families": len(fam_metrics),
        "gate2_accept": gate2.get("accept_all", False),
    }
    finalize_run(status="success", metrics_dict=summary)
