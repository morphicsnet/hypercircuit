from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Any

import numpy as np

from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.discovery.coactivation import mine_rank_weighted_coactivations
from hypercircuit.discovery.synergy import score_candidates, filter_candidates
from hypercircuit.dictionary.builder import build_ensemble_dictionary
from hypercircuit.surrogate.train import fit_surrogates_for_family
from hypercircuit.causal.harness import CausalHarness
from hypercircuit.causal.reporting import assemble_gate2_report
from hypercircuit.utils.config import load_config, stage_path, Config
from hypercircuit.utils.io import load_jsonl, save_jsonl, write_json, read_json


def _make_family_candidates(tmp_path: Path, overlay_path: str, seed: int = 77) -> Tuple[str, Mapping[str, object]]:
    """
    Generate deterministic logs and discovery candidates for a single family,
    returning the family name and a builder-ready inputs_by_family block.
    Mirrors tests/test_ensemble_dictionary.py with minimal differences.
    """
    cfg = load_config(["configs/base.yaml", "configs/discovery.yaml", overlay_path])
    ds = cfg.dataset
    fam_name = (ds.task_family or ds.name or "unknown").strip()
    split = ds.split

    logs_path = tmp_path / f"{fam_name}_logs.jsonl"
    logger = ActivationLogger(
        tokens_per_sample=cfg.logging.tokens_per_sample,
        threshold=cfg.logging.threshold,
        hysteresis=cfg.logging.hysteresis,
        sparsity=cfg.logging.sparsity,
        n_features=cfg.dataset.n_features,
        seed=seed,
        instrumented_layers=cfg.logging.instrumented_layers,
        token_window=cfg.logging.token_window,
        thresholds=cfg.logging.thresholds,
        node_types=cfg.logging.node_types.model_dump(),
    )
    logger.run(out_path=logs_path, n_samples=cfg.dataset.n_samples)
    assert logs_path.exists() and logs_path.stat().st_size > 0, f"logs.jsonl should exist for {fam_name}"

    # Mine candidates and score
    events = load_jsonl(logs_path)
    disc = cfg.discovery
    candidates, ws_index, rep_ws = mine_rank_weighted_coactivations(
        events,
        min_weighted_support=disc.min_weighted_support,
        candidate_caps=disc.candidate_caps.model_dump() if hasattr(disc.candidate_caps, "model_dump") else dict(disc.candidate_caps),  # type: ignore[arg-type]
        temporal_span=disc.candidate_caps.temporal_span,
        dedup_jaccard_min=disc.dedup_jaccard_min,
        max_set_size=3,
    )
    scored = score_candidates(candidates, ws_index, rep_ws)
    passed = filter_candidates(
        scored,
        synergy_threshold=disc.synergy_threshold,
        stability_score_min=disc.stability_score_min,
    )

    return fam_name, {
        "candidates": passed,
        "split": split,
        "run_id": f"{fam_name}-run",
        "source_artifact_paths": [str(logs_path)],
        "discovered_at": None,
    }


def _train_logs(run_dir: Path, cfg: Config, seed: int = 0) -> Path:
    """Emit a unified logs.jsonl in run_dir for surrogate training and causal eval."""
    run_logs = stage_path(run_dir, "logs.jsonl")
    logger = ActivationLogger(
        tokens_per_sample=cfg.logging.tokens_per_sample,
        threshold=cfg.logging.threshold,
        hysteresis=cfg.logging.hysteresis,
        sparsity=cfg.logging.sparsity,
        n_features=cfg.dataset.n_features,
        seed=seed,
        instrumented_layers=cfg.logging.instrumented_layers,
        token_window=cfg.logging.token_window,
        thresholds=cfg.logging.thresholds,
        node_types=cfg.logging.node_types.model_dump(),
    )
    logger.run(out_path=run_logs, n_samples=cfg.dataset.n_samples)
    assert run_logs.exists() and run_logs.stat().st_size > 0
    return run_logs


def test_surrogate_cv_and_gate2_report(tmp_path: Path) -> None:
    # 1) Build deterministic ensemble dictionary for two families
    overlays = [
        "configs/datasets/sycophancy.yaml",
        "configs/datasets/jailbreak.yaml",
    ]
    inputs_by_family: Dict[str, Mapping[str, object]] = {}
    families: List[str] = []
    for ov in overlays:
        fam, block = _make_family_candidates(tmp_path, ov, seed=77)
        inputs_by_family[fam] = block
        families.append(fam)

    # Single run_dir hosting all downstream artifacts
    run_dir = tmp_path / "week3_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure dictionary builder with relaxed thresholds for mock determinism
    dict_cfg = {
        "synergy_min": 0.0,
        "stability_min": 0.5,
        "max_per_family": 50,
        "dedup_jaccard_min": 0.5,
        "exemplars_top_k": 3,
        "min_passed_per_top_family": 1,
        "families": families,
    }
    _ = build_ensemble_dictionary(
        inputs_by_family=inputs_by_family,
        config=dict_cfg,
        run_dir=run_dir,
        seed=1234,
        families_to_evaluate=families,
    )
    ensembles_path = run_dir / "ensembles.jsonl"
    assert ensembles_path.exists() and ensembles_path.stat().st_size > 0, "ensembles.jsonl must exist"

    # 2) Emit unified training logs for this run_dir
    cfg_train = load_config(["configs/base.yaml", "configs/surrogate.yaml", "configs/dictionary.yaml"])
    _ = _train_logs(run_dir, cfg_train, seed=0)

    # 3) Train interpretable surrogates with CV + calibration
    fit_res = fit_surrogates_for_family(run_dir=run_dir, cfg=cfg_train, families=families)
    params_path = run_dir / "surrogates_params.jsonl"
    index_path = run_dir / "surrogates_index.json"
    assert params_path.exists() and params_path.stat().st_size > 0, "surrogates_params.jsonl should exist"
    assert index_path.exists() and index_path.stat().st_size > 0, "surrogates_index.json should exist"

    # Validate schema of at least one record
    params = load_jsonl(params_path)
    assert len(params) > 0, "At least one surrogate should be trained"
    r0 = params[0]
    for k in ["ensemble_id", "family", "cv_score", "calibration_error", "model_state", "hyperparams"]:
        assert k in r0, f"Missing required key in surrogates_params: {k}"

    # Family summary exists
    summary = read_json(index_path)
    assert "summary" in summary and isinstance(summary["summary"], dict)
    for fam in families:
        fam_block = summary["summary"].get(fam, {})
        for k in ["n_surrogates_trained", "median_cv_score", "median_calibration_error"]:
            assert k in fam_block, f"{k} missing in per-family summary"

    # 4) Targeted causal evaluation on top-K ensembles per family (mock)
    cfg_causal = load_config(["configs/base.yaml", "configs/causal.yaml", "configs/dictionary.yaml"])
    events = load_jsonl(stage_path(run_dir, "logs.jsonl"))
    ensembles = load_jsonl(ensembles_path)
    n_samples = len(events)

    # Helper: presence matrix per sample for ensemble member node_types
    def _sample_features_for_members(members: List[str]) -> np.ndarray:
        X = np.zeros((n_samples, len(members)), dtype=float)
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

    # Select top-K per family by CV score
    n_k = int(getattr(cfg_causal.causal, "n_ensembles_per_family", 10))
    by_family: Dict[str, List[Mapping[str, Any]]] = {}
    for e in params:
        fam = str(e.get("family"))
        by_family.setdefault(fam, []).append(e)
    selected: List[Mapping[str, Any]] = []
    for fam, recs in by_family.items():
        recs = list(recs)
        recs.sort(key=lambda d: -float(d.get("cv_score", 0.0)))
        selected.extend(recs[:n_k])

    # Run mock ablations/transfers and aggregate results
    harness = CausalHarness(ablation_strength=cfg_causal.causal.ablation_strength)
    results: List[Mapping[str, Any]] = []
    for rec in selected:
        eid = str(rec.get("ensemble_id"))
        fam = str(rec.get("family"))
        members: List[str] = []
        for e in ensembles:
            if e.get("id") == eid:
                members = list(e.get("members", []))
                break
        if not members:
            continue
        X = _sample_features_for_members(members)
        y = X.sum(axis=1) + 0.01
        state = rec.get("model_state") or {}
        weights = np.array(state.get("weights") or [0.0] * len(members), dtype=float)
        intercept = float(state.get("intercept", 0.0))
        y_pred = X @ weights + intercept
        nz = [i for i, w in enumerate(weights) if w > 0]
        X_abl = harness.ablate(X, features=nz) if nz else X
        y_pred_abl = X_abl @ weights + intercept
        delta = float(np.mean((y - y_pred_abl) ** 2) - np.mean((y - y_pred) ** 2))
        # Transfer sufficiency: copy nz cols first->second half
        X_src = X[: max(1, n_samples // 2)]
        X_tgt = X[max(1, n_samples // 2) :]
        if len(X_tgt) == 0:
            suff_reinst = 0.0
        else:
            Xt = harness.transfer(X_src, X_tgt, features=nz) if nz else X_tgt
            ytp = Xt @ weights + intercept
            ybp = X_tgt @ weights + intercept
            suff_reinst = float(np.mean(ytp) - np.mean(ybp))
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

    # Persist causal artifacts
    causal_path = stage_path(run_dir, "causal_eval_results.jsonl")
    save_jsonl(causal_path, results)
    assert causal_path.exists() and causal_path.stat().st_size > 0, "causal_eval_results.jsonl should exist"

    gate2 = assemble_gate2_report(
        results=results,
        params=params,
        ensembles=ensembles,
        cfg_snapshot=cfg_causal.model_dump(),
        seeds=cfg_causal.run.seeds,
    )
    gate2_path = stage_path(run_dir, "gate2_report.json")
    write_json(gate2_path, gate2)
    assert gate2_path.exists() and gate2_path.stat().st_size > 0, "gate2_report.json should exist"

    # Structure checks
    g2 = read_json(gate2_path)
    assert "families" in g2 and isinstance(g2["families"], dict), "Gate 2 families block missing"
    assert "acceptance" in g2 and isinstance(g2["acceptance"], dict), "Gate 2 acceptance block missing"
    # Per-family required fields
    for fam, fb in g2["families"].items():
        for k in [
            "n_surrogates_trained",
            "median_cv_score",
            "median_calibration_error",
            "n_ensembles_evaluated",
            "median_causal_impact_delta",
            "sufficiency_reinstatement_rate",
            "minimality_pass_rate",
            "top10_signature_md5",
            "top10_ids",
            "accept",
        ]:
            assert k in fb, f"Gate 2 per-family field missing: {k}"

    # 5) Determinism: re-train surrogates with same seed; top-10 ids unchanged
    # Re-run training (overwrites files deterministically)
    _ = fit_surrogates_for_family(run_dir=run_dir, cfg=cfg_train, families=families)
    params2 = load_jsonl(params_path)
    # Select top ids by family from both runs and compare
    def _top_ids(perms: List[Mapping[str, Any]], fam: str) -> List[str]:
        recs = [r for r in perms if r.get("family") == fam]
        recs.sort(key=lambda d: -float(d.get("cv_score", 0.0)))
        return [str(x.get("ensemble_id")) for x in recs[:10] if x.get("ensemble_id")]

    for fam in families:
        assert _top_ids(params, fam) == _top_ids(params2, fam), "Top-10 ids must be deterministic across runs with same seed"