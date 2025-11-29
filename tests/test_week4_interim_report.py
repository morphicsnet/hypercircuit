from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.discovery.coactivation import mine_rank_weighted_coactivations
from hypercircuit.discovery.synergy import score_candidates, filter_candidates
from hypercircuit.dictionary.builder import build_ensemble_dictionary
from hypercircuit.surrogate.aggregate_train import train_all_families
from hypercircuit.eval.matrix import run_matrix_evaluation
from hypercircuit.eval.reporting import assemble_interim_report
from hypercircuit.utils.config import Config, load_config, stage_path
from hypercircuit.utils.io import load_jsonl, read_json
from hypercircuit.utils.registry import start_run, finalize_run


def _family_inputs(tmp_path: Path, overlay_path: str, seed: int = 11) -> Tuple[str, Mapping[str, object]]:
    cfg = load_config(["configs/base.yaml", "configs/discovery.yaml", overlay_path])
    ds = cfg.dataset
    fam_name = (ds.task_family or ds.name or "unknown").strip()
    split = ds.split

    # Emit deterministic per-family logs to construct discovery candidates
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
    events = load_jsonl(logs_path)

    # Mine, score, filter candidates
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
        "run_id": f"{fam_name}-week4",
        "source_artifact_paths": [str(logs_path)],
        "discovered_at": None,
    }


def test_week4_interim_matrix_and_report(tmp_path: Path) -> None:
    # Families (≥3)
    overlays = [
        "configs/datasets/sycophancy.yaml",
        "configs/datasets/jailbreak.yaml",
        "configs/datasets/truthfulness_qa.yaml",
    ]
    inputs_by_family: Dict[str, Mapping[str, object]] = {}
    families: List[str] = []
    for ov in overlays:
        fam, block = _family_inputs(tmp_path, ov, seed=11)
        inputs_by_family[fam] = block
        families.append(fam)

    # Load full config with overrides for speed/determinism
    overrides = [
        "causal.n_ensembles_per_family=5",
        "matrix.per_cell.n_seeds=3",
        "matrix.per_cell.n_prompts=100",
        "matrix.half_matrix=true",
    ]
    cfg: Config = load_config(
        ["configs/base.yaml", "configs/surrogate.yaml", "configs/causal.yaml", "configs/dictionary.yaml", "configs/matrix.yaml"],
        overrides=overrides,
    )
    cfg_dict = cfg.model_dump()

    # Start a registry run
    run_id, run_dir = start_run(cfg_dict, stage_name="week4_interim_test", config_paths=["configs/base.yaml"])
    # Build dictionary into this run_dir
    _ = build_ensemble_dictionary(
        inputs_by_family=inputs_by_family,
        config={
            "synergy_min": 0.0,
            "stability_min": 0.5,
            "max_per_family": 50,
            "dedup_jaccard_min": 0.5,
            "exemplars_top_k": 3,
            "min_passed_per_top_family": 1,
            "families": families,
        },
        run_dir=run_dir,
        seed=123,
        families_to_evaluate=families,
    )

    # Emit unified logs.jsonl into run_dir
    logger = ActivationLogger(
        tokens_per_sample=cfg.logging.tokens_per_sample,
        threshold=cfg.logging.threshold,
        hysteresis=cfg.logging.hysteresis,
        sparsity=cfg.logging.sparsity,
        n_features=cfg.dataset.n_features,
        seed=0,
        instrumented_layers=cfg.logging.instrumented_layers,
        token_window=cfg.logging.token_window,
        thresholds=cfg.logging.thresholds,
        node_types=cfg.logging.node_types.model_dump(),
    )
    logger.run(out_path=stage_path(run_dir, "logs.jsonl"), n_samples=cfg.dataset.n_samples)

    # Train surrogates across all families
    tr = train_all_families(cfg=cfg, run_dir=run_dir, families=families)
    idx_all = stage_path(run_dir, "surrogates_index_all.json")
    assert idx_all.exists() and idx_all.stat().st_size > 0, "surrogates_index_all.json must exist"

    # Matrix evaluation (half matrix)
    mat = run_matrix_evaluation(cfg=cfg, run_dir=run_dir, families=families, methods=None, half_matrix=True)
    mat_path = stage_path(run_dir, "matrix_results.jsonl")
    assert mat_path.exists() and mat_path.stat().st_size > 0, "matrix_results.jsonl must exist"
    rows = load_jsonl(mat_path)
    assert len(rows) > 0 and {"family", "method", "effect_size", "checksum_id"}.issubset(rows[0].keys())

    # Interim report
    rep = assemble_interim_report(cfg=cfg, run_dir=run_dir, matrix_path=mat_path, seeds=cfg.run.seeds)
    rpt_path = rep["report_path"]
    assert rpt_path.exists() and rpt_path.stat().st_size > 0, "interim_report.json must exist"

    report = read_json(rpt_path)
    # Top-level keys and acceptance flags
    for k in ["families", "methods", "n_cells", "coverage_ratio", "topline", "fdr", "acceptance", "determinism"]:
        assert k in report, f"Missing key in report: {k}"
    acc = report["acceptance"]
    for k in ["coverage_ok", "effect_ok", "stability_ok"]:
        assert k in acc and isinstance(acc[k], bool), f"Acceptance flag missing or wrong type: {k}"

    # Determinism: re-run matrix and report; checksums identical, effects stable
    mat2 = run_matrix_evaluation(cfg=cfg, run_dir=run_dir, families=families, methods=None, half_matrix=True)
    rep2 = assemble_interim_report(cfg=cfg, run_dir=run_dir, matrix_path=mat2["matrix_path"], seeds=cfg.run.seeds)
    rep_a = read_json(rep["report_path"])
    rep_b = read_json(rep2["report_path"])
    assert (
        rep_a["determinism"]["top_cells_by_family_md5"] == rep_b["determinism"]["top_cells_by_family_md5"]
    ), "Top-cells checksums must match across identical runs"
    # Effects near-identical
    med_a = float(rep_a["topline"]["median_effect_hypercircuit"])
    med_b = float(rep_b["topline"]["median_effect_hypercircuit"])
    assert abs(med_a - med_b) <= 1e-9, "Median effects should be deterministic to within tolerance"

    finalize_run(status="success", metrics_dict={"n_cells": int(report.get("n_cells", 0))})