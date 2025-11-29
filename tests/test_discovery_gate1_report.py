from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.discovery.coactivation import mine_rank_weighted_coactivations
from hypercircuit.discovery.synergy import score_candidates, filter_candidates
from hypercircuit.discovery.reporting import assemble_gate1_report
from hypercircuit.utils.io import load_jsonl, save_jsonl, write_json
from hypercircuit.utils.config import load_config


def _top_signature(exemplars: List[dict]) -> List[Tuple[Tuple[str, ...], float, float]]:
    """Return a compact signature of the top candidates for determinism checks."""
    sig: List[Tuple[Tuple[str, ...], float, float]] = []
    for x in exemplars:
        members = tuple(x.get("members", []))
        ws = float(x.get("weighted_support", 0.0))
        syn = float(x.get("synergy_score", 0.0))
        sig.append((members, ws, syn))
    return sig


def test_discovery_gate1_report(tmp_path: Path) -> None:
    # Load configs: base + discovery + dataset overlay
    cfg = load_config(
        [
            "configs/base.yaml",
            "configs/discovery.yaml",
            "configs/datasets/synthetic_parity.yaml",
        ]
    )

    # Generate a small deterministic mock log
    logs_path = tmp_path / "logs.jsonl"
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
    logger.run(out_path=logs_path, n_samples=cfg.dataset.n_samples)
    assert logs_path.exists() and logs_path.stat().st_size > 0

    # Load events and run co-activation mining (rank-weighted)
    events = load_jsonl(logs_path)
    disc = cfg.discovery
    candidates, ws_index, rep_ws = mine_rank_weighted_coactivations(
        events,
        min_weighted_support=disc.min_weighted_support,
        candidate_caps=disc.candidate_caps.model_dump(),
        temporal_span=disc.candidate_caps.temporal_span,
        dedup_jaccard_min=disc.dedup_jaccard_min,
        max_set_size=3,
    )

    # Score synergy and stability, then filter
    scored = score_candidates(candidates, ws_index, rep_ws)
    passed = filter_candidates(
        scored,
        synergy_threshold=disc.synergy_threshold,
        stability_score_min=disc.stability_score_min,
    )

    # Write artifacts
    cand_path = tmp_path / "candidates.jsonl"
    save_jsonl(cand_path, scored)
    assert cand_path.exists() and cand_path.stat().st_size > 0, "candidates.jsonl should exist and be non-empty"

    report = assemble_gate1_report(
        all_candidates=scored,
        after_synergy=[c for c in scored if (not c.get("redundancy_flag")) and float(c.get("synergy_score", 0.0)) >= disc.synergy_threshold],
        after_stability=passed,
        input_paths=[logs_path],
        output_paths=[cand_path],
        thresholds={
            "min_weighted_support": disc.min_weighted_support,
            "synergy_threshold": disc.synergy_threshold,
            "stability_score_min": disc.stability_score_min,
            "dedup_jaccard_min": disc.dedup_jaccard_min,
            "caps": {
                "size2": disc.candidate_caps.size2,
                "size3": disc.candidate_caps.size3,
                "temporal_span": disc.candidate_caps.temporal_span,
            },
        },
        run_meta={"task_family": cfg.dataset.name, "split": cfg.run.split},
    )
    report_path = tmp_path / "gate1_report.json"
    write_json(report_path, report)

    # Report existence and structure
    assert report_path.exists() and report_path.stat().st_size > 0, "gate1_report.json should exist and be non-empty"
    # Top-level convenience keys and nested counts
    for k in [
        "n_candidates_total",
        "n_candidates_size2",
        "n_candidates_size3",
        "n_passed_synergy",
        "n_passed_stability",
    ]:
        assert k in report, f"report should include top-level key: {k}"
        assert k in report["counts"], f"report['counts'] should include: {k}"
    assert "acceptance" in report and "flags" in report["acceptance"], "acceptance.flags should be present"

    # Determinism: same seed/events produce identical top-10 signature
    report2 = assemble_gate1_report(
        all_candidates=scored,
        after_synergy=[c for c in scored if (not c.get("redundancy_flag")) and float(c.get("synergy_score", 0.0)) >= disc.synergy_threshold],
        after_stability=passed,
        input_paths=[logs_path],
        output_paths=[cand_path],
        thresholds={
            "min_weighted_support": disc.min_weighted_support,
            "synergy_threshold": disc.synergy_threshold,
            "stability_score_min": disc.stability_score_min,
            "dedup_jaccard_min": disc.dedup_jaccard_min,
            "caps": {
                "size2": disc.candidate_caps.size2,
                "size3": disc.candidate_caps.size3,
                "temporal_span": disc.candidate_caps.temporal_span,
            },
        },
        run_meta={"task_family": cfg.dataset.name, "split": cfg.run.split},
    )
    sig1 = _top_signature(report.get("top_exemplars", []))
    sig2 = _top_signature(report2.get("top_exemplars", []))
    assert sig1[:10] == sig2[:10], "Top-10 candidate member sets and scores should be identical under same seed"