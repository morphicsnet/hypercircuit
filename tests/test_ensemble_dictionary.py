from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from hypercircuit.logging.activations import ActivationLogger
from hypercircuit.discovery.coactivation import mine_rank_weighted_coactivations
from hypercircuit.discovery.synergy import score_candidates, filter_candidates
from hypercircuit.dictionary.builder import build_ensemble_dictionary
from hypercircuit.utils.config import load_config
from hypercircuit.utils.io import load_jsonl, read_json


def _make_family_candidates(tmp_path: Path, overlay_path: str, seed: int = 123) -> Tuple[str, Mapping[str, object]]:
    """
    Generate deterministic logs and discovery candidates for a single family,
    returning the family name and a builder-ready inputs_by_family block.
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
    # Keep both the full scored list and a filtered version (builder will re-filter)
    passed = filter_candidates(
        scored,
        synergy_threshold=disc.synergy_threshold,
        stability_score_min=disc.stability_score_min,
    )

    # Use the filtered list to be closer to post-screening input
    return fam_name, {
        "candidates": passed,
        "split": split,
        "run_id": f"{fam_name}-run",
        "source_artifact_paths": [str(logs_path)],
        "discovered_at": None,
    }


def test_ensemble_dictionary_build_and_go_nogo(tmp_path: Path) -> None:
    # Arrange: build deterministic candidates for two families
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

    # Build 1 with mock thresholds configured to ensure a 'go' decision
    run_dir1 = tmp_path / "dict_run1"
    run_dir1.mkdir(parents=True, exist_ok=True)
    cfg_block = {
        "synergy_min": 0.0,
        "stability_min": 0.5,
        "max_per_family": 50,
        "dedup_jaccard_min": 0.5,
        "exemplars_top_k": 3,
        "min_passed_per_top_family": 1,  # keep small for mock determinism
        "families": families,
    }
    res1 = build_ensemble_dictionary(
        inputs_by_family=inputs_by_family,
        config=cfg_block,
        run_dir=run_dir1,
        seed=1234,
        families_to_evaluate=families,
    )

    # Artifacts existence
    assert res1.ensembles_path.exists() and res1.ensembles_path.stat().st_size > 0, "ensembles.jsonl must exist and be non-empty"
    assert res1.dictionary_path.exists() and res1.dictionary_path.stat().st_size > 0, "ensemble_dictionary.json must exist"
    assert res1.go_no_go_path.exists() and res1.go_no_go_path.stat().st_size > 0, "gate1_go_no_go.json must exist"

    # Validate schema fields on a few entries
    entries1 = load_jsonl(res1.ensembles_path)
    assert len(entries1) > 0, "Should produce at least one ensemble entry"
    first = entries1[0]
    for k in [
        "id",
        "members",
        "size",
        "synergy_score",
        "stability_score",
        "weighted_support",
        "family",
        "split",
        "run_id",
        "source_artifact_paths",
        "discovered_at",
        "exemplars",
        "semantics",
    ]:
        assert k in first, f"Entry missing required field: {k}"
    assert isinstance(first["members"], list) and len(first["members"]) >= 2, "members should be a list of size>=2"
    assert isinstance(first["exemplars"], list) and len(first["exemplars"]) == cfg_block["exemplars_top_k"], "exemplars length should match top_k"

    # Go/no-go flags and final decision
    gate1 = read_json(res1.go_no_go_path)
    assert "flags" in gate1 and "counts_ok" in gate1["flags"] and "storage_ok" in gate1["flags"], "flags should include counts_ok and storage_ok"
    assert gate1.get("final") in {"go", "no_go"}, "final must be 'go' or 'no_go'"
    assert gate1.get("final") == "go", "Mock configuration should produce a 'go' decision"

    # Determinism: rebuild with same inputs/seed in a separate directory and compare id/member sets
    run_dir2 = tmp_path / "dict_run2"
    run_dir2.mkdir(parents=True, exist_ok=True)
    res2 = build_ensemble_dictionary(
        inputs_by_family=inputs_by_family,
        config=cfg_block,
        run_dir=run_dir2,
        seed=1234,
        families_to_evaluate=families,
    )
    entries2 = load_jsonl(res2.ensembles_path)

    sig1 = {(e["id"], tuple(e["members"])) for e in entries1}
    sig2 = {(e["id"], tuple(e["members"])) for e in entries2}
    assert sig1 == sig2, "id and member sets should be deterministic under same seed and inputs"