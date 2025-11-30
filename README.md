# HYPERCIRCUIT

Interpretability framework skeleton for SAE-first workflows: hyperedge discovery, surrogate modeling, ensemble-level causal evaluation, semantic labeling, and safety edits. This repository provides an implementation-ready scaffold with clear module boundaries, typed interfaces, configs, scripts, and smoke tests. Heavy algorithms are intentionally stubbed.

## Installation
- Python 3.11+
- Create and activate a virtualenv
- Install in editable mode:
  - pip install -e .[dev]

## Quickstart (mock, <1 minute)
- Run mock logging to create event logs:
  - python -m hypercircuit.cli.run_log --config configs/base.yaml configs/logging.yaml
- Mine candidates and screen:
  - python -m hypercircuit.cli.run_discovery --config configs/base.yaml configs/discovery.yaml
- Train surrogates (CV+calibration):
  - python -m hypercircuit.cli.run_surrogate --config configs/base.yaml configs/surrogate.yaml configs/dictionary.yaml
- Causal eval + Gate 2:
  - python -m hypercircuit.cli.run_causal_eval --config configs/base.yaml configs/causal.yaml configs/dictionary.yaml
- Safety edits:
  - python -m hypercircuit.cli.run_edit_eval --config configs/base.yaml configs/editing.yaml

Artifacts are written under a run directory defined in the config (defaults to runs/mock or a timestamped run).

## Repository map
- src/hypercircuit: source package with submodules:
  - utils: config load/merge/validate, IO helpers, seeding
  - sae_io: SAE adapters and a fake dictionary for mocks
  - logging: activation logging stubs (no external model calls)
  - discovery: coactivation mining, synergy proxies, candidate pruning
  - surrogate: interpretable monotone combiner scaffold
  - causal: harness and patching utilities for ablations/transfers
  - semantics: labeling schema, exemplar selection, agreement metrics
  - steering: safety edit simulation via surrogate sensitivities
  - eval: metrics with mock yet typed computations
  - dashboards: export flat JSON artifacts + provenance
  - cli: script entry points that wire configs to modules
- configs: base and stage-specific knobs; dataset presets
- scripts: thin wrappers calling cli entry points (for local runs)
- tests: smoke tests and config schema checks
- examples: quickstart walkthrough

## Safety note
This scaffold includes mock data paths and deterministic seeds. It does not connect to external model servers. Ensemble-level edits and interventions are simulated and must be validated against controls before any real deployment. Always evaluate off-target impacts and rollback criteria.

## Run registry and seeds

The run registry is a lightweight, local JSON-based tracker that creates a run directory and writes minimal provenance.

- Entry points:
  - [registry.start_run()](src/hypercircuit/utils/registry.py:60) initializes a run, returns (run_id, run_dir), and writes a manifest to runs/{run_id}/manifest.json.
  - [registry.log_artifact()](src/hypercircuit/utils/registry.py:125) appends relative paths and metadata to runs/{run_id}/artifacts.jsonl.
  - [registry.finalize_run()](src/hypercircuit/utils/registry.py:140) updates manifest status and summary metrics.

- Manifest fields: run_id, created_at, stage, config_paths, model_tag, task_family, split, seeds, versions, git_sha (best effort), status, summary.

- Run IDs are deterministic-friendly: timestamp + short uuid. You can override via config:
  - run.output_dir (default "runs"), run.run_id ("mock" in base config). Using CLI --run-dir also works (legacy).

- Seeds policy:
  - Set seeds in config at run.seeds (default [0,1,2,3,4]). Stage CLIs pick an active seed deterministically (Week 1 uses index 0).
  - The seeds list is recorded in the manifest for each run.

- Mock logging sanity:
  - [cli.run_log.main()](src/hypercircuit/cli/run_log.py:1) prints an event-density sanity line:
    - event_density_per_token vs expected band from configs/logging.yaml (expected_event_density_range).
  - Artifacts include logs.jsonl, candidates.jsonl, surrogates_params.jsonl, surrogates_index.json, causal_eval_results.jsonl, gate2_report.json, edits.json (depending on stage).

Quickstart remains unchanged; the registry is automatically used by the CLIs listed above.

## Discovery and Gate 1

The discovery stage reads recent logging artifacts, mines size-2/3 co-activations with rank-weighted supports, applies lightweight synergy and stability screens, and emits:
- candidates.jsonl (full candidate registry)
- gate1_report.json (aggregated metrics, exemplars, and acceptance flags)

Key components:
- Mining and stability splits: transactions are built per (sample, layer, window_index) with a configurable temporal span, and supports are computed globally and across two mock replicates (even/odd sample_id) for stability.
- Synergy proxy: candidate weighted_support minus max weighted_support of proper subsets (clamped at 0).
- Stability score: normalized Spearman rank correlation between replicate supports over the candidate list, mapped to [0,1].

Outputs are written to the active run directory:
- candidates.jsonl: full scored candidates with fields members, size, support, weighted_support, synergy_score, stability_score, window_span.
- gate1_report.json: counts, storage usage, top exemplars, thresholds, and acceptance flags.

Config defaults (Week 1) in configs/discovery.yaml:
- discovery.min_weighted_support: 0.01
- discovery.candidate_caps.size2: 1000
- discovery.candidate_caps.size3: 300
- discovery.candidate_caps.temporal_span: 3
- discovery.synergy_threshold: 0.0
- discovery.stability_score_min: 0.5
- discovery.dedup_jaccard_min: 0.5

## Week 2 synergy screening

Week 2 adds a multi-family aggregator that extends dataset overlays, supports multi-replicate stability, and produces a consolidated report.

- Aggregator entrypoint: [run_week2_screening()](src/hypercircuit/discovery/aggregate.py:1)
- CLI wrapper: [scripts/run_week2_screening.py](scripts/run_week2_screening.py:1)
- Reporting builder: [assemble_week2_synergy_report()](src/hypercircuit/discovery/reporting.py:1)
- Multi-replicate stability helper: [compute_stability_multi()](src/hypercircuit/discovery/synergy.py:1)
- Candidate scoring with optional multi-replicate mode: [score_candidates()](src/hypercircuit/discovery/synergy.py:60)

Configuration (new block in [configs/discovery.yaml](configs/discovery.yaml:1)):
- discovery.week2_screening.top_families: ["sycophancy", "jailbreak"]
- discovery.week2_screening.replicates_k: 3
- discovery.week2_screening.paraphrase_replicates: 2
Existing Week 1 thresholds (min_weighted_support, candidate_caps, temporal_span, synergy_threshold, stability_score_min, dedup_jaccard_min) remain active.

Dataset overlays added:
- [configs/datasets/deceptive_reasoning.yaml](configs/datasets/deceptive_reasoning.yaml:1) (dev, ~96, ~12, tiny)
- [configs/datasets/truthfulness_qa.yaml](configs/datasets/truthfulness_qa.yaml:1) (dev, ~128, ~12, tiny)

How it works (mock/deterministic):
- For each overlay’s task_family/split, the aggregator locates the latest logs.jsonl via the run registry.
- Reuses mining ([mine_rank_weighted_coactivations()](src/hypercircuit/discovery/coactivation.py:101)) and synergy logic.
- If replicates_k > 2, partitions events by hash(sample_id) % k to compute k replicate supports, then averages pairwise Spearman rank correlations via compute_stability_multi().
- Writes a consolidated week2_synergy_report.json and logs artifacts via the registry.

Consolidated report fields (week2_synergy_report.json):
- families.{family}.counts: total, size2, size3, after-synergy, after-stability
- families.{family}.stability: {replicates_k, stability_multi}
- families.{family}.top_exemplars: top-10 exemplars after filtering
- families.{family}.top10_signature_md5: determinism checksum of top-10 (members, ws, synergy)
- thresholds: discovery thresholds used
- week2: {replicates_k, paraphrase_replicates, top_families}
- acceptance: flags for configured top_families based on stability pass count

Example usage (mock):
- Generate logs per family (see Quickstart logging), then:
  - python scripts/run_week2_screening.py --config configs/base.yaml configs/discovery.yaml --overlay configs/datasets/sycophancy.yaml configs/datasets/jailbreak.yaml configs/datasets/deceptive_reasoning.yaml configs/datasets/truthfulness_qa.yaml -o discovery.week2_screening.replicates_k=3
- The script prints per-family summary lines and writes week2_synergy_report.json under the active run directory.

The discovery script locates the latest compatible logs.jsonl (matching run.task_family and run.split), runs mining and screening, writes artifacts, logs them in the registry, and finalizes the manifest with summary metrics.



## Ensemble dictionary and Gate 1 go/no-go

Phase 1 Week 2 adds an ensemble dictionary builder that consolidates screened discovery candidates into a machine-readable dictionary with provenance, plus a Gate 1 go/no-go decision artifact.

- Core API:
  - [build_ensemble_dictionary()](src/hypercircuit/dictionary/builder.py:1)
  - Helpers: [select_and_dedup_candidates()](src/hypercircuit/dictionary/builder.py:1), [compute_provenance()](src/hypercircuit/dictionary/builder.py:1), [assemble_dictionary_entries()](src/hypercircuit/dictionary/builder.py:1), [emit_go_no_go()](src/hypercircuit/dictionary/builder.py:1)
- Loader utility for discovery outputs:
  - [load_candidates_for_family()](src/hypercircuit/discovery/reporting.py:1)
- Optional dashboard export:
  - [export_dictionary()](src/hypercircuit/dashboards/export.py:1)
- Registry wiring:
  - [registry.start_run()](src/hypercircuit/utils/registry.py:60), [registry.log_artifact()](src/hypercircuit/utils/registry.py:125), [registry.finalize_run()](src/hypercircuit/utils/registry.py:140)

Artifacts (written to the active run dir):
- ensembles.jsonl: one JSON object per ensemble entry (with id, members, synergy_score, stability_score, provenance, exemplars, semantics placeholders)
- ensemble_dictionary.json: header/index with schema_version, config snapshot, counts per family, and a small index
- gate1_go_no_go.json: flags (counts_ok, storage_ok), families_checked, counts_by_family, and final "go" or "no_go" with reasons

Config
- New block in [utils.config.Config](src/hypercircuit/utils/config.py:167) under run.dictionary, defaults provided in [configs/dictionary.yaml](configs/dictionary.yaml:1):
  - synergy_min: 0.0
  - stability_min: 0.5
  - max_per_family: 50
  - dedup_jaccard_min: 0.5
  - exemplars_top_k: 3
  - min_passed_per_top_family: 5 (mock)
  - families: null (use discovery.week2_screening.top_families); or provide explicit list

CLI usage (mock/deterministic)
- Ensure discovery artifacts exist (per-family logs and discovery have produced candidates.jsonl and gate1_report.json under runs/)
- Build dictionary:
  - python -m hypercircuit.cli.run_build_dictionary --config configs/base.yaml configs/discovery.yaml configs/dictionary.yaml --families sycophancy jailbreak --export-flat runs/mock/dictionary_flat.json
  - Script wrapper: [scripts/run_build_dictionary.py](scripts/run_build_dictionary.py:1)
- The CLI:
  - Resolves families (CLI --families, dictionary.families, or discovery.week2_screening.top_families)
  - Loads latest per-family discovery artifacts via [load_candidates_for_family()](src/hypercircuit/discovery/reporting.py:1)
  - Calls [build_ensemble_dictionary()](src/hypercircuit/dictionary/builder.py:1)
  - Logs all artifacts and finalizes the run manifest with summary metrics and the go/no-go decision

Go/no-go mock rule
- counts_ok: for each top family, at least dictionary.min_passed_per_top_family ensembles pass thresholds
- storage_ok: dictionary artifacts created and non-zero
- final: "go" if all flags true; otherwise "no_go" with reasons

Determinism notes
- Selection uses stable sorts and [seed_context()](src/hypercircuit/utils/seed.py:1) during build
- IDs are stable hashes of family + member set
- Test: [tests/test_ensemble_dictionary.py](tests/test_ensemble_dictionary.py:1) validates schema, determinism, and go/no-go evaluation

## Week 3: Surrogate training and Gate 2

Phase 2 Week 3 adds a deterministic surrogate-training pipeline and a targeted causal evaluation pass.

Artifacts
- surrogates_params.jsonl — one record per ensemble with fields {ensemble_id, family, cv_score, calibration_error, model_state, hyperparams}
- surrogates_index.json — per-family summary {n_surrogates_trained, median_cv_score, median_calibration_error}
- causal_eval_results.jsonl — per-ensemble causal metrics {causal_impact_delta, sufficiency_reinstatement, minimality_pass}
- gate2_report.json — consolidated Gate 2 report per family with top10_signature_md5, acceptance flags, and a config snapshot. Built by [assemble_gate2_report()](src/hypercircuit/causal/reporting.py:1).

How to run (mock)
- Ensure you have a dictionary and logs in the same run directory, or run the CLIs in order:
  - python -m hypercircuit.cli.run_log --config configs/base.yaml configs/logging.yaml
  - python -m hypercircuit.cli.run_discovery --config configs/base.yaml configs/discovery.yaml
  - python -m hypercircuit.cli.run_build_dictionary --config configs/base.yaml configs/discovery.yaml configs/dictionary.yaml
  - python -m hypercircuit.cli.run_surrogate --config configs/base.yaml configs/surrogate.yaml configs/dictionary.yaml
  - python -m hypercircuit.cli.run_causal_eval --config configs/base.yaml configs/causal.yaml configs/dictionary.yaml

Training orchestrator
- [fit_surrogates_for_family()](src/hypercircuit/surrogate/train.py:1) loads ensembles/logs, assembles mock training data via [assemble_mock_training_data()](src/hypercircuit/surrogate/train.py:1), runs [cross_validate_and_calibrate()](src/hypercircuit/surrogate/train.py:1), and serializes artifacts with [serialize_surrogate()](src/hypercircuit/surrogate/train.py:1).
- The model is a monotone combiner exposing [fit()](src/hypercircuit/surrogate/model.py:1), [predict()](src/hypercircuit/surrogate/model.py:1), [state_dict()](src/hypercircuit/surrogate/model.py:1), [load_state_dict()](src/hypercircuit/surrogate/model.py:1).

Acceptance (mock)
- Per-family acceptance requires: at least 10 ensembles evaluated AND median_causal_impact_delta > 0 AND sufficiency_reinstatement_rate ≥ 0.7.
- accept_all is true only if all observed families pass.

Determinism
- CV shuffling and synthetic targets use [seed_context()](src/hypercircuit/utils/seed.py:56) seeded from run.seeds[0]; top-10 IDs per family are hashed into a checksum.

## Week 4 interim matrix and report

Run end-to-end (mock/deterministic) to train surrogates across all families, evaluate a half-sized causal matrix, and assemble the interim report.

- CLI:
  - python -m hypercircuit.cli.run_week4_interim --config configs/base.yaml configs/surrogate.yaml configs/causal.yaml configs/dictionary.yaml configs/matrix.yaml
  - Script wrapper: [scripts/run_week4_interim.py](scripts/run_week4_interim.py:1)
  - Entrypoint: [main()](src/hypercircuit/cli/run_week4_interim.py:1)

- Config knobs:
  - Matrix config: [configs/matrix.yaml](configs/matrix.yaml:1) (families=null, half_matrix=true, methods preset)
  - Additional defaults: causal.matrix_eval in [configs/causal.yaml](configs/causal.yaml:1)
  - Typed schema: matrix block in [Config](src/hypercircuit/utils/config.py:184)

- Pipeline steps and APIs:
  1) Train surrogates across all families
     - [train_all_families()](src/hypercircuit/surrogate/aggregate_train.py:1)
     - Writes surrogates_index_all.json and logs via registry
  2) Evaluate half-sized matrix (families × methods)
     - [run_matrix_evaluation()](src/hypercircuit/eval/matrix.py:1)
     - Writes matrix_results.jsonl with per-cell: effect_size, ci_low, ci_high, p_value, stability, passed, checksum_id
  3) Assemble interim report
     - [assemble_interim_report()](src/hypercircuit/eval/reporting.py:1)
     - Writes interim_report.json (coverage_ok, effect_ok, stability_ok, FDR BH, determinism checksums)

- Registry wiring:
  - Start: [start_run()](src/hypercircuit/utils/registry.py:60)
  - Artifact: [log_artifact()](src/hypercircuit/utils/registry.py:125)
  - Finalize: [finalize_run()](src/hypercircuit/utils/registry.py:140)

- Artifacts (under the active run directory):
  - surrogates_index_all.json, surrogates_params.jsonl, surrogates_index.json
  - matrix_results.jsonl
  - interim_report.json

- Acceptance (mock):
  - coverage_ok: at least half of the configured grid evaluated
  - effect_ok: median hypercircuit_full &gt; baselines in ≥ 50% of families
  - stability_ok: median seed stability ≥ 0.8

## Roadmap

- Full plan: [docs/ROADMAP_V1_to_V2.md](docs/ROADMAP_V1_to_V2.md)
  - Executive overview; target architecture and principles; data and storage plan; interfaces and contracts; phased roadmap with deliverables, gates, risks; migration plan; metrics; compute/storage/scheduling; roles and QA; risks and mitigations; timeline; and forward-compatible V1 changes.
- Contracts and schemas: [docs/CONTRACTS_AND_SCHEMAS.md](docs/CONTRACTS_AND_SCHEMAS.md)
  - Prose schemas for feature activation logs, candidate hyperedges, dynamic circuit graph snapshots, manifold snapshots, time-window indices and state bundles; six core contracts (ensemble discovery, circuit graph abstraction, Sidecar intake, manifold constraint, surrogate and causal evaluation, safety edit protocols); governance, retention, and migration notes.

## Week 5: Safety edits and Gate 3

Run the safety edit planning, staged monitoring, and Gate 3 reporting (mock/deterministic).

- CLI (module):
  - python -m hypercircuit.cli.run_week5_safety --config configs/base.yaml configs/editing.yaml configs/causal.yaml configs/dictionary.yaml
- Script wrapper:
  - python [scripts/run_week5_safety.py](scripts/run_week5_safety.py:1) --config configs/base.yaml configs/editing.yaml configs/causal.yaml configs/dictionary.yaml

What it does (mock):
- Starts registry stage "week5_safety"
- Resolves risky families from config
- Proposes a deterministic edit plan and simulates impact
- Runs a monitored loop over the configured step schedule with per-step safety evaluation and specificity checks
- Writes artifacts and assembles Gate 3 report with acceptance flags and determinism checksums

Artifacts (under the active run directory):
- safety_edit_plans.jsonl — proposed_scales (by family), ranking, applied_scales_per_step, seed
- safety_eval_results.jsonl — per-step per-family metrics: harmful_rate_reduction, benign_deg_pct, specificity_ratio, thresholds, breach flags
- gate3_report.json — per-family acceptance flags and global accept_all, reasons_for_fail, determinism checksums, config snapshot, seeds

Acceptance bars (defaults, mock):
- harmful_rate_reduction ≥ 0.20 (target)
- benign_deg_pct ≤ 0.50
- specificity_ratio ≥ 1.20

## Week 7: 24-layer logging, semantic labeling, and dashboards

This phase expands activation logging to 24 relative layers for top behaviors, finalizes semantic labels deterministically, and exports lightweight dashboard JSONs. All pipelines remain mock/deterministic and reuse existing artifacts and registry wiring.

Key modules and configs:
- [configs/logging.yaml](configs/logging.yaml)
- [configs/datasets/sycophancy.yaml](configs/datasets/sycophancy.yaml), [configs/datasets/jailbreak.yaml](configs/datasets/jailbreak.yaml), [configs/datasets/deceptive_reasoning.yaml](configs/datasets/deceptive_reasoning.yaml)
- [src/hypercircuit/logging/activations.py](src/hypercircuit/logging/activations.py)
- [src/hypercircuit/semantics/labeling.py](src/hypercircuit/semantics/labeling.py)
- [src/hypercircuit/dashboards/export.py](src/hypercircuit/dashboards/export.py)
- [src/hypercircuit/cli/run_week7_labels_dash.py](src/hypercircuit/cli/run_week7_labels_dash.py)
- [scripts/run_week7_labels_dash.py](scripts/run_week7_labels_dash.py)
- Tests: [tests/test_week7_labels_dash.py](tests/test_week7_labels_dash.py)

### 1) 24-layer logging for top behaviors
The logger can expand from 12 to 24 relative layers for top families while keeping event density within the configured band.

- Config toggles in [configs/logging.yaml](configs/logging.yaml):
  - logging.top_behaviors_only: false by default; set true to enable expansion for top families
  - logging.top_behaviors_families: ["sycophancy","jailbreak","deceptive_reasoning"] default
  - logging.layers_profile_24: [-24..-1] default profile
- Per-family overlays set a dataset flag to trigger expansion:
  - dataset.top_behaviors: true set in [configs/datasets/sycophancy.yaml](configs/datasets/sycophancy.yaml), [configs/datasets/jailbreak.yaml](configs/datasets/jailbreak.yaml), [configs/datasets/deceptive_reasoning.yaml](configs/datasets/deceptive_reasoning.yaml)

Run mock logging with 24-layer expansion (example):
- Module:
  - python -m hypercircuit.cli.run_log --config configs/base.yaml configs/logging.yaml configs/datasets/sycophancy.yaml -o logging.top_behaviors_only=true -o run.run_dir=runs/week7_logging_syc
- Script wrapper:
  - python scripts/run_log.py --config configs/base.yaml configs/logging.yaml configs/datasets/sycophancy.yaml -o logging.top_behaviors_only=true -o run.run_dir=runs/week7_logging_syc

Expected:
- Prints sanity line with events_per_token within the band from logging.expected_event_density_range
- Writes runs/&lt;run_id&gt;/logs.jsonl
- Manifest summary includes n_layers and n_layers_used=24

### 2) Semantic labeling finalization
Deterministically finalizes semantic labels for accepted ensembles, emitting JSONL records and an aggregated report. Selection is seed-stable and fast.

- Entrypoint:
  - Module: python -m hypercircuit.cli.run_week7_labels_dash --config configs/base.yaml configs/discovery.yaml configs/dictionary.yaml -o run.run_dir=runs/week7_labels_dash
  - Script: python [scripts/run_week7_labels_dash.py](scripts/run_week7_labels_dash.py)
- Output artifacts (under the run directory):
  - labels.jsonl — fields: ensemble_id, family, label_text, evidence_tokens, context_summary, confidence, uncertainty, exemplars, annotator_stub, seed, schema_version
  - label_report.json — per-family coverage, mock agreement kappa, drift checks

Implementation notes:
- Label pipeline: [src/hypercircuit/semantics/labeling.py](src/hypercircuit/semantics/labeling.py)
  - finalize_labels() writes labels.jsonl and label_report.json deterministically
  - compute_agreement_metrics() returns mock Cohen’s kappa and drift stubs

### 3) Dashboards JSON export
Lightweight, dashboard-ready JSONs cross-link ensembles with labels and summarize coverage.

- Export functions: [src/hypercircuit/dashboards/export.py](src/hypercircuit/dashboards/export.py)
  - export_dashboard_ensembles(...) — dashboard_ensembles.json
  - export_dashboard_labels(...) — dashboard_labels.json
  - export_dashboard_summary(...) — dashboard_summary.json
- The Week 7 CLI emits these by default with provenance and stable payloads.

CLI acceptance and manifest:
- The Week 7 CLI finalizes the registry with:
  - label_coverage, agreement_kappa_mock
  - dashboard_sections_emitted
  - accept_week7 (bool; non-empty labels, kappa ≥ target, and all sections emitted)

### Determinism and tests
- Deterministic seeds are reused throughout. Re-running with identical config yields identical labels and dashboard checksums (timestamps sanitized in tests).
- See test coverage in [tests/test_week7_labels_dash.py](tests/test_week7_labels_dash.py)

### Quick reference
- 24-layer logging (top behaviors): enable logging.top_behaviors_only=true and use a dataset overlay with dataset.top_behaviors=true
- Labels+Dashboards CLI: python -m hypercircuit.cli.run_week7_labels_dash --config configs/base.yaml configs/discovery.yaml configs/dictionary.yaml
- Artifacts written under runs/&lt;run_id&gt;:
  - logs.jsonl (if you ran logging)
  - labels.jsonl
  - label_report.json
  - dashboard_ensembles.json
  - dashboard_labels.json
  - dashboard_summary.json


## Week 8: Freeze and Release

This phase freezes the current ensemble dictionary into an immutable snapshot with provenance, consolidates Gate 1–4, labeling, and dashboard summaries into a final report, emits a scaling recommendation, and registers a single release bundle manifest.

Key modules:
- Freezer: [freeze.py](src/hypercircuit/dictionary/freeze.py:1) providing [freeze_dictionary()](src/hypercircuit/dictionary/freeze.py:178) and [assemble_release_manifest()](src/hypercircuit/dictionary/freeze.py:291)
- Final report: [final_report.py](src/hypercircuit/eval/final_report.py:1) providing [assemble_final_report()](src/hypercircuit/eval/final_report.py:52)
- CLI orchestrator: [run_week8_release.py](src/hypercircuit/cli/run_week8_release.py:1) entrypoint [main()](src/hypercircuit/cli/run_week8_release.py:119)
- Release bundle checksums (helper): [export.py](src/hypercircuit/dashboards/export.py:1) providing [compute_release_bundle_checksum()](src/hypercircuit/dashboards/export.py:228)

Artifacts produced under the active run directory:
- ensemble_dictionary_frozen.json
- release_manifest.json
- final_report.json
- final_summary.md
- docs/SCALING_RECOMMENDATION.md

Determinism and IDs:
- snapshot_id = sha1 over {timestamp_bucket (YYYYMMDD from dictionary header), seed, sorted family:id:members} truncated to 12 chars
- release_bundle_checksum = stable aggregate MD5 over declared artifacts; JSON timestamps and markdown timestamps are sanitized before hashing

How to run (mock/deterministic):
- Module:
  - python -m hypercircuit.cli.run_week8_release --config configs/base.yaml configs/discovery.yaml configs/dictionary.yaml -o run.run_dir=runs/week8_release_mock
- Script wrapper:
  - python [scripts/run_week8_release.py](scripts/run_week8_release.py:1) --config configs/base.yaml configs/discovery.yaml configs/dictionary.yaml -o run.run_dir=runs/week8_release_mock

CLI orchestration:
1) Freeze dictionary via [freeze_dictionary()](src/hypercircuit/dictionary/freeze.py:178) → ensemble_dictionary_frozen.json with snapshot metadata
2) Assemble final_report + final_summary via [assemble_final_report()](src/hypercircuit/eval/final_report.py:52) consolidating Gate 1–4, labels, and dashboard summary
3) Emit SCALING_RECOMMENDATION.md (templated deterministic note)
4) Assemble release manifest via [assemble_release_manifest()](src/hypercircuit/dictionary/freeze.py:291) with artifact list, per-file checksums, acceptance flags, and release_bundle_checksum
5) Finalize registry stage “week8_release” with snapshot_id, n_ensembles_frozen, accept_gate1..accept_gate4, accept_release, release_bundle_checksum, and reasons (if any)

Tests:
- End-to-end coverage in [tests/test_week8_release.py](tests/test_week8_release.py:1) verifies artifact existence, schema, and determinism of snapshot_id and release_bundle_checksum.

<!-- BEGIN UPCOMING_VERSIONS -->

## Upcoming Versions

This repository is release‑ready for Version 1.0 (GO with conditions). The roadmap below summarizes near‑term versions and the V1 → V2 transition. For full details, see:
- Roadmap: [docs/ROADMAP_V1_to_V2.md](docs/ROADMAP_V1_to_V2.md)
- Contracts and Schemas: [docs/CONTRACTS_AND_SCHEMAS.md](docs/CONTRACTS_AND_SCHEMAS.md)
- Scaling Recommendation: [docs/SCALING_RECOMMENDATION.md](docs/SCALING_RECOMMENDATION.md)

Quick links:
- Dashboards/labels export: [scripts/run_week7_labels_dash.py](scripts/run_week7_labels_dash.py)
- Release bundle (freeze + manifest): [scripts/run_week8_release.py](scripts/run_week8_release.py)

### V1.1 Patch Series (stability and polish)

Status: Planned (near‑term)

Objectives
- Tighten safety edit specificity floors; expand benign control suites.
- Add stronger paraphrase/adversarial robustness sets beyond mocks.
- Improve necessity diagnostics and per‑family rollback drills.
- Documentation polish and dashboards usability improvements.

Key changes (planned)
- Editing/safety tunables in [configs/editing.yaml](configs/editing.yaml) and evaluation knobs in [configs/causal.yaml](configs/causal.yaml).
- Robustness evaluation corpus expansion (no breaking schema changes).
- Additional tests around determinism, schema, and acceptance flags.

Compatibility
- Backward compatible. Existing data contracts in [docs/CONTRACTS_AND_SCHEMAS.md](docs/CONTRACTS_AND_SCHEMAS.md) remain stable.

Migration from V1.0
- Re‑run Week 5/6 safety + robustness suites using existing CLIs; no code changes required for consumers.

### V1.5 Sidecar Preview (passive co‑evolving interpretability)

Status: Planned

Objectives
- Introduce a passive Sidecar that mirrors the hypergraph and applies manifold‑constrained hyperedge creation and limited LLC‑driven structural updates (preview only).
- Preserve dynamic→frozen graph split and provenance while adding Sidecar‑ingestible exports.

Key changes (planned)
- Optional Sidecar preview outputs (new artifacts) aligned to contracts in [docs/CONTRACTS_AND_SCHEMAS.md](docs/CONTRACTS_AND_SCHEMAS.md).
- No behavioral changes to core pipelines by default; opt‑in via configs (to be documented in the preview release notes).

Compatibility
- Backward compatible; Sidecar artifacts are additive/optional.

Migration from V1.0/1.1
- Enable preview flags (documented in the release notes). Existing consumers need not change unless ingesting Sidecar outputs.

### V2.0 HMCS Sidecar: Co‑evolving Interpretability

Status: Planned

Objectives
- Full Sidecar controller mode with manifold‑guided hyperedge creation, implicit differentiation, and spectral smoothing of structural updates.
- First‑class “ensemble‑level” safety edits and monitoring inside the Sidecar with rollback guarantees.

Key changes (planned)
- Stable Sidecar artifact sets and live coupling to circuit updates; extended dashboards.
- Strengthened acceptance criteria and o.o.d. robustness gates.

Compatibility
- Strong effort toward backward compatibility. Some config and artifact shape extensions likely; migration scripts will be provided.

Migration from V1.x
- Projected one‑click migration with a conversion CLI and automatic registry backfills (documented at release).

### Compatibility & Migration Summary

- Data contracts: Stable for V1.x; V2 adds optional Sidecar artifacts (see [docs/CONTRACTS_AND_SCHEMAS.md](docs/CONTRACTS_AND_SCHEMAS.md)).
- CLIs: Current CLIs remain; V1.5/V2 may add Sidecar‑focused entry points without breaking existing ones.
- Configs: V1.1 tightens thresholds; V1.5 introduces opt‑in preview flags; V2 adds Sidecar sections.
- Artifacts/Manifests: V1.0 freeze/manifest remain canonical; V2 adds Sidecar bundles alongside them.
- Determinism: Seeds and canonicalization remain enforced across versions.

### Safety Risks & Mitigations (tracked forward)

- Sycophancy: Maintain high specificity ratio and low benign degradation; broaden control domains.
- Jailbreak: Require transfer robustness under template mutations; reject if persistence falls below floor.
- Deception: Enforce minimality/sufficiency stability across contexts; add long‑context tests.
- Rollback: Keep per‑family rollback checkpoints in safety edits; practice smoke drills.

<!-- END UPCOMING_VERSIONS -->
