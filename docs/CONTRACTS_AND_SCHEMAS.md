# Contracts and Schemas (Prose)

This document consolidates the core data contracts and prose schemas underpinning the
hypercircuit corridor. It focuses on stable shapes, field meanings, provenance,
determinism, and retention for an adjacent platform whose job is to turn upstream traces
and manifests into explicit higher-order mechanism artifacts.

The north-star thesis is defined in [NORTH_STAR.md](NORTH_STAR.md). These contracts should
be read as the artifact discipline that makes that thesis auditable. Some current producers
are still mock-friendly or scaffolded, but the contracts themselves are written to support
both structure-validation mode and the more fully real downstream path.

Operational notes:
- CLI entry points are exposed via console scripts (for example, `hypercircuit-run-log`, `hypercircuit-run-week5-safety`).
- Artifacts are written under `runs/<run_id>/` by default (configurable via `run.output_dir`).

Sections:
- Feature activation logs (temporal spike sequences)
- Candidate hyperedges (co-activation, synergy, stability, windows)
- Dynamic circuit graph and frozen snapshot artifacts
- Manifold snapshots and geodesic adjacency
- Time-window indices and state snapshot bundles
- Six core operational contracts: ensemble discovery, circuit graph abstraction, sidecar intake, manifold constraint, surrogate and causal evaluation, safety edit protocols
- Governance, retention, and migration notes

## 1) Feature activation logs (temporal spike sequences)

Purpose:
- Capture instrumented feature events per sample with minimal yet sufficient provenance for
  downstream mining, stability checks, and later mechanism evaluation.

Prose schema:
- One record per event (canonical event stream).
- Required canonical fields:
  - schema_version: string, event schema version.
  - run_id: string, run identifier.
  - source_kind: mock | hf_local | api_trace | posthoc_import | hardware_capture.
  - feature_space_id: string, feature space identifier.
  - feature_space_version: string, feature space version.
  - sample_id: integer, stable per dataset ordering.
  - sequence_id (or doc_id): integer, stable sequence/document identifier.
  - token_index: integer, position within the sample.
  - layer: integer or signed index (negative indicates backward-from-end convention).
  - node_type: string label (e.g., sae_features, attn_heads).
  - node_id: integer feature index.
  - value: float activation value (mock uses a constant).
  - step_index: integer token step for ordering.
- Identity lineage fields:
  - dictionary_id, dictionary_version, dictionary_type.
  - feature_origin_layer.
  - member_key: canonical member id (feature_space_id:layer:node_type:node_id).
  - feature_key: canonical feature id (includes feature_space_version and lineage).
- Optional tags:
  - task_family, prompt_family, split, label, capability_tag, safety_tag.
- Determinism:
  - Event density is controlled by configuration thresholds; seeds fix any randomized selections (mock).
- Storage:
  - JSONL file with one mapping per line.
  - Accompanying `events_manifest.json` records schema version, feature space, model/tokenizer, and dataset provenance.

## 2) Candidate hyperedges (co-activation, synergy, stability, windows)

Purpose:
- Represent discovered sets of features (size 2 or 3 in mock) that co-activate with measurable support and synergy, with stability across replicates or partitions.

Prose schema:
- One JSONL record per candidate with:
  - candidate_id: stable string derived from members and candidate_type.
  - candidate_type: string (coactivation by default).
  - members: ordered list of member keys; order is stable for determinism.
  - arity: integer cardinality (2 or 3 in mock).
  - support_count: integer global support count.
  - weighted_support: float rank-weighted support (monotone in support).
  - synergy_score: float, candidate score minus max proper-subset score, floored at zero.
  - stability_score: float in [0,1], rank-correlation-based across replicates; optional multi-replicate variant.
  - window_span: integer temporal span used for co-activation within windowing.
  - formation_rule: string identifier of the mining algorithm.
  - scores: {coactivation_score, synergy_score, stability_score, calibration_score, causal_score, final_rank_score}.
- Back-compat fields are preserved in mock (size/support).
- Accompanying `candidates_manifest.json` records member granularity and score bundle availability.

## 2.5) Calibration + reconciliation artifacts

Purpose:
- Persist real-path calibration thresholds and stable-feature reconciliation outputs for reproducibility.

Artifacts:
- `calibration.json`:
  - dataset_slice: counts of events and candidates used for calibration.
  - thresholds: discovery thresholds explored or selected.
  - score_distributions: quantiles of weighted_support, synergy_score, stability_score.
- `feature_alignment.json`:
  - mapping from member_key → stable_node_id (identity for now).
- `consensus_features.json`:
  - consensus feature list (identity passthrough by default).

## 3) Dynamic circuit graph and frozen snapshot artifacts

Purpose:
- Provide a graph abstraction that can be frozen at a point in time for downstream reporting, safety planning, and manifold constraints.

Prose schema:
- Snapshot artifact (single JSON) with:
  - snapshot_id: stable string (run_id plus checksum).
  - nodes: list of nodes with fields:
    - id: string node identifier (aligned with dictionary member ids where applicable).
    - type: node category (feature, head, channel, residual).
    - attrs: small bag of numeric attributes (degree, frequency, mock centrality).
  - edges: list of edges with fields:
    - src, dst: node ids.
    - weight: float in [0,1] (mock). Represents association strength or directed influence proxy.
    - window_constraints: optional structure indicating active time windows.
  - ensembles_index: optional mapping from ensemble_id to member node ids.
  - provenance: config snapshot minimal fields, seeds list, and determinism checksum for top-K structures.
  - created_at: ISO 8601 string.
- Frozen snapshots are append-only; later rebuilds must either keep id stable or create a new id with explicit supersedes pointer.

## 4) Manifold snapshots and geodesic adjacency

Purpose:
- Summarize a local geometric view suitable for constraint checking and adjacency-aware operations during safety planning (mock distances).

Prose schema:
- Manifold snapshot (single JSON) with:
  - manifold_id: stable string (run_id plus checksum).
  - points: list of items with:
    - id: node or ensemble identifier embedded into manifold space.
    - chart: small fixed-dimensional coordinate list (mock values).
    - mass: optional float weighting.
  - geodesic_adjacency: list of adjacency records with:
    - id_a, id_b: point identifiers.
    - distance: float distance in manifold coordinates (mock).
    - within_threshold: boolean flag indicating adjacency under configured threshold.
  - constraints: optional set of inequalities capturing allowed edit regions (prose format).
  - provenance: seeds, config parameters affecting coordinates, and determinism checksums.

## 5) Time-window indices and state snapshot bundles

Purpose:
- Provide quick lookup for events and states per time window, and to encapsulate state needed to resume or rollback.

Prose schema:
- Time-window index (single JSON) with:
  - window_size: integer as configured.
  - windows: list of entries:
    - window_index: integer.
    - sample_ids: list of integers.
    - active_feature_counts: mapping of feature id to count in window.
- State snapshot bundle (single JSON) with:
  - bundle_id: stable string (run_id plus checksum).
  - includes: URIs or relative paths for required artifacts to resume or rollback (logs, candidates, surrogate index, edit plan state).
  - seeds: list of integers in effect.
  - created_at: ISO 8601 string.

## 6) Six core operational contracts

These describe how modules expect and provide data across the pipeline.

Ensemble discovery contract:
- Input: feature activation logs and discovery thresholds.
- Output: candidate hyperedges with synergy and stability annotations and acceptance flags.
- Guarantees:
  - Stable sorting and deterministic id formation.
  - Thresholds and provenance embedded in each record.

Circuit graph abstraction contract:
- Input: ensembles/dictionary, surrogate signal strengths, and optional co-activation edges.
- Output: dynamic graph snapshot with typed nodes and weighted edges.
- Guarantees:
  - Node ids align with dictionary members.
  - Edge weights are normalized and bounded.

Sidecar intake contract:
- Input: existing run artifacts referenced by relative paths and verified via checksums.
- Output: in-memory indexable views for downstream steps and optional flat exports for dashboards.
- Guarantees:
  - Read-only behavior; original artifacts are not modified.
  - Determinism via sorted enumerations and fixed-seed sampling when needed.

Manifold constraint contract:
- Input: graph snapshot and optional embeddings to define a local manifold.
- Output: manifold snapshot with geodesic adjacency and optional constraint descriptors.
- Guarantees:
  - Distances are consistent with configured metric and thresholds.
  - Constraints are serializable and auditable in prose.

Surrogate and causal evaluation contract:
- Input: dictionary entries and logs; surrogate params (weights, intercepts); causal harness knobs.
- Output: causal_eval_results.jsonl with deltas, sufficiency reinstatement, and minimality pass indicators; gate reports summarizing family metrics.
- Guarantees:
  - Stable selection of ensembles per family.
  - Seeded partitioning for mock source and target sets.

Safety edit protocols contract:
- Input: surrogate sensitivities, risky families list, safety thresholds (benign degradation cap, specificity minimum ratio, target harmful reduction), and a step schedule.
- Output: safety_edit_plans.jsonl with proposed and applied per-step scales; safety_eval_results.jsonl with per-step metrics; gate3_report.json summarizing acceptance.
- Guarantees:
  - Monotonic application of scales up to max edit scale.
  - Rollback points created before application; rollback triggered on threshold breach.

## Governance and retention

Governance:
- Each stage writes artifacts.jsonl entries with kind and minimal metadata.
- Gate reports embed seeds, thresholds, and determinism checksums.
- Changes to schemas are additive; schema_version tagging is used for JSON payloads.

Retention:
- Program policy persists any artifact referenced by a gate report.
- Non-referenced intermediates may be pruned after a grace period.
- Manifests and artifacts.jsonl are retained for audit.

## Migration notes

Compatibility:
- Existing artifacts remain valid; new shapes are additive and optional.
- Where a new artifact supersedes an older one, a supersedes pointer is added to the snapshot.

Incremental adoption:
- Safety edit protocols and monitoring can operate on existing dictionary and surrogate artifacts without modification.
- Manifold/circuit abstractions can be introduced behind optional configuration blocks, producing additional summaries without altering prior reports.

Determinism:
- All new modules follow the seeds policy and record checksums of selection signatures and top-K aggregates for reproducibility.
