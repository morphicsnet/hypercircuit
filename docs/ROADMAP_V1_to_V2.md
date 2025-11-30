# V1 → V2 Integrated Roadmap

This document describes the approved plan to evolve Hypercircuit from the V1 scaffold to a V2-ready, deterministic and testable pipeline with ensemble discovery, surrogate modeling, causal evaluation, safety edit protocols, and gating. It consolidates architecture, data plan, interfaces, migration strategy, and a phased execution roadmap with acceptance criteria.

## Target architecture and principles

The system is a layered pipeline with clear module boundaries:
- Logging: deterministic activation/event logging and provenance.
- Discovery: co‑activation mining, synergy and stability screening, candidate pruning.
- Dictionary: ensemble selection and provenance with per‑family indices.
- Surrogate: monotone combiner with calibration and cross‑validation.
- Causal: ablations, transfers, and mock harness evaluation.
- Steering: safety edit planning, simulation, application, and monitoring.
- Evaluation and reporting: matrix evaluation, interim and gate reports.
- Utilities: typed configs, registry, IO helpers, seeds.

Architecture principles:
- Determinism first: every stage uses explicit seeds with recorded checksums.
- Artifacts as contracts: JSON/JSONL schemas are stable and versioned.
- Separation of concerns: discovery, modeling, evaluation, steering are independent.
- Progressive disclosure of risk: Gate 1 (discovery), Gate 2 (surrogate + causal), Gate 3 (safety edits).
- Back‑compat: V1 CLIs and artifacts remain usable while V2 features are added.

## Data and storage plan

Storage layout:
- runs/{run_id}/manifest.json with stage, seeds, config paths, summary, and status.
- runs/{run_id}/artifacts.jsonl with relative paths and metadata.
- Stage artifacts written under the run directory; file formats are JSON and JSONL.
- Artifact schema version tag is added to JSON via SCHEMA_VERSION; JSONL records are line‑delineated mappings.

Retention and lifecycle:
- Keep last N runs per stage (policy configurable), with optional offline archiving.
- Artifacts referenced by gate reports are retained as long as the report is part of program records.
- Manifests capture minimal provenance to reconstruct a report from artifacts.

Governance:
- All stage CLIs record config snapshots, seed lists, and determinism checksums in reports.
- Only mock/deterministic pipelines are implemented; no external model calls are permitted.

## Interfaces and contracts (overview)

Data contracts are expressed via stable JSON/JSONL shapes (detailed in CONTRACTS_AND_SCHEMAS):
- Feature activation logs: temporal spike sequences, per‑sample event bundles, and density summaries.
- Candidate hyperedges: members, support, synergy, stability, windows.
- Dynamic circuit graph and snapshot artifacts: edges, weights, provenance, freeze points.
- Manifold snapshots and geodesic adjacency: local charts, distances, and constraints.
- Time‑window indices and state snapshot bundles: per‑window indices, seeds, and pointers to inputs/outputs.

Core operational contracts (further defined in CONTRACTS_AND_SCHEMAS):
- Ensemble discovery contract.
- Circuit graph abstraction contract.
- Sidecar intake contract.
- Manifold constraint contract.
- Surrogate and causal evaluation contract.
- Safety edit protocols contract.

## Four‑phase roadmap with deliverables, gates, and risks

Phase 1 — Discovery and Gate 1 (Weeks 1–2)
- Deliverables: logs.jsonl, candidates.jsonl, gate1_report.json with counts, stability, exemplars, thresholds.
- Method: deterministic mining of co‑activation patterns, rank‑weighted supports, synergy, stability.
- Acceptance (Gate 1): storage_ok and non‑trivial stability counts; informational flag in mock CI.
- Key risks: over‑pruning due to tight thresholds; mitigated by explicit config and exemplars review.

Phase 2 — Surrogates, Causal Evaluation, and Interim (Weeks 3–4)
- Deliverables: surrogates_params.jsonl, surrogates_index.json, causal_eval_results.jsonl, gate2_report.json, matrix_results.jsonl, interim_report.json.
- Method: CV + calibration for monotone combiners; targeted causal evaluation; half‑matrix evaluation; FDR BH.
- Acceptance (Gate 2): per‑family thresholds on median causal impact and sufficiency reinstatement; aggregate stability and coverage checks in interim.
- Key risks: surrogate mis‑calibration and seed instability; mitigated by fixed seeding and calibration summaries.

Phase 2 — Week 5 Safety Edits and Gate 3 (this task)
- Deliverables: safety_edit_plans.jsonl, safety_eval_results.jsonl, gate3_report.json.
- Method: propose deterministic edit plans from surrogate sensitivities; simulate impact; staged application; monitoring loop with specificity ratio and benign degradation; rollback on breach; consolidated Gate 3 report.
- Acceptance (Gate 3): harmful_rate_reduction ≥ target, benign_deg_pct ≤ threshold, specificity_ratio ≥ threshold; global accept only if all observed families pass.
- Key risks: off‑target regressions; mitigated by control families, specificity metric floor, and rollback points.

Phase 3 — Manifold and Circuit Abstraction (Weeks 6–7)
- Deliverables: manifold snapshot artifacts, geodesic adjacency index, dynamic circuit graph snapshots.
- Method: freeze graph snapshots from ensembles and surrogate weights; approximate manifold charts from mock embeddings; geodesic adjacency via distance thresholds.
- Acceptance: construction completeness and referential integrity checks; determinism checksums for snapshot content.
- Risks: schema creep and artifact bloat; mitigated by schema consolidation and pruning policies.

Phase 4 — Hardening and Delivery (Week 8+)
- Deliverables: stabilized schemas, governance and retention policy, CI gates wired, developer docs, and quickstart scripts for all stages.
- Acceptance: all gates produce artifacts and acceptance flags under deterministic seeds in CI; documentation complete and linked from README.
- Risks: back‑compat regression; mitigated by schema versioning and integration tests.

## Migration plan (V1 → V2)

Back‑compatibility:
- Keep existing CLIs and artifact names; add new Week 5 CLI alongside without breaking prior tests.
- Extend editing and causal configs with optional blocks; defaults chosen to preserve V1 behavior.
- Reports retain prior fields; new fields are additive and non‑breaking.

Progressive migration steps:
- Introduce new steering modules (policies, monitor) and safety reporting under separate namespaces.
- Add evaluate_safety_suite entry to the causal harness with mock implementations; wire monitor to call it.
- Add Week 5 CLI and script wrapper; route registry wiring through start_run/log_artifact/finalize_run.
- Update README with Roadmap and Week 5 quick start; add CONTRACTS_AND_SCHEMAS doc.

Deprecations (none immediate):
- No removals; future deprecations will follow a two‑release notice with shims maintained.

## Metrics and acceptance bars

Program metrics:
- Coverage ratios for matrix evaluation; BH FDR threshold and discoveries count.
- Surrogate training medians: CV, calibration error; evaluation deltas.
- Gate 3 metrics per family: harmful_rate_reduction, benign_deg_pct, specificity_ratio, sufficiency_reinstatement_after_edit (mock).
- Determinism: per‑family top‑K checksums, cell‑set checksum, and plan selection signature.

Acceptance bars (mock defaults):
- Gate 1: informational counts_ok and storage_ok.
- Gate 2: per‑family median_causal_impact_delta > 0 and sufficiency_reinstatement_rate ≥ 0.7.
- Gate 3: benign_deg_pct ≤ 0.5; specificity_ratio ≥ 1.2; harmful_rate_reduction ≥ 0.2 global target.

## Compute, storage, and scheduling

Compute model:
- CPU‑only, mock numeric workloads; no external model servers.
- Seeded RNG for NumPy and Torch with context‑managed restoration.

Storage:
- JSON/JSONL artifacts only; schema tag embedded in JSON; size‑aware retention policy.
- Consolidated families reports to minimize redundant per‑family artifacts.

Scheduling:
- Local CLI orchestration; per‑stage run contexts; seeds recorded in manifest and reports.
- Optional parallelization reserved for future non‑mock execution.

## Roles and QA

Roles:
- Maintainers: own config schema, registry, and report assemblers.
- Contributors: implement mock algorithms and tests per stage.

QA strategy:
- Unit tests for schema and determinism.
- Integration tests per stage ensuring artifacts exist and contain required keys.
- Golden‑run reproducibility under fixed seeds and tight tolerances.

## Risks and mitigations

Key risks:
- Non‑determinism due to implicit RNG use.
- Config sprawl and silent default changes.
- Artifact footprint growth and retention failures.
- Over‑fitted acceptance thresholds or false positives in Gate 3.

Mitigations:
- seed_context and active seed propagation across all CLIs and modules.
- Typed Pydantic config with defaults and explicit schema extension for new features.
- Registry‑tracked artifacts and governance rules.
- Specificity floors, control families, and rollback points in monitoring.

## Timeline (indicative)

- Week 1–2: Discovery and Gate 1 reports.
- Week 3–4: Surrogates, Gate 2, interim matrix and report.
- Week 5: Safety edits planning, monitoring, and Gate 3 report (this task).
- Week 6–7: Manifold and circuit abstraction artifacts.
- Week 8+: Hardening, governance, and documentation finalization.

## Forward‑compatible V1 changes summary

- Add steering policies and monitoring modules; no changes to existing function signatures elsewhere.
- Extend config with SafetyEditConfig under editing and safety_eval block under causal; defaults preserve V1 behavior.
- Add evaluate_safety_suite in the causal harness and simulate_impact/apply_edit_plan in steering edits; previous compute_edit_map/apply_edits remain available.
- Introduce Week 5 CLI and wrapper; registry stage name "week5_safety" aligns with prior stage naming.
- README gains a Roadmap section and Week 5 quick start; docs directory includes ROADMAP_V1_to_V2 and CONTRACTS_AND_SCHEMAS.