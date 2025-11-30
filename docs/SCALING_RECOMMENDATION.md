# Scaling Recommendation Template (V1 → V1.5 → V2)

This template is programmatically updated by the Week 8 release CLI. The CLI writes a deterministic, metrics‑aware version under each run directory at:
- runs/<run_id>/docs/SCALING_RECOMMENDATION.md

Snapshot
- snapshot_id: {{SNAPSHOT_ID}}
- n_ensembles_frozen: {{N_ENSEMBLES}}
- release_bundle_checksum: {{RELEASE_BUNDLE_CHECKSUM}}

Gate acceptance (V1 status)
- Gate 1: {{GATE1_STATUS}}   # PASS/FAIL
- Gate 2: {{GATE2_STATUS}}   # PASS/FAIL
- Gate 3: {{GATE3_STATUS}}   # PASS/FAIL
- Gate 4: {{GATE4_STATUS}}   # PASS/FAIL
- Release decision: {{RELEASE_DECISION}}   # ACCEPT/REJECT

Budgets (mock/deterministic)
- Storage (current run artifacts): {{STORAGE_MB}} MB
- Compute envelope (stub): deterministic-mock

Observed outcomes and gaps
- {{REASONS_MULTILINE}}

Migration steps
- V1 → V1.5:
  - Expand evaluation coverage to all configured families; address failing gates from reasons.
  - Tighten stability/necessity thresholds as needed; re-run Gate 4 to confirm persistence.
  - Grow dashboard surface: add per-family drill-down and error budgets.
- V1.5 → V2:
  - Scale dataset variants and seeds; increase CV folds for surrogates.
  - Add regression guardrails: freeze API + manifests, enforce bundle checksums in CI.
  - Productionize run registry with artifact lineage queries and retention policy.

Resource envelopes
- Data: maintain 2× overhead on storage for concurrent runs and snapshots.
- Compute: allocate capacity for 2× current cell grid with half-matrix parity.
- People/time: reserve 1 sprint for remediation of reasons above.

Provenance
- Generated deterministically by the Week 8 CLI.
- snapshot_id: {{SNAPSHOT_ID}}
- bundle_checksum: {{RELEASE_BUNDLE_CHECKSUM}}