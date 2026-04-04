# hypercircuit North Star

hypercircuit exists to turn higher-order mechanism structure into explicit, testable
geometric and causal artifacts. It is the part of the system that takes upstream traces,
feature events, and manifests and asks the harder question: what mechanism is actually
there, what survives intervention, what composes, and what edits are admissible without
destroying the surrounding structure?

## Core Objects

The repo is organized around a small set of mathematical and operational objects that must
become explicit artifacts rather than hidden intermediate state:

- activation events over layers, token positions, node types, and feature identities
- candidate ensembles representing higher-order co-activation or mechanism structure
- surrogate structure that approximates how those ensembles behave across tasks or families
- circuit snapshots that freeze a view of the active relational structure
- manifold constraints and local geometric neighborhoods
- causal deltas and intervention outcomes
- edit protocols and monitoring records
- gated release records that say what evidence was strong enough to carry forward

These objects matter because they separate three questions that are too often blurred
together:

1. what pattern appears in the trace
2. what mechanism survives stronger tests
3. what intervention is justified by the evidence

## North-Star Corridor

The intended end-state corridor is:

1. ingest real traces and manifests from upstream systems
2. construct higher-order mechanism candidates with explicit provenance
3. fit and test surrogate structure without collapsing the candidate identity story
4. run causal and intervention checks against real evidence
5. emit stage-gated records stating what mechanisms exist, how they compose, and what edits
   are admissible

The destination is not “better dashboards.” The destination is a disciplined mechanism
evidence system in which geometry, causality, and release gating are all part of the same
artifact chain.

## Current Truth

### What Is Already Real

- the repo already treats artifacts and manifests as first-class objects
- the stage boundaries and CLI surfaces are concrete
- deterministic mock mode exercises the corridor shape end to end
- real-model logging exists as an attachment path
- BLT/MAIR-backed downstream analysis exists through installed-package integration

### What Is Still Mock or Scaffolded

- several downstream discovery, surrogate, causal, and editing surfaces still depend on mock
  or placeholder logic
- some stages still inject deterministic stand-ins to keep the corridor stable
- the default causal path still preserves a mock compatibility mode
- “fully real end-to-end hypercircuit” is not the current state of the repo

That gap should be stated plainly. The north star is not an excuse to pretend the current
pipeline is already there. It is the standard against which the repo should be judged.

## What hypercircuit Is

- an adjacent platform for mechanism geometry and intervention evidence
- a downstream bridge from trace/manifests to higher-order structure and gated conclusions
- a math-first corridor that keeps stage boundaries explicit

## What hypercircuit Is Not

- a replacement for upstream capture and manifest systems
- a generic explainability workbench
- a polished product core simply because it has many stages and reports

## Why It Stays Adjacent

hypercircuit is strongest when it remains explicit about its system role.

- upstream systems such as BLT own trace capture and artifact production
- manifest systems such as MAIR own contract discipline and receipt structure
- hypercircuit takes those surfaces and asks what higher-order mechanism claims can survive
  discovery, surrogate fitting, causal testing, and edit evaluation

If it tries to absorb everything, it weakens its own sharpest contribution. Its leverage is
not “being the whole stack.” Its leverage is being the place where higher-order mechanism
claims are made explicit, audited, and tested.

## Why The Repo Matters Before It Is Fully Real

Even in its current state, hypercircuit already does something important: it preserves the
right corridor shape. That shape matters because once the interfaces are explicit, the mock
surfaces can be replaced stage by stage without losing the provenance story.

The repo matters now because it already encodes:

- explicit stage contracts
- explicit artifact surfaces
- explicit gate boundaries
- an explicit distinction between structure-validation mode and real evidence mode

That is the right substrate for hardening the real path.

## North-Star Test

hypercircuit should count as having reached its intended form only when all of the following
are true:

- real traces and manifests are first-class inputs, not optional edge paths
- downstream candidate construction, surrogate structure, causal evaluation, and edit
  protocols are real implementations rather than deterministic stand-ins
- every major stage emits artifacts that are both auditable and meaningful
- the geometry/manifold language corresponds to computed constraints, not decorative framing
- gate and release records summarize real mechanism evidence rather than scaffolded plumbing
- the repo can state, with discipline, not just what it found but what interventions are
  justified and what remains unproven

That is the north star: not a corridor that merely runs, but a corridor that can defend its
mechanism claims.
