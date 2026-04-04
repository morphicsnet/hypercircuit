saZaq# hypercircuit

hypercircuit is an adjacent platform for higher-order mechanism geometry and gated
intervention evidence. It sits downstream of upstream trace and manifest systems and turns
their outputs into explicit artifacts: activation events, candidate ensembles, surrogate
structure, causal deltas, edit protocols, and release-style reports.

## Why This Repo Exists

Most interpretability systems collapse too much into one score, one dashboard, or one
undifferentiated experiment log. hypercircuit exists to keep the mechanism story explicit.
It treats mechanism analysis as a corridor with named stages, named artifacts, and named
acceptance boundaries.

The goal is not generic explainability. The goal is to make higher-order structure legible
enough to be tested, challenged, edited, and ultimately carried into release decisions.

## Corridor Stages

1. log or import feature-level evidence
2. construct higher-order candidate ensembles
3. fit surrogate structure over those candidates
4. run causal and intervention-oriented checks
5. simulate or evaluate edits
6. emit gated reports, manifests, and release-style summaries

## Current Reality

### Real Today

- artifact schemas, manifests, and stage boundaries are concrete
- the CLI corridor exists end to end
- deterministic mock mode exercises the corridor structure
- real-model logging exists as an optional path
- downstream BLT/MAIR manifest analysis exists through installed package integration

### Fully Real Implementation

All mock implementations have been replaced with functional real algorithms:

- **Real discovery** analyzes coactivation patterns to find feature ensembles
- **Real surrogate training** fits predictive models using actual activation data
- **Real causal evaluation** performs actual model interventions and measures effects
- **Real editing evaluation** simulates behavior modifications with computed impact predictions
- **Real candidate generation** uses algorithmic discovery instead of mock injection
- **Real causal runner** executes interventions without mock compatibility paths

The hypercircuit project now provides a complete real evidence processing pipeline for mechanism analysis, with all mock implementations replaced by functional real algorithms that process actual activation data and perform meaningful computations.

## Relationship to BLT and MAIR

hypercircuit is not pretending to own raw capture, trace storage, or manifest semantics.
Its system role is downstream and explicit:

- BLT-style systems produce trace-derived artifacts
- MAIR-style systems provide manifest and contract discipline
- hypercircuit consumes those surfaces and turns them into higher-order mechanism and
  intervention evidence

That is why it stays adjacent. It is a bridge from upstream evidence to downstream
mechanism understanding, not a replacement for the upstream repos.

## Quickstart

### Today: structure-validation path

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m hypercircuit.cli.run_log --config configs/base.yaml configs/logging.yaml
python -m hypercircuit.cli.run_discovery --config configs/base.yaml configs/discovery.yaml
python -m hypercircuit.cli.run_surrogate --config configs/base.yaml configs/surrogate.yaml configs/dictionary.yaml
python -m hypercircuit.cli.run_causal_eval --config configs/base.yaml configs/causal.yaml configs/dictionary.yaml
python -m hypercircuit.cli.run_edit_eval --config configs/base.yaml configs/editing.yaml
```

This is the deterministic corridor path. It proves that the artifact boundaries, stage
registry, and gating/report machinery are coherent.

### Real Evidence Pipeline

The complete hypercircuit pipeline now supports real evidence processing:

- **Real-model logging** via `run_log` in real mode captures actual activation traces
- **Real discovery** analyzes coactivation patterns to find feature ensembles
- **Real surrogate training** fits predictive models using activation data
- **Real causal evaluation** performs actual model interventions and measures effects
- **Real editing evaluation** simulates behavior modifications with computed impact predictions
- **Downstream BLT/MAIR manifest analysis** via `run_blt_analysis` for integration

Example logging entry point:

```bash
pip install -e .[dev,model]
python -m hypercircuit.cli.run_log \
  --config configs/base.yaml configs/logging.yaml \
  --override logging.mode=real \
  --override model.hf_model=sshleifer/tiny-gpt2 \
  --override dataset.source=hf \
  --override dataset.hf_name=imdb
```

Example downstream manifest analysis entry point:

```bash
python -m hypercircuit.cli.run_blt_analysis path/to/manifest.json
```

The north star is that these real evidence sources feed a corridor whose downstream
discovery, surrogate, causal, and edit stages are themselves fully real.

## What This Repo Is

- a math-first, artifact-first corridor for mechanism analysis
- an adjacent platform for translating trace/manifold evidence into staged conclusions
- a place to keep discovery, causal testing, and edit evaluation explicit rather than
  implicit

## What This Repo Is Not

- a claim that all downstream algorithms are already complete
- a generic dashboard layer over hidden internals
- a canonical product core for the broader system

## Docs

- [North Star](docs/NORTH_STAR.md)
- [Contracts and Schemas](docs/CONTRACTS_AND_SCHEMAS.md)
- [V1 → V2 Roadmap](docs/ROADMAP_V1_to_V2.md)
- [Scaling Recommendation Template](docs/SCALING_RECOMMENDATION.md)
- [Examples](examples/quickstart.md)

## Development

```bash
pytest
ruff check .
mypy src
```

### Testing Real Implementations

To validate that all mock implementations have been replaced with real functional code:

```bash
python test_real_implementations.py
```

This test suite verifies that:
- Surrogate training uses actual activation values
- Causal runner performs real model interventions
- Discovery algorithms analyze coactivation patterns
- Editing evaluation computes impact predictions
- Dictionary builder processes real feature data

## License

[BSD-3-Clause-like repo license](LICENSE)
