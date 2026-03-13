# Hypercircuit

## Overview
Hypercircuit is an interpretability workflow scaffold for SAE-first pipelines: discovery, surrogate modeling, causal evaluation, and safety edits. It supports both deterministic mock runs and real-model activation logging via local Hugging Face Transformers, with a versioned canonical event schema and feature-space adapters.

## Quickstart
Prerequisites:
- Python >= 3.9

Install (editable) and run the mock pipeline:
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

Real-model logging (optional):
```bash
pip install -e .[dev,model]
python -m hypercircuit.cli.run_log --config configs/base.yaml configs/logging.yaml \
  --override logging.mode=real \
  --override model.hf_model=sshleifer/tiny-gpt2 \
  --override dataset.source=hf \
  --override dataset.hf_name=imdb
```

Console script equivalents (from `pyproject.toml`):
- `hypercircuit-run-log`
- `hypercircuit-run-discovery`
- `hypercircuit-run-surrogate`
- `hypercircuit-run-causal-eval`
- `hypercircuit-run-edit-eval`
- `hypercircuit-run-week5-safety`
- `hypercircuit-run-week6-gate4`
- `hypercircuit-run-week7-labels-dash`
- `hypercircuit-run-week8-release`
- `hypercircuit-run-reconcile`

## Repository Map
- `src/hypercircuit/` core package and CLI entry points
- `src/hypercircuit/model_io/` model adapters (local HF capture)
- `src/hypercircuit/feature_io/` feature-space adapters and reconciliation stage
- `configs/` base and stage-specific YAML configs
- `scripts/` thin wrappers for local runs
- `tests/` smoke tests and config schema checks
- `docs/` contracts, roadmap, scaling template
- `examples/` walkthroughs

## Configuration
- Base config: `configs/base.yaml`
- Stage configs: `configs/logging.yaml`, `configs/discovery.yaml`, `configs/surrogate.yaml`, `configs/causal.yaml`, `configs/editing.yaml`, `configs/dictionary.yaml`
- Real-model sections: `model` (HF model + activation targets), `sae` (dictionary path/format), `dataset.source` (hf | jsonl)
- Run registry writes to `runs/<run_id>/` by default.

## Docs Index
- Contracts and schemas: `docs/CONTRACTS_AND_SCHEMAS.md`
- Roadmap: `docs/ROADMAP_V1_to_V2.md`
- Scaling template: `docs/SCALING_RECOMMENDATION.md`

## Development and Testing
```bash
pytest
ruff check .
mypy src
```

## Status and Limitations
- Mock mode is deterministic; real mode requires optional dependencies and local model weights.
- Safety edits are simulated and must be validated before real deployment.
