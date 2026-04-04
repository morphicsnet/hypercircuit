# Quickstart

This repo currently has two distinct stories:

- a deterministic structure-validation path that proves the corridor shape and artifact
  contracts
- an early real-evidence attachment path that connects logging or BLT/MAIR-backed inputs
  into a corridor that is still partly scaffolded downstream

Keep those stories distinct. Mock mode proves corridor integrity. Real mode proves that the
corridor can attach to real evidence sources.

## Today: Structure-Validation Path

1. Install: `pip install -e .[dev]`
2. Generate mock activations (JSONL): `python -m hypercircuit.cli.run_log --config configs/base.yaml configs/logging.yaml`
3. Mine candidates and synergy screen: `python -m hypercircuit.cli.run_discovery --config configs/base.yaml configs/discovery.yaml`
4. Fit surrogate: `python -m hypercircuit.cli.run_surrogate --config configs/base.yaml configs/surrogate.yaml`
5. Causal evaluation: `python -m hypercircuit.cli.run_causal_eval --config configs/base.yaml configs/causal.yaml`
6. Safety edits (simulation): `python -m hypercircuit.cli.run_edit_eval --config configs/base.yaml configs/editing.yaml`

Artifacts are written under the configured `run_dir`, including `logs.jsonl`,
`candidates.jsonl`, surrogate outputs, causal outputs, and edit outputs.

## Real-Path Intent

The real path is not yet a fully real end-to-end corridor, but two real attachments already
exist:

- real-model logging through `run_log` with `logging.mode=real`
- downstream manifest analysis through `run_blt_analysis` once BLT and MAIR are installed

Example real logging entry point:

```bash
pip install -e .[dev,model]
python -m hypercircuit.cli.run_log \
  --config configs/base.yaml configs/logging.yaml \
  --override logging.mode=real \
  --override model.hf_model=sshleifer/tiny-gpt2 \
  --override dataset.source=hf \
  --override dataset.hf_name=imdb
```

Example downstream BLT/MAIR analysis entry point:

```bash
python -m hypercircuit.cli.run_blt_analysis path/to/manifest.json
```

When that path hardens, the expectation is:

- upstream traces and manifests arrive with real provenance
- hypercircuit turns them into higher-order candidates and testable structure
- causal and edit stages operate on real evidence rather than deterministic stand-ins
- stage-gated outputs become defensible mechanism records rather than corridor-only proofs

See [README.md](../README.md) for the repo-level story and
[NORTH_STAR.md](../docs/NORTH_STAR.md) for the full destination statement.
