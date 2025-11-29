# Quickstart (Mock)

1) Install: `pip install -e .[dev]`
2) Generate mock activations (JSONL): `python -m hypercircuit.cli.run_log --config configs/base.yaml configs/logging.yaml`
3) Mine candidates and synergy screen: `python -m hypercircuit.cli.run_discovery --config configs/base.yaml configs/discovery.yaml`
4) Fit surrogate: `python -m hypercircuit.cli.run_surrogate --config configs/base.yaml configs/surrogate.yaml`
5) Causal evaluation: `python -m hypercircuit.cli.run_causal_eval --config configs/base.yaml configs/causal.yaml`
6) Safety edits (simulation): `python -m hypercircuit.cli.run_edit_eval --config configs/base.yaml configs/editing.yaml`

Artifacts: logs.jsonl, candidates.jsonl, surrogate.json, causal.json, edits.json under the configured run_dir.
