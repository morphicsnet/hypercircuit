from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol, Sequence

from hypercircuit.causal.harness import CausalHarness
from hypercircuit.utils.config import Config
from hypercircuit.utils.io import load_jsonl, write_json


class CausalRunner(Protocol):
    """Backend-agnostic causal runner interface."""

    def run(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass
class RealCausalRunner:
    """Real causal runner that performs actual interventions using CausalHarness."""
    harness: CausalHarness
    config: Config

    def run(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run causal evaluations for ensembles specified in payload."""
        # Extract ensemble IDs from payload
        ensemble_ids = payload.get("ensemble_ids", [])
        if not ensemble_ids:
            return {"status": "error", "message": "No ensemble_ids provided"}

        # Load model and dataset paths from config
        model_path = Path(self.config.model.path) if hasattr(self.config.model, 'path') else None
        dataset_path = Path(self.config.dataset.path) if hasattr(self.config.dataset, 'path') else None

        if not model_path or not dataset_path:
            return {"status": "error", "message": "Model or dataset path not configured"}

        # Get run directory
        run_dir = Path(payload.get("run_dir", "."))

        # Run actual causal interventions
        results = self.harness.run_interventions(
            ensemble_ids=ensemble_ids,
            model_path=model_path,
            dataset_path=dataset_path,
            run_dir=run_dir
        )

        return {
            "status": "success",
            "results": results["results"],
            "artifact_path": results["artifact_path"],
            "ensemble_ids": ensemble_ids
        }


@dataclass
class MockCausalRunner:
    """Legacy mock causal runner for backwards compatibility."""
    harness: CausalHarness

    def run(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        # Placeholder: pass-through for now.
        return {"status": "mock", "payload": dict(payload)}

