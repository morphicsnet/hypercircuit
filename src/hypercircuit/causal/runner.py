from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol

from hypercircuit.causal.harness import CausalHarness


class CausalRunner(Protocol):
    """Backend-agnostic causal runner interface."""

    def run(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass
class MockCausalRunner:
    """Default mock causal runner (wraps CausalHarness)."""
    harness: CausalHarness

    def run(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        # Placeholder: pass-through for now.
        return {"status": "mock", "payload": dict(payload)}

