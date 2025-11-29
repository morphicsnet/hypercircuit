from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

from hypercircuit.sae_io.loaders import FakeSAEDictionary
from hypercircuit.utils.io import save_jsonl
from hypercircuit.utils.seed import seed_context
from hypercircuit.utils.config import LoggingThresholds


@dataclass
class ActivationLogger:
    """Mock activation logger with per-node-type toggles and metrics.

    Generates deterministic mock activation events across instrumented layers
    and node types. Event rates are calibrated to land within the configured
    expected density band when using the baseline logging.yaml.
    """

    # Existing mock knobs (kept for back-compat)
    tokens_per_sample: int
    threshold: float
    hysteresis: float
    sparsity: float
    n_features: int
    seed: int

    # Week 1 instrumentation controls
    instrumented_layers: List[int] = field(default_factory=lambda: list(range(-12, 0)))
    token_window: int = 160
    thresholds: LoggingThresholds = LoggingThresholds()
    node_types: Mapping[str, bool] = field(
        default_factory=lambda: {
            "sae_features": True,
            "attn_heads": True,
            "mlp_channels": True,
            "residual_dirs": True,
        }
    )

    def _node_spaces(self) -> Dict[str, int]:
        """Mock sizes for each node type."""
        return {
            "sae_features": self.n_features,
            "attn_heads": max(4, min(16, self.n_features if self.n_features > 0 else 8)),
            "mlp_channels": max(8, self.n_features * 2),
            "residual_dirs": 16,
        }

    def _enabled_types(self) -> List[str]:
        return [k for k, v in self.node_types.items() if v]

    def _target_p_event(self) -> float:
        """Set per-(layer,node_type) event probability targeting ~6 events/token overall."""
        n_layers = max(1, len(self.instrumented_layers))
        n_types = max(1, len(self._enabled_types()))
        target_events_per_token = 6.0
        p = target_events_per_token / (n_layers * n_types)
        return max(1e-6, min(0.9, p))

    def _generate(self, n_samples: int) -> Tuple[List[Mapping[str, object]], Mapping[str, object]]:
        events: List[Mapping[str, object]] = []
        counts_by_type: Dict[str, int] = {t: 0 for t in self._enabled_types()}
        coverage_by_layer: Dict[int, int] = {L: 0 for L in self.instrumented_layers}

        with seed_context(self.seed):
            # Constructed for realism; sampling below uses RNG directly
            _ = FakeSAEDictionary(n_features=self.n_features, sparsity=self.sparsity)

            rng = np.random.default_rng(self.seed + 1337)
            p_event = self._target_p_event()
            node_spaces = self._node_spaces()
            enabled = self._enabled_types()

            for s in range(n_samples):
                for tok in range(self.tokens_per_sample):
                    for L in self.instrumented_layers:
                        any_layer_event = False
                        for nt in enabled:
                            if rng.random() < p_event:
                                space = node_spaces.get(nt, self.n_features)
                                node_id = int(rng.integers(0, max(1, space)))
                                events.append(
                                    {
                                        "sample_id": s,
                                        "token_index": tok,
                                        "layer": L,
                                        "node_type": nt,
                                        "node_id": node_id,
                                        "tokens": self.tokens_per_sample,
                                    }
                                )
                                counts_by_type[nt] = counts_by_type.get(nt, 0) + 1
                                any_layer_event = True
                        if any_layer_event:
                            coverage_by_layer[L] += 1

            # Ensure non-zero per-node-type counts in mock mode
            if enabled:
                for nt in enabled:
                    if counts_by_type.get(nt, 0) == 0:
                        L = self.instrumented_layers[0]
                        events.append(
                            {
                                "sample_id": 0,
                                "token_index": 0,
                                "layer": L,
                                "node_type": nt,
                                "node_id": 0,
                                "tokens": self.tokens_per_sample,
                            }
                        )
                        counts_by_type[nt] = 1
                        coverage_by_layer[L] += 1

        tokens_logged = n_samples * self.tokens_per_sample
        total_events = sum(counts_by_type.values())
        events_per_token = total_events / max(1, tokens_logged)
        ept_by_type = {k: v / max(1, tokens_logged) for k, v in counts_by_type.items()}

        covered_layers = sum(1 for L in self.instrumented_layers if coverage_by_layer.get(L, 0) > 0)
        layer_coverage_fraction = covered_layers / max(1, len(self.instrumented_layers))

        metrics: Dict[str, object] = {
            "events_per_token": float(events_per_token),
            "events_per_token_by_node_type": ept_by_type,
            "events_by_node_type": counts_by_type,
            "layer_coverage_fraction": float(layer_coverage_fraction),
            "covered_layers_count": int(covered_layers),
            "total_events": int(total_events),
            "tokens_logged": int(tokens_logged),
            "n_layers": int(len(self.instrumented_layers)),
        }
        return events, metrics

    def compute_metrics(self, n_samples: int) -> Mapping[str, object]:
        """Simulate and return metrics without writing artifacts."""
        _, metrics = self._generate(n_samples)
        return metrics

    def run(self, out_path: Path, n_samples: int) -> Mapping[str, object]:
        """Generate events, persist them to JSONL, and return metrics."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        events, metrics = self._generate(n_samples)
        save_jsonl(out_path, events)
        return metrics
