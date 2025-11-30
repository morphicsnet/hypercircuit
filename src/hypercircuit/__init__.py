"""
HYPERCIRCUIT package root.

This package provides a scaffold for an SAE-first interpretability workflow:
- hyperedge discovery (coactivation + synergy screening)
- surrogate modeling (interpretable monotone combiner)
- ensemble-level causal evaluation (ablations, transfers, patching)
- semantics (labeling)
- safety edits (ensemble scaling/dampening)
- evaluation (metrics)
- dashboards (export)
"""

from . import cli as cli  # re-export subpackage for static analyzers
__all__ = ["__version__", "cli"]
__version__ = "0.1.0"
