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

__all__ = ["__version__"]
__version__ = "0.1.0"
