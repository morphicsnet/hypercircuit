"""
CLI entrypoints that wire configs to module stubs.
"""

# Re-export CLIs to satisfy static analyzers and ease discovery
from . import run_log as run_log  # noqa: F401
from . import run_discovery as run_discovery  # noqa: F401
from . import run_build_dictionary as run_build_dictionary  # noqa: F401
from . import run_surrogate as run_surrogate  # noqa: F401
from . import run_causal_eval as run_causal_eval  # noqa: F401
from . import run_edit_eval as run_edit_eval  # noqa: F401
from . import run_week4_interim as run_week4_interim  # noqa: F401
from . import run_week5_safety as run_week5_safety  # noqa: F401
from . import run_week6_gate4 as run_week6_gate4  # noqa: F401
from . import run_week7_labels_dash as run_week7_labels_dash  # noqa: F401
from . import run_week8_release as run_week8_release  # noqa: F401

__all__ = [
    "run_log",
    "run_discovery",
    "run_build_dictionary",
    "run_surrogate",
    "run_causal_eval",
    "run_edit_eval",
    "run_week4_interim",
    "run_week5_safety",
    "run_week6_gate4",
    "run_week7_labels_dash",
    "run_week8_release",
]
