"Causal evaluation harness and patching utilities."

from .runner import MockCausalRunner
from hypercircuit.utils.adapter_registry import register_adapter

register_adapter("causal_runner", "mock", MockCausalRunner)

__all__ = ["MockCausalRunner"]

# Re-export common submodules for static analyzers and convenient imports
from . import harness as harness
from . import patching as patching
from . import reporting as reporting

__all__ = ["harness", "patching", "reporting"]
